"""复用冻结40组缓存，运行独立L1生命周期候选并按原3m完整评价。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_registry import CanonicalLifecycle, LifecycleSettings
from scripts import multiview_20261123_run_graph as original


P = 'mvrepair_lifecycle_20261123_'
REPORT = PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'
OLD = PROJECT/'outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1'


def bridge_check():
    # 接口反例，不是实测成绩：front A与front C分别支持side B，A/C不得传递合并。
    common = dict(category='car', feature=[1., 0.], center=[0., 0., 0.], capture_timestamp_s=1.)
    obs = [dict(**common, track_id='A', source_key=['front', 1., 0], real_points=[[0., 0., 0.], [.1, 0., 0.]]),
           dict(**common, track_id='B', source_key=['side', 1., 0], real_points=[[0., 0., 0.], [.1, 0., 0.], [.4, 0., 0.], [.5, 0., 0.]]),
           dict(**common, track_id='C', source_key=['front', 1., 1], real_points=[[.4, 0., 0.], [.5, 0., 0.]])]
    reg = CanonicalLifecycle()
    audit = reg.observe_group(0, 1., obs)
    assert reg.resolve('A') == reg.resolve('B') and reg.resolve('A') != reg.resolve('C')
    assert any(w['reason'] == 'component_complete_link_conflict' for w in audit['witnesses'])
    assert max(audit['unique_group_hits'].values()) == 1
    # 同源双state例外：相同相机、曝光、检测index可统一；同组仍一次命中。
    twin = copy.deepcopy(obs[0]); twin['track_id'] = 'dense_twin'
    audit2 = reg.observe_group(1, 1.5, [dict(o, source_key=[o['source_key'][0], 1.5, o['source_key'][2]], capture_timestamp_s=1.5)
                                    for o in [obs[0], twin]])
    assert reg.resolve('A') == reg.resolve('dense_twin')
    assert audit2['unique_group_hits'][reg.resolve('A')] == 2
    return dict(bridge_rejected=True, identical_sourcekey_allowed=True, same_group_hits_not_added=True,
                synthetic_interface_checks_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    registration = json.loads((REPORT/'registration.json').read_text())
    previous = json.loads((OLD/'multiview_20261123_release.json').read_text())
    manifest = json.loads((PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json').read_text())
    run = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    destination = REPORT/args.run_id
    destination.mkdir()
    source_paths = [Path(__file__).resolve(), PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_registry.py',
                    PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_mapper.py',
                    PROJECT/'conceptgraph/slam/mviter_20261123_diagnostics.py']
    source_paths += [Path(p) for p in previous['source_sha256']]
    frozen = {}
    for i, path in enumerate(dict.fromkeys(source_paths)):
        snapshot = run/(P+f'source_{i:02d}_'+path.name)
        shutil.copy2(path, snapshot)
        frozen[str(path)] = dict(sha256=digest(path), snapshot=str(snapshot))
    for path, sha in previous['input_sha256'].items():
        assert digest(path) == sha
    check = bridge_check()
    dump(run/(P+'release.json'), dict(started_utc=datetime.now(timezone.utc).isoformat(),
        candidate='L1_current_real_witness_persistent_canonical_retirement', seed=17,
        settings=asdict(LifecycleSettings()), source_sha256=frozen,
        inputs_sha256=previous['input_sha256'], original_config=previous['config'],
        scope='same 40groups/2scenes/1log;105union77front28side;original3m;development',
        mechanism='component-complete current real support -> persistent canonical; retire covered state only; no global TTL edit',
        guardrails='retain C side14/union37, target dup<=A2/frontTP>=A24; report failure unchanged',
        GT_in_inference=False, interface_checks=check, registration=registration))
    audit_totals = dict(accepted_links=0, retired_states=0, reduced_outputs=0, ambiguous_groups=0)
    for scene in manifest['scenes']:
        name = scene['scene']
        old = list(map(json.loads, gzip.open(OLD/name/'multiview_20261123_predictions.jsonl.gz', 'rt')))
        mapper = LifecycleRetiringMapper(OmegaConf.create(previous['config']), CLASSES)
        outdir = run/name
        outdir.mkdir()
        with gzip.open(outdir/(P+'predictions.jsonl.gz'), 'wt') as out:
            for group, reference in zip(scene['groups'], old):
                if time.monotonic()-started > 1200 or shutil.disk_usage(PROJECT).free < 20*2**30:
                    raise RuntimeError('bounded run budget/disk protection')
                np.random.seed(17+group['frame_idx'])
                frames = [(c, original.load_frame(group, c)) for c in group['views']]
                mapper.capture_times = {c: group['views'][c]['camera_timestamp_s'] for c, _ in frames}
                mapper.update_group(frames)
                predictions = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                audit = mapper.lifecycle_audit
                audit_totals['accepted_links'] += sum(w['accepted'] for w in audit['registry']['witnesses'])
                audit_totals['retired_states'] += len(audit['retired_states'])
                audit_totals['reduced_outputs'] += audit['raw_output_count']-len(predictions)
                audit_totals['ambiguous_groups'] += bool(audit['registry']['ambiguous_current_roots'])
                out.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'], sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'], predictions={
                        'A_front': reference['predictions']['A_front'],
                        'C_three_dedup': reference['predictions']['C_three_dedup'],
                        'L1_lifecycle': predictions}, audit={'L1_lifecycle':audit}, GT_in_inference=False), allow_nan=False)+'\n')
    original.DEST, original.P = destination, P
    truth = PROJECT/'dataset/processed/multiview_20261123_pilot_v1/multiview_20261123_evaluation_ground_truth.jsonl'
    results = original.evaluate(run, manifest, truth)
    report = dict(run=str(run), complete=True, results=results, audit_totals=audit_totals,
        wall_s=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        interface_checks=check, GT_in_inference=False, new_frontend_model_GPU_training_CARLA_calls=0,
        ground_truth_sha256={str(truth):digest(truth)}, source_sha256=frozen,
        frozen_sources_unchanged=all(digest(p)==v['sha256'] for p,v in frozen.items()))
    dump(run/(P+'report.json'), report)
    dump(destination/'result.json', report)
    print(json.dumps({k:report[k] for k in ['run','results','audit_totals','wall_s','interface_checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
