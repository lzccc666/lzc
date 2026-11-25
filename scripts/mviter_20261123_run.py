"""固定A/C/D三方法重放；D仅一个登记机制，复用相同双范围评价。"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
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
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mviter_20261123_front_anchor import FrontAnchoredMapper
from scripts import multiview_20261123_run_graph as original


P = 'mviter_20261123_'
REPORT = PROJECT/'outputs/reports/multiview_task_iteration_20261123/mapping'
OLD = PROJECT/'outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1'


def array_payload(mapper):
    """状态数组/点云逻辑payload；不把它冒充进程RSS或Python总内存。"""
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes
        if hasattr(value, 'element_size') and hasattr(value, 'numel'):
            return value.element_size()*value.numel()
        if hasattr(value, 'points'):
            return np.asarray(value.points).nbytes+np.asarray(value.colors).nbytes
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return 0
    return size([mapper.objects, mapper.last_observations, mapper.position_layer.tracks,
                 mapper.position_layer.dense_states, mapper.measurement_sources])


def main():
    start = time.monotonic()
    manifest_path = PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json'
    truth = manifest_path.parent/'multiview_20261123_evaluation_ground_truth.jsonl'
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads((OLD/'multiview_20261123_release.json').read_text())
    registration = json.loads((REPORT/(P+'candidate_registration.json')).read_text())
    run = experiment_directory('nuscenes', 'validation', P+'candidate_v1', create=True)
    source_paths = [Path(__file__).resolve(), PROJECT/'conceptgraph/slam/mviter_20261123_diagnostics.py',
                    PROJECT/'conceptgraph/slam/mviter_20261123_front_anchor.py',
                    PROJECT/'tests/mviter_20261123_test_anchor.py']
    source_paths.extend(Path(p) for p in previous['source_sha256'])
    sources = {}
    for i, path in enumerate(dict.fromkeys(source_paths)):
        target = run/(P+f'source_{i:02d}_'+path.name)
        shutil.copy2(path, target)
        sources[str(path)] = dict(sha256=digest(path), snapshot=str(target))
    for path, sha in previous['input_sha256'].items():
        assert digest(path) == sha
    settings = FrontAnchoredMapper(OmegaConf.create(previous['config']), CLASSES).settings_record()
    assert settings['anchor'] == registration['parameters']
    dump(run/(P+'release.json'), dict(started_utc=datetime.now(timezone.utc).isoformat(),
        config=previous['config'], candidate='D_front_anchor_current_support', settings=settings,
        methods=['A_front', 'C_three_dedup', 'D_front_anchor'], candidate_count=1,
        registration_sha256=digest(REPORT/(P+'candidate_registration.json')),
        input_sha256=previous['input_sha256'], source_sha256=sources, seed=17,
        GT_in_inference=False, test_accessed=False, default_changed=False,
        scope='same 2 scenes40groups105union77front same1log; not independent test',
        peak_RSS_definition='one CPU process with A/C/D(2 maps) co-resident; includes imports/images/audit; not isolated method RSS',
        state_payload_definition='per method sum logical numpy/tensor/pointcloud payload, object-identity dedup; excludes Python overhead; D sums both maps',
        maximum_wall_s=1800, maximum_new_bytes=512*2**20, minimum_free_bytes=20*2**30))
    baseline_exact, costs, timing = Counter(), defaultdict(Counter), []
    for scene in manifest['scenes']:
        name = scene['scene']
        old = list(map(json.loads, gzip.open(OLD/name/'multiview_20261123_predictions.jsonl.gz', 'rt')))
        assert len(scene['groups']) == len(old) == 20
        destination = run/name
        destination.mkdir()
        a = SourceAuditedMapper(OmegaConf.create(previous['config']), CLASSES, deduplication=False)
        c = SourceAuditedMapper(OmegaConf.create(previous['config']), CLASSES, deduplication=True)
        d = FrontAnchoredMapper(OmegaConf.create(previous['config']), CLASSES)
        with gzip.open(destination/(P+'predictions.jsonl.gz'), 'wt') as out:
            for group, reference in zip(scene['groups'], old):
                if time.monotonic()-start > 1800 or shutil.disk_usage(PROJECT).free < 20*2**30:
                    raise RuntimeError('预算/磁盘保护')
                assert (group['frame_idx'], group['sample_token']) == (reference['frame_idx'], reference['sample_token'])
                loaded = {cam: original.load_frame(group, cam) for cam in group['views']}
                predictions, audit = {}, {}
                for method, mapper, cameras in [('A_front', a, ['CAM_FRONT']),
                                                ('C_three_dedup', c, list(group['views']))]:
                    np.random.seed(17+group['frame_idx'])
                    mapper.capture_times = {cam: group['views'][cam]['camera_timestamp_s'] for cam in cameras}
                    begun = time.perf_counter()
                    frames = [(cam, copy.deepcopy(loaded[cam])) for cam in cameras]
                    mapper.update_group(frames)
                    predictions[method] = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                    elapsed = time.perf_counter()-begun
                    assert predictions[method] == reference['predictions'][method]
                    assert mapper.frame_audit == reference['audit'][method]['mapping']
                    baseline_exact[method] += 1
                    audit[method] = dict(mapping=mapper.frame_audit, crossview=mapper.crossview_audit)
                    costs[method]['mapping_s'] += elapsed
                    payload = array_payload(mapper)
                    costs[method]['peak_state_array_payload_bytes'] = max(costs[method]['peak_state_array_payload_bytes'], payload)
                    timing.append(dict(scene=name, frame_idx=group['frame_idx'], method=method,
                        mapping_s=elapsed, state_array_payload_bytes=payload))
                begun = time.perf_counter()
                predictions['D_front_anchor'] = d.update_group(list(loaded.items()),
                    {cam: view['camera_timestamp_s'] for cam, view in group['views'].items()})
                elapsed = time.perf_counter()-begun
                costs['D_front_anchor']['mapping_s'] += elapsed
                costs['D_front_anchor']['removed_outputs'] += d.audit['canonicalization']['removed']
                payload = array_payload(d.front)+array_payload(d.multi)
                costs['D_front_anchor']['peak_state_array_payload_bytes'] = max(costs['D_front_anchor']['peak_state_array_payload_bytes'], payload)
                timing.append(dict(scene=name, frame_idx=group['frame_idx'], method='D_front_anchor',
                    mapping_s=elapsed, state_array_payload_bytes=payload))
                audit['D_front_anchor'] = d.audit
                # Audit the independently anchored front map as an implementation invariant.
                assert d.raw_predictions['front'] == predictions['A_front']
                assert d.raw_predictions['multi'] == predictions['C_three_dedup']
                assert d.front.frame_audit == a.frame_audit
                assert d.multi.frame_audit == c.frame_audit
                out.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'], sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'], predictions=predictions,
                    audit=audit, GT_in_inference=False), allow_nan=False)+'\n')
                print(name, group['frame_idx'], 'A_C_D_complete', flush=True)
    # Only module-local output path/prefix is redirected in this isolated process.
    # The frozen evaluation function and historical source files are unchanged.
    original.DEST, original.P = REPORT, P
    result = original.evaluate(run, manifest, truth)
    original.write_csv(REPORT/(P+'cost_frame.csv'), timing)
    report = dict(complete=True, scenes=2, groups=40, logs=1, results=result,
        baseline_exact=dict(baseline_exact), costs={m: dict(v) for m, v in costs.items()},
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        wall_s=time.monotonic()-start, source_sha256=sources, input_sha256=previous['input_sha256'],
        ground_truth_sha256={str(truth): digest(truth)},
        outputs_sha256={str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()},
        csv_sha256={str(p): digest(p) for p in REPORT.glob(P+'*.csv')},
        GT_in_inference=False, frontend_calls=0, default_changed=False, test_accessed=False)
    dump(run/(P+'report.json'), report)
    dump(REPORT/(P+'result.json'), dict(run=str(run), **report))
    print(json.dumps(dict(node=result['node_main'], side=result['side_coverage'], costs=report['costs'],
                         peak_RSS=report['peak_process_RSS_bytes'], wall_s=report['wall_s']), indent=2))


if __name__ == '__main__':
    main()
