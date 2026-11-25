"""40组同配置确定性重放、得失审计及历史指纹保护核验。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.multiview_20261123_fusion import GroupedR1Mapper
from scripts.multiview_20261123_run_graph import load_frame


P = 'multiview_20261123_'
REPORT = PROJECT/'outputs/reports/multiview_semantic_tasks_20261123/graph'


def read(path):
    return json.loads(path.read_text())


def records(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    release = read(args.input_run/(P+'release.json'))
    report = read(args.input_run/(P+'report.json'))
    manifest = read(args.manifest)
    for path, sha in release['input_sha256'].items():
        assert digest(path) == sha, path
    for path, source in release['source_sha256'].items():
        assert digest(path) == digest(source['snapshot']) == source['sha256'], path
    for path, sha in report['output_sha256'].items():
        assert digest(args.input_run/path) == sha, path
    for path, sha in report['report_csv_sha256'].items():
        assert digest(path) == sha, path
    protected = read(REPORT/(P+'start.json'))['preserve']
    for path, sha in protected.items():
        assert digest(path) == sha, path
    destination = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    dump(destination/(P+'release.json'), dict(input_run=str(args.input_run),
        input_release_sha256=digest(args.input_run/(P+'release.json')), source_sha256=digest(__file__),
        maximum_wall_s=600, maximum_new_bytes=32*2**20, seed=release['seed'],
        purpose='same-config exact prediction and inference-audit replay; no new candidate', GT_in_inference=False))
    exact, gains, losses, removed_cases = Counter(), [], [], []
    gain_loss_counts, edge_totals = Counter(), Counter()
    with gzip.open(destination/(P+'frame_checks.jsonl.gz'), 'wt') as stream:
        for scene in manifest['scenes']:
            name = scene['scene']
            old = records(args.input_run/name/(P+'predictions.jsonl.gz'))
            evaluation = records(args.input_run/name/(P+'evaluation.jsonl.gz'))
            assert len(old) == len(evaluation) == len(scene['groups'])
            mappers = {m: GroupedR1Mapper(OmegaConf.create(release['config']), CLASSES,
                deduplication=spec['deduplication']) for m, spec in release['methods'].items()}
            for group, reference, evaluated in zip(scene['groups'], old, evaluation):
                if time.monotonic()-start > 600 or shutil.disk_usage(PROJECT).free < 20*2**30:
                    raise RuntimeError('核验预算/磁盘保护')
                loaded = {c: load_frame(group, c) for c in group['views']}
                for method, mapper in mappers.items():
                    np.random.seed(release['seed']+group['frame_idx'])
                    cameras = release['methods'][method]['cameras']
                    mapper.capture_times = {c: group['views'][c]['camera_timestamp_s'] for c in cameras}
                    frames = [(c, copy.deepcopy(loaded[c])) for c in cameras]
                    mapper.update_group(frames)
                    actual = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                    assert actual == reference['predictions'][method], (name, group['frame_idx'], method)
                    assert mapper.crossview_audit == reference['audit'][method]['crossview']
                    assert mapper.frame_audit == reference['audit'][method]['mapping']
                    exact[method] += 1
                for scope in ['front', 'union']:
                    truth = evaluated['evaluation_only_official'][scope]
                    a, c = [evaluated['result'][m][scope] for m in ['A_front', 'C_three_dedup']]
                    ma, mc = [{r['truth_index']: r for r in v['metrics']['matches']} for v in [a, c]]
                    for i in sorted(set(mc)^set(ma)):
                        kind = 'gained' if i in mc else 'lost'
                        obj = truth[i]
                        case = dict(scene=name, frame_idx=group['frame_idx'], scope=scope,
                            kind=kind, truth=obj, A_match=ma.get(i), C_match=mc.get(i),
                            A_prediction=a['predictions'][ma[i]['prediction_index']] if i in ma else None,
                            C_prediction=c['predictions'][mc[i]['prediction_index']] if i in mc else None,
                            input_views={cam: dict(rgb=v['rgb'], detections=v['detections'], depth=v['depth'],
                                projection_evidence=v['projection_evidence']) for cam, v in group['views'].items()})
                        (gains if kind == 'gained' else losses).append(case)
                        gain_loss_counts[scope+':'+kind] += 1
                        gain_loss_counts[scope+':'+kind+':'+obj['category']] += 1
                        gain_loss_counts[scope+':'+kind+':'+('front_GT' if 'CAM_FRONT' in obj['visible_cameras'] else 'side_only_GT')] += 1
                    for method, scopes in evaluated['result'].items():
                        item = scopes[scope]
                        # Independent set path: do not call the metric implementation.
                        gt_edges = {tuple(e) for e in item['gt_relations']}
                        pred_edges = {tuple(e) for e in item['predicted_relations']}
                        mapping = {int(k): v for k, v in item['node_mapping'].items()}
                        resolved = {(mapping[s], r, mapping[t]) for s, r, t in pred_edges if s in mapping and t in mapping}
                        tp = len(resolved & gt_edges)
                        assert tp == item['relation_counts']['TP']
                        assert len(pred_edges) == item['relation_counts']['all_pred']
                        assert len(gt_edges) == item['relation_counts']['all_truth']
                        edge_totals[method+':'+scope+':TP'] += tp
                audit = reference['audit']['C_three_dedup']['crossview']
                if audit['removed']:
                    accepted = [r for r in reference['audit']['C_three_dedup']['mapping']
                                if r['stage'] == 'geometry' and r['reason'] == 'accepted']
                    for pair in audit['pairs']:
                        if not pair['eligible']:
                            continue
                        details = []
                        for key in ['index_a', 'index_b']:
                            observed = accepted[pair[key]]
                            camera = observed['camera']
                            details.append(dict(observed, rgb=group['views'][camera]['rgb'],
                                                detections=group['views'][camera]['detections']))
                        removed_cases.append(dict(scene=name, frame_idx=group['frame_idx'], pair=pair,
                            observations=details, clusters=audit['clusters']))
                stream.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'], exact_methods=list(mappers),
                    inference_GT_access=False))+'\n')
    assert dict(exact) == dict.fromkeys(release['methods'], 40)
    assert gain_loss_counts['union:gained']-gain_loss_counts['union:lost'] == 37-25
    assert gain_loss_counts['front:gained']-gain_loss_counts['front:lost'] == 23-24
    dump(REPORT/(P+'gain_loss_cases.json'), dict(counts=dict(gain_loss_counts), gains=gains, losses=losses))
    dump(REPORT/(P+'crossview_dedup_cases.json'), removed_cases)
    # Diagnostic real-image panel; no generated scene, no manuscript figure claim.
    if removed_cases:
        panels = []
        for observed in removed_cases[0]['observations']:
            with gzip.open(observed['detections'], 'rb') as stream:
                import pickle
                detection = pickle.load(stream)
            original = Image.open(observed['rgb']).convert('RGB')
            box = detection['xyxy'][observed['detection_index']]
            draw = ImageDraw.Draw(original)
            draw.rectangle(tuple(map(float, box)), outline=(255, 200, 0), width=7)
            original = original.resize((800, 450))
            panel = Image.new('RGB', (800, 480), 'white')
            panel.paste(original, (0, 30))
            ImageDraw.Draw(panel).text((10, 8), observed['camera']+' car; actual RGB and existing detection', fill='black')
            panels.append(panel)
        combined = Image.new('RGB', (800*len(panels), 480), 'white')
        for i, panel in enumerate(panels):
            combined.paste(panel, (800*i, 0))
        combined.save(REPORT/(P+'real_dedup_diagnostic.jpg'), quality=90)
    # All protected old files remain unchanged after the replay too.
    for path, sha in protected.items():
        assert digest(path) == sha
    for path, sha in release['input_sha256'].items():
        assert digest(path) == sha
    dump(REPORT/(P+'verification.json'), dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        exact_replay_groups_by_method=dict(exact), identity_and_all_predictions_exact=True,
        inference_audit_exact=True, protected_sha256_checked=len(protected),
        input_sha256_checked=len(release['input_sha256']), source_snapshots_checked=len(release['source_sha256']),
        independently_recounted_relation_frame_scopes=40*3*2, gain_loss_counts=dict(gain_loss_counts),
        crossview_removed_observation_cases=len(removed_cases), same_log_pilot_only=True,
        bootstrap_or_significance_claim=False, new_candidate_count=0,
        replay_run=str(destination), wall_s=time.monotonic()-start, free_disk_bytes=shutil.disk_usage(PROJECT).free,
        verifier_sha256=digest(__file__)))
    print(json.dumps(dict(exact=dict(exact), gains_losses=dict(gain_loss_counts),
                         removed_cases=len(removed_cases), wall_s=time.monotonic()-start), indent=2))


if __name__ == '__main__':
    main()
