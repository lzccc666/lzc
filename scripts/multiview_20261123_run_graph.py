"""冻结单前/三视角同入口有界对照；官方GT只在全部推理落盘后读取。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.quality import object_frame_metrics, relations_from_centers
from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.scenegraph.paper_graph_20261123_metrics import edge_counts, scores
from conceptgraph.slam.multiview_20261123_fusion import GroupedR1Mapper, in_view
from conceptgraph.slam.streamlined_mapping import MappingFrame


P = 'multiview_20261123_'
METHODS = dict(A_front=dict(cameras=['CAM_FRONT'], deduplication=False),
               B_three_no_dedup=dict(cameras=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'], deduplication=False),
               C_three_dedup=dict(cameras=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'], deduplication=True))
DEST = PROJECT/'outputs/reports/multiview_semantic_tasks_20261123/graph'


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def node_scores(counts):
    tp, gt, pred = [counts[k] for k in ['true_positive', 'truth', 'predictions']]
    return dict(precision=tp/pred if pred else None, recall=tp/gt if gt else None,
                F1=2*tp/(gt+pred) if gt+pred else None,
                center_error_m=counts['error_sum_m']/tp if tp else None,
                IDF1=2*counts['IDTP']/(counts['GT_observations']+counts['prediction_observations'])
                if counts['GT_observations']+counts['prediction_observations'] else None)


def guard(start, wall_budget_s):
    if time.monotonic()-start > wall_budget_s or shutil.disk_usage(PROJECT).free < 20*2**30:
        raise RuntimeError('预算/磁盘保护，保留真实前缀')


def load_frame(group, camera):
    view = group['views'][camera]
    rgb = np.asarray(Image.open(view['rgb']).convert('RGB'))
    depth = np.asarray(Image.open(view['depth']), dtype=float)/float(view.get('depth_scale', 1000.))
    if rgb.shape[:2] != depth.shape or rgb.shape[1] != view['width'] or rgb.shape[0] != view['height']:
        raise ValueError('相机RGB/depth/K尺寸不一致')
    if view.get('lidar_timestamp_s') is not None and view['lidar_timestamp_s'] >= view['camera_timestamp_s']:
        raise ValueError('必须严格过去LiDAR')
    if view['camera_timestamp_s'] > group['reference_timestamp_s']:
        raise ValueError('使用了参考时刻以后的相机')
    with gzip.open(view['detections'], 'rb') as stream:
        detections = pickle.load(stream)
    return MappingFrame(group['frame_idx'], rgb, depth, np.asarray(view['intrinsic']),
        np.asarray(view['camera_pose']), detections, view['rgb'], timestamp_s=group['reference_timestamp_s'])


def evaluate(run, manifest, ground_truth):
    # This function is only called after every inference frame has been written.
    all_gt = list(map(json.loads, ground_truth.open()))
    gt_rows = {(r['scene'], r['frame_idx']): r for r in all_gt}
    expected = {(s['scene'], g['frame_idx']) for s in manifest['scenes'] for g in s['groups']}
    assert len(gt_rows) == len(all_gt) and set(gt_rows) == expected
    counts, relations = defaultdict(Counter), defaultdict(Counter)
    scene_rows, frames, side_rows = [], [], []
    case = []
    for scene in manifest['scenes']:
        name = scene['scene']
        sequence = list(map(json.loads, gzip.open(run/name/(P+'predictions.jsonl.gz'), 'rt')))
        assert len(sequence) == len(scene['groups'])
        scene_counts, prepared = defaultdict(Counter), defaultdict(list)
        with gzip.open(run/name/(P+'evaluation.jsonl.gz'), 'wt') as out:
            for row, group in zip(sequence, scene['groups']):
                assert row['frame_idx'] == group['frame_idx']
                official = gt_rows[name, row['frame_idx']]
                assert official['sample_token'] == group['sample_token']
                union = official['objects']
                assert len(union) == len({o['instance_token'] for o in union})
                truth = dict(union=union, front=[o for o in union if 'CAM_FRONT' in o['visible_cameras']])
                forward = np.asarray(group['views']['CAM_FRONT']['camera_pose'])[:2, 2]
                result = {}
                for method, all_predictions in row['predictions'].items():
                    result[method] = {}
                    for scope in ['front', 'union']:
                        pred = [p for p in all_predictions if (in_view(p['center'], group['views']['CAM_FRONT'])
                            if scope == 'front' else any(in_view(p['center'], v) for v in group['views'].values()))]
                        metric = object_frame_metrics(pred, truth[scope], threshold_m=3.)
                        c = scene_counts[method, scope]
                        for key in ['true_positive', 'truth', 'predictions', 'unmatched_predictions', 'false_negative', 'duplicate_nodes']:
                            c[key] += metric[key]
                        c['error_sum_m'] += sum(m['distance_m'] for m in metric['matches'])
                        prepared[method, scope].append(dict(frame_idx=row['frame_idx'],
                            camera_timestamp_s=row['reference_timestamp_s'], predictions={method: pred},
                            evaluation_only_official=truth[scope], metrics={method: metric}))
                        predicted_edges = relations_from_centers([dict(id=i, center=p['center']) for i, p in enumerate(pred)],
                            forward_xy=forward, max_distance_m=12., near_m=6., component_margin_m=1.)
                        gt_edges = relations_from_centers([dict(id=o['instance_token'], center=o['center']) for o in truth[scope]],
                            forward_xy=forward, max_distance_m=12., near_m=6., component_margin_m=1.)
                        mapping = {m['prediction_index']: truth[scope][m['truth_index']]['instance_token'] for m in metric['matches']}
                        edge = edge_counts(predicted_edges, gt_edges, mapping)
                        relations[method, scope].update(edge)
                        if scope == 'union':
                            matched = {m['truth_index'] for m in metric['matches']}
                            front_indices = {i for i, o in enumerate(union) if 'CAM_FRONT' in o['visible_cameras']}
                            side = set(range(len(union)))-front_indices
                            side_rows.append(dict(scene=name, frame_idx=row['frame_idx'], method=method,
                                common_front_GT=len(front_indices), common_front_TP=len(matched & front_indices),
                                side_only_GT=len(side), side_only_TP=len(matched & side)))
                            if method != 'A_front':
                                case.append(dict(scene=name, frame_idx=row['frame_idx'], method=method,
                                    side_matched_instances=[union[i]['instance_token'] for i in sorted(matched & side)]))
                        result[method][scope] = dict(predictions=pred, metrics=metric,
                            relation_counts=dict(edge), predicted_relations=sorted(predicted_edges),
                            gt_relations=sorted(gt_edges), node_mapping=mapping)
                        frames.append(dict(scene=name, log_token=scene['log_token'], frame_idx=row['frame_idx'],
                            method=method, scope=scope, **{k: metric[k] for k in ['truth', 'predictions', 'true_positive',
                            'unmatched_predictions', 'false_negative', 'duplicate_nodes']},
                            center_error_m=metric['center_error_m_mean'], **scores(edge)))
                out.write(json.dumps(dict(scene=name, frame_idx=row['frame_idx'], evaluation_only_official=truth,
                    result=result), allow_nan=False)+'\n')
            for key, sequence in prepared.items():
                method, scope = key
                identity = identity_counts(prepare_sequence(sequence, method), 'all_gated')['counts']
                scene_counts[key].update(identity)
                counts[key].update(scene_counts[key])
                scene_rows.append(dict(scene=name, log_token=scene['log_token'], method=method, scope=scope,
                    **scene_counts[key], **node_scores(scene_counts[key])))
    node_rows = [dict(method=m, scope=s, **c, **node_scores(c)) for (m, s), c in counts.items()]
    edge_rows = [dict(method=m, scope=s, **c, **scores(c)) for (m, s), c in relations.items()]
    side_total = defaultdict(Counter)
    for row in side_rows:
        side_total[row['method']].update({k: row[k] for k in ['common_front_GT', 'common_front_TP', 'side_only_GT', 'side_only_TP']})
    side_main = [dict(method=m, **c) for m, c in side_total.items()]
    files = dict(node_main=node_rows, relation_main=edge_rows, side_coverage=side_main,
                 scene_metrics=scene_rows, frame_metrics=frames, side_frame=side_rows)
    for name, rows in files.items():
        write_csv(DEST/(P+name+'.csv'), rows)
    dump(run/(P+'side_cases.json'), case)
    return dict(node_main=node_rows, relation_main=edge_rows, side_coverage=side_main)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-budget-s', type=float, default=5400.)
    args = parser.parse_args()
    start = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    baseline_path = PROJECT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2/release.json'
    baseline = json.loads(baseline_path.read_text())
    scenes = manifest['scenes']
    if len(scenes) > 2 or any(len(s['groups']) > 20 for s in scenes):
        raise ValueError('超出已冻结小试范围')
    allowed = {s['scene'] for s in baseline['scenes']}
    if any(s['scene'] not in allowed for s in scenes):
        raise ValueError('禁止使用非38开发场景')
    inputs = {str(args.manifest): digest(args.manifest), str(baseline_path): digest(baseline_path)}
    for scene in scenes:
        if [g['frame_idx'] for g in scene['groups']] != list(range(len(scene['groups']))):
            raise ValueError('组编号必须从0连续且唯一')
        times = [g['reference_timestamp_s'] for g in scene['groups']]
        if not np.isfinite(times).all() or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('组时钟必须有限且严格递增')
        for group in scene['groups']:
            assert set(group['views']) == {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'}
            if group['reference_timestamp_s'] != max(v['camera_timestamp_s'] for v in group['views'].values()):
                raise ValueError('组时钟必须是三相机最新采集时间')
            for view in group['views'].values():
                for key in ['rgb', 'depth', 'detections', 'projection_evidence']:
                    if key in view and view[key] is not None:
                        inputs[view[key]] = digest(view[key])
    run = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    source_paths = [Path(__file__).resolve(), PROJECT/'conceptgraph/slam/multiview_20261123_fusion.py',
                    PROJECT/'scripts/multiview_20261123_ground_truth.py',
                    PROJECT/'conceptgraph/scenegraph/paper_graph_20261123_metrics.py',
                    PROJECT/'conceptgraph/completion/quality.py',
                    PROJECT/'conceptgraph/completion/temporal_identity_evaluation.py']
    source_paths += [Path(p) for p in baseline['sources_sha256'] if Path(p).name != 'run_mapping_refinement.py']
    sources = {}
    for i, path in enumerate(dict.fromkeys(source_paths)):
        target = run/(P+f'source_{i:02d}_'+path.name)
        shutil.copy2(path, target)
        sources[str(path)] = dict(sha256=digest(path), snapshot=str(target))
    settings = GroupedR1Mapper(OmegaConf.create(baseline['config']), CLASSES, deduplication=True).settings_record()
    dump(run/(P+'release.json'), dict(started_utc=datetime.now(timezone.utc).isoformat(),
        name='three-camera pilot new R1-derived entry; not strict historical R1', methods=METHODS,
        scenes=[dict(scene=s['scene'], log_token=s['log_token'], groups=len(s['groups'])) for s in scenes],
        config=baseline['config'], settings=settings, seed=17, input_sha256=inputs, source_sha256=sources,
        maximum_wall_s=args.wall_budget_s, maximum_new_bytes=2**30, minimum_free_bytes=20*2**30,
        primary_comparison='C_vs_A at identical union GT; front scope separately paired; no cross-scope percentage gain',
        diagnostic_comparison='C_vs_B deduplication; not a threshold sweep', GT_in_inference=False,
        graph_evaluation=dict(near_m=6., directional_m=12., component_margin_m=1., forward='reference CAM_FRONT world XY'),
        test_accessed=False, default_changed=False, new_frontend_calls_by_graph=0))
    total_audit, time_rows = defaultdict(Counter), []
    for scene in scenes:
        name = scene['scene']
        destination = run/name
        destination.mkdir()
        mappers = {m: GroupedR1Mapper(OmegaConf.create(baseline['config']), CLASSES,
                   deduplication=spec['deduplication']) for m, spec in METHODS.items()}
        with gzip.open(destination/(P+'predictions.jsonl.gz'), 'wt') as stream:
            for group in scene['groups']:
                guard(start, args.wall_budget_s)
                loaded = {c: load_frame(group, c) for c in group['views']}
                predictions, audits = {}, {}
                for method, mapper in mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras = METHODS[method]['cameras']
                    mapper.capture_times = {c: group['views'][c]['camera_timestamp_s'] for c in cameras}
                    begun = time.perf_counter()
                    frames = [(c, copy.deepcopy(loaded[c])) for c in cameras]
                    mapper.update_group(frames)
                    predictions[method] = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                    elapsed = time.perf_counter()-begun
                    time_rows.append(dict(scene=name, frame_idx=group['frame_idx'], method=method,
                        cameras=len(cameras), mapping_s=elapsed))
                    audits[method] = dict(crossview=mapper.crossview_audit, mapping=mapper.frame_audit)
                    for key in ['input_observations', 'output_observations', 'removed', 'sparse_observations_not_deduplicated']:
                        total_audit[method][key] += mapper.crossview_audit[key]
                    total_audit[method]['mapping_s'] += elapsed
                    total_audit[method]['output_node_frames'] += len(predictions[method])
                stream.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'],
                    sample_token=group['sample_token'], reference_timestamp_s=group['reference_timestamp_s'],
                    predictions=predictions, audit=audits, GT_in_inference=False), allow_nan=False)+'\n')
                print(name, group['frame_idx'], 'all3methods_complete', flush=True)
    # No mapper has been given the contents or path of this evaluation-only source.
    result = evaluate(run, manifest, args.ground_truth)
    write_csv(DEST/(P+'mapping_time.csv'), time_rows)
    report = dict(complete=True, scenes=len(scenes), groups=sum(len(s['groups']) for s in scenes),
        scope='bounded repeatedly-used development prefix; not full38; not independent test',
        results=result, audit={m: dict(c) for m, c in total_audit.items()}, wall_s=time.monotonic()-start,
        ground_truth_sha256={str(args.ground_truth): digest(args.ground_truth)},
        source_sha256=sources, input_sha256=inputs,
        output_sha256={str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()},
        report_csv_sha256={str(p): digest(p) for p in DEST.glob(P+'*.csv')},
        GT_in_inference=False, test_accessed=False, default_changed=False)
    dump(run/(P+'report.json'), report)
    dump(DEST/(P+'result.json'), dict(run=str(run), **report))
    print(json.dumps(dict(run=str(run), node_main=result['node_main'], audit=report['audit'], wall_s=report['wall_s']),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
