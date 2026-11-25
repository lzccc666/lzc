"""通用冻结U1单状态对照入口：固定缓存、A/C及仅前视结构消融。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import importlib
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
from scripts import multiview_20261123_run_graph as original


P = 'mvrepair_20261123_'
BASE = PROJECT/'outputs/results/nuscenes/validation/iterations'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--tag', default='U1')
    parser.add_argument('--mapper', default='conceptgraph.slam.mvrepair_20261123_unified:UnifiedMapper')
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--wall-budget-s', type=float, default=1800.)
    args = parser.parse_args()
    start = time.monotonic()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    baseline_path = BASE/'multiview_20261123_graph_pilot_v1/multiview_20261123_release.json'
    baseline = json.loads(baseline_path.read_text())
    module_name, class_name = args.mapper.split(':')
    module = importlib.import_module(module_name)
    mapper_class = getattr(module, class_name)
    settings = None
    if args.settings:
        settings = module.UnifiedSettings(**json.loads(args.settings.read_text()))
    run = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    settings_record = mapper_class(OmegaConf.create(baseline['config']), CLASSES, settings).settings_record()
    paths = [Path(__file__).resolve(), Path(module.__file__), PROJECT/'tests/mvrepair_20261123_test_unified.py']
    paths += [Path(p) for p in baseline['source_sha256']]
    paths += [PROJECT/'conceptgraph/slam/mviter_20261123_diagnostics.py']
    paths += [PROJECT/p for p in getattr(module, 'EXTRA_SOURCES', [])]
    sources = {}
    for i, path in enumerate(dict.fromkeys(paths)):
        target = run/(P+f'source_{i:02d}_'+path.name)
        shutil.copy2(path, target)
        sources[str(path)] = dict(sha256=digest(path), snapshot=str(target))
    inputs = {str(args.manifest): digest(args.manifest)}
    for scene in manifest['scenes']:
        for group in scene['groups']:
            for view in group['views'].values():
                for key in ['rgb', 'depth', 'detections', 'projection_evidence']:
                    inputs[view[key]] = digest(view[key])
    dump(run/(P+'release.json'), dict(tag=args.tag, started_utc=datetime.now(timezone.utc).isoformat(),
        config=baseline['config'], settings=settings_record, mapper=args.mapper,
        source_sha256=sources, input_sha256=inputs, seed=17, GT_in_inference=False,
        primary='same union105 and front77 pilot guardrails; no per-threshold sweep',
        ablation=args.tag+'_front vs '+args.tag+'_three; identical config',
        maximum_wall_s=args.wall_budget_s, maximum_new_bytes=768*2**20, minimum_free_bytes=20*2**30,
        no_new_frontend=True, no_test=True, default_changed=False,
        ground_truth_sha256={str(args.ground_truth): digest(args.ground_truth)}))
    costs, total_groups, exact_baseline = {}, 0, {'A_front': 0, 'C_three_dedup': 0}
    for scene in manifest['scenes']:
        name = scene['scene']
        reference_path = BASE/'multiview_20261123_graph_pilot_v1'/name/'multiview_20261123_predictions.jsonl.gz'
        reference = list(map(json.loads, gzip.open(reference_path, 'rt'))) if reference_path.exists() else None
        if reference is not None:
            assert len(reference) >= len(scene['groups'])
        old_mappers = dict(A_front=SourceAuditedMapper(OmegaConf.create(baseline['config']), CLASSES, deduplication=False),
            C_three_dedup=SourceAuditedMapper(OmegaConf.create(baseline['config']), CLASSES, deduplication=True))
        new_mappers = {args.tag+'_'+scope: mapper_class(OmegaConf.create(baseline['config']), CLASSES, settings) for scope in ['front', 'three']}
        destination = run/name
        destination.mkdir()
        with gzip.open(destination/(P+'predictions.jsonl.gz'), 'wt') as out:
            for offset, group in enumerate(scene['groups']):
                if time.monotonic()-start > args.wall_budget_s or shutil.disk_usage(PROJECT).free < 20*2**30:
                    raise RuntimeError('预算/磁盘保护，保留真实前缀')
                loaded = {cam: original.load_frame(group, cam) for cam in group['views']}
                projection = {}
                for cam, view in group['views'].items():
                    with np.load(view['projection_evidence']) as source:
                        projection[cam] = {k: source[k].copy() for k in source.files}
                    projection[cam]['lidar_token'] = view.get('lidar_token')
                predictions, audit, timing = {}, {}, {}
                for method, mapper in old_mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras = ['CAM_FRONT'] if method == 'A_front' else list(group['views'])
                    mapper.capture_times = {cam: group['views'][cam]['camera_timestamp_s'] for cam in cameras}
                    begun = time.perf_counter()
                    frames = [(cam, copy.deepcopy(loaded[cam])) for cam in cameras]
                    mapper.update_group(frames)
                    predictions[method] = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                    timing[method] = time.perf_counter()-begun
                    audit[method] = dict(mapping=mapper.frame_audit, crossview=mapper.crossview_audit)
                    if reference:
                        assert (group['frame_idx'], group['sample_token']) == (reference[offset]['frame_idx'], reference[offset]['sample_token'])
                        assert predictions[method] == reference[offset]['predictions'][method]
                        exact_baseline[method] += 1
                for method, mapper in new_mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras = ['CAM_FRONT'] if method.endswith('_front') else list(group['views'])
                    begun = time.perf_counter()
                    frames = [(cam, copy.deepcopy(loaded[cam])) for cam in cameras]
                    predictions[method] = mapper.update_group(frames,
                        {cam: group['views'][cam]['camera_timestamp_s'] for cam in cameras},
                        {cam: projection[cam] for cam in cameras})
                    timing[method] = time.perf_counter()-begun
                    audit[method] = mapper.audit
                for method, elapsed in timing.items():
                    costs[method] = costs.get(method, 0.)+elapsed
                out.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'], sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'], predictions=predictions,
                    audit=audit, timing_s=timing, GT_in_inference=False), allow_nan=False)+'\n')
                total_groups += 1
                print(name, group['frame_idx'], args.tag, 'complete', flush=True)
    assert total_groups == sum(len(scene['groups']) for scene in manifest['scenes'])
    original.DEST, original.P = args.report_dir, P
    result = original.evaluate(run, manifest, args.ground_truth)
    report = dict(complete=True, tag=args.tag, groups=total_groups, results=result,
        cost_total_s=costs, wall_s=time.monotonic()-start, baseline_exact=exact_baseline,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        RSS_scope='all A/C/candidate-front/candidate-three co-resident plus imports and input images; not isolated method memory',
        input_sha256=inputs, ground_truth_sha256={str(args.ground_truth): digest(args.ground_truth)},
        source_sha256=sources, outputs_sha256={str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()},
        no_new_frontend=True, no_test=True, default_changed=False)
    assert report['ground_truth_sha256'] == json.loads((run/(P+'release.json')).read_text())['ground_truth_sha256']
    dump(run/(P+'report.json'), report)
    dump(args.report_dir/(P+'result.json'), dict(run=str(run), **report))
    print(json.dumps(dict(node=result['node_main'], side=result['side_coverage'], cost=costs, wall_s=report['wall_s']), indent=2))


if __name__ == '__main__':
    main()
