"""固定T3 v2和原A/C：120时刻1/3/6视角统一参考clock真实缓存回放。"""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.mv136_20261123_protocol import SIX, SUBSETS, clock_evidence, evaluate
from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_20261123_native_representative_v2 import NativeRepresentativeMapper
from scripts.multiview_20261123_run_graph import load_frame, write_csv

P = 'mv136_20261123_'
SCENES = ['scene-0012','scene-0013','scene-0019','scene-0035','scene-0042','scene-0047']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--deadline-utc', default='2026-11-23T15:24:00+00:00')
    args = parser.parse_args()
    start = time.monotonic()
    deadline = datetime.fromisoformat(args.deadline_utc)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    assert sorted(s['scene'] for s in manifest['scenes']) == SCENES
    clocks = []
    for scene in manifest['scenes']:
        assert [g['frame_idx'] for g in scene['groups']] == list(range(20))
        assert len({g['sample_token'] for g in scene['groups']}) == 20
        prior = -float('inf')
        for group in scene['groups']:
            evidence = clock_evidence(group)
            assert group['reference_timestamp_s'] > prior
            prior = group['reference_timestamp_s']
            clocks.append(dict(scene=scene['scene'],frame_idx=group['frame_idx'],**evidence))
    assert len(clocks) == 120
    baseline_path = PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json'
    frozen = json.loads(baseline_path.read_text())
    assert digest(PROJECT/'conceptgraph/slam/mvrepair_20261123_native_representative_v2.py') == '16f16aad31f235081183e9f4f18843b7e103d244bcf8571dc326c5b5c3582c92'
    cfg = frozen['config']
    run = experiment_directory('nuscenes','validation',args.run_id,create=True)
    paths = [Path(__file__).resolve(),PROJECT/'conceptgraph/completion/mv136_20261123_protocol.py',
        PROJECT/'tests/mv136_20261123_test_protocol.py',*map(Path,frozen['source_sha256'])]
    sources = {}
    for i,path in enumerate(dict.fromkeys(paths)):
        target = run/(P+f'source_{i:02d}_'+path.name)
        shutil.copy2(path,target)
        sources[str(path)] = dict(sha256=digest(path),snapshot=str(target))
    inputs = {str(args.manifest):digest(args.manifest),str(baseline_path):digest(baseline_path)}
    for scene in manifest['scenes']:
        for group in scene['groups']:
            for view in group['views'].values():
                for key in ['rgb','depth','detections','projection_evidence']:
                    inputs[view[key]] = digest(view[key])
    gt_hash = {str(args.ground_truth):digest(args.ground_truth)}
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),
        methods=SUBSETS,mapper='fixed NativeRepresentativeMapper T3 v2, A/C SourceAuditedMapper background',
        config=cfg,T3_settings=frozen['settings'],seed=17,source_sha256=sources,input_sha256=inputs,
        ground_truth_sha256=gt_hash,GT_in_inference=False,scenes=SCENES,groups=120,
        common_reference='max six actual exposures; only already captured images; LiDAR strictly past own camera',
        scopes=['union6 primary','front3 auxiliary','front auxiliary','rear-only from union6 matched GT'],
        background='A/C replays share new max6 clock; no claim of exact historical-clock equality',
        maximum_new_bytes=256*2**20,minimum_free_bytes=20*2**30,deadline_utc=args.deadline_utc,
        default_changed=False,no_new_frontend_by_mapper=True,no_test=True))
    total, costs, timing_rows = 0, {m:0. for m in SUBSETS}, []
    for scene in manifest['scenes']:
        mappers = {m:(SourceAuditedMapper(OmegaConf.create(cfg),CLASSES,deduplication=m!='A_front_context')
            if m in ['A_front_context','C_3','C_6'] else NativeRepresentativeMapper(OmegaConf.create(cfg),CLASSES)) for m in SUBSETS}
        folder = run/scene['scene']
        folder.mkdir()
        with gzip.open(folder/(P+'predictions.jsonl.gz'),'wt') as stream:
            for group in scene['groups']:
                if datetime.now(timezone.utc) >= deadline or shutil.disk_usage(PROJECT).free < 20*2**30:
                    raise RuntimeError('时间/空闲盘硬保护，保留已写真实前缀')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) > 256*2**20:
                    raise RuntimeError('本线输出预算保护，保留前缀')
                loaded = {c:load_frame(group,c) for c in SIX}
                projection = {}
                for c,view in group['views'].items():
                    with np.load(view['projection_evidence']) as source:
                        projection[c] = {k:source[k].copy() for k in source.files}
                    projection[c]['lidar_token'] = view.get('lidar_token')
                predictions,audits,timing = {},{},{}
                for method,cameras in SUBSETS.items():
                    np.random.seed(17+group['frame_idx'])
                    mapper = mappers[method]
                    frames = [(c,copy.deepcopy(loaded[c])) for c in cameras]
                    captures = {c:group['views'][c]['camera_timestamp_s'] for c in cameras}
                    begun = time.perf_counter()
                    if method in ['A_front_context','C_3','C_6']:
                        mapper.capture_times = captures
                        mapper.update_group(frames)
                        predictions[method] = mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frames[0][1])
                        audits[method] = dict(mapping=mapper.frame_audit,crossview=mapper.crossview_audit)
                    else:
                        predictions[method] = mapper.update_group(frames,captures,{c:projection[c] for c in cameras})
                        audits[method] = mapper.audit
                    elapsed = time.perf_counter()-begun
                    timing[method] = elapsed
                    costs[method] += elapsed
                    timing_rows.append(dict(scene=scene['scene'],log_token=scene['log_token'],frame_idx=group['frame_idx'],
                        method=method,cameras=len(cameras),mapper_update_wall_s=elapsed))
                stream.write(json.dumps(dict(scene=scene['scene'],frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],clock=clock_evidence(group),
                    predictions=predictions,audit=audits,timing_s=timing,GT_in_inference=False),allow_nan=False,
                    default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
                total += 1
                print(scene['scene'],group['frame_idx'],'all six method conditions complete',flush=True)
    assert total == 120
    assert gt_hash == {str(args.ground_truth):digest(args.ground_truth)}
    result = evaluate(run,manifest,args.ground_truth,args.report_dir,P)
    write_csv(args.report_dir/(P+'mapper_time.csv'),timing_rows)
    dump(run/(P+'clock_audit.json'),clocks)
    report = dict(complete=True,run=str(run),groups=total,scenes=6,results=result,cost_total_s=costs,
        wall_s=time.monotonic()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        RSS_scope='six method conditions simultaneously resident plus imports/input images; not isolated method memory',
        source_sha256=sources,input_sha256=inputs,ground_truth_sha256=gt_hash,GT_in_inference=False,
        default_changed=False,no_new_frontend_by_mapper=True,no_test=True)
    dump(run/(P+'report.json'),report)
    dump(args.report_dir/(P+'result.json'),report)
    print(json.dumps(result['node_main'],ensure_ascii=False))


if __name__ == '__main__':
    main()
