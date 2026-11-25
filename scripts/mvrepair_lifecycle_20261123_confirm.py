"""冻结L1的一次跨log确认入口：同输入A/C/L1，预测写完后才读取GT。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper
from scripts import multiview_20261123_run_graph as original


P = 'mvrepair_lifecycle_20261123_'
FROZEN = PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_lifecycle_20261123_candidate_L1_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    freeze = json.loads((FROZEN/(P+'release.json')).read_text())
    for path, record in freeze['source_sha256'].items():
        if digest(path) != record['sha256']:
            raise RuntimeError('L1冻结源码已变化，停止确认：'+path)
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest['scenes']) <= 2 and all(len(s['groups']) <= 20 for s in manifest['scenes'])
    inputs = {str(args.manifest.resolve()):digest(args.manifest)}
    for scene in manifest['scenes']:
        assert [g['frame_idx'] for g in scene['groups']] == list(range(len(scene['groups'])))
        for group in scene['groups']:
            assert set(group['views']) == {'CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'}
            for view in group['views'].values():
                for key in ['rgb','depth','detections','projection_evidence']:
                    if view.get(key): inputs[view[key]] = digest(view[key])
    run = experiment_directory('nuscenes','validation',args.run_id,create=True)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).resolve()
    shutil.copy2(source,run/source.name)
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=inputs,frozen_L1_release_sha256=digest(FROZEN/(P+'release.json')),
        runner_sha256={str(source):digest(source)},seed=17,GT_in_inference=False,
        ground_truth_read_only_after_all_predictions=True,threshold_m=3.,
        methods=['A_front','C_three_dedup','L1_lifecycle'],frozen_config=freeze['original_config'],
        role='development cross-log confirmation; no candidate change after seeing result'))
    for scene in manifest['scenes']:
        name=scene['scene']; outdir=run/name; outdir.mkdir()
        methods={'A_front':SourceAuditedMapper(OmegaConf.create(freeze['original_config']),CLASSES,deduplication=False),
                 'C_three_dedup':SourceAuditedMapper(OmegaConf.create(freeze['original_config']),CLASSES,deduplication=True),
                 'L1_lifecycle':LifecycleRetiringMapper(OmegaConf.create(freeze['original_config']),CLASSES)}
        with gzip.open(outdir/(P+'predictions.jsonl.gz'),'wt') as out:
            for group in scene['groups']:
                if time.monotonic()-started>1200 or shutil.disk_usage(PROJECT).free<20*2**30:
                    raise RuntimeError('确认预算/磁盘保护停止')
                loaded={c:original.load_frame(group,c) for c in group['views']}
                predictions,audit={},{}
                for method,mapper in methods.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras=['CAM_FRONT'] if method=='A_front' else list(group['views'])
                    mapper.capture_times={c:group['views'][c]['camera_timestamp_s'] for c in cameras}
                    frames=[(c,copy.deepcopy(loaded[c])) for c in cameras]
                    mapper.update_group(frames)
                    predictions[method]=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frames[0][1])
                    audit[method]=mapper.lifecycle_audit if method=='L1_lifecycle' else dict(mapping=mapper.frame_audit,crossview=mapper.crossview_audit)
                out.write(json.dumps(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=predictions,audit=audit,
                    GT_in_inference=False),allow_nan=False)+'\n')
    original.DEST,original.P=args.report_dir,P
    # GT首次访问明确在所有预测文件关闭后；不以真值重构任何输入或身份。
    result=original.evaluate(run,manifest,args.ground_truth)
    report=dict(run=str(run),complete=True,results=result,wall_s=time.monotonic()-started,
                ground_truth_sha256={str(args.ground_truth.resolve()):digest(args.ground_truth)},
                frozen_sources_unchanged=all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items()),
                GT_in_inference=False,new_model_GPU_training_CARLA_calls=0)
    dump(run/(P+'report.json'),report);dump(args.report_dir/'L1_confirmation_result.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
