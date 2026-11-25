"""S同配置复现：核对完整预测／身份／图像辅助关联审计，不产生第二候选。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.paper_graph_20261123_sparse_continuity import ImageContinuityR1
from conceptgraph.slam.streamlined_mapping import MappingFrame


P='paper_graph_20261123_'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wall-budget-s',type=float,default=1500.)
    args=parser.parse_args()
    start=time.monotonic()
    original=PROJECT/'outputs/results/nuscenes/validation/iterations/paper_graph_20261123_S_full_v1'
    release=json.loads((original/(P+'release.json')).read_text())
    report=json.loads((original/(P+'report.json')).read_text())
    if report['frames']!=1508 or not report['full_validation_scope']:
        raise AssertionError('候选不完整')
    for source,record in release['sources_sha256'].items():
        if digest(source)!=record['sha256'] or digest(record['snapshot'])!=record['sha256']:
            raise AssertionError('冻结/当前执行源码变更')
    dest=experiment_directory('nuscenes','validation',args.run_id,create=True)
    dump(dest/(P+'release.json'),dict(mode='deterministic_verification_only',original_run=str(original),
        original_release_sha256=digest(original/(P+'release.json')),source_sha256=digest(__file__),
        candidate_count_added=0,wall_budget_s=args.wall_budget_s,seed=release['seed'],new_frontend_calls=0,test_accessed=False))
    frames=0
    for scene in release['scenes']:
        name=scene['scene']
        mapper=ImageContinuityR1(OmegaConf.create(release['config']),CLASSES)
        if mapper.settings_record()!=release['experimental_settings']:
            raise AssertionError('候选参数变化')
        path=OUTPUT/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
        raw=list(map(json.loads,path.open()))
        expected=list(map(json.loads,gzip.open(original/name/(P+'frames.jsonl.gz'),'rt')))
        if len(raw)!=len(expected):
            raise AssertionError('输入帧范围变化')
        check=[]
        for source,reference in zip(raw,expected):
            if time.monotonic()-start>args.wall_budget_s or shutil.disk_usage(dest).free<20*2**30:
                raise RuntimeError('核验达到预算/磁盘保护线')
            idx=source['frame_idx']
            path=Path(reference['input']['detections'])
            if digest(path)!=reference['input']['detections_sha256']:
                raise AssertionError('缓存指纹变化')
            with gzip.open(path,'rb') as stream:
                detections=pickle.load(stream)
            frame=MappingFrame(idx,np.asarray(Image.open(source['rgb']).convert('RGB')),
                np.asarray(Image.open(source['depth']),dtype=float)/1000,np.asarray(source['intrinsic']),
                np.asarray(source['camera_pose']),detections,source['rgb'],timestamp_s=source['camera_timestamp_s'])
            np.random.seed(release['seed']+idx)
            mapper.update(frame)
            output=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
            actual=dict(predictions=output,audit=mapper.frame_audit,image_audit=mapper.position_layer.fallback_used)
            wanted=dict(predictions=reference['predictions'][release['condition']],audit=reference['audit'],image_audit=reference['image_continuity_audit'])
            if json.loads(json.dumps(actual,allow_nan=False))!=wanted:
                raise AssertionError(f'候选预测或审计复现失败 {name}/{idx}')
            check.append(dict(frame_idx=idx,prediction_and_all_audit_exact=True))
            frames+=1
        dump(dest/(P+name+'_verification.json'),dict(scene=name,frames=check,complete=True))
        print(name,frames,'exact',flush=True)
    if frames!=1508:
        raise AssertionError('未完整复现')
    dump(dest/(P+'report.json'),dict(complete=True,frames=frames,scenes=38,every_prediction_and_audit_exact=True,
        wall_s=time.monotonic()-start,new_candidate=False,new_frontend_calls=0,test_accessed=False,
        outputs_sha256={p.name:digest(p) for p in dest.glob(P+'scene-*.json')}))


if __name__=='__main__':
    main()
