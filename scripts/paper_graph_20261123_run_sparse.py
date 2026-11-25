"""固定单一S候选；仅重放已有冻结缓存，小试与完整38场景分开落盘。"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.paper_graph_20261123_sparse_continuity import ImageContinuityR1
from conceptgraph.slam.streamlined_mapping import MappingFrame


P='paper_graph_20261123_'
R1=PROJECT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
PILOT=('scene-0012','scene-0019','scene-0021','scene-0039')
CONDITION='R1_S_image_continuity'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--wall-budget-s',type=float,default=2400.)
    args=parser.parse_args()
    start=time.monotonic()
    baseline=json.loads((R1/'release.json').read_text())
    scenes=[s for s in baseline['scenes'] if not args.pilot or s['scene'] in PILOT]
    dest=experiment_directory('nuscenes','validation',args.run_id,create=True)
    inputs={str(p):digest(p) for p in [R1/'release.json',R1/'report.json',R1/'input_fingerprints.json']}
    for s in scenes:
        for p in [R1/s['scene']/'frames.jsonl.gz',OUTPUT/'mapping_evaluation/causal_mapping_full_001'/s['scene']/'frames.jsonl']:
            inputs[str(p)]=digest(p)
    sources=[Path(__file__).resolve(),PROJECT/'conceptgraph/slam/paper_graph_20261123_sparse_continuity.py']
    sources.extend(Path(p) for p in baseline['sources_sha256'] if Path(p).name!='run_mapping_refinement.py')
    source_record={}
    for i,p in enumerate(dict.fromkeys(sources)):
        target=dest/(P+f'source_{i:02d}_'+p.name)
        shutil.copy2(p,target)
        source_record[str(p)]=dict(sha256=digest(p),snapshot=str(target))
    settings=ImageContinuityR1(OmegaConf.create(baseline['config']),CLASSES).settings_record()
    release=dict(condition=CONDITION,baseline='R1',pilot=args.pilot,scenes=scenes,
        config=baseline['config'],experimental_settings=settings,seed=baseline['seed'],
        input_sha256=inputs,sources_sha256=source_record,wall_budget_s=args.wall_budget_s,
        campaign_extra_wall_budget_s=5400,maximum_new_bytes=512*2**20,minimum_free_bytes=20*2**30,
        threshold_m=3.,hypothesis='sparse image continuity improves full-sequence IDF1; report all node quality/duplicates/tradeoffs',
        comparisons_registered=['S_vs_R1'],bootstrap_repetitions=10000,bootstrap_seed=20261123,
        preserved=['original sparse confirmation','original TTL','original dense association','reliable dense position protection','full GT denominator'],
        new_frontend_calls=0,GT_in_inference=False,test_accessed=False,default_changed=False,
        candidate_count=1,pilot_scenes_fixed=list(PILOT),no_config_sweep=True,
        pilot_gate='functional/no-nonfinite/time/confirmation violations; quality not used to tune configuration',
        storage_file_prefix=P)
    dump(dest/(P+'release.json'),release)
    total,identity,frames=Counter(),Counter(),0
    fallback_total=Counter()
    for scene in scenes:
        name=scene['scene']
        mapper=ImageContinuityR1(OmegaConf.create(baseline['config']),CLASSES)
        raw_path=OUTPUT/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
        raw=list(map(json.loads,raw_path.open()))
        expected=list(map(json.loads,gzip.open(R1/name/'frames.jsonl.gz','rt')))
        if len(raw)!=len(expected):
            raise AssertionError('输入范围变化')
        target=dest/name
        target.mkdir()
        scene_rows=[]
        scene_fallback=Counter()
        with gzip.open(target/(P+'frames.jsonl.gz'),'wt') as out:
            for source,reference in zip(raw,expected):
                if time.monotonic()-start>args.wall_budget_s or shutil.disk_usage(dest).free<20*2**30:
                    raise RuntimeError('预算/磁盘保护，保存真实前缀')
                idx=source['frame_idx']
                if idx!=reference['frame_idx'] or source['evaluation_only_official']!=reference['evaluation_only_official']:
                    raise AssertionError('帧或GT变更')
                p=Path(reference['input']['detections'])
                if digest(p)!=reference['input']['detections_sha256']:
                    raise ValueError('冻结检测缓存改变')
                with gzip.open(p,'rb') as stream:
                    gobs=pickle.load(stream)
                frame=MappingFrame(idx,np.asarray(Image.open(source['rgb']).convert('RGB')),
                    np.asarray(Image.open(source['depth']),dtype=float)/1000,np.asarray(source['intrinsic']),
                    np.asarray(source['camera_pose']),gobs,source['rgb'],timestamp_s=source['camera_timestamp_s'])
                np.random.seed(baseline['seed']+idx)
                begun=time.perf_counter()
                mapper.update(frame)
                predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
                mapping_s=time.perf_counter()-begun
                truth=source['evaluation_only_official']
                metric=object_frame_metrics(predictions,truth,threshold_m=3.)
                record=dict(scene=name,frame_idx=idx,camera_timestamp_s=source['camera_timestamp_s'],
                    predictions={CONDITION:predictions},metrics={CONDITION:metric},evaluation_only_official=truth,
                    gt_in_inference=False,input=reference['input'],audit=mapper.frame_audit,
                    image_continuity_audit=mapper.position_layer.fallback_used,mapping_s=mapping_s)
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                scene_rows.append(record)
                for key in ['true_positive','truth','predictions','false_negative','unmatched_predictions','duplicate_nodes']:
                    total[key]+=metric[key]
                total['center_error_sum_m']+=sum(m['distance_m'] for m in metric['matches'])
                total['mapping_s']+=mapping_s
                for link in mapper.position_layer.fallback_used:
                    scene_fallback['eligible_image_links']+=1
                    scene_fallback['selected_image_links']+=link['selected_update']
                frames+=1
        counts=identity_counts(prepare_sequence(scene_rows,CONDITION),'all_gated')['counts']
        identity.update(counts)
        fallback_total.update(scene_fallback)
        dump(target/(P+'scene_report.json'),dict(scene=name,log_token=scene['log_token'],frames=len(scene_rows),identity=counts,image_links=dict(scene_fallback)))
        print(name,frames,'complete',dict(scene_fallback),flush=True)
    if not args.pilot and (len(scenes),frames,total['truth'])!=(38,1508,5723):
        raise AssertionError('完整候选范围错误')
    counts=dict(total)
    counts.update(identity)
    metrics=dict(recall=total['true_positive']/total['truth'],precision=total['true_positive']/total['predictions'],
        F1=2*total['true_positive']/(total['truth']+total['predictions']),
        IDF1=2*identity['IDTP']/(identity['GT_observations']+identity['prediction_observations']),
        center_error_m=total['center_error_sum_m']/total['true_positive'])
    report=dict(condition=CONDITION,pilot=args.pilot,full_validation_scope=not args.pilot,scenes=len(scenes),
        logs=len({s['log_token'] for s in scenes}),frames=frames,result=dict(counts=counts,metrics=metrics),
        image_links=dict(fallback_total),wall_s=time.monotonic()-start,threshold_m=3.,
        actual_frontend_calls=0,test_accessed=False,default_changed=False,
        output_sha256={str(p.relative_to(dest)):digest(p) for p in dest.glob('scene-*/*') if p.is_file()})
    dump(dest/(P+'report.json'),report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
