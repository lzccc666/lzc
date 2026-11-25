"""同一真实YOLO缓存下重放完整建图，比较当前几何更新，禁止GT进入mapper。"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.current_geometry_mapper import CurrentGeometryMapper
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame
from run_mapping_iteration import dump, ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-budget-s', type=float, default=1200)
    args = parser.parse_args()
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    begun = time.monotonic()
    scenes = available_development()
    reference = PROJECT/'outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1'
    cfg = mapping_config()
    cfg.merge_with(dict(mask_conf_threshold=.1, strong_confidence=.2, weak_confirmation_observations=2))
    from omegaconf import OmegaConf
    sources = [Path(__file__).resolve(), PROJECT/'conceptgraph/slam/current_geometry_mapper.py',
               PROJECT/'conceptgraph/slam/evidence_mapper.py', PROJECT/'conceptgraph/slam/mapping.py',
               PROJECT/'conceptgraph/slam/streamlined_mapping.py', PROJECT/'conceptgraph/slam/utils.py',
               PROJECT/'conceptgraph/completion/depth.py', PROJECT/'conceptgraph/completion/quality.py',
               PROJECT/'conceptgraph/completion/association.py', PROJECT/'conceptgraph/completion/stage9.py']
    fingerprints = {str(p): digest(p) for p in sources}
    for p in sources:
        target = dest/'source_snapshot'/p.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    dump(dest/'release.json', dict(scenes=scenes, config=OmegaConf.to_container(cfg),
        sources_sha256=fingerprints, reference=str(reference), seed=17, threshold_m=3.,
        wall_budget_s=args.wall_budget_s, minimum_free_gib=20, maximum_new_gib=1,
        GT_in_inference=False, actual_GPU_calls=0, test_accessed=False,
        hypothesis='对交通参与者采用当前实测几何，减少静态累计点云偏移；负结果保留',
        input_frames_sha256={s['scene']:digest(reference/s['scene']/'frames.jsonl.gz') for s in scenes}))
    reports=[]
    try:
        for scene in scenes:
            name=scene['scene']
            source=OUTPUT/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
            rows=[json.loads(x) for x in source.open()]
            with gzip.open(reference/name/'frames.jsonl.gz','rt') as stream:
                cached={r['frame_idx']:r for r in map(json.loads,stream)}
            mappers={'baseline':EvidenceAwareMapper(copy.deepcopy(cfg),CLASSES),
                     'current_geometry':CurrentGeometryMapper(copy.deepcopy(cfg),CLASSES)}
            records=[]
            case=dest/name
            case.mkdir()
            with gzip.open(case/'frames.jsonl.gz','wt') as stream:
                for row in rows:
                    if time.monotonic()-begun > args.wall_budget_s:
                        raise TimeoutError('达到1200秒预算，保留前缀')
                    if shutil.disk_usage(dest).free < 20*1024**3:
                        raise OSError('空闲空间低于20GiB')
                    idx=row['frame_idx']
                    path=Path(cached[idx]['detections'])
                    assert digest(path)==cached[idx]['detection_sha256']
                    with gzip.open(path,'rb') as f:gobs=pickle.load(f)
                    image=np.asarray(Image.open(row['rgb']).convert('RGB'))
                    depth=np.asarray(Image.open(row['depth']),dtype=float)/1000
                    pose,k=np.asarray(row['camera_pose']),np.asarray(row['intrinsic'])
                    predicted={}
                    for method,mapper in mappers.items():
                        np.random.seed(17+idx)
                        mapper.update(MappingFrame(idx,image,depth,k,pose,copy.deepcopy(gobs),
                            row['rgb'],timestamp_s=row['camera_timestamp_s']))
                        predicted[method]=predict(mapper.confirmed_objects(),CLASSES,pose=pose,intrinsic=k)
                    predicted['single_frame_low_conf']=predict(mappers['baseline'].last_observations,CLASSES,pose=pose,intrinsic=k)
                    truth=row['evaluation_only_official']
                    metrics={m:object_frame_metrics(p,truth) for m,p in predicted.items()}
                    assert metrics['baseline']['true_positive']==cached[idx]['metrics']['frontend_v2_evidence']['true_positive']
                    before={m['truth_index'] for m in metrics['single_frame_low_conf']['matches']}
                    after={m['truth_index'] for m in metrics['baseline']['matches']}
                    record=dict(scene=name,frame_idx=idx,camera_timestamp_s=row['camera_timestamp_s'],predictions=predicted,metrics=metrics,
                        evaluation_only_official=truth,gt_in_inference=False,
                        input=dict(rgb=row['rgb'],depth=row['depth'],detections=str(path),detections_sha256=digest(path)),
                        diagnostic=dict(single_correct_map_wrong=len(before-after),single_wrong_map_correct=len(after-before),
                        note='包含确认与投影变化，不是纯关联损失'))
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    records.append(record)
            conditions={}
            for method in records[0]['metrics']:
                c={k:sum(r['metrics'][method][k] for r in records) for k in ['true_positive','truth','predictions','duplicate_nodes']}
                c['center_error_sum_m']=sum(m['distance_m'] for r in records for m in r['metrics'][method]['matches'])
                c.update(identity_counts(prepare_sequence(records,method),'all_gated')['counts'])
                conditions[method]=dict(counts=c,metrics=ratios(c))
            report=dict(scene=name,log=scene['log_token'],frames=len(records),conditions=conditions,
                diagnostic={k:sum(r['diagnostic'][k] for r in records) for k in ['single_correct_map_wrong','single_wrong_map_correct']})
            dump(case/'report.json',report)
            reports.append(report)
            print(json.dumps(dict(scene=name,metrics={k:v['metrics'] for k,v in conditions.items()})),flush=True)
        totals={}
        for method in reports[0]['conditions']:
            c={k:sum(r['conditions'][method]['counts'][k] for r in reports) for k in reports[0]['conditions'][method]['counts']}
            totals[method]=dict(counts=c,metrics=ratios(c))
        assert all(digest(Path(p))==h for p,h in fingerprints.items())
        dump(dest/'report.json',dict(scenes=len(reports),frames=sum(r['frames'] for r in reports),conditions=totals,
            diagnostic={k:sum(r['diagnostic'][k] for r in reports) for k in reports[0]['diagnostic']},
            wall_s=time.monotonic()-begun,baseline_reproduced_every_frame=True,test_accessed=False,actual_arrival_evaluated=False))
        print(json.dumps(totals,ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(dest/'interrupted.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_scenes=len(reports)))
        raise


if __name__=='__main__':main()
