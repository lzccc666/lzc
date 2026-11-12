"""冻结真实录制前端上的感知缺失、稀疏深度和位姿扰动开发实验。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import tarfile
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.quality import object_frame_metrics, identity_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump, predict


CONDITIONS={
    'unperturbed':{},
    'detection_dropout_30pct':dict(detection_keep=.7),
    'lidar_dropout_80pct':dict(lidar_keep=.2),
    'depth_noise_020m':dict(depth_sigma_m=.2),
    'pose_noise_025m_2deg':dict(position_sigma_m=.25,yaw_sigma_deg=2.),
    'three_frame_perception_blackout':dict(blackout_indices=[8,9,10]),
}


def retain_detections(gobs,keep):
    out=copy.deepcopy(gobs)
    for key in ['xyxy','confidence','class_id','mask','image_crops','image_feats','text_feats']:
        if key in out and out[key] is not None:
            out[key]=[out[key][i] for i in np.flatnonzero(keep)] if isinstance(out[key],list) else out[key][keep]
    return out


def perturb(depth,pose,gobs,condition,rng,index):
    cfg=CONDITIONS[condition]
    noisy=depth.copy()
    observed_pose=pose.copy()
    keep=np.ones(len(gobs['xyxy']),bool)
    if 'detection_keep' in cfg:
        keep=rng.random(len(keep))<cfg['detection_keep']
    if index in cfg.get('blackout_indices',[]):
        keep[:]=False
    observation=retain_detections(gobs,keep)
    if 'lidar_keep' in cfg:
        noisy[rng.random(depth.shape)>cfg['lidar_keep']]=0
    if 'depth_sigma_m' in cfg:
        valid=noisy>0
        noisy[valid]=np.maximum(.001,noisy[valid]+rng.normal(0,cfg['depth_sigma_m'],int(valid.sum())))
    if 'position_sigma_m' in cfg:
        observed_pose[:2,3]+=rng.normal(0,cfg['position_sigma_m'],2)
        yaw=rng.normal(0,math.radians(cfg['yaw_sigma_deg']))
        rotation=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        observed_pose[:3,:3]=rotation@observed_pose[:3,:3]
    return noisy,observed_pose,observation,dict(input_detections=len(keep),retained_detections=int(keep.sum()),
        input_depth_pixels=int(np.count_nonzero(depth)),retained_depth_pixels=int(np.count_nonzero(noisy)),
        position_delta_m=(observed_pose[:3,3]-pose[:3,3]).tolist(),
        gobs_classes_and_frame_provenance_unchanged=True)


def mapping_config():
    cfg=OmegaConf.load(PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml')
    cfg.merge_with(dict(device='cpu',min_points_threshold=4,obj_min_detections=1,
        dbscan_remove_noise=False,match_method='sim_sum',sim_threshold=.35,phys_bias=.5,
        spatial_sim_type='iou',frame_dt=1.,skip_bg=True,association_policy='one_to_one',
        dynamic_ttl_s=2.,initial_motion_unknown=True,static_confirmation_observations=3,
        depth_policy='largest_supported_cluster',motion_estimator='robust_xy'))
    return cfg


def run_case(frames,condition,seed,dest,classes,deadline,*,inference_config=None):
    from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper
    mapper=StreamlinedMapper(mapping_config() if inference_config is None else inference_config,classes)
    metrics=[]
    identities=[]
    log=dest/'frames.jsonl'
    with log.open('x') as handle:
        for source in frames:
            if time.monotonic()>deadline:
                raise TimeoutError('fault experiment wall budget; raw prefix retained')
            index=source['frame_idx']
            # 稳定种子绑定场景/帧/条件；不使用目标或GT决定扰动位置。
            token=f"{seed}/{source['scene']}/{index}/{condition}".encode()
            rng=np.random.default_rng(int(hashlib.sha256(token).hexdigest()[:16],16))
            image=np.asarray(Image.open(source['rgb']).convert('RGB'))
            depth=np.asarray(Image.open(source['depth']),dtype=float)/1000.
            with gzip.open(source['detections'],'rb') as f:
                gobs=pickle.load(f)
            true_pose=np.asarray(source['camera_pose'])
            observed_depth,observed_pose,observation,audit=perturb(depth,true_pose,gobs,condition,rng,index)
            np.random.seed(seed+index)
            mapper.update(MappingFrame(index,image,observed_depth,np.asarray(source['intrinsic']),
                observed_pose,observation,source['rgb'],timestamp_s=source['camera_timestamp_s']))
            # 真值和真实相机只在推理结束后的隔离评估中使用；不因噪声改变标注范围。
            predicted=predict(mapper.objects,classes,pose=true_pose,intrinsic=np.asarray(source['intrinsic']))
            truth=source['evaluation_only_official']
            m=object_frame_metrics(predicted,truth)
            metrics.append(m)
            identities.append((index,[(predicted[p['prediction_index']]['node_id'],
                truth[p['truth_index']]['instance_token']) for p in m['matches']]))
            row=dict(scene=source['scene'],condition=condition,seed=seed,frame_idx=index,
                camera_timestamp_s=source['camera_timestamp_s'],perturbation=audit,
                predictions=predicted,metrics=m,gt_sent_to_mapper=False)
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush()
    totals={key:sum(m[key] for m in metrics) for key in
        ['predictions','truth','true_positive','unmatched_predictions','false_negative','duplicate_nodes']}
    errors=[p['distance_m'] for m in metrics for p in m['matches']]
    totals.update(recall=totals['true_positive']/totals['truth'] if totals['truth'] else None,
        precision=totals['true_positive']/totals['predictions'] if totals['predictions'] else None,
        center_error_m_mean=float(np.mean(errors)) if errors else None,
        **identity_metrics(identities))
    report=dict(scene=frames[0]['scene'],condition=condition,seed=seed,frames=len(frames),
        metrics=totals,records_sha256=digest(log),gt_in_inference=False)
    dump(dest/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--scene',action='append')
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    parser.add_argument('--wall-budget-s',type=float,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    known={r['scene']:r for r in available_development()}
    scenes=args.scene or sorted(known)
    if not set(scenes)<=known.keys() or len(scenes)!=len(set(scenes)):
        raise ValueError('only unique development scenes allowed')
    dest=OUTPUT/'faults'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    sources=[OUTPUT/'mapping_evaluation/robust_motion_full_001'/s/'frames.jsonl' for s in scenes]
    py_sources=[Path(__file__),PROJECT/'conceptgraph/slam/streamlined_mapping.py',
        PROJECT/'conceptgraph/slam/mapping.py',PROJECT/'conceptgraph/slam/utils.py',
        PROJECT/'conceptgraph/completion/depth.py',PROJECT/'conceptgraph/completion/association.py',
        PROJECT/'conceptgraph/completion/motion.py',PROJECT/'conceptgraph/completion/quality.py',
        PROJECT/'conceptgraph/completion/stage9.py',PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml']
    dump(dest/'release.json',dict(conditions=CONDITIONS,scene_logs={s:known[s]['log_token'] for s in scenes},
        seeds=args.seeds,wall_budget_s=args.wall_budget_s,storage_budget_mb=350,
        source_frames={str(p):digest(p) for p in sources},
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in py_sources},
        inference_config=OmegaConf.to_container(mapping_config()),
        temporal_rule='逐帧当前及过去观测；扰动不读取GT；评估范围保持原真实相机范围',
        physical_limit='合成白噪声/随机缺失压力测试，不代表经过标定的真实传感器误差分布',
        independent_test=False,gt_in_inference=False,goal_complete=False))
    with tarfile.open(dest/'source_snapshot.tar.gz','x:gz') as archive:
        for p in py_sources:
            archive.add(p,arcname=str(p.relative_to(PROJECT)))
    reports=[]
    for path in sources:
        frames=[json.loads(line) for line in path.open()]
        classes_path=Path(frames[0]['detections']).parents[2]/'gsa_classes_streamlined_yolo.json'
        frontend_manifest=classes_path.parent/'frontend_complete.json'
        frozen_frontend=json.loads(frontend_manifest.read_text())
        for name,sha in frozen_frontend['output_sha256'].items():
            if digest(name)!=sha:
                raise ValueError('actual frontend evidence changed')
        classes=json.loads(classes_path.read_text())
        for seed in args.seeds:
            for condition in CONDITIONS:
                case=dest/path.parent.name/f'seed{seed}'/condition
                case.mkdir(parents=True)
                try:
                    report=run_case(frames,condition,seed,case,classes,started+args.wall_budget_s)
                except Exception as exc:
                    dump(case/'failed.json',dict(error=repr(exc),goal_complete=False))
                    raise
                reports.append(report)
                print(json.dumps(report,ensure_ascii=False),flush=True)
    conditions=defaultdict(list)
    for report in reports:
        conditions[report['condition']].append(report)
    dump(dest/'summary.json',dict(cases=len(reports),scenes=len(scenes),seeds=args.seeds,
        input_frames=sum(len(list(p.open())) for p in sources),conditions=conditions,
        elapsed_s=time.monotonic()-started,all_failures_retained=True,actual_frontend_saved_outputs=True,
        gpu_inference_this_run=False,navigation_arrival_evaluated=False,
        independent_test=False,stage12_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
