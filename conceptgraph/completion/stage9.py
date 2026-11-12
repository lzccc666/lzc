"""在允许的开发场景真实重建，并用隔离官方真值诊断对象质量。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time
import gzip
import pickle
import tarfile

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.quality import object_frame_metrics, identity_metrics

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
HISTORY = PROJECT/'outputs/project_v2/lane_qwen_expansion_v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def available_development():
    rows=[]
    for base in [HISTORY, HISTORY/'prospective_supplement_v2']:
        split=json.loads((base/'split_manifest.json').read_text())['scene_to_split']
        selection=json.loads((base/'scene_selection_manifest.json').read_text())['scenes']
        for scene in selection:
            if split.get(scene['scene'])=='development':
                rows.append(dict(scene=scene['scene'], log_token=scene['log_token'], root=str(base)))
    return sorted(rows, key=lambda r:r['scene'])


def predict(objects, classes, *, pose=None, intrinsic=None):
    rows=[]
    for index, obj in enumerate(objects):
        class_ids=np.asarray(obj['class_id'],dtype=int)
        unique,counts=np.unique(class_ids,return_counts=True)
        category=classes[int(unique[np.argmax(counts)])]
        if category not in {'car','truck','bus','bicycle','motorcycle','pedestrian'}:
            continue
        center=np.asarray(obj['bbox'].get_center(),dtype=float)
        if pose is not None:
            camera=pose[:3,:3].T@(center-pose[:3,3])
            uv=intrinsic@camera
            if not (1<camera[2]<65 and 0<=uv[0]/uv[2]<1600 and 0<=uv[1]/uv[2]<900):
                continue
        color=np.asarray(obj['inst_color']).round(6).tolist()
        identity=obj.get('stable_object_id', 'color-'+','.join(map(str,color)))
        rows.append(dict(node_id=identity, category=category, center=center.tolist(),
                         extent=np.asarray(obj['bbox'].get_extent()).tolist(),
                         num_detections=int(obj['num_detections']),
                         motion_state=obj.get('motion_state','unknown'),
                         velocity_mps=float(obj.get('velocity',0)),
                         last_seen_frame=int(obj.get('last_seen_frame',-1)),
                         timestamp_s=obj.get('timestamp_s'),
                         depth_evidence=copy.deepcopy(obj.get('depth_evidence', [])[-1:]),
                         source_index=index))
    return rows


def run_scene(scene, destination, *, frontend_run=None):
    from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper

    name=scene['scene']
    base=Path(scene['root'])
    converted=base/'converted'/name
    if frontend_run is not None:
        converted=OUTPUT/'mapping_evaluation'/frontend_run/'inputs'/name
        completed=json.loads((converted/'frontend_complete.json').read_text())
        for p,h in completed['output_sha256'].items():
            if digest(p)!=h:
                raise ValueError('前端输出指纹不一致')
    alignment=json.loads((converted/'frame_alignment.json').read_text())
    poses=np.loadtxt(converted/'poses.txt').reshape(-1,4,4)
    classes=json.loads((converted/'gsa_classes_streamlined_yolo.json').read_text())
    # 评估专用GT在观测送入mapper后才读取；不加入MappingFrame、缓存或关联输入。
    gt={int(r['frame_idx']):r['objects'] for r in map(json.loads,
        (base/'ground_truth_frames.jsonl').open()) if r['scene']==name}
    cfg=OmegaConf.load(PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml')
    cfg.merge_with(dict(device='cpu', min_points_threshold=4, obj_min_detections=1,
        dbscan_remove_noise=False, match_method='sim_sum', sim_threshold=0.35,
        phys_bias=0.5, spatial_sim_type='iou', frame_dt=1.0, skip_bg=True))
    old=StreamlinedMapper(copy.deepcopy(cfg),classes)
    newcfg=copy.deepcopy(cfg)
    newcfg.association_policy='one_to_one'
    newcfg.dynamic_ttl_s=2.0
    newcfg.initial_motion_unknown=True
    newcfg.static_confirmation_observations=3
    improved=StreamlinedMapper(newcfg,classes)
    timestamp_only=StreamlinedMapper(copy.deepcopy(cfg),classes)
    no_expiry_cfg=copy.deepcopy(newcfg)
    no_expiry_cfg.dynamic_ttl_s=1e9
    methods={'original_fusion':old,'timestamp_only':timestamp_only,
             'one_to_one_no_expiry':StreamlinedMapper(no_expiry_cfg,classes),
             'one_to_one_timestamp':improved}
    depth_cfg=copy.deepcopy(newcfg)
    depth_cfg.depth_policy='largest_supported_cluster'
    methods['one_to_one_depth_quality']=StreamlinedMapper(depth_cfg,classes)
    motion_cfg=copy.deepcopy(depth_cfg)
    motion_cfg.motion_estimator='robust_xy'
    methods['one_to_one_depth_robust_motion']=StreamlinedMapper(motion_cfg,classes)
    if scene.get('methods'):
        methods={k:v for k,v in methods.items() if k in scene['methods']}
    metrics=defaultdict(list)
    identities=defaultdict(list)
    inputs=[]
    time_rows=[]
    started=time.monotonic()
    with (destination/'frames.jsonl').open('x') as frame_out:
        for idx,row in enumerate(alignment):
            color=converted/'color'/f'{idx:06d}.jpg'
            depthpath=converted/'depth'/f'{idx:06d}.png'
            detpath=converted/'exp_streamlined_yolo_outdoor_objects_v1/detections'/f'{idx:06d}.pkl.gz'
            image=np.asarray(Image.open(color).convert('RGB'))
            depth=np.asarray(Image.open(depthpath),dtype=float)/1000
            with gzip.open(detpath,'rb') as f:
                gobs=pickle.load(f)
            # 相机原始采集时间，保留样本/LiDAR时间用于差异审计。
            camera_us=int(Path(row['camera_source']).stem.split('__')[-1])
            timestamp_s=camera_us/1e6
            k=np.asarray(row['intrinsic'],dtype=float)
            inputs.append({str(p):digest(p) for p in [color,depthpath,detpath]})
            time_rows.append(dict(frame_idx=idx,sample_us=row['timestamp'],camera_us=camera_us))
            predictions={}
            for method,mapper in methods.items():
                np.random.seed(17+idx)
                frame=MappingFrame(idx,image,depth,k,poses[idx],copy.deepcopy(gobs),str(color),
                    timestamp_s=timestamp_s if method!='original_fusion' else None)
                mapper.update(frame)
                predictions[method]=predict(mapper.objects,classes,pose=poses[idx],intrinsic=k)
            observation_source=improved if 'one_to_one_timestamp' in methods else next(iter(methods.values()))
            predictions['single_frame']=predict(observation_source.last_observations,classes,pose=poses[idx],intrinsic=k)
            results={}
            for method,objects in predictions.items():
                m=object_frame_metrics(objects,gt[idx])
                results[method]=m
                metrics[method].append(m)
                identities[method].append((idx,[(objects[p['prediction_index']]['node_id'],
                    gt[idx][p['truth_index']]['instance_token']) for p in m['matches']]))
            record=dict(scene=name,frame_idx=idx,camera_timestamp_s=timestamp_s,
                rgb=str(color),depth=str(depthpath),detections=str(detpath),
                camera_pose=poses[idx].tolist(),intrinsic=k.tolist(),
                predictions=predictions,evaluation_only_official=gt[idx],metrics=results)
            frame_out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    summary={}
    for method,frames in metrics.items():
        totals={key:sum(f[key] for f in frames) for key in ['predictions','truth',
            'true_positive','unmatched_predictions','false_negative','duplicate_nodes']}
        tp=totals['true_positive']
        errors=[m['distance_m'] for f in frames for m in f['matches']]
        totals.update(recall=tp/totals['truth'] if totals['truth'] else None,
            precision=tp/totals['predictions'] if totals['predictions'] else None,
            center_error_m_mean=float(np.mean(errors)) if errors else None,
            **identity_metrics(identities[method]))
        summary[method]=totals
    dts=np.diff([r['camera_us'] for r in time_rows])/1e6
    report=dict(scene=name,log_token=scene['log_token'],frame_count=len(alignment),
        runtime_s=time.monotonic()-started,methods=summary,inputs=inputs,
        scope='CAM_FRONT中心可见、1<z<65米、官方visibility>=2；仅六个可核验类别；非官方nuScenes评测',
        unmatched_prediction_meaning='范围内未匹配，含评测关联不足；不得全部解释为误检',
        time_audit=dict(actual_camera_dt_s_min=float(dts.min()),actual_camera_dt_s_median=float(np.median(dts)),
            actual_camera_dt_s_max=float(dts.max()),historical_frame_dt_s=1.0,
            historical_config_evidence=str(base/'preprocessing/mapping'/name/'complete.json'),
            frames=time_rows))
    dump(destination/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--scene',action='append')
    parser.add_argument('--frontend-run')
    parser.add_argument('--methods',nargs='+')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():
        raise ValueError('非法run-id')
    run=OUTPUT/'mapping_evaluation'/args.run_id
    scenes=available_development()
    if args.scene:
        scenes=[s for s in scenes if s['scene'] in args.scene]
        if len(scenes)!=len(set(args.scene)):
            raise ValueError('只允许开发场景')
    if args.limit:
        scenes=scenes[:args.limit]
    if args.methods:
        known={'original_fusion','timestamp_only','one_to_one_no_expiry','one_to_one_timestamp',
               'one_to_one_depth_quality','one_to_one_depth_robust_motion'}
        if not set(args.methods)<=known:
            raise ValueError('未知方法')
        scenes=[dict(s,methods=args.methods) for s in scenes]
    sources=[Path(__file__),PROJECT/'conceptgraph/slam/mapping.py',
        PROJECT/'conceptgraph/slam/streamlined_mapping.py', PROJECT/'conceptgraph/slam/utils.py',
        PROJECT/'conceptgraph/completion/association.py',PROJECT/'conceptgraph/completion/quality.py',
        PROJECT/'conceptgraph/completion/depth.py',PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml']
    sources.append(PROJECT/'conceptgraph/completion/motion.py')
    frozen=dict(scenes=scenes,source_sha256={str(p):digest(p) for p in sources},
                purpose='开发诊断，不开启新封存测试',ground_truth_in_inference=False)
    frozen['evaluation_only_truth_sha256']={s['root']:digest(Path(s['root'])/'ground_truth_frames.jsonl') for s in scenes}
    if args.frontend_run:
        frontend=OUTPUT/'mapping_evaluation'/args.frontend_run
        frozen['frontend_release_sha256']=digest(frontend/'release.json')
        frozen['frontend_run']=args.frontend_run
        frozen['frontend_scene_manifests']={s['scene']:digest(frontend/'inputs'/s['scene']/'frontend_complete.json')
                                            for s in scenes}
    if run.exists():
        if not args.resume or json.loads((run/'release.json').read_text())!=frozen:
            raise ValueError('已有run或指纹改变；请使用新run-id')
    else:
        run.mkdir(parents=True)
        dump(run/'release.json',frozen)
        with tarfile.open(run/'frozen_source_snapshot.tar.gz','x:gz') as archive:
            for p in sources:
                archive.add(p,arcname=str(p.relative_to(PROJECT)))
    reports=[]
    for scene in scenes:
        dest=run/scene['scene']
        if (dest/'report.json').exists():
            reports.append(json.loads((dest/'report.json').read_text()))
            continue
        if dest.exists():
            raise ValueError('保留中断目录；请另建run-id，不覆盖失败')
        dest.mkdir()
        try:
            report=run_scene(scene,dest,frontend_run=args.frontend_run)
        except Exception as e:
            dump(dest/'failed.json',dict(error=repr(e),goal_complete=False))
            raise
        reports.append(report)
        print(json.dumps({k:report[k] for k in ['scene','frame_count','runtime_s','methods']},ensure_ascii=False),flush=True)
    if not (run/'summary.json').exists():
        dump(run/'summary.json',dict(scene_count=len(reports),frames=sum(r['frame_count'] for r in reports),
            reports=[str(run/r['scene']/'report.json') for r in reports],
            independent_test=False,stage9_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
