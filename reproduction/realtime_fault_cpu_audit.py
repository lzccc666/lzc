"""CPU实际传感器/真实检测故障的逐帧输入审计；保留未暴露与无制动负结果。"""
from pathlib import Path
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import pickle

import numpy as np
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT,dump
from conceptgraph.completion.kinematic_benchmark import setup,SPEC
from conceptgraph.navigation.vehicle_geometry import VehicleState,audit_executed_states


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('nonstandard JSON number '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def main():
    root=OUTPUT/'faults/realtime_fault_cpu_smoke_002'
    assert (root/'summary.json').exists(),'CPU真实试跑未完整，不按前缀评价'
    destination=root/'audit_001';destination.mkdir(exist_ok=False)
    release=read(root/'release.json');summary=read(root/'summary.json')
    tasks=read(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json')
    configs={c['config_id']:c for c in tasks['configs']}
    task=next(t for t in tasks['tasks'] if t['task_id']==release['task'])
    sources={root/'release.json':sha(root/'release.json'),root/'summary.json':sha(root/'summary.json')}
    reports=[]
    for index,trial in enumerate(release['plan']):
        record=read(root/f'{index:02d}_record.json');assert record['returncode']==0
        case=Path(record['raw_case']);result=read(case/'result.json')
        assert sha(case/'result.json')==record['result_sha256']
        assert not result['gt_in_agent'] and result['semantic_applied']==0
        source_frames=sorted((case/'sensor_evidence').glob('*.json'))
        control=[json.loads(s) for s in (case/'execution.jsonl').open()]
        events=[json.loads(s) for s in (case/'fault_events.jsonl').open()]
        detectors={e['frame_idx']:e for e in events if e['event']=='detector'}
        sensor_events={e['frame_idx']:e for e in events if e['event']=='sensor'}
        assert len(source_frames)==len(detectors)==record['exposure']['actual_detector_calls']
        active_observations=valid_depth=retained_depth=removed_detections=raw_detections=0
        depth_delta=[];pose_delta=[];field_elements=0
        for frame_path in source_frames:
            frame=read(frame_path);i=frame['frame_idx']
            raw_path=Path(frame['unmodified_sensor_evidence']);raw=read(raw_path)
            assert frame['state']==raw['state'] and frame['timestamp_s']==raw['timestamp_s']
            assert frame['rgb_sha256']==raw['rgb_sha256']==sha(frame['rgb'])==sha(raw['rgb'])
            assert frame['depth_sha256']==sha(frame['depth']) and raw['depth_sha256']==sha(raw['depth'])
            actual,original=np.asarray(Image.open(frame['depth'])),np.asarray(Image.open(raw['depth']))
            assert np.issubdtype(actual.dtype,np.integer) and np.issubdtype(original.dtype,np.integer) and actual.shape==original.shape
            assert Path(frame['depth']).read_bytes()[24]==Path(raw['depth']).read_bytes()[24]==16
            assert np.all(actual[original==0]==0)
            active=frame['fault_active'];condition=trial['condition']
            assert active==sensor_events[i]['active']==detectors[i]['active']
            changed=np.abs(actual.astype(float)-original.astype(float))/1000
            if active:
                active_observations+=1
                valid_depth+=int(np.count_nonzero(original));retained_depth+=int(np.count_nonzero(actual))
                depth_delta.extend(changed[original>0].tolist())
            if condition=='lidar_dropout_80pct' and active:
                assert np.all((actual==0)|(actual==original))
            elif condition!='depth_noise_02m' or not active:
                assert np.array_equal(actual,original)
            original_pose,pose=np.asarray(raw['camera_pose_cv']),np.asarray(frame['camera_pose_cv'])
            assert np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-9)
            if condition!='pose_noise_025m_2deg' or not active:
                assert np.array_equal(pose,original_pose)
            else:
                pose_delta.append((pose[:3,3]-original_pose[:3,3]).tolist())
            raw_gobs_path=case/'unmodified_detections'/f'{i:06d}.pkl.gz'
            effective_path=case/'detections'/f'{i:06d}.pkl.gz'
            assert sha(raw_gobs_path)==detectors[i]['unmodified_detection_sha256']
            with gzip.open(raw_gobs_path) as h:gobs=pickle.load(h)
            with gzip.open(effective_path) as h:effective=pickle.load(h)
            assert set(effective)==set(gobs) and 'fault_injection' not in effective
            assert gobs['classes']==effective['classes']
            assert gobs['frame_provenance']==effective['frame_provenance']
            assert gobs['frame_provenance']['rgb_sha256']==frame['rgb_sha256']
            keep=detectors[i]['retained_indices'];n=len(gobs['xyxy'])
            assert keep==sorted(set(keep)) and all(0<=j<n for j in keep)
            assert len(effective['xyxy'])==len(keep)==detectors[i]['retained_detections']
            if condition!='detection_dropout_30pct' or not active:assert keep==list(range(n))
            if active:
                raw_detections+=n;removed_detections+=n-len(keep)
            for field in ['xyxy','confidence','class_id','mask','image_crops','image_feats','text_feats']:
                for original_index,new in zip(keep,effective[field]):
                    assert np.array_equal(np.asarray(gobs[field][original_index]),np.asarray(new))
                    field_elements+=1
            for p in [frame_path,raw_path,Path(frame['rgb']),Path(raw['rgb']),Path(frame['depth']),Path(raw['depth']),raw_gobs_path,effective_path]:
                sources[p]=sha(p)
        config=configs[task['config_id']];rng=np.random.default_rng(release['seed'])
        initial=VehicleState(*(np.asarray(config['start_xy'])+rng.normal(0,.03,2)),
            config['start_yaw_rad']+rng.normal(0,math.radians(.5)))
        _,road,obstacles,_=setup(config,task)
        states=[initial]+[VehicleState(**v['state']) for v in control]
        audit=audit_executed_states(states,road,obstacles,SPEC)
        assert audit==result['independent_execution_audit']
        assert bool(active_observations)==record['exposure']['actual_fault_exposed']
        report=dict(condition=trial['condition'],actual_frames=len(source_frames),active_frames=active_observations,
            active_input_depth_pixels=valid_depth,active_retained_depth_pixels=retained_depth,
            active_retained_depth_fraction=retained_depth/valid_depth if valid_depth else None,
            active_raw_detections=raw_detections,active_removed_detections=removed_detections,
            active_depth_delta_mean_absolute_m=float(np.mean(depth_delta)) if depth_delta else None,
            active_depth_delta_RMS_m=float(np.sqrt(np.mean(np.square(depth_delta)))) if depth_delta else None,
            actual_pose_translation_deltas_m=pose_delta,aligned_field_elements_checked=field_elements,
            original_and_effective_RGB_bytes_equal=True,invalid_depth_not_densified=True,
            control_ticks=len(control),existing_independent_execution_auditor_rerun_equal=True,
            actual_status=result['status'],actual_full_perception_arrival=result['actual_full_perception_arrival'],
            exposure=record['exposure'])
        reports.append(report)
    assert summary['attempted']==summary['expected']==len(reports)==6
    result=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),cases=reports,
        actual_frontend_calls=sum(r['actual_frames'] for r in reports),actual_control_ticks=sum(r['control_ticks'] for r in reports),
        source_files_checked=len(sources),source_files={str(k):v for k,v in sources.items()},
        audit_source_sha256=sha(__file__),extra_unique_tasks=0,unique_tasks=1,unique_configurations=1,
        three_conditions_not_conflated='CPU输入试跑、已知路径控制功能、正式GPU故障矩阵分别评价',
        second_independent_vehicle_geometry_implementation=False,real_human_records_added=0,formal_GPU_faults_complete=False,
        scope='CPU/OSMesa规则真实前端六条件单任务单种子；TTL15秒，不横比GPU1.5秒或独占性能',goal_complete=False)
    dump(destination/'report.json',result)
    rows=['# CPU真实前端六故障条件逐帧复核','',
        '本批全部六条件、同一任务config00-positive0/seed17。CPU/llvmpipe、规则前端、15秒TTL；不新增独立任务/配置覆盖，不是正式GPU故障或独占性能。',
        '逐帧原图哈希相同、零深度未加密、位姿与物理车态隔离、真实检测七字段保留索引全部核对。复跑既有独立车身执行审计与保存结果一致；本次不是第二套独立几何实现。','',
        '|条件|帧/真实前端调用|注入帧|控制tick|终止|实际到达|','|---|---:|---:|---:|---|---|']
    for report in reports:
        rows.append(f"|{report['condition']}|{report['actual_frames']}|{report['active_frames']}|{report['control_ticks']}|{report['actual_status']}|{report['actual_full_perception_arrival']}|")
    blackout=next(r for r in reports if r['condition']=='sensor_blackout_4s')['exposure']
    rows+=['',f"实际断流{blackout.get('blackout_duration_s')}秒；期间{blackout['blackout_control_ticks']}控制tick，过期制动{blackout['blackout_stale_braking_ticks']}tick，断流期停止={blackout['blackout_stopped_during_gap']}，随后新观测下运动={blackout['after_gap_fresh_motion_observed']}。15秒TTL下未触发过期制动的负结果保留，不能称该CPU条件证明过期安全停止。",'全部原始成功/失败按实际记录保留，真实人工0，正式GPU矩阵仍待前置主矩阵及缺一任务完整结束。']
    (destination/'report.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_files']},ensure_ascii=False))


if __name__=='__main__':main()
