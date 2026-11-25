"""第7辅助完整轨迹控制：原六方法原样；同六RGB特征、共同最早past扫描。"""
from __future__ import annotations
import argparse,copy,gzip,json,shutil,time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from conceptgraph.completion import nuscenes_full_20261125_run as base
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.slam.streamlined_mapping import MappingFrame
from conceptgraph.scenegraph.relation_nav_unified_20261125 import CurrentRelationCache
_ORIGINAL_CPU_SCENE=base.cpu_scene
CONTROL='S2_6_sharedscan'


def freeze():
    paths=[Path(__file__).resolve(),source.ROOT/'scripts/nuscenes_full_20261125_sharedscan.py',source.ROOT/'conceptgraph/completion/nuscenes_full_20261125_sharedscan_evaluate.py',source.ROOT/'scripts/nuscenes_full_20261125_sharedscan_evaluate.py']
    record=dict(created_utc=source.now(),control_method=CONTROL,source_sha256={str(p):source.sha(p) for p in paths},
        original65_source_preseal_sha256=source.sha(base.DATA/'phaseB_source_preseal_v1.json'),all_original_six_unchanged=True,primary_five_unchanged=True,
        selection='all850 original scenes/all34149groups full independent mapper trajectories, including groups with no extra scan; not a per-frame subset',
        protocol='nearest requested LiDAR strictly before earliest six exposures, then eachcamera0<age<=100ms else emptydepth; earliest absence entiregroupempty',
        matches_historical_control='scripts/mv136_20261123_sharedscan.py',historical120_does_not_prove_full850=True,
        inherited_config='exact S2 frozen settings; same6actualRGB and original per-image detections/features',GT_in_method=False,new_GPU_calls=0,
        auxiliary_evaluation_scopes=['front','front3','union6'],auxiliary_only_no_primary_significance_test=True,
        estimates=dict(extra_CPU_hours=[2,3],extra_persistent_GiB=[.2,.7],needs_first8_resource_measurement=True),
        budgets_unchanged=dict(nu_total_persistent_bytes=int(4.5*2**30),temporary_bytes=512*2**20,full_wall_s=36*3600,GPU_occupied_s=12*3600),
        cleanup='originalsix CPUworker cleans current scene only after sharedscan auxiliary output also closed; failures retain temporary/prediction prefixes')
    source.dump(base.DATA/'sharedscan_auxiliary_protocol_v1.json',record)


def control_scene(scene,request,temp,receipts,geometry_path,gpu_result,output,config=None):
    import pickle,torch
    torch.set_num_threads(2);torch.manual_seed(17)
    frozen=json.loads((base.DATA/'method_manifest.json').read_text()) if config is None else dict(configs=dict(S2=config));geom=json.loads(Path(geometry_path).read_text());poses,calibrations=geom['poses'],geom['calibrations']
    mapper=base.FullS2(OmegaConf.create(frozen['configs']['S2']),base.CLASSES);registry={};cache=CurrentRelationCache();counts=Counter();tick=time.monotonic()
    with gzip.open(output/'sharedscan_frames.jsonl.gz','xt',compresslevel=6) as out:
        for group in request['groups']:
            earliest=min(group['views'],key=lambda c:request['sensors'][group['views'][c]['camera']]['timestamp_us'])
            chosen=group['views'][earliest]['lidar'];lidar=request['sensors'].get(chosen);stamp=group['reference_timestamp_us']/1e6;index=group['frame_idx']
            frames=[];projections={};contexts={};view_protocol={}
            for camera in base.CHANNELS:
                v=group['views'][camera];cam=request['sensors'][v['camera']];pose=base.sensor_transform(cam,poses,calibrations)
                intrinsic=np.asarray(calibrations[cam['calibrated_sensor_token']]['camera_intrinsic']);width,height=cam['width'],cam['height'];rgbpath=temp/cam['filename']
                cached=gpu_result['detections'][v['camera']];assert source.sha(cached['path'])==cached['sha256']
                with gzip.open(cached['path'],'rb') as f:det=pickle.load(f)
                assert base.prediction_digest(det)==cached['prediction_sha256'];age=cam['timestamp_us']-lidar['timestamp_us'] if lidar else None
                eligible=age is not None and 0<age<=100000
                if eligible:
                    points=np.fromfile(temp/lidar['filename'],np.float32).reshape(-1,5);world=base.sensor_transform(lidar,poses,calibrations)
                    depth,pixels=base.projection(points,base.inverse_transform(pose)@world,intrinsic,height,width)
                else:
                    depth=np.zeros((height,width),np.uint16);pixels=dict(pixel_y=np.empty(0,int),pixel_x=np.empty(0,int),point_index=np.empty(0,int),depth_m=np.empty(0,np.float32));counts['empty_depth_views']+=1
                applied=chosen if eligible else None;pixels['lidar_token']=applied;projections[camera]=pixels
                contexts[camera]=dict(camera_token=v['camera'],lidar_token=applied,rgb_sha256=receipts[v['camera']]['sha256'],lidar_sha256=receipts[applied]['sha256'] if applied else None)
                view_protocol[camera]=dict(requested_shared_lidar_token=chosen,applied_lidar_token=applied,lidar_age_s=age/1e6 if age is not None else None,
                    age_exceeded=age is not None and age>100000,main_protocol_lidar_token=v['lidar'],eligible=eligible)
                frames.append((camera,MappingFrame(index,np.asarray(Image.open(rgbpath).convert('RGB')),depth.astype(float)/1000.,intrinsic,pose,det,str(rgbpath),timestamp_s=stamp)))
            mapper.begin_evidence(scene['scene'],index);mapper._full_context=contexts;np.random.seed(17+index)
            preds=mapper.update_group(frames,{c:request['sensors'][group['views'][c]['camera']]['timestamp_us']/1e6 for c in base.CHANNELS},projections)
            payload=base.node_record('S2_6',mapper,preds,scene['scene'],index,stamp,frames[0][1].camera_pose,registry,cache)
            out.write(json.dumps(base.clean(dict(scene=scene['scene'],log=scene['log_token'],frame_idx=index,sample_token=group['sample_token'],render_timestamp_s=stamp,
                method=CONTROL,shared_scan_requested_token=chosen,earliest_camera=earliest,view_source_protocol=view_protocol,
                camera_geometry='identical same sample/camera metadata in canonical main frames views',payload=payload,GT_in_method=False)),separators=(',',':'))+'\n')
            counts['frames']+=1;counts['predictions']+=len(preds);counts['groups_no_earliest_scan']+=int(chosen is None)
            counts['groups_input_differs_from_main']+=int(any(view_protocol[c]['applied_lidar_token']!=view_protocol[c]['main_protocol_lidar_token'] for c in base.CHANNELS))
    assert counts['frames']==scene['frames'] and not torch.cuda.is_initialized()
    return dict(complete=True,method=CONTROL,counts=dict(counts),CPU_wall_s=time.monotonic()-tick,frames_sha256=source.sha(output/'sharedscan_frames.jsonl.gz'),new_GPU_calls=0,GT_in_method=False)


def cpu_scene_with_control(scene,request,temp_path,receipts,geometry_path,gpu_result,source_io_s,output_path):
    output=Path(output_path);retained=False
    try:
        control=control_scene(scene,request,Path(temp_path),receipts,geometry_path,gpu_result,output)
        source.dump(output/'sharedscan_complete.json',control)
    except Exception as error:
        retained=True;control=dict(complete=False,method=CONTROL,error_type=type(error).__name__,error=str(error),full_auxiliary_denominator_retained=True,temporary_retained_path=temp_path)
        source.dump(output/'sharedscan_failed.json',dict(control,utc=source.now()))
    # Original six methods always execute, independently of auxiliary success.
    # If the auxiliary failed, retain only this own temporary directory for a
    # repair; never let missing auxiliary mark primary six methods incomplete.
    original_remove=base.shutil.rmtree
    def preserve_failed_aux(path,*args,**kwargs):
        if Path(path).resolve()==Path(temp_path).resolve():return None
        return original_remove(path,*args,**kwargs)
    if retained:base.shutil.rmtree=preserve_failed_aux
    try:original=_ORIGINAL_CPU_SCENE(scene,request,temp_path,receipts,geometry_path,gpu_result,source_io_s,output_path)
    finally:base.shutil.rmtree=original_remove
    root=base.RUN/scene['scene']
    if not retained:
        for name in ['sharedscan_frames.jsonl.gz','sharedscan_complete.json']:(root/name).symlink_to((output/name).relative_to(root))
    else:source.dump(root/'sharedscan_pending.json',dict(control,preserved_attempt=str(output)))
    return dict(original,auxiliary_sharedscan=control)



def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('freeze');sub.add_parser('prepare')
    r=sub.add_parser('run');r.add_argument('--resource-pilot-only',action='store_true');r.add_argument('--kitti-release-reference',required=True);args=p.parse_args()
    if args.command=='freeze':freeze();return
    protocol=json.loads((base.DATA/'sharedscan_auxiliary_protocol_v1.json').read_text())
    for path,h in protocol['source_sha256'].items():assert source.sha(path)==h,path
    if args.command=='prepare':
        base.prepare()
        for path in protocol['source_sha256']:
            target=base.RUN/'source_snapshot'/Path(path).relative_to(source.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    else:
        base.cpu_scene=cpu_scene_with_control;base.run(args)

if __name__=='__main__':main()
