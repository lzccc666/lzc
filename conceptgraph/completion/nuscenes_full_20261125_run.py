"""冻结六条件全场景建图：逐scene临时源、一次前端、多mapper和紧凑证据。"""
from __future__ import annotations
import argparse,atexit,copy,fcntl,gzip,hashlib,json,os,pickle,random,shutil,time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion.stage9 import predict
from conceptgraph.completion.vehicle_frontend import CLASSES,PROMPTS
from conceptgraph.project_paths import experiment_directory
from conceptgraph.scenegraph.node_motion_unified_20261125 import UnifiedR1Mapper,UnifiedS2Mapper,unified_nodes,FRAME_SOURCE
from conceptgraph.scenegraph.relation_nav_unified_20261125 import CurrentRelationCache
from conceptgraph.scenegraph.relations import RelationNode
from conceptgraph.slam.streamlined_mapping import MappingFrame,StreamlinedMapper
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform,inverse_transform
from scripts.multiview_20261123_prepare_inputs import projection

DATA=source.DATA;REPORT=source.REPORT
RUN_ID='nuscenes_full_20261125_main_v1'
RUN=experiment_directory('nuscenes','validation',RUN_ID)
TMP=source.ROOT/'dataset/processed/nuscenes_full_20261125_tmp'
LOCK=Path('/root/autodl-tmp/lizichong/outputs/reports/shared_gpu_20261125.lock')
METHODS=['single_frame','original_fusion','R1','S2_1','S2_3','S2_6']
CHANNELS=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']


def clean(value):
    return json.loads(json.dumps(value,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))


class PixelEvidenceMixin:
    """Only enrich already selected real measurement provenance; no mask change."""
    def observation_policy(self,frame):
        original=super().observation_policy(frame)
        camera=next((c for c,f in getattr(self,'group_frames',[]) if f is frame),'CAM_FRONT')
        capture=float(getattr(self,'capture_times',{}).get(camera,frame.timestamp_s))
        def select(depth,mask,gobs,index):
            selected,evidence=original(depth,mask,gobs,index)
            raw_index=int(gobs.get('_audit_indices',np.arange(len(gobs['mask'])))[index])
            record=self._nm_raw.get((camera,capture,raw_index))
            if record is None:raise AssertionError('raw_pixel_source_key_missing')
            v,u=np.nonzero(selected & np.isfinite(depth) & (depth>1) & (depth<65))
            record['pixels_flat']=(v*depth.shape[1]+u).tolist()
            record['depth_mm']=np.rint(depth[v,u]*1000).astype(int).tolist()
            record['input_sensor_tokens']=self._full_context[camera]
            return selected,evidence
        return select


class FullR1(PixelEvidenceMixin,UnifiedR1Mapper):pass
class FullS2(PixelEvidenceMixin,UnifiedS2Mapper):pass


def configs():
    node=json.loads((source.ROOT/'dataset/processed/node_motion_unified_20261125_v2/manifest.json').read_text())
    release=Path(source.ROOT/'outputs/results/nuscenes/test/mapping/original_fusion').resolve().parents[1]/'release.json'
    previous=json.loads(release.read_text())
    assert previous['configs']['single_frame']==previous['configs']['original_fusion']
    return dict(original_fusion=previous['configs']['original_fusion'],R1=node['R1_config'],S2=node['S2_config']),release,node


def prepare():
    certification=json.loads((DATA/'source_certification.json').read_text());assert certification['complete'] and certification['fully_member_available_scenes']==850
    assert json.loads((REPORT/'CPU_contract_summary_v3.json').read_text())['passed']
    assert (DATA/'evaluation_protocol.json').exists()
    preseal=json.loads((DATA/'phaseB_source_preseal_v1.json').read_text())
    for p,h in preseal['source_sha256'].items():assert source.sha(p)==h,p
    cfg,old_release,node=configs();experiment_directory('nuscenes','validation',RUN_ID,create=True)
    files={Path(p) for p in node['source_sha256']}
    files.update([Path(__file__).resolve(),source.ROOT/'scripts/nuscenes_full_20261125_run.py',Path(source.__file__),
        source.ROOT/'conceptgraph/completion/vehicle_frontend.py',source.ROOT/'conceptgraph/scripts/streamlined_detections.py',
        source.ROOT/'conceptgraph/utils/model_utils.py',source.ROOT/'scripts/multiview_20261123_prepare_inputs.py',
        source.ROOT/'conceptgraph/scripts/preprocess_nuscenes_to_minimal.py',
        source.ROOT/'conceptgraph/scenegraph/relation_nav_unified_20261125.py',source.ROOT/'conceptgraph/scenegraph/relations.py',source.ROOT/'scripts/nuscenes_full_20261125_preflight_v3.py'])
    weights=[source.ROOT/'checkpoint/yolov8l-world.pt',source.ROOT/'checkpoint/mobile_sam.pt',
        source.ROOT/'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']
    frozen=dict(created_utc=source.now(),seed=17,configuration_count=1,parameter_search=False,configs=cfg,
        source_manifest_sha256=source.sha(DATA/'manifest.json'),source_certification_sha256=source.sha(DATA/'source_certification.json'),
        role_overlay_sha256=source.sha(DATA/'scene_role_access_overlay_v2.json'),source_sha256={str(p):source.sha(p) for p in sorted(files)},
        source_release_sha256={str(old_release):source.sha(old_release)},weights_sha256={str(p):source.sha(p) for p in weights},
        frontend=dict(prompts=PROMPTS,classes=CLASSES,detection_image_size=960,confidence=.1,train=False,
            model='YOLOv8l-world/MobileSAM/CLIP ViT-H-14',one_inference_per_unique_camera_view=True),
        methods=METHODS,single_frame='same original_fusion current extracted observations; frame-specific IDs; no accumulated history',
        original_fusion_clock='frozen legacy fusion configuration, true shared reference time supplied; no deliberate legacy timestamp bug',
        common_clock='max6 actual exposures for all methods; each view uses own strictpast<=100ms LiDAR',
        evaluation=dict(class_same=True,planar_XY_threshold_m=3.,one_to_one=True,scopes=['front','front3','union6'],
            visibility_minimum=2,center_camera_depth_m=[1,65],instance_deduplicated=True,GT_only_after_predictions=True,
            primary_algorithm_scope='front same inputs for first4 methods',camera_ablation='S2_1/3/6 use identical GT universe for each reported scope',
            near_m=6.,directional_m=12.,component_deadzone_m=1.),
        motion='legacy estimate and frozen verified raw state distinct; unsupported legacy raw identity history unknown',
        shape='unknown shapes remain unknown; all vehicle_safety_authorized false',
        evaluation_protocol_sha256=source.sha(DATA/'evaluation_protocol.json'),
        evaluation_source_sha256={str(source.ROOT/n):source.sha(source.ROOT/n) for n in ['conceptgraph/completion/nuscenes_full_20261125_evaluate.py','scripts/nuscenes_full_20261125_evaluate.py']},
        required_semantic_tasks=dict(existing1115='same new front-node outputs; re-evaluate all fixed tasks/denominators',
            existing1200='same new S2_3 scene/sample/front20 outputs; 141/999 and60pending kept',
            full850_new_tasks='outside primary mapping matrix; not fabricated parking/closed-loop evaluation'),
        GPU_lock=str(LOCK),GPU_start_requires='explicit KITTI release plus exclusive fcntl lock; completed source8scene resource pilot',
        budgets=dict(persistent_bytes=int(4.5*2**30),temp_bytes=512*2**20,GPU_wall_s=12*3600,full_wall_s=36*3600,min_free_bytes=20*2**30),
        preserved_partial_failures=True,cleanup='only this run-created TMP/scene, only after complete original member hash+all six outputs sealed; never old/raw/cache',
        raw_mask_storage='full masks/features only in memory; retain every used source measurement pixel/depth and original inference hash for CPU reprojection',
        GPU_zero_before_lock=True,GT_in_method=False)
    frozen['resource_pipeline']=dict(GPU_producers=1,CPU_spawn_workers=2,maximum_pending_scenes=3,aggregate_temporary_bytes=512*2**20,det_cache='temporary only until CPU scene sealed',failed_attempts='retained under scene/attempts; canonical links only after successful seal',GPU_budget_cumulative_including_resource_pilot=True,resource_regression_extra_GPU_calls=48)
    frozen['runtime_environment']=dict(Python='/root/miniconda3/envs/conceptgraph/bin/python',LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',PYTHONDONTWRITEBYTECODE='1')
    frozen['CPU_contract_sha256']=source.sha(REPORT/'CPU_contract_summary_v3.json')
    frozen['source_preseal_sha256']=source.sha(DATA/'phaseB_source_preseal_v1.json')
    source.dump(DATA/'method_manifest.json',frozen);source.dump(RUN/'manifest.json',frozen)
    for p in files:
        target=RUN/'source_snapshot'/p.relative_to(source.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ['conceptgraph/completion/nuscenes_full_20261125_evaluate.py','scripts/nuscenes_full_20261125_evaluate.py']:
        p=source.ROOT/name;q=RUN/'source_snapshot'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    print(json.dumps(dict(prepared=True,run=str(RUN),sources=len(files))),flush=True)


class SourceStore:
    def __init__(self):
        self.ig=source.indexed_module();self.archives={};self.handles={};self.by_scene={}
        for number in range(1,11):self.archives[number]=json.loads((DATA/'archives'/f'archive{number:02d}_complete.json').read_text())
        for r in source.rows(DATA/'member_binding.jsonl.gz'):self.by_scene.setdefault(r['scene'],{})[r['token']]=r

    def extract(self,request,attempt='initial'):
        folder=TMP/(request['scene']+'__'+attempt);folder.mkdir(parents=True,exist_ok=False)
        bindings=self.by_scene.get(request['scene'],{});total=sum(v['bytes'] for v in bindings.values())
        if total>512*2**20:raise OSError('scene temp size exceeds512MiB; source request retained')
        receipts={}
        for token,b in sorted(bindings.items(),key=lambda p:(p[1]['archive'],p[1]['offset'])):
            number=b['archive'];a=self.archives[number]
            if number not in self.handles:
                assert Path(a['archive']).stat().st_size==a['archive_bytes'] and Path(a['archive']).stat().st_mtime_ns==a['archive_mtime_ns']
                self.handles[number]=self.ig.IndexedGzipFile(filename=a['archive'],index_file=a['seek_index'],drop_handles=False)
            handle=self.handles[number];handle.seek(b['offset']);raw=handle.read(b['bytes']);assert len(raw)==b['bytes']
            target=folder/b['filename'];assert folder.resolve() in target.resolve().parents
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
            fingerprint=hashlib.sha256(raw).hexdigest();receipts[token]=dict(**b,sha256=fingerprint,path=str(target),archive_sha256=a['archive_sha256'])
        return folder,receipts


def geometry_tables(manifest):
    wanted=set()
    for scene in manifest['scenes']:
        req=next(source.rows(scene['request']));wanted.update(s['ego_pose_token'] for s in req['sensors'].values())
    poses={r['token']:r for r in source.iter_records(source.META/'ego_pose.json') if r['token'] in wanted}
    calibrations={r['token']:r for r in source.iter_records(source.META/'calibrated_sensor.json')}
    return poses,calibrations


def sensor_transform(sensor,poses,calibrations):
    pose=poses[sensor['ego_pose_token']];cal=calibrations[sensor['calibrated_sensor_token']]
    return build_transform(pose['translation'],pose['rotation'])@build_transform(cal['translation'],cal['rotation'])


def node_record(method,mapper,predictions,scene,index,stamp,pose,registry,cache):
    if method in ['R1','S2_1','S2_3','S2_6']:
        states=mapper.evidence_states();nodes=unified_nodes(predictions,states,mapper.resolve_evidence_id,stamp)
        compact_states=[{**{k:v for k,v in s.items() if k!='raw_trace'},'raw_trace_keys':[r['key'] for r in s['raw_trace']]} for s in states]
        raw=list(mapper._nm_raw.values());events=mapper._nm_events
    else:
        compact_states=[];raw=[];events=[];nodes=[]
        for i,p in enumerate(predictions):
            extent=p.get('extent');shape=extent is not None and np.isfinite(extent).all() and np.all(np.asarray(extent)>0)
            nodes.append(dict(original_index=i,node_id=p['node_id'],category=p['category'],center_world_m=p['center'],extent_world_aabb_m=extent if shape else None,
                shape_available=bool(shape),motion_state='unknown',motion_speed_xy_mps=None,motion_reference_frame='world_xy',motion_heading_world_rad=None,
                body_heading_world_rad=None,body_heading_source='unknown',motion_time=dict(render_timestamp_s=stamp,current_update=p.get('last_seen_frame')==index),
                measurement=dict(unknown_reasons=['legacy_baseline_has_no_audited_multiframe_identity_trace'],current_points=None,current_support_fraction=None,source_closed=False,observation_keys=[]),
                uncertainty=dict(diag_m2=None,kind='unavailable'),semantic_position_candidate=False,vehicle_safety_authorized=False,
                legacy_motion_state=p.get('motion_state','unknown'),legacy_velocity_mps=p.get('velocity_mps'),motion_history_cross_id_merged=False))
    relation_nodes={}
    for p,node in zip(predictions,nodes):
        node.update(estimated_motion_state=node['legacy_motion_state'],estimated_motion_state_source='frozen_original_method_estimate',
            verified_motion_state=node['motion_state'],verified_motion_state_source='frozen_causal_raw_confirmation' if method in ['R1','S2_1','S2_3','S2_6'] else 'unsupported_no_audited_history',
            body_heading_available=False)
        registry.setdefault(p['node_id'],len(registry));local=registry[p['node_id']]
        relation_nodes[local]=RelationNode(local,p['center'],dynamic=node['motion_state']=='dynamic',observation_count=p['num_detections'],
            last_seen_frame=p.get('last_seen_frame'),metadata=dict(timestamp_s=p.get('timestamp_s'),motion_state=node['motion_state'],shape_available=node['shape_available'],source_node_id=p['node_id']))
    edges=cache.update_snapshot(relation_nodes,frame_index=index,heading=pose[:2,2],timestamp_s=stamp).edges
    return dict(predictions=predictions,nodes=nodes,states=compact_states,raw_observations=raw,lineage_events=events,
        edges=[e.to_dict() for e in edges],relation_node_registry={str(v):k for k,v in registry.items()},
        vehicle_safety_authorized=False,body_heading_available=False)


def footprint():
    return sum(p.stat().st_size for folder in [DATA,RUN,REPORT] if folder.exists() for p in folder.rglob('*') if p.is_file() and not p.is_symlink())


def snapshot_size(folder):
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())


def prediction_digest(det):
    digest=hashlib.sha256()
    for key in ['xyxy','confidence','class_id','mask','image_feats','text_feats']:
        value=np.ascontiguousarray(det[key]);digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def gpu_scene(models,request,temp,*,regression=False):
    import torch
    from conceptgraph.completion.vehicle_frontend import detect_vehicle_frame
    saved={};counts=Counter();tick=time.monotonic();regressions=[]
    detection_dir=temp/'_this_run_detections';detection_dir.mkdir()
    for group in request['groups']:
        for camera in CHANNELS:
            token=group['views'][camera]['camera'];cam=request['sensors'][token];rgbpath=temp/cam['filename']
            assert rgbpath.exists(),'certified RGB member extraction missing'
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():det=detect_vehicle_frame(models,rgbpath,image_size=960)
            torch.cuda.synchronize();elapsed=time.monotonic()-start;det.pop('image_crops',None)
            digest=prediction_digest(det)
            # Resource-only paired check of explicit synchronization. Same model,
            # same actual RGB, exact boxes/masks/features; never reads annotations.
            if regression and group is request['groups'][0]:
                with torch.inference_mode():paired=detect_vehicle_frame(models,rgbpath,image_size=960)
                torch.cuda.synchronize();paired.pop('image_crops',None)
                assert prediction_digest(paired)==digest,'synchronization output regression failed'
                regressions.append(dict(camera=camera,token=token,prediction_sha256=digest,exact=True))
            target=detection_dir/(token+'.pkl.gz')
            with gzip.open(target,'xb',compresslevel=1) as f:pickle.dump(det,f,protocol=5)
            saved[token]=dict(path=str(target),sha256=source.sha(target),prediction_sha256=digest,model_s=elapsed,
                detections=len(det['xyxy']),boxes=det['xyxy'].tolist(),confidence=det['confidence'].tolist(),class_id=det['class_id'].tolist(),
                mask_sha256=hashlib.sha256(np.packbits(det['mask']).tobytes()).hexdigest(),features_sha256=hashlib.sha256(det['image_feats'].tobytes()).hexdigest())
            counts['actual_GPU_image_calls']+=1;counts['model_microseconds']+=int(elapsed*1e6)
            counts['resource_regression_extra_GPU_calls']+=int(regression and group is request['groups'][0])
            if snapshot_size(TMP)>512*2**20:raise OSError('aggregate temporary guard during GPU scene; prefix retained')
    return dict(detections=saved,counts=dict(counts),GPU_frontend_wall_s=time.monotonic()-tick,regressions=regressions,
        temporary_detection_bytes=snapshot_size(detection_dir),GT_in_method=False)


def cpu_scene(scene,request,temp_path,receipts,geometry_path,gpu_result,source_io_s,output_path):
    import torch
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.manual_seed(17);random.seed(17);np.random.seed(17)
    frozen=json.loads((DATA/'method_manifest.json').read_text());cfg=frozen['configs'];temp=Path(temp_path);folder=Path(output_path);started=time.monotonic()
    geom=json.loads(Path(geometry_path).read_text());poses,calibrations=geom['poses'],geom['calibrations']
    old=StreamlinedMapper(OmegaConf.create(cfg['original_fusion']),CLASSES)
    mappers={'R1':FullR1(OmegaConf.create(cfg['R1']),CLASSES)}
    mappers.update({k:FullS2(OmegaConf.create(cfg['S2']),CLASSES) for k in ['S2_1','S2_3','S2_6']})
    registry={k:{} for k in METHODS};caches={k:CurrentRelationCache() for k in METHODS};counts=Counter(gpu_result['counts']);seconds=Counter()
    with gzip.open(folder/'frames.jsonl.gz','xt',compresslevel=6) as output:
        for group in request['groups']:
            frames={};projections={};views={};frontends=[];contexts={};start=time.monotonic()
            for camera in CHANNELS:
                v=group['views'][camera];cam=request['sensors'][v['camera']];pose=sensor_transform(cam,poses,calibrations)
                intrinsic=np.asarray(calibrations[cam['calibrated_sensor_token']]['camera_intrinsic']);rgbpath=temp/cam['filename'];width,height=cam['width'],cam['height']
                detection=gpu_result['detections'][v['camera']];assert source.sha(detection['path'])==detection['sha256']
                with gzip.open(detection['path'],'rb') as f:det=pickle.load(f)
                assert prediction_digest(det)==detection['prediction_sha256']
                image=np.asarray(Image.open(rgbpath).convert('RGB'))
                lidar=request['sensors'].get(v['lidar']);lidarpath=temp/lidar['filename'] if lidar else None
                if lidarpath and lidarpath.exists():
                    points=np.fromfile(lidarpath,np.float32).reshape(-1,5);lw=sensor_transform(lidar,poses,calibrations)
                    depth,pixels=projection(points,inverse_transform(pose)@lw,intrinsic,height,width)
                else:
                    depth=np.zeros((height,width),np.uint16);pixels=dict(pixel_y=np.empty(0,int),pixel_x=np.empty(0,int),point_index=np.empty(0,int),depth_m=np.empty(0,np.float32));counts['empty_depth_views']+=1
                pixels['lidar_token']=v['lidar'];projections[camera]=pixels
                contexts[camera]=dict(camera_token=v['camera'],lidar_token=v['lidar'],rgb_sha256=receipts.get(v['camera'],{}).get('sha256'),lidar_sha256=receipts.get(v['lidar'],{}).get('sha256'))
                views[camera]=dict(camera_pose=pose.tolist(),intrinsic=intrinsic.tolist(),width=width,height=height,camera_timestamp_s=cam['timestamp_us']/1e6,
                    lidar_timestamp_s=lidar['timestamp_us']/1e6 if lidar else None,**contexts[camera])
                frames[camera]=MappingFrame(group['frame_idx'],image,depth.astype(float)/1000.,intrinsic,pose,det,str(rgbpath),timestamp_s=group['reference_timestamp_us']/1e6)
                frontends.append(dict(camera=camera,**{k:v for k,v in detection.items() if k not in ['path','sha256']},**contexts[camera]))
            seconds['cached_detection_load_and_projection']+=time.monotonic()-start
            index=group['frame_idx'];stamp=group['reference_timestamp_us']/1e6;front=frames['CAM_FRONT'];results={};timing={}
            np.random.seed(17+index);t=time.monotonic();old.update(copy.deepcopy(front));timing['original_fusion']=time.monotonic()-t
            preds={'original_fusion':predict(old.objects,CLASSES,pose=front.camera_pose,intrinsic=front.cam_K),
                'single_frame':predict(old.last_observations,CLASSES,pose=front.camera_pose,intrinsic=front.cam_K)}
            for p in preds['single_frame']:p['node_id']=f"frame{index}:{p['node_id']}"
            for name,mapper in mappers.items():
                mapper.begin_evidence(scene['scene'],index);mapper._full_context=contexts;np.random.seed(17+index);t=time.monotonic()
                if name=='R1':
                    mapper.capture_times={'CAM_FRONT':views['CAM_FRONT']['camera_timestamp_s']};mapper.update(copy.deepcopy(front))
                    predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),front)
                else:
                    cams=CHANNELS[:int(name.split('_')[-1])]
                    predictions=mapper.update_group([(c,copy.deepcopy(frames[c])) for c in cams],{c:views[c]['camera_timestamp_s'] for c in cams},{c:projections[c] for c in cams})
                preds[name]=predictions;timing[name]=time.monotonic()-t
            start=time.monotonic()
            for name in METHODS:results[name]=node_record(name,old if name in ['single_frame','original_fusion'] else mappers[name],preds[name],scene['scene'],index,stamp,front.camera_pose,registry[name],caches[name])
            seconds['node_schema_and_relations']+=time.monotonic()-start
            for name,value in timing.items():seconds[name]+=value
            record=dict(scene=scene['scene'],log=scene['log_token'],frame_idx=index,sample_token=group['sample_token'],render_timestamp_s=stamp,
                official_sample_timestamp_s=group['official_sample_timestamp_us']/1e6,views=views,frame_source=FRAME_SOURCE,frontends=frontends,
                methods=results,mapper_seconds=timing,GT_in_method=False,vehicle_safety_authorized=False)
            output.write(json.dumps(clean(record),separators=(',',':'))+'\n');counts['frames']+=1
            for name in METHODS:counts[name+'_predictions']+=len(preds[name])
    assert counts['frames']==scene['frames']
    for r in receipts.values():assert source.sha(r['path'])==r['sha256']
    complete=dict(complete=True,scene=scene['scene'],log=scene['log_token'],counts=dict(counts),frames_sha256=source.sha(folder/'frames.jsonl.gz'),
        source_members_sha256=source.sha(folder/'source_members.json'),request_sha256=scene['request_sha256'],CPU_scene_wall_s=time.monotonic()-started,
        stage_seconds=dict(seconds),source_seek_extract_s=source_io_s,GPU_frontend_wall_s=gpu_result['GPU_frontend_wall_s'],
        model_s=counts['model_microseconds']/1e6,ended_utc=source.now(),original_role=scene['role'],GT_in_method=False,vehicle_safety_authorized=False,
        source_prediction_regressions=gpu_result['regressions'],temporary_cleanup_path=str(temp),temporary_detection_bytes=gpu_result['temporary_detection_bytes'])
    assert not torch.cuda.is_initialized(),'CPU mapping worker initialized CUDA'
    assert all(str(m.cfg.device)=='cpu' for m in [old,*mappers.values()])
    complete['CPU_worker_CUDA_initialized']=False
    source.dump(folder/'complete.json',complete)
    assert temp.parent.resolve()==TMP.resolve();shutil.rmtree(temp)
    return complete


def run(args):
    import torch
    from conceptgraph.completion.vehicle_frontend import configure_vehicle_prompts
    from conceptgraph.scripts.streamlined_detections import load_detection_models
    frozen=json.loads((DATA/'method_manifest.json').read_text());manifest=json.loads((DATA/'manifest.json').read_text())
    assert json.loads((REPORT/'resource_pilot_summary.json').read_text())['passed']
    if not args.resource_pilot_only:assert json.loads((REPORT/'GPU_resource_pilot_summary.json').read_text())['passed']
    for p,h in frozen['source_sha256'].items():assert source.sha(p)==h,p
    for p,h in frozen['weights_sha256'].items():assert source.sha(p)==h,p
    wall_start=time.monotonic();TMP.mkdir(exist_ok=True);scenes=manifest['scenes']
    assert os.environ.get('LD_LIBRARY_PATH','').split(':')[0]=='/root/miniconda3/envs/conceptgraph/lib'
    pilot_names=[s['scene'] for s in json.loads((DATA/'resource_pilot_preregistration.json').read_text())['scenes']]
    if args.resource_pilot_only:scenes=[s for s in scenes if s['scene'] in pilot_names]
    completed={};failed={};pending=[];peak_temp=0
    attempts=RUN/'runtime_attempts';attempts.mkdir(exist_ok=True)
    previous=list(attempts.iterdir());prior_GPU_s=0.
    for prior in previous:
        release=prior/'GPU_release.json'
        if not (prior/'GPU_acquired.json').exists():continue
        if not release.exists():raise RuntimeError('prior GPU attempt lacks closed occupancy receipt; preserve and audit before resume')
        prior_GPU_s+=json.loads(release.read_text())['GPU_occupied_wall_s']
    attempt_id=f'run{len(previous)+1:04d}';attempt=attempts/attempt_id;attempt.mkdir()
    Path(args.kitti_release_reference).read_text()
    source.dump(attempt/'preregistration.json',dict(created_utc=source.now(),source_sha256=source.sha(__file__),pilot=args.resource_pilot_only,prior_GPU_occupied_s=prior_GPU_s,expected_scenes=[s['scene'] for s in scenes]))
    with LOCK.open('a+') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_start=time.monotonic()
        source.dump(attempt/'GPU_acquired.json',dict(acquired_utc=source.now(),lock=str(LOCK),explicit_release_reference=args.kitti_release_reference,release_notice_sha256=source.sha(args.kitti_release_reference),PID=os.getpid(),GPU='cuda:0'))
        def occupancy_receipt():
            path=attempt/'GPU_release.json'
            if not path.exists():source.dump(path,dict(released_utc=source.now(),GPU_occupied_wall_s=time.monotonic()-gpu_start,PID=os.getpid(),prior_GPU_occupied_s=prior_GPU_s))
        atexit.register(occupancy_receipt)
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(4)
        models=load_detection_models(type('Cfg',(),dict(device='cuda:0'))());configure_vehicle_prompts(models)
        for model in models[:3]:
            if hasattr(model,'eval'):model.eval()
        store=SourceStore();poses,calibrations=geometry_tables(manifest)
        def finish(wait=False):
            nonlocal pending
            for future,name,output_path in list(pending):
                if wait or future.done():
                    try:
                        record=future.result();completed[name]=record
                        scene_root=RUN/name
                        for filename in ['frames.jsonl.gz','source_members.json','frontend_receipt.json']:
                            (scene_root/filename).symlink_to((Path(output_path)/filename).relative_to(scene_root))
                        source.dump(scene_root/'complete.json',dict(record,accepted_attempt=str(output_path)))
                    except Exception as error:
                        failed[name]=dict(scene=name,error_type=type(error).__name__,error=str(error),preserved_attempt=str(output_path),full_denominator_retained=True)
                        source.dump(Path(output_path)/'failed.json',dict(**failed[name],utc=source.now()))
                        print(json.dumps(dict(technical_failure=failed[name])),flush=True)
                    pending.remove((future,name,output_path))
                    total=Counter()
                    for result in completed.values():total.update(result['counts'])
                    progress=dict(utc=source.now(),scene=name,completed_scenes=len(completed),failed_scenes=failed,expected_scenes=len(scenes),counts=dict(total),persistent_bytes=footprint(),
                        GPU_occupied_wall_s=time.monotonic()-gpu_start,wall_s=time.monotonic()-wall_start,peak_total_temp_bytes=peak_temp)
                    if len(completed)%100==0:
                        progress['resource_forecast']=dict(completed_scenes=len(completed),projected_GPU_occupied_hours=(time.monotonic()-gpu_start)/max(total['actual_GPU_image_calls'],1)*204894/3600,projected_scene_output_bytes=sum(snapshot_size(RUN/name) for name in completed)/max(len(completed),1)*850,fixed_metadata_and_source_bytes=footprint()-sum(snapshot_size(RUN/name) for name in completed),method_parameters_unchanged=True)
                    source.dump(REPORT/'main_progress.json',progress,False);print(json.dumps(progress),flush=True)
                    if len(failed)>=3:raise RuntimeError('3 technical scene failures; preserve all850 denominator and prefixes for repair')
                    if wait:break
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            for ordinal,s in enumerate(scenes):
                finish()
                while len(pending)>=3:finish(wait=True)
                folder=RUN/s['scene'];completion=folder/'complete.json'
                if completion.exists():
                    record=json.loads(completion.read_text());assert source.sha(folder/'frames.jsonl.gz')==record['frames_sha256'];completed[s['scene']]=record;continue
                folder.mkdir(exist_ok=True);output_path=folder/'attempts'/attempt_id;output_path.mkdir(parents=True)
                if shutil.disk_usage(RUN.resolve()).free<20*2**30:raise OSError('disk20GiB guard before scene')
                if footprint()>int(4.5*2**30):raise OSError('nu4.5GiB persistent cap; prefixes retained')
                if prior_GPU_s+time.monotonic()-gpu_start>12*3600 or time.monotonic()-wall_start>36*3600:raise TimeoutError('full/GPU occupied wall budget exhausted')
                req=next(source.rows(s['request']))
                expected_source_bytes=sum(v['bytes'] for v in store.by_scene[s['scene']].values())
                while pending and snapshot_size(TMP)+expected_source_bytes+32*2**20>512*2**20:finish(wait=True)
                if snapshot_size(TMP)+expected_source_bytes>512*2**20:raise OSError('retained temporary prefixes exhaust512MiB; preserve for repair')
                tick=time.monotonic();temp,receipts=store.extract(req,attempt_id);source_io=time.monotonic()-tick
                source.dump(output_path/'source_members.json',dict(request_sha256=s['request_sha256'],members=receipts,GT_in_method=False))
                needed={r['ego_pose_token'] for r in req['sensors'].values()};cals={r['calibrated_sensor_token'] for r in req['sensors'].values()}
                geom=temp/'_this_run_geometry.json';source.dump(geom,dict(poses={t:poses[t] for t in needed},calibrations={t:calibrations[t] for t in cals}))
                gpu_result=gpu_scene(models,req,temp,regression=args.resource_pilot_only)
                current_temp=snapshot_size(TMP);peak_temp=max(peak_temp,current_temp)
                if current_temp>512*2**20:raise OSError('aggregate active scene temporary exceeds512MiB')
                source.dump(output_path/'frontend_receipt.json',dict(**{k:v for k,v in gpu_result.items() if k!='detections'},
                    detections={t:{k:v for k,v in d.items() if k!='path'} for t,d in gpu_result['detections'].items()},GT_in_method=False))
                pending.append((pool.submit(cpu_scene,s,req,str(temp),receipts,str(geom),gpu_result,source_io,str(output_path)),s['scene'],str(output_path)))
            while pending:finish(wait=True)
        occupied=time.monotonic()-gpu_start;occupancy_receipt();atexit.unregister(occupancy_receipt);fcntl.flock(gpu_lock,fcntl.LOCK_UN)
    total=Counter()
    for record in completed.values():total.update(record['counts'])
    result=dict(complete=len(completed)==len(scenes),failed_scenes=failed,attempt=attempt_id,prior_GPU_occupied_s=prior_GPU_s,total_GPU_occupied_s=prior_GPU_s+occupied,scenes=len(completed),expected_scenes=len(scenes),counts=dict(total),GPU_occupied_wall_s=occupied,
        elapsed_s=time.monotonic()-wall_start,peak_total_temp_bytes=peak_temp,source_manifest_sha256=frozen['source_manifest_sha256'],GT_in_method=False)
    if args.resource_pilot_only:
        result.update(passed=len(completed)==8 and not failed,source_prediction_regression_pairs=sum(len(r['source_prediction_regressions']) for r in completed.values()),
            projected_full_GPU_occupied_hours=occupied/sum(r['counts']['actual_GPU_image_calls'] for r in completed.values())*204894/3600,
            stage_totals_s=dict(source_seek_extract=sum(r['source_seek_extract_s'] for r in completed.values()),
                GPU_model=sum(r['model_s'] for r in completed.values()),CPU_mapper=sum(sum(r['stage_seconds'].values()) for r in completed.values())))
        source.dump(REPORT/'GPU_resource_pilot_summary.json',result)
    else:source.dump(RUN/'inference_complete.json' if result['complete'] else attempt/'inference_partial.json',result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');r=sub.add_parser('run');r.add_argument('--kitti-release-reference',required=True);r.add_argument('--resource-pilot-only',action='store_true')
    args=p.parse_args();prepare() if args.command=='prepare' else run(args)

if __name__=='__main__':main()
