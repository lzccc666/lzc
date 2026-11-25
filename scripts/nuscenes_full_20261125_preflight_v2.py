"""纯CPU实际缓存接口回归，确认像素来源旁路不改冻结预测；不评价GT。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,gzip,json,time
import numpy as np
import torch
from omegaconf import OmegaConf
from conceptgraph.completion import nuscenes_full_20261125_run as main
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.scenegraph.node_motion_unified_20261125 import UnifiedR1Mapper,UnifiedS2Mapper
from conceptgraph.scenegraph.relation_nav_unified_20261125 import CurrentRelationCache
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper
from conceptgraph.completion.stage9 import predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from scripts.multiview_20261123_run_graph import load_frame


def execute():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    path=source.ROOT/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json'
    manifest=json.loads(path.read_text());scene=manifest['scenes'][0];groups=scene['groups'][:20];cfg,_,_=main.configs()
    source.dump(source.DATA/'cpu_contract_preregistration_v2.json',dict(created_utc=source.now(),source_sha256=source.sha(__file__),main_draft_sha256=source.sha(main.__file__),
        input_manifest_sha256=source.sha(path),scene=scene['scene'],frame_indices=[g['frame_idx'] for g in groups],selection='first scene first20 fixed existing6view causal frames; no labels opened',
        purpose='raw pixel provenance enrichment exact-prediction CPU interface regression',GPU_calls=0,GT_in_method=False))
    old=StreamlinedMapper(OmegaConf.create(cfg['original_fusion']),CLASSES)
    mappers={n:c(OmegaConf.create(cfg['R1' if n=='R1' else 'S2']),CLASSES) for n,c in [('R1',main.FullR1),('S2_1',main.FullS2),('S2_3',main.FullS2),('S2_6',main.FullS2)]}
    refs={n:c(OmegaConf.create(cfg['R1' if n=='R1' else 'S2']),CLASSES) for n,c in [('R1',UnifiedR1Mapper),('S2_1',UnifiedS2Mapper),('S2_3',UnifiedS2Mapper),('S2_6',UnifiedS2Mapper)]}
    registries={n:{} for n in main.METHODS};caches={n:CurrentRelationCache() for n in main.METHODS};started=time.monotonic();records=[]
    for group in groups:
        frames={c:load_frame(group,c) for c in main.CHANNELS};projections={}
        contexts={c:dict(camera_token=group['views'][c]['camera_token'],lidar_token=group['views'][c]['lidar_token']) for c in main.CHANNELS}
        for c in main.CHANNELS:
            with np.load(group['views'][c]['projection_evidence']) as f:projections[c]={k:f[k].copy() for k in f.files}
            projections[c]['lidar_token']=group['views'][c]['lidar_token']
        index=group['frame_idx'];stamp=group['reference_timestamp_s'];front=frames['CAM_FRONT'];preds={};checks=[]
        np.random.seed(17+index);old.update(copy.deepcopy(front))
        preds['original_fusion']=predict(old.objects,CLASSES,pose=front.camera_pose,intrinsic=front.cam_K)
        preds['single_frame']=predict(old.last_observations,CLASSES,pose=front.camera_pose,intrinsic=front.cam_K)
        for p in preds['single_frame']:p['node_id']=f"frame{index}:{p['node_id']}"
        for name,mapper in mappers.items():
            results=[]
            for current in [mapper,refs[name]]:
                current.begin_evidence(scene['scene'],index);current._full_context=contexts;np.random.seed(17+index)
                if name=='R1':
                    current.capture_times={'CAM_FRONT':group['views']['CAM_FRONT']['camera_timestamp_s']};current.update(copy.deepcopy(front))
                    result=current.position_predictions(predict(current.confirmed_objects(),CLASSES),front)
                else:
                    cams=main.CHANNELS[:int(name.split('_')[-1])]
                    result=current.update_group([(c,copy.deepcopy(frames[c])) for c in cams],{c:group['views'][c]['camera_timestamp_s'] for c in cams},{c:projections[c] for c in cams})
                results.append(main.clean(result))
            assert results[0]==results[1],(name,index,'pixel_provenance_changed_original_prediction')
            preds[name]=results[0];checks.append(dict(method=name,exact=True,predictions=len(results[0]),raw_observations=len(mapper._nm_raw)))
        outputs={n:main.node_record(n,old if n in ['single_frame','original_fusion'] else mappers[n],preds[n],scene['scene'],index,stamp,front.camera_pose,registries[n],caches[n]) for n in main.METHODS}
        records.append(dict(frame_idx=index,checks=checks,methods=outputs))
        print(json.dumps(dict(frame=index,checks=checks,elapsed_s=time.monotonic()-started)),flush=True)
    path=source.REPORT/'CPU_contract_frames_v2.jsonl.gz'
    with gzip.open(path,'xt',compresslevel=6) as f:
        for r in records:f.write(json.dumps(main.clean(r),separators=(',',':'))+'\n')
    source.dump(source.REPORT/'CPU_contract_summary_v2.json',dict(passed=True,frames=len(groups),scene=scene['scene'],four_causal_mapper_exact_comparisons=80,
        all6condition_schema_exercised=True,nonempty_confirmed_node_frames=sum(any(r['methods'][n]['nodes'] for n in main.METHODS) for r in records),source_sha256=source.sha(__file__),main_draft_sha256=source.sha(main.__file__),frames_sha256=source.sha(path),elapsed_s=time.monotonic()-started,GPU_calls=0,GT_in_method=False))

if __name__=='__main__':execute()
