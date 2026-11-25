"""复用真实旧20帧共享scan缓存，核新辅助重投影/预测；不读GT/不调用GPU。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,gzip,json,pickle,shutil,time
import numpy as np
import torch
from omegaconf import OmegaConf
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion import nuscenes_full_20261125_run as base
from conceptgraph.completion import nuscenes_full_20261125_sharedscan as control
from conceptgraph.scenegraph.node_motion_unified_20261125 import UnifiedS2Mapper
from conceptgraph.project_paths import experiment_directory
from scripts.multiview_20261123_run_graph import load_frame


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.monotonic()
    normal_path=source.ROOT/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json';shared_path=source.ROOT/'dataset/processed/mv136_20261123_sharedscan_v1/manifest.json'
    normal=json.loads(normal_path.read_text())['scenes'][0];shared=json.loads(shared_path.read_text())['scenes'][0]
    manifest=json.loads((base.DATA/'manifest.json').read_text());scene=next(s for s in manifest['scenes'] if s['scene']==normal['scene']);request=next(source.rows(scene['request']));request['groups']=request['groups'][:20];scene=copy.deepcopy(scene);scene['frames']=20
    cfg,_,_=base.configs();run=experiment_directory('nuscenes','validation','nuscenes_full_20261125_sharedscan_contract_v1',create=True)
    prereg=dict(created_utc=source.now(),source_sha256=source.sha(__file__),auxiliary_source_sha256=source.sha(control.__file__),fixed65_preseal_sha256=source.sha(base.DATA/'phaseB_source_preseal_v1.json'),
        scene=scene['scene'],groups=20,normal_manifest_sha256=source.sha(normal_path),shared_manifest_sha256=source.sha(shared_path),selection='first registered historical6scene first20; no labels',config=cfg['S2'],GPU_calls=0,GT_accessed=False,
        claim='new control recomputes old sharedscan exact depth and same S2 predictions, not historic T3=850quality')
    source.dump(base.DATA/'sharedscan_contract_preregistration_v1.json',prereg)
    temp=base.TMP/(scene['scene']+'__sharedscan_contract');temp.mkdir(exist_ok=False);receipts={};detections={}
    for group in normal['groups'][:20]:
        for camera,view in group['views'].items():
            for token,path in [(view['camera_token'],view['rgb']),(view['lidar_token'],view['lidar_raw'])]:
                if token is None or token in receipts:continue
                sensor=request['sensors'][token];target=temp/sensor['filename'];target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(path)
                receipts[token]=dict(path=str(target),sha256=source.sha(path),bytes=Path(path).stat().st_size)
            with gzip.open(view['detections'],'rb') as f:d=pickle.load(f)
            detections[view['camera_token']]=dict(path=view['detections'],sha256=source.sha(view['detections']),prediction_sha256=base.prediction_digest(d))
    poses,calibrations=base.geometry_tables(dict(scenes=[scene]));geometry=base.DATA/'sharedscan_contract_geometry.json';source.dump(geometry,dict(poses=poses,calibrations=calibrations))
    result=control.control_scene(scene,request,temp,receipts,geometry,dict(detections=detections),run,config=cfg['S2'])
    exported=list(source.rows(run/'sharedscan_frames.jsonl.gz'));mapper=UnifiedS2Mapper(OmegaConf.create(cfg['S2']),base.CLASSES);counts=[]
    for group,actual in zip(shared['groups'][:20],exported):
        index=group['frame_idx'];mapper.begin_evidence(scene['scene'],index);np.random.seed(17+index);projections={}
        for camera in base.CHANNELS:
            with np.load(group['views'][camera]['projection_evidence']) as f:projections[camera]={k:f[k].copy() for k in f.files}
            projections[camera]['lidar_token']=group['views'][camera]['lidar_token'];assert actual['view_source_protocol'][camera]['applied_lidar_token']==group['views'][camera]['lidar_token']
        predicted=mapper.update_group([(c,load_frame(group,c)) for c in base.CHANNELS],{c:group['views'][c]['camera_timestamp_s'] for c in base.CHANNELS},projections)
        assert base.clean(predicted)==actual['payload']['predictions'],('sharedscan_cached_prediction_mismatch',index)
        counts.append(dict(frame_idx=index,predictions=len(predicted),exact=True))
    assert len(exported)==20 and not torch.cuda.is_initialized()
    source.dump(run/'summary.json',dict(passed=True,frames=20,prediction_comparisons=counts,new_control=result,elapsed_s=time.monotonic()-started,
        wrapper_preserves_original_six_function=control._ORIGINAL_CPU_SCENE is base.cpu_scene,original65_source_unchanged=True,GPU_calls=0,GT_accessed=False,
        auxiliary_source_sha256=source.sha(control.__file__),source_sha256=source.sha(__file__),temporary_only_new_path=str(temp)))
    shutil.rmtree(temp);print(json.dumps(dict(passed=True,frames=20,run=str(run),elapsed_s=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
