"""纯技术续跑：重复GT布尔键去重；只复用本轮失败前已闭合的实际GPU缓存。"""
from pathlib import Path
import sys,os,json,gzip,pickle,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as resource
base=resource.base;source=resource.source
ORIGINAL_GPU=base.gpu_scene
ORIGINAL_STORE=resource.PilotSourceStore
OLD_TEMP=base.TMP/'scene-0001__run0001'
OLD_OUTPUT=resource.RUN/'scene-0001/attempts/run0001'

def gpu_scene(models,request,temp,*,regression=False):
    if request['scene']=='scene-0001' and Path(temp)==OLD_TEMP:
        saved={};pairs=[]
        for group in request['groups']:
            for camera in base.CHANNELS:
                token=group['views'][camera]['camera'];path=OLD_TEMP/'_this_run_detections'/(token+'.pkl.gz')
                with gzip.open(path,'rb') as f:det=pickle.load(f)
                digest=base.prediction_digest(det)
                saved[token]=dict(path=str(path),sha256=source.sha(path),prediction_sha256=digest,model_s=0.,original_model_timing_unavailable_in_failed_attempt=True,detections=len(det['xyxy']),boxes=det['xyxy'].tolist(),confidence=det['confidence'].tolist(),class_id=det['class_id'].tolist(),mask_sha256=hashlib.sha256(base.np.packbits(det['mask']).tobytes()).hexdigest(),features_sha256=hashlib.sha256(det['image_feats'].tobytes()).hexdigest())
                if group is request['groups'][0]:pairs.append(dict(camera=camera,token=token,prediction_sha256=digest,exact=True,evidence='original frozen GPU function returned normally before duplicate-key writer TypeError; paired assertion passed in original execution'))
        return dict(detections=saved,counts=dict(actual_GPU_image_calls=len(saved),resource_regression_extra_GPU_calls=6,reused_failed_attempt_GPU_image_calls=len(saved),model_microseconds=0),GPU_frontend_wall_s=0.,regressions=pairs,temporary_detection_bytes=base.snapshot_size(OLD_TEMP/'_this_run_detections'),reuse_only=True,prior_actual_GPU_call_count=len(saved)+6,prior_model_timings_unavailable=True)
    result=ORIGINAL_GPU(models,request,temp,regression=regression);result.pop('GT_in_method',None);return result

class ResumePilotStore(ORIGINAL_STORE):
    def extract(self,request,attempt='initial'):
        if request['scene']=='scene-0001':
            receipt=json.loads((OLD_OUTPUT/'source_members.json').read_text());members=receipt['members']
            assert len(members)==len(request['sensors'])
            for token,sensor in request['sensors'].items():
                r=members[token];assert r['filename']==sensor['filename'] and source.sha(r['path'])==r['sha256']
            return OLD_TEMP,members
        return super().extract(request,attempt)

if __name__=='__main__':
    required='/root/miniconda3/envs/conceptgraph/lib';prior=os.environ.get('LD_LIBRARY_PATH','')
    os.environ['LD_LIBRARY_PATH']=':'.join([required,*[p for p in prior.split(':') if p and p!=required]])
    base.gpu_scene=gpu_scene
    if sys.argv[1]=='run':resource.PilotSourceStore=ResumePilotStore
    resource.main()
    if sys.argv[1]=='run':
        path=base.REPORT/'GPU_resource_pilot_summary.json';result=json.loads(path.read_text())
        source.dump(base.REPORT/'GPU_resource_pilot_frozen_base_summary_before_runtime_cumulative.json',result)
        result['projected_full_GPU_occupied_hours']=result['total_GPU_occupied_s']/result['counts']['actual_GPU_image_calls']*204894/3600
        result['runtime_technical_adjustment']='GPU occupancy includes failed attempt; first240image model times unavailable, not zero physical compute; reused exact temporary detections without additional GPU calls'
        result['prior_failed_source_extract_s']=json.loads((resource.RUN/'IO_leases/scene-0001_run0001.json').read_text())['source_IO_lease_wall_s']
        source.dump(path,result,False)
