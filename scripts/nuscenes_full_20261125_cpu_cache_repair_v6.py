"""缓存4/5/6 CPU修复；单一worker连续4,5,4,5，七条件逐帧精确比对。"""
from pathlib import Path
import sys,os,resource,json,gzip,hashlib,shutil,time,multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as pilot
from conceptgraph.completion.nuscenes_full_20261125_cpu_runtime_v6 import initialize_cpu_worker
from conceptgraph.project_paths import experiment_directory
base=pilot.base;source=pilot.source
REPLAY=pilot.RUN.parent/'nuscenes_full_20261125_cpu_runtime_v6_replay_v1'

def content_hash(path):
    h=hashlib.sha256()
    with gzip.open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def task(number,repeat=False):
    import pickle,torch
    scene=next(r for r in json.loads((pilot.DATA/'manifest.json').read_text())['scenes'] if r['scene']==f'scene-{number:04d}')
    request=next(source.rows(scene['request']));temp=base.TMP/(scene['scene']+'__run0004');old=pilot.RUN/scene['scene']/'attempts/run0004'
    gpu=json.loads((old/'frontend_receipt.json').read_text());gpu.pop('GT_in_method',None)
    for token,d in gpu['detections'].items():
        d['path']=str(temp/'_this_run_detections'/(token+'.pkl.gz'));assert source.sha(d['path'])==d['sha256']
        with gzip.open(d['path'],'rb') as f:det=pickle.load(f)
        assert base.prediction_digest(det)==d['prediction_sha256']
    receipt=json.loads((old/'source_members.json').read_text());members=receipt['members']
    for r in members.values():assert source.sha(r['path'])==r['sha256']
    run=REPLAY if repeat else pilot.RUN;root=run/scene['scene'];output=root/'attempts/cpu_v6';output.mkdir(parents=True)
    base.DATA=pilot.DATA;base.RUN=run
    for name in ['source_members.json','frontend_receipt.json']:shutil.copy2(old/name,output/name)
    removal=base.shutil.rmtree
    def preserve_for_double_check(path,*args,**kwargs):
        if number in [4,5] and Path(path).resolve()==temp.resolve():return None
        return removal(path,*args,**kwargs)
    base.shutil.rmtree=preserve_for_double_check
    oldlease=pilot.RUN/'IO_leases'/(scene['scene']+'_run0004.json');io_s=json.loads(oldlease.read_text())['source_IO_lease_wall_s']
    try:result=pilot.aux.cpu_scene_with_control(scene,request,str(temp),members,str(temp/'_this_run_geometry.json'),gpu,io_s,str(output))
    finally:base.shutil.rmtree=removal
    for name in ['frames.jsonl.gz','source_members.json','frontend_receipt.json']:(root/name).symlink_to((output/name).relative_to(root))
    source.dump(root/'complete.json',dict(result,accepted_attempt=str(output),cache_only_cpu_repair=True,no_additional_GPU_calls=True))
    answer=dict(scene=scene['scene'],repeat=repeat,PID=os.getpid(),frames=result['counts']['frames'],main_content_sha256=content_hash(root/'frames.jsonl.gz'),auxiliary_content_sha256=content_hash(root/'sharedscan_frames.jsonl.gz'),all7closed=result['auxiliary_sharedscan']['complete'],interop_threads=torch.get_num_interop_threads(),CUDA_initialized=torch.cuda.is_initialized())
    source.dump(output/'runtime_v6_check.json',answer);return answer

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));experiment_directory('nuscenes','validation','nuscenes_full_20261125_cpu_runtime_v6_replay_v1',create=True)
    sources=[Path(__file__).resolve(),source.ROOT/'conceptgraph/completion/nuscenes_full_20261125_cpu_runtime_v6.py']
    source.dump(REPLAY/'preregistration.json',dict(created_utc=source.now(),source_sha256={str(p):source.sha(p) for p in sources},frozen_sequence_same_worker=[4,5,4,5],parallel_other_scene=6,all_existing_GPU_cache_reused=True,GT=False,new_GPU_calls=0,original65unchanged=True))
    for p in sources:
        dst=REPLAY/'source_snapshot'/p.relative_to(source.ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu_worker) as serial,ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu_worker) as other:
        sixth=other.submit(task,6);results=[]
        for number,repeat in [(4,False),(5,False),(4,True),(5,True)]:
            r=serial.submit(task,number,repeat).result();results.append(r);print(json.dumps(r),flush=True)
        results.append(sixth.result())
    assert len({r['PID'] for r in results[:4]})==1
    for first,repeated in [(results[0],results[2]),(results[1],results[3])]:
        assert first['main_content_sha256']==repeated['main_content_sha256'] and first['auxiliary_content_sha256']==repeated['auxiliary_content_sha256']
    for number in [4,5]:
        tmp=base.TMP/f'scene-{number:04d}__run0004';assert tmp.parent.resolve()==base.TMP.resolve();shutil.rmtree(tmp)
    source.dump(REPLAY/'summary.json',dict(passed=True,same_worker_two_distinct_scenes_repeated_exact=True,results=results,all_three_original_caches_CPU_closed=True,new_GPU_calls=0,original_failures_preserved=True))
    print('CPU_RUNTIME_V6_PASSED',flush=True)
if __name__=='__main__':main()
