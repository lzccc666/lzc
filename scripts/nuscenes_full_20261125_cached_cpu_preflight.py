"""本轮缓存端到端CPU预演：不调用/加载GPU，全部7mapper与独占写收口。"""
from pathlib import Path
import sys,json,importlib.util,time,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as resource
from conceptgraph.project_paths import experiment_directory
base=resource.base;source=resource.source

def main():
    import torch
    tick=time.monotonic();run=experiment_directory('nuscenes','validation','nuscenes_full_20261125_cached_cpu_preflight_v1',create=True)
    source.dump(run/'preregistration.json',dict(created_utc=source.now(),source_sha256=source.sha(__file__),GT=False,GPU_calls=0,scope='complete existing40frame scene0001, same7conditions, only own temporary cache reused read-only'))
    spec=importlib.util.spec_from_file_location('runtime_v4',source.ROOT/'scripts/nuscenes_full_20261125_resource_runtime_v4.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    scene=json.loads((resource.DATA/'manifest.json').read_text())['scenes'][0];request=next(source.rows(scene['request']));temp=runtime.OLD_TEMP
    members=json.loads((runtime.OLD_OUTPUT/'source_members.json').read_text())['members']
    for token,r in members.items():assert source.sha(r['path'])==r['sha256']
    gpu=runtime.gpu_scene(None,request,temp,regression=True);assert gpu['reuse_only']
    base.DATA=resource.DATA;base.RUN=run
    root=run/scene['scene'];output=root/'attempts/cpu_only';output.mkdir(parents=True)
    source.dump(output/'source_members.json',dict(request_sha256=scene['request_sha256'],members=members,GT_in_method=False))
    source.dump(output/'frontend_receipt.json',dict(**{k:v for k,v in gpu.items() if k!='detections'},detections={t:{k:v for k,v in d.items() if k!='path'} for t,d in gpu['detections'].items()},GT_in_method=False))
    original=base.shutil.rmtree
    def preserve_only_original_cache(path,*args,**kwargs):
        if Path(path).resolve()==temp.resolve():return None
        return original(path,*args,**kwargs)
    base.shutil.rmtree=preserve_only_original_cache
    try:record=resource.aux.cpu_scene_with_control(scene,request,str(temp),members,str(temp/'_this_run_geometry.json'),gpu,0.,str(output))
    finally:base.shutil.rmtree=original
    for name in ['frames.jsonl.gz','source_members.json','frontend_receipt.json']:(root/name).symlink_to((output/name).relative_to(root))
    source.dump(root/'complete.json',dict(record,accepted_attempt=str(output)))
    assert source.sha(root/'frames.jsonl.gz')==record['frames_sha256'] and record['auxiliary_sharedscan']['complete']
    alias=run/'promotion_mechanics_only';alias.mkdir();(alias/scene['scene']).symlink_to(root,target_is_directory=True)
    assert source.sha(alias/scene['scene']/'frames.jsonl.gz')==record['frames_sha256']
    assert not (resource.FULL_DATA/'source_certification.json').exists(),'actual full850 promote still gated; do not fake certification'
    source.dump(run/'summary.json',dict(passed=True,frames=scene['frames'],methods=base.METHODS,auxiliary=resource.aux.CONTROL,GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),all_source_members_exact=True,closed_hashes_exact=True,alias_mechanics_passed=True,actual_full850_promote_still_waits_certificate=True,original_failure_cache_preserved=temp.exists(),elapsed_s=time.monotonic()-tick))
    print((run/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()
