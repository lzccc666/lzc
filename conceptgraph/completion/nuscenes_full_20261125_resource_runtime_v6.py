"""资源调度：CPU预备真实几何、3CPU消费者、收据键去重；建图数学保持。"""
from __future__ import annotations
import argparse,gzip,json,os,resource as system_resource,signal,time
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as pilot
base=pilot.base;source=pilot.source
ORIGINAL_GPU=base.gpu_scene;ORIGINAL_GEOMETRY=base.geometry_tables;ORIGINAL_EXTRACT=pilot._ORIGINAL_STORE.extract
from conceptgraph.completion.nuscenes_full_20261125_cpu_runtime_v6 import initialize_cpu_worker
ORIGINAL_EXECUTOR=base.ProcessPoolExecutor
DEADLINE=1795617120

def physical_bytes():
    seen=set();total=0
    folders=[pilot.FULL_DATA,pilot.DATA,base.REPORT]
    folders += list((source.ROOT/'outputs/results/nuscenes/validation/iterations').glob('nuscenes_full_20261125_*'))
    for folder in folders:
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if p.is_symlink() or not p.is_file():continue
            st=p.stat();key=st.st_dev,st.st_ino
            if key not in seen:seen.add(key);total+=st.st_size
    return total

def io_stats():
    try:return {k:int(v) for k,v in [line.split(':') for line in Path('/proc/self/io').read_text().splitlines()]}
    except OSError:return {}

def profile_extract(self,request,attempt='initial'):
    progress=source.REPORT/'phaseA_progress_v3.json';paused=[];tick=time.monotonic();usage=system_resource.getrusage(system_resource.RUSAGE_SELF);before_io=io_stats()
    if progress.exists() and time.time()<DEADLINE:
        for number,pid in json.loads(progress.read_text()).get('active_PIDs',{}).items():
            p=Path('/proc')/str(pid)
            try:
                cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split()[2]
                if b'nuscenes_full_20261125_index_' in cmd and state not in ['T','Z']:
                    os.kill(pid,signal.SIGSTOP);paused.append(dict(archive=number,PID=pid,cmdline=cmd.decode().replace('\0',' ')))
            except FileNotFoundError:pass
    try:return ORIGINAL_EXTRACT(self,request,attempt)
    finally:
        resumed=[]
        for r in paused:
            p=Path('/proc')/str(r['PID'])
            try:
                if time.time()<DEADLINE and b'nuscenes_full_20261125_index_' in (p/'cmdline').read_bytes():os.kill(r['PID'],signal.SIGCONT);resumed.append(r['PID'])
            except FileNotFoundError:pass
        after=system_resource.getrusage(system_resource.RUSAGE_SELF);wall=time.monotonic()-tick;cpu=(after.ru_utime+after.ru_stime)-(usage.ru_utime+usage.ru_stime);after_io=io_stats()
        folder=base.RUN/'IO_leases';folder.mkdir(exist_ok=True)
        source.dump(folder/(request['scene']+'_'+attempt+'.json'),dict(ended_utc=source.now(),paused=paused,resumed_PIDs=resumed,source_IO_lease_wall_s=wall,parent_process_CPU_s=cpu,wall_minus_parent_CPU_s=wall-cpu,process_IO_counter_delta={k:after_io.get(k,0)-before_io.get(k,0) for k in after_io},limits='wall-minus-CPU is not a unique cause attribution; OS readbytes may exclude remote-page reads; cannot prove cold cache cause',index_deadline_unix=DEADLINE))

class PilotStore(pilot.PilotSourceStore):
    extract=profile_extract
class FullStore(pilot._ORIGINAL_STORE):
    extract=profile_extract

def gpu_scene(*args,**kwargs):
    clock=json.loads((pilot.FULL_DATA/'overall_execution_clock_v1.json').read_text());assert time.time()<clock['absolute_deadline_unix'],'absolute overall36h budget reached; retain scope/progress'
    result=ORIGINAL_GPU(*args,**kwargs);result.pop('GT_in_method',None);return result

def three_CPU_workers(*args,**kwargs):
    assert kwargs.get('max_workers')==2;kwargs['max_workers']=3;kwargs['initializer']=initialize_cpu_worker
    return ORIGINAL_EXECUTOR(*args,**kwargs)

def freeze_geometry(data):
    manifest=json.loads((data/'manifest.json').read_text());cache=data/'runtime_geometry_v5.json.gz';receipt=data/'runtime_geometry_v5_receipt.json'
    if cache.exists():
        record=json.loads(receipt.read_text());assert record['manifest_sha256']==source.sha(data/'manifest.json') and record['geometry_sha256']==source.sha(cache)
        with gzip.open(cache,'rt') as f:geometry=json.load(f)
        return geometry['poses'],geometry['calibrations']
    tick=time.monotonic();poses,cals=ORIGINAL_GEOMETRY(manifest)
    with gzip.open(cache,'xt',compresslevel=6) as f:json.dump(dict(poses=poses,calibrations=cals),f,separators=(',',':'))
    source.dump(receipt,dict(created_utc=source.now(),poses=len(poses),calibrations=len(cals),manifest_sha256=source.sha(data/'manifest.json'),geometry_sha256=source.sha(cache),CPU_preparation_wall_s=time.monotonic()-tick,GPU_lock_held=False,GT_in_geometry=False,source='original frozen geometry_tables using current source-only ego_pose/calibrated_sensor tables',source_sha256=source.sha(base.__file__)))
    return poses,cals

def idempotent_promote():
    full=json.loads((pilot.FULL_DATA/'method_manifest.json').read_text());old=json.loads((pilot.DATA/'method_manifest.json').read_text())
    cert=json.loads((pilot.FULL_DATA/'source_certification.json').read_text());assert cert['complete'] and cert['fully_member_available_scenes']==850
    assert json.loads((base.REPORT/'GPU_resource_pilot_summary.json').read_text())['passed']
    assert old['master_manifest_sha256']==full['source_manifest_sha256'] and old['configs']==full['configs'] and old['weights_sha256']==full['weights_sha256']
    for path,h in full['source_sha256'].items():assert old['source_sha256'][path]==h
    def alias(target,origin):
        if target.exists() or target.is_symlink():assert target.is_symlink() and target.resolve()==origin.resolve(),'existing different alias/entity retained; refuse overwrite'
        else:target.symlink_to(origin,target_is_directory=origin.is_dir())
    names=[s['scene'] for s in json.loads((pilot.DATA/'manifest.json').read_text())['scenes']]
    for name in names:
        origin=pilot.RUN/name;r=json.loads((origin/'complete.json').read_text());assert r['auxiliary_sharedscan']['complete'] and source.sha(origin/'frames.jsonl.gz')==r['frames_sha256'];alias(pilot.FULL_RUN/name,origin)
    attempts=pilot.FULL_RUN/'runtime_attempts';attempts.mkdir(exist_ok=True)
    for origin in (pilot.RUN/'runtime_attempts').iterdir():
        assert (origin/'GPU_release.json').exists();alias(attempts/origin.name,origin)
    record=dict(scenes=names,pilot_run=str(pilot.RUN),no_duplicate_GPU=True,source_manifest_and_configs_exact=True,idempotent_existing_aliases_verified=True)
    receipt=pilot.FULL_RUN/'resource_pilot_reuse.json'
    if receipt.exists():assert json.loads(receipt.read_text())==record
    else:source.dump(receipt,record)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','fullrun','promote']);parser.add_argument('--kitti-release-reference',default=str(pilot.FULL_DATA/'KITTI_GPU_released_notice.json'));args=parser.parse_args()
    required='/root/miniconda3/envs/conceptgraph/lib';prior=os.environ.get('LD_LIBRARY_PATH','');os.environ['LD_LIBRARY_PATH']=':'.join([required,*[p for p in prior.split(':') if p and p!=required]])
    system_resource.setrlimit(system_resource.RLIMIT_CORE,(0,0));os.environ['NUMEXPR_NUM_THREADS']='2'
    if args.command=='promote':idempotent_promote();return
    clock=json.loads((pilot.FULL_DATA/'overall_execution_clock_v1.json').read_text());assert time.time()<clock['absolute_deadline_unix'],'absolute36h task budget'
    reused=pilot.RUN.parent/'nuscenes_full_20261125_cpu_runtime_v6_replay_v1/summary.json';assert json.loads(reused.read_text())['passed']
    data=pilot.DATA if args.command=='run' else pilot.FULL_DATA
    preflight=pilot.RUN.parent/'nuscenes_full_20261125_cached_cpu_preflight_v1/summary.json';assert json.loads(preflight.read_text())['passed']
    own={str(Path(__file__).resolve()):source.sha(__file__),str(source.ROOT/'scripts/nuscenes_full_20261125_resource_runtime_v6.py'):source.sha(source.ROOT/'scripts/nuscenes_full_20261125_resource_runtime_v6.py'),str(source.ROOT/'conceptgraph/completion/nuscenes_full_20261125_cpu_runtime_v6.py'):source.sha(source.ROOT/'conceptgraph/completion/nuscenes_full_20261125_cpu_runtime_v6.py')}
    receipt=data/'resource_runtime_v6_freeze.json'
    if receipt.exists():assert json.loads(receipt.read_text())['source_sha256']==own
    else:source.dump(receipt,dict(created_utc=source.now(),source_sha256=own,changes=['geometry CPU beforeGPUlock, same frozen source-only transforms','3CPUspawn consumers, original pending<=3/temp512MiB unchanged','remove duplicate receipt boolean key only','per-sourceextract14:32deadline IOleases+wall/CPU counters','all actual GPU occupied attempts including technical failures cumulative'],CPU_quota_cores=16,CPU_threads_per_mapper=2,per_frame_math_or_seeds_changed=False,method_parameter_search=False,original65_sources_unchanged=True))
    geometry=freeze_geometry(data);expected=source.sha(data/'manifest.json')
    def cached_geometry(manifest):
        assert manifest==json.loads((data/'manifest.json').read_text());return geometry
    base.geometry_tables=cached_geometry;base.gpu_scene=gpu_scene;base.ProcessPoolExecutor=three_CPU_workers;pilot.footprint=physical_bytes
    if args.command=='run':
        pilot.PilotSourceStore=PilotStore;pilot.run(args)
        path=base.REPORT/'GPU_resource_pilot_summary.json';result=json.loads(path.read_text());source.dump(base.REPORT/'GPU_resource_pilot_frozen_base_summary_before_v5_cumulative.json',result)
        result['projected_full_GPU_occupied_hours']=result['total_GPU_occupied_s']/result['counts']['actual_GPU_image_calls']*204894/3600
        result['runtime_technical_adjustment']='all closed failed GPU attempts included; first240 image model timing unavailable, not zero physical compute; scene1 uses CPU-preflight exact real cache; remaining images normal inference'
        result['CPU_consumers']=3;result['geometry_prepared_before_GPU_lock']=True;source.dump(path,result,False)
    else:
        base.SourceStore=FullStore;pilot.fullrun(args)
if __name__=='__main__':main()
