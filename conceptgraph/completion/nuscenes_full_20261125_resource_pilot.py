"""仅资源调度修订：真实8场景证书先试跑，正式850门保持。"""
from __future__ import annotations
import argparse,gzip,json,os,shutil,signal,time
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125_run as base
from conceptgraph.completion import nuscenes_full_20261125_sharedscan as aux
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.project_paths import experiment_directory
FULL_DATA=base.DATA;FULL_RUN=base.RUN
DATA=source.ROOT/'dataset/processed/nuscenes_full_20261125_resource_v1'
RUN_ID='nuscenes_full_20261125_resource_v1'
RUN=source.ROOT/'outputs/results/nuscenes/validation/iterations'/RUN_ID
_ORIGINAL_STORE=base.SourceStore

def footprint():
    seen=set();total=0
    for folder in [FULL_DATA,FULL_RUN,DATA,RUN,base.REPORT]:
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if p.is_symlink() or not p.is_file():continue
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);total+=st.st_size
    return total

class PilotSourceStore(_ORIGINAL_STORE):
    def __init__(self):
        self.ig=source.indexed_module();self.handles={};self.by_scene={}
        cert=json.loads((DATA/'source_certification.json').read_text());assert cert['complete'] and cert['fully_member_available_scenes']==8
        self.archives={1:json.loads((FULL_DATA/'archives/archive01_complete.json').read_text())}
        for r in source.rows(DATA/'member_binding.jsonl.gz'):self.by_scene.setdefault(r['scene'],{})[r['token']]=r
    def extract(self,request,attempt='initial'):
        progress=source.REPORT/'phaseA_progress_v3.json';paused=[];started=time.monotonic()
        if progress.exists() and time.time()<1795613520:
            for number,pid in json.loads(progress.read_text()).get('active_PIDs',{}).items():
                p=Path('/proc')/str(pid)
                try:
                    cmd=(p/'cmdline').read_bytes();state=(p/'stat').read_text().split()[2]
                    if b'nuscenes_full_20261125_index_' in cmd and state not in ['T','Z']:
                        os.kill(pid,signal.SIGSTOP);paused.append(dict(archive=number,PID=pid,cmdline=cmd.decode().replace('\0',' ')))
                except FileNotFoundError:pass
        try:return super().extract(request,attempt)
        finally:
            resumed=[]
            for row in paused:
                p=Path('/proc')/str(row['PID'])
                try:
                    if time.time()<1795613520 and b'nuscenes_full_20261125_index_' in (p/'cmdline').read_bytes():os.kill(row['PID'],signal.SIGCONT);resumed.append(row['PID'])
                except FileNotFoundError:pass
            target=RUN/'IO_leases';target.mkdir(exist_ok=True)
            source.dump(target/(request['scene']+'_'+attempt+'.json'),dict(ended_utc=source.now(),paused=paused,resumed_PIDs=resumed,source_IO_lease_wall_s=time.monotonic()-started,reason='brief source extraction priority only; original C-index memory retained; no algorithm change'))

def cpu_scene(*args,**kwargs):
    base.DATA=DATA;base.RUN=RUN
    return aux.cpu_scene_with_control(*args,**kwargs)

def prepare():
    assert json.loads((FULL_DATA/'KITTI_GPU_released_notice.json').read_text())['explicit_release']
    pilot=json.loads((FULL_DATA/'resource_pilot_preregistration.json').read_text());scenes=pilot['scenes'];assert [r['scene'] for r in scenes]==[f'scene-{i:04d}' for i in range(1,9)]
    master=json.loads((FULL_DATA/'manifest.json').read_text());archive=json.loads((FULL_DATA/'archives/archive01_complete.json').read_text())
    assert archive['complete'] and source.sha(archive['members'])==archive['members_sha256'] and source.sha(archive['seek_index'])==archive['seek_index_sha256']
    members={r[0]:r for r in source.rows(archive['members'])};bindings=[]
    for scene in scenes:
        assert source.sha(scene['request'])==scene['request_sha256'];request=next(source.rows(scene['request']))
        for token,sensor in request['sensors'].items():
            row=members[sensor['filename']];bindings.append(dict(scene=scene['scene'],token=token,filename=row[0],archive=1,offset=row[1],bytes=row[2]))
    DATA.mkdir(parents=True,exist_ok=False);experiment_directory('nuscenes','validation',RUN_ID,create=True)
    with gzip.open(DATA/'member_binding.jsonl.gz','xt') as f:
        for row in bindings:f.write(json.dumps(row,separators=(',',':'))+'\n')
    source.dump(DATA/'manifest.json',dict(created_utc=source.now(),phase='eight_scene_resource_only_not_full850_certification',scenes=scenes,scene_count=8,frames=sum(s['frames'] for s in scenes),master_manifest_sha256=source.sha(FULL_DATA/'manifest.json'),no_GT=True))
    source.dump(DATA/'source_certification.json',dict(complete=True,fully_member_available_scenes=8,total_scenes=8,member_bindings=len(bindings),unique_sensor_members=len({r['token'] for r in bindings}),source_archive_receipt_sha256=source.sha(FULL_DATA/'archives/archive01_complete.json'),full850_complete=False,member_binding_sha256=source.sha(DATA/'member_binding.jsonl.gz')))
    for name in ['resource_pilot_preregistration.json','scene_role_access_overlay_v2.json']:shutil.copy2(FULL_DATA/name,DATA/name)
    cfg,release,node=base.configs();preseal=json.loads((FULL_DATA/'phaseB_source_preseal_v1.json').read_text());auxiliary=json.loads((FULL_DATA/'sharedscan_auxiliary_protocol_v1.json').read_text())
    files=dict(preseal['source_sha256']);files.update(auxiliary['source_sha256'])
    for path in [Path(__file__).resolve(),source.ROOT/'scripts/nuscenes_full_20261125_resource_pilot.py']:files[str(path)]=source.sha(path)
    for path,digest in files.items():assert source.sha(path)==digest,path
    weights=[source.ROOT/'checkpoint/yolov8l-world.pt',source.ROOT/'checkpoint/mobile_sam.pt',source.ROOT/'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']
    frozen=dict(created_utc=source.now(),seed=17,configuration_count=1,parameter_search=False,configs=cfg,source_sha256=files,weights_sha256={str(p):source.sha(p) for p in weights},source_manifest_sha256=source.sha(DATA/'manifest.json'),source_certification_sha256=source.sha(DATA/'source_certification.json'),master_manifest_sha256=source.sha(FULL_DATA/'manifest.json'),methods=base.METHODS,auxiliary=aux.CONTROL,GT_in_method=False,resource_only=True,full850_certificate_unchanged=True,io_policy='brief pause currently active index workers only during each source extraction; resume own PIDs in finally before deadline; index C memory preserved',GPU_budget_cumulative_including_this_pilot=True)
    source.dump(DATA/'method_manifest.json',frozen);source.dump(RUN/'manifest.json',frozen)
    for path in files:
        dest=RUN/'source_snapshot'/Path(path).relative_to(source.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    print(json.dumps(dict(prepared=True,scenes=8,bindings=len(bindings),run=str(RUN))),flush=True)

def run(args):
    base.DATA=DATA;base.RUN=RUN;base.SourceStore=PilotSourceStore;base.cpu_scene=cpu_scene;base.footprint=footprint
    args.resource_pilot_only=True;base.run(args)

def promote():
    pilot=json.loads((base.REPORT/'GPU_resource_pilot_summary.json').read_text());assert pilot['passed']
    original=json.loads((DATA/'method_manifest.json').read_text());full=json.loads((FULL_DATA/'method_manifest.json').read_text())
    assert original['master_manifest_sha256']==full['source_manifest_sha256'] and original['configs']==full['configs'] and original['weights_sha256']==full['weights_sha256']
    for p,h in full['source_sha256'].items():assert original['source_sha256'][p]==h
    cert=json.loads((FULL_DATA/'source_certification.json').read_text());assert cert['complete'] and cert['fully_member_available_scenes']==850
    names=[s['scene'] for s in json.loads((DATA/'manifest.json').read_text())['scenes']]
    for name in names:
        r=json.loads((RUN/name/'complete.json').read_text());assert r['auxiliary_sharedscan']['complete'] and source.sha(RUN/name/'frames.jsonl.gz')==r['frames_sha256'];(FULL_RUN/name).symlink_to(RUN/name,target_is_directory=True)
    target=FULL_RUN/'runtime_attempts';target.mkdir()
    for attempt in (RUN/'runtime_attempts').iterdir():
        assert (attempt/'GPU_release.json').exists();(target/attempt.name).symlink_to(attempt,target_is_directory=True)
    source.dump(FULL_RUN/'resource_pilot_reuse.json',dict(created_utc=source.now(),scenes=names,pilot_run=str(RUN),prior_GPU_occupied_s=pilot['total_GPU_occupied_s'],no_duplicate_GPU=True,source_manifest_and_configs_exact=True))

def fullrun(args):
    base.footprint=footprint;base.cpu_scene=aux.cpu_scene_with_control;args.resource_pilot_only=False;base.run(args)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','promote','fullrun']);p.add_argument('--kitti-release-reference',default=str(FULL_DATA/'KITTI_GPU_released_notice.json'));args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='promote':promote()
    elif args.command=='fullrun':fullrun(args)
    else:run(args)
if __name__=='__main__':main()
