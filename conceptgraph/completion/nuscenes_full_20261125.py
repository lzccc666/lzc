"""本机全nuScenes源成员认证与可恢复gzip索引；不读取方法分数或GT答案。"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tarfile
import time

from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory

PREFIX = 'nuscenes_full_20261125'
DATA = ROOT/'dataset/processed'/f'{PREFIX}_v1'
REPORT = ROOT/'outputs/reports'/PREFIX
ARCHIVES = Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')
META = ROOT/'dataset/raw/nuscenes_trainval/v1.0-trainval'
INDEX_ENV = Path('/tmp/nuscenes_full_index_20261125')
SPACING = 16*2**20
MIN_FREE = 20*2**30


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*2**20),b''): h.update(b)
    return h.hexdigest()


def dump(path,value,exclusive=True):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def now(): return datetime.now(timezone.utc).isoformat()


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream: yield json.loads(line)


def guard(deadline):
    if time.time()>=deadline: raise TimeoutError('phaseA deadline; preserve completed source/index prefixes')
    if shutil.disk_usage(DATA.resolve()).free<MIN_FREE: raise OSError('actual target free space below20GiB')


def prepare(args):
    DATA.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    (DATA/'requests').mkdir();(DATA/'archives').mkdir()
    run=experiment_directory('nuscenes','validation',f'{PREFIX}_index_v1',create=True)
    (run/'source_snapshot/conceptgraph/completion').mkdir(parents=True)
    (run/'source_snapshot/scripts').mkdir()
    sources=[Path(__file__).resolve(), ROOT/'scripts'/f'{PREFIX}.py',ROOT/'conceptgraph/nuscenes/expansion_benchmark.py']
    for p in sources:
        target=run/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    scene_rows=json.loads((META/'scene.json').read_text());scenes={r['token']:r for r in scene_rows}
    samples=json.loads((META/'sample.json').read_text());sample_scene={r['token']:r['scene_token'] for r in samples}
    grouped_samples=defaultdict(list)
    for r in samples: grouped_samples[r['scene_token']].append(r)
    roles={}; role_inputs=[]
    for base in [ROOT/'outputs/project_v2/lane_qwen_expansion_v1',ROOT/'outputs/project_v2/lane_qwen_expansion_v1/prospective_supplement_v2']:
        path=base/'split_manifest.json'; d=json.loads(path.read_text());role_inputs.append(path)
        for name,role in d['scene_to_split'].items():
            roles[name]=dict(original_role=role,source=str(path),source_test_labels_opened=d.get('test_labels_opened'),
                access_role='used_validation' if role=='development' else 'historical_regression' if role=='legacy_regression' else 'historical_holdout_access_to_audit')
    path=ROOT/'dataset/splits/historical/reserved_test_scenes_v2.json';role_inputs.append(path)
    for s in json.loads(path.read_text())['scenes']:
        roles[s['scene']]=dict(original_role='reserved_test_v2',access_role='previously_used_test_regression',source=str(path))
    seen_logs={s['log_token'] for s in scene_rows if s['name'] in roles}
    cameras={}; lidars=defaultdict(list); channels=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    print(json.dumps(dict(stage='metadata_sensor_scan',utc=now(),expected_scenes=len(scenes),expected_samples=len(samples))),flush=True)
    for r in iter_records(META/'sample_data.json'):
        ch=r['filename'].split('/')[1]
        if ch=='LIDAR_TOP': lidars[sample_scene[r['sample_token']]].append(r)
        elif r['is_key_frame'] and ch in channels: cameras[r['sample_token'],ch]=r
    def small(r):
        return dict(token=r['token'],filename=r['filename'],timestamp_us=r['timestamp'],ego_pose_token=r['ego_pose_token'],calibrated_sensor_token=r['calibrated_sensor_token'],
            **({'width':r['width'],'height':r['height']} if r['filename'].endswith('.jpg') else {}))
    entries=[]; stats=Counter()
    for scene_token,scene in sorted(scenes.items(),key=lambda p:p[1]['name']):
        sensor={};groups=[];li=sorted(lidars[scene_token],key=lambda x:x['timestamp']);times=[r['timestamp'] for r in li]
        for frame_idx,sample in enumerate(sorted(grouped_samples[scene_token],key=lambda x:(x['timestamp'],x['token']))):
            views={}
            for channel in channels:
                cam=cameras.get((sample['token'],channel))
                if cam is None: views[channel]=dict(camera=None,lidar=None,reason='missing_official_camera_metadata');stats['missing_camera_metadata']+=1;continue
                pos=bisect_left(times,cam['timestamp'])-1
                lidar=li[pos] if pos>=0 and 0<cam['timestamp']-li[pos]['timestamp']<=100000 else None
                sensor[cam['token']]=small(cam)
                if lidar:sensor[lidar['token']]=small(lidar)
                views[channel]=dict(camera=cam['token'],lidar=lidar['token'] if lidar else None,
                    reason=None if lidar else 'no_strictly_past_lidar_within_100ms_keep_empty_depth')
                stats['camera_views']+=1;stats['views_with_causal_lidar']+=int(lidar is not None)
            groups.append(dict(frame_idx=frame_idx,sample_token=sample['token'],official_sample_timestamp_us=sample['timestamp'],
                reference_timestamp_us=max(sensor[v['camera']]['timestamp_us'] for v in views.values() if v['camera']),views=views))
        role=roles.get(scene['name'],dict(original_role='not_previously_selected',access_role='fixed_method_new_scene',source='official scene table and frozen absence from historical selection'))
        request=dict(scene=scene['name'],scene_token=scene_token,log_token=scene['log_token'],role=role,log_previously_used=scene['log_token'] in seen_logs,
            body_pose_policy='camera optical axis XY proxy declared; no GT heading',sensors=sensor,groups=groups)
        target=DATA/'requests'/(scene['name']+'.jsonl.gz')
        with gzip.open(target,'wt',compresslevel=6) as f: f.write(json.dumps(request,separators=(',',':'))+'\n')
        entries.append(dict(scene=scene['name'],scene_token=scene_token,log_token=scene['log_token'],frames=len(groups),role=role,log_previously_used=request['log_previously_used'],request=str(target),request_sha256=sha(target)))
    source_files=[META/n for n in ['scene.json','sample.json','sample_data.json','calibrated_sensor.json','sensor.json','log.json','ego_pose.json']]
    manifest=dict(created_utc=now(),git_base='dc290332033d1b7879eec3f4a5d54f8f4b6cf6bc',phase='source_index_only_no_inference',seed=17,source_sha256={str(p):sha(p) for p in sources},
        metadata_sha256={str(p):sha(p) for p in source_files},role_source_sha256={str(p):sha(p) for p in role_inputs},
        scenes=entries,counts=dict(scenes=len(entries),frames=sum(s['frames'] for s in entries),logs=len({s['log_token'] for s in entries}),**dict(stats)),
        archives=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(ARCHIVES.glob('*_blobs.tgz'))],
        cameras=channels,clock='max actual camera exposures; LiDAR strictly before own camera at most100ms; missing depth retained',
        members_certified=False,GT_in_method=False,GPU_calls=0,index_package=dict(name='indexed_gzip',version='1.10.3',isolated_target=str(INDEX_ENV),spacing=SPACING,window_bytes=32768,readbuf_bytes=2**20,
            upstream='https://github.com/pauldmccarthy/indexed_gzip',crc_checked_on_original_build=True,imported_index_crc_disabled_by_library=True),
        budgets=dict(phaseA_wall_s=5400,phaseA_metadata_bytes=128*2**20,index_bytes=2**30,nu_total_persistent_bytes=int(4.5*2**30),temporary_scene_bytes=512*2**20,minimum_free_bytes=MIN_FREE,full_wall_s=36*3600,GPU_wall_s=12*3600),
        deadline_unix=args.deadline,run=str(run),GPU_lock='/root/autodl-tmp/lizichong/outputs/reports/shared_gpu_20261125.lock',GPU_order='KITTI Object first; wait explicit release and exclusive fcntl lock',
        fixed_conditions=['single_frame_CAM_FRONT','original_fusion_CAM_FRONT','R1_CAM_FRONT','S2_CAM_FRONT','S2_front3','S2_six'],
        temporary_cleanup_policy='only dataset/processed/nuscenes_full_20261125_tmp/<scene>; after all methods and measurement-source evidence sealed and checked; preserve raw/old caches and failed prefixes',
        no_scores_accessed_for_selection=True,body_heading_unknown_preserved=True)
    dump(DATA/'manifest.json',manifest);dump(run/'manifest.json',manifest)
    print(json.dumps(dict(stage='prepared',counts=manifest['counts'],manifest=str(DATA/'manifest.json'),run=str(run))),flush=True)


def indexed_module():
    sys.path.insert(0,str(INDEX_ENV));import indexed_gzip
    assert indexed_gzip.__version__=='1.10.3'
    return indexed_gzip


def index_archive(number,deadline):
    ig=indexed_module();archive=ARCHIVES/f'v1.0-trainval{number:02d}_blobs.tgz';name=f'archive{number:02d}'
    complete=DATA/'archives'/(name+'_complete.json')
    if complete.exists():
        previous=json.loads(complete.read_text());assert previous['archive_bytes']==archive.stat().st_size and previous['archive_mtime_ns']==archive.stat().st_mtime_ns
        assert sha(previous['members'])==previous['members_sha256'];assert sha(previous['seek_index'])==previous['seek_index_sha256'];return previous
    output=DATA/'archives'/(name+'_members.jsonl.gz');index=DATA/'archives'/(name+'.gzidx')
    if output.exists() or index.exists(): raise FileExistsError('unclosed prior prefix retained; use explicit technical revision')
    start=time.monotonic();counter=Counter();stat=archive.stat();last=start
    with ThreadPoolExecutor(max_workers=1) as pool:
        full_hash=pool.submit(sha,archive)
        with ig.IndexedGzipFile(filename=str(archive),spacing=SPACING,window_size=32768,readbuf_size=2**20,drop_handles=False,skip_crc_check=False) as stream,gzip.open(output,'xt',compresslevel=6) as out:
            with tarfile.open(fileobj=stream,mode='r:') as tf:
                for member in tf:
                    counter['tar_headers']+=1
                    if member.isfile():
                        clean=member.name.removeprefix('./')
                        if (clean.startswith('samples/CAM_') and clean.endswith('.jpg')) or ('/LIDAR_TOP/' in clean and clean.endswith('.bin')):
                            out.write(json.dumps([clean,member.offset_data,member.size],separators=(',',':'))+'\n');counter['retained_members']+=1;counter[clean.split('/')[1]]+=1
                    tf.members.clear()
                    if time.monotonic()-last>30:
                        guard(deadline);last=time.monotonic()
                        progress=dict(archive=number,stage='indexing',counts=dict(counter),uncompressed_position=stream.tell(),elapsed_s=time.monotonic()-start,utc=now())
                        dump(REPORT/(name+'_progress.json'),progress,False);print(json.dumps(progress),flush=True)
            stream.build_full_index();stream.export_index(filename=str(index))
        fingerprint=full_hash.result()
    guard(deadline);assert archive.stat().st_size==stat.st_size and archive.stat().st_mtime_ns==stat.st_mtime_ns
    total_index=sum(p.stat().st_size for p in (DATA/'archives').glob('*.gzidx'))
    if total_index>2**30:raise OSError('gzip index total exceeds1GiB; retain current index')
    result=dict(complete=True,archive=str(archive),archive_bytes=stat.st_size,archive_mtime_ns=stat.st_mtime_ns,archive_sha256=fingerprint,
        members=str(output),members_sha256=sha(output),seek_index=str(index),seek_index_sha256=sha(index),seek_index_bytes=index.stat().st_size,
        spacing=SPACING,counts=dict(counter),elapsed_s=time.monotonic()-start,ended_utc=now(),gzip_CRC_validation=True,
        maximum_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_calls=0)
    dump(complete,result);print(json.dumps(result),flush=True);return result


def certify(args):
    manifest=json.loads((DATA/'manifest.json').read_text());members={};duplicates=[]
    for i in range(1,11):
        p=DATA/'archives'/f'archive{i:02d}_complete.json';done=json.loads(p.read_text())
        for filename,offset,size in rows(done['members']):
            if filename in members:duplicates.append(filename)
            else:members[filename]=(i,offset,size)
    all_counts=Counter();scene_rows=[]
    with gzip.open(DATA/'member_binding.jsonl.gz','xt',compresslevel=6) as output:
        for s in manifest['scenes']:
            req=next(rows(s['request']));counter=Counter();missing=[]
            for token,sensor in req['sensors'].items():
                binding=members.get(sensor['filename']);ch=sensor['filename'].split('/')[1]
                counter['expected_'+ch]+=1;counter['available_'+ch]+=int(binding is not None)
                if binding is None:missing.append(dict(token=token,filename=sensor['filename']))
                else:output.write(json.dumps(dict(scene=s['scene'],token=token,filename=sensor['filename'],archive=binding[0],offset=binding[1],bytes=binding[2]),separators=(',',':'))+'\n')
            for group in req['groups']:
                have={ch for ch,v in group['views'].items() if v['camera'] and req['sensors'][v['camera']]['filename'] in members}
                for label,chs in [('1',['CAM_FRONT']),('3',manifest['cameras'][:3]),('6',manifest['cameras'])]:counter['complete_RGB_groups_'+label]+=int(set(chs)<=have)
                for ch,v in group['views'].items():
                    counter['requested_views']+=1
                    if v['lidar'] is None:counter['legitimate_missing_past_lidar']+=1
                    elif req['sensors'][v['lidar']]['filename'] not in members:counter['missing_causal_lidar_member']+=1
            all_counts.update(counter);scene_rows.append(dict(**s,counts=dict(counter),missing_members=missing,all_requested_source_members_available=not missing))
    result=dict(complete=True,utc=now(),scenes=scene_rows,counts=dict(all_counts),all_scenes=len(scene_rows),all_frames=sum(s['frames'] for s in scene_rows),
        logs=len({s['log_token'] for s in scene_rows}),fully_member_available_scenes=sum(s['all_requested_source_members_available'] for s in scene_rows),
        duplicate_member_filenames=len(duplicates),duplicates=duplicates,GT_accessed=False,GPU_calls=0,missing_source_or_depth_never_removes_denominator=True,
        source_manifest_sha256=sha(DATA/'manifest.json'),member_binding_sha256=sha(DATA/'member_binding.jsonl.gz'))
    dump(DATA/'source_certification.json',result);dump(REPORT/'phaseA_source_summary.json',{k:v for k,v in result.items() if k not in ['scenes','duplicates']})
    print(json.dumps({k:v for k,v in result.items() if k not in ['scenes','duplicates']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--deadline',type=float,required=True)
    p=sub.add_parser('index');p.add_argument('--archive',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    sub.add_parser('certify');args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='index':index_archive(args.archive,args.deadline)
    else:certify(args)

