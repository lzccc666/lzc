"""第7共享scan独立增量检查；辅助失败不改变原六完整性。"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import time
from conceptgraph.completion.nuscenes_full_stream_audit_20261125 import (
    DATA, RUN, REPORT, CHANNELS, sha, rows, write, check_frame)


def audit(scene):
    folder=RUN/scene['scene']
    completion_path=folder/'sharedscan_complete.json'
    completion=json.loads(completion_path.read_text())
    fingerprint=sha(completion_path)
    target=REPORT/'nuscenes_auxiliary_scenes'/(scene['scene']+'__'+fingerprint[:16]+'.json')
    stream_path=folder/'sharedscan_frames.jsonl.gz'
    stat=stream_path.stat()
    if target.exists():
        prior=json.loads(target.read_text())
        assert prior['passed'] and prior['stream_sha256']==completion['frames_sha256']
        assert prior['audited_stream_stat']==dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        assert prior['audit_source_sha256']==sha(__file__)
        assert prior['common_checker_sha256']==sha(Path(__file__).with_name('nuscenes_full_stream_audit_20261125.py'))
        return prior
    tick=time.monotonic()
    assert completion['complete'] and completion['new_GPU_calls']==0 and completion['GT_in_method'] is False
    assert completion['method']=='S2_6_sharedscan'
    main_receipt=REPORT/'nuscenes_scenes'/(scene['scene']+'__'+sha(folder/'complete.json')[:16]+'.json')
    main_audit=json.loads(main_receipt.read_text())
    assert main_audit['passed'] and sha(folder/'frames.jsonl.gz')==main_audit['frames_sha256']
    assert sha(stream_path)==completion['frames_sha256']
    request_records=list(rows(scene['request']));assert len(request_records)==1
    request=request_records[0]
    members=json.loads((folder/'source_members.json').read_text())['members']
    histories={'S2_6':dict(raw={},assignment={})};counts=Counter()
    main=iter(rows(folder/'frames.jsonl.gz'));aux=iter(rows(stream_path))
    for group in request['groups']:
        original=next(main);record=next(aux)
        assert record['scene']==scene['scene'] and record['frame_idx']==original['frame_idx']==group['frame_idx']
        assert record['sample_token']==original['sample_token']==group['sample_token']
        assert record['method']=='S2_6_sharedscan' and record['GT_in_method'] is False
        assert record['render_timestamp_s']==original['render_timestamp_s']
        earliest=min(request['sensors'][v['camera']]['timestamp_us'] for v in group['views'].values())
        camera=record['earliest_camera']
        assert request['sensors'][group['views'][camera]['camera']]['timestamp_us']==earliest
        chosen=group['views'][camera]['lidar']
        assert record['shared_scan_requested_token']==chosen
        assert set(record['view_source_protocol'])==set(CHANNELS)
        derived=copy.deepcopy(original);derived['methods']={'S2_6':record['payload']}
        expected_group=copy.deepcopy(group)
        differs=False
        for name in CHANNELS:
            view=group['views'][name];protocol=record['view_source_protocol'][name]
            age=request['sensors'][view['camera']]['timestamp_us']-request['sensors'][chosen]['timestamp_us'] if chosen else None
            eligible=age is not None and 0<age<=100000
            applied=chosen if eligible else None
            assert protocol['requested_shared_lidar_token']==chosen
            assert protocol['applied_lidar_token']==applied
            assert protocol['main_protocol_lidar_token']==view['lidar']
            assert protocol['eligible']==eligible and protocol['age_exceeded']==(age is not None and age>100000)
            assert protocol['lidar_age_s']==(age/1e6 if age is not None else None)
            differs |= applied!=view['lidar']
            expected_group['views'][name]['lidar']=applied
            derived['views'][name].update(lidar_token=applied,
                lidar_timestamp_s=request['sensors'][applied]['timestamp_us']/1e6 if applied else None,
                lidar_sha256=members[applied]['sha256'] if applied else None)
        check_frame(derived,expected_group,request,members,histories,counts,methods=('S2_6',))
        counts['groups_no_earliest_scan']+=int(chosen is None)
        counts['groups_input_differs_from_main']+=int(differs)
    assert next(main,None) is None and next(aux,None) is None
    assert counts['frames']==completion['counts']['frames']==scene['frames']
    assert counts['S2_6_nodes']==completion['counts']['predictions']
    for key in ('empty_depth_views','groups_no_earliest_scan','groups_input_differs_from_main'):
        assert counts[key]==completion['counts'].get(key,0)
    result=dict(passed=True,scene=scene['scene'],counts=dict(counts),stream_sha256=completion['frames_sha256'],
        completion_sha256=fingerprint,main_stream_sha256=sha(folder/'frames.jsonl.gz'),audit_source_sha256=sha(__file__),
        common_checker_sha256=sha(Path(__file__).with_name('nuscenes_full_stream_audit_20261125.py')),
        audited_stream_stat=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),no_GPU=True,GT_opened=False,
        elapsed_s=time.monotonic()-tick,primary_six_completeness_independent=True,
        scope='all requested auxiliary groups, independent earliest-scan/age/empty logic, same actual RGB metadata, stored-pixel reprojection and verified raw motion gates',
        limits='No raw sensor archive/mask reread; full source/initial pixel audit is separately recorded. Auxiliary output is not vehicle safety authorization.')
    write(target,result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--maximum-scenes',type=int,default=100);parser.add_argument('--run-directory',type=Path,default=RUN);args=parser.parse_args();RUN=args.run_directory
    assert args.maximum_scenes>0
    manifest=json.loads((DATA/'manifest.json').read_text());counts=Counter();newly_audited=0;statuses=[]
    for scene in manifest['scenes']:
        folder=RUN/scene['scene'];state='not_closed_pending'
        if (folder/'sharedscan_complete.json').exists() and (folder/'complete.json').exists():
            path=REPORT/'nuscenes_auxiliary_scenes'/(scene['scene']+'__'+sha(folder/'sharedscan_complete.json')[:16]+'.json')
            cached=path.exists()
            if cached or newly_audited<args.maximum_scenes:
                audit(scene);newly_audited+=int(not cached);state='independently_audited'
            else:state='closed_wait_next_batch'
        elif (folder/'sharedscan_pending.json').exists():state='auxiliary_failed_pending'
        counts[state]+=1;statuses.append(dict(scene=scene['scene'],expected_frames=scene['frames'],state=state))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record=dict(scope='auxiliary850 separately from primary six',expected=850,statuses=dict(counts),newly_audited=newly_audited,no_GPU=True,GT_opened=False,
        audit_source_sha256=sha(__file__),utc=stamp,scenes=statuses)
    write(REPORT/'nuscenes_auxiliary_batches'/(stamp+'.json'),record)
    print(json.dumps({k:v for k,v in record.items() if k!='scenes'}))
