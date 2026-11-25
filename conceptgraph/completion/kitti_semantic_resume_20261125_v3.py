"""Continue only uncertified RGB headers, reuse prior receipts, no GPU."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from conceptgraph.completion import kitti_semantic_pair_inventory_20261125 as original


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.nice(10)
    root=original.ROOT;report=original.REPORT
    priorpath=report/'semantic_RGB_LiDAR_pose_pair_inventory_v2.json';prior=json.loads(priorpath.read_text())
    fullpath=report/'depth_tar_header_inventory_v1.json';full=json.loads(fullpath.read_text())
    mappingpath=report/'official_odometry_raw_mapping_receipt_v1.json';mapping=json.loads(mappingpath.read_text())
    archive=Path(full['archive']);assert archive.stat().st_size==full['archive_bytes'] and archive.stat().st_mtime_ns==full['archive_mtime_ns']
    known={r['sequence']:set() for r in mapping['rows']};last={}
    for pair in prior['pairs']:
        seq=pair['sequence'];known[seq].add(pair['frame_idx']);rgb=pair['RGB']
        last[seq]=max(last.get(seq,0),rgb['offset_data']+((rgb['bytes']+511)//512)*512)
    complete_ranges={r['sequence'] for r in mapping['rows'] if len(known[r['sequence']])==r['raw_end']-r['raw_start']+1}
    targets=[]
    for row in mapping['rows']:
        if row['sequence'] in complete_ranges:continue
        drive=row['raw_drive']+'_sync'
        candidates=[r for r in full['metadata'] if f'/{drive}/image_02/timestamps.txt' in r['name'] and '/raw/' in r['name']]
        if candidates:
            assert len(candidates)==1;targets.append(dict(**row,metadata=candidates[0]))
        else:assert full['drives'].get(drive,{}).get('RGB2',0)==0
    targets.sort(key=lambda r:r['metadata']['offset_data'])
    started=time.monotonic();deadline=started+1500.;limited=False;new_headers=0
    amendment=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in
        [Path(__file__),root/'scripts/kitti_semantic_resume_20261125_v3.py',Path(original.__file__)]},
        prior_receipt_sha256=sha(priorpath),full_directory_inventory_sha256=sha(fullpath),mapping_sha256=sha(mappingpath),
        maximum_wall_s=1500,maximum_persistent_new_bytes=20*1024**2,CPU_nice=10,readers=1,GPU_calls=0,
        reused_certified_pairs=len(prior['pairs']),skipped_certified_sequences=sorted(complete_ranges),
        resume_offsets_after_last_certified_image_payload=last,
        operation='reuse all certified metadata; continue after last certified payload in partial07 and seek only previously unscanned mapped directories',
        range_completion='all requested official frame indices certified; no need to re-read non-task RGB beyond official interval')
    original.write(report/'semantic_resume_amendment_v3.json',amendment)
    tar_open=tarfile.open
    class TargetHeaders:
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def __iter__(self):
            nonlocal limited,new_headers
            for pair in prior['pairs']:
                rgb=pair['RGB'];member=tarfile.TarInfo(rgb['name']);member.size=rgb['bytes'];member.offset_data=rgb['offset_data'];member.type=tarfile.REGTYPE
                yield member
            for target in targets:
                seq=target['sequence'];meta=target['metadata'];prefix=meta['name'].rsplit('/',1)[0]+'/'
                if time.monotonic()>deadline:limited=True;return
                offset=last.get(seq,meta['offset_data']-512)
                with archive.open('rb') as stream:
                    stream.seek(offset)
                    with tar_open(fileobj=stream,mode='r:') as reader:
                        for member in reader:
                            if not member.name.startswith(prefix):break
                            if member.isfile() and '/data/' in member.name and member.name.endswith('.png'):
                                new_headers+=1;raw_frame=int(Path(member.name).stem)
                                if target['raw_start']<=raw_frame<=target['raw_end']:
                                    frame=raw_frame-target['raw_start']
                                    if frame in known[seq]:raise AssertionError('already_certified_RGB_header_was_rescanned')
                                    known[seq].add(frame);yield member
                                time.sleep(.002)
                            if time.monotonic()>deadline:limited=True;return
                expected=set(range(target['raw_end']-target['raw_start']+1))
                if known[seq]!=expected:raise AssertionError('official_target_RGB_indices_missing_after_directory_complete')
                complete_ranges.add(seq)
                progress=dict(utc=datetime.now(timezone.utc).isoformat(),complete_ranges=sorted(complete_ranges),
                    certified_per_sequence={k:len(v) for k,v in known.items()},new_headers_read=new_headers,elapsed_s=time.monotonic()-started)
                with (report/'semantic_resume_progress_v3.json').open('w') as f:json.dump(progress,f,indent=2)
                print(json.dumps(progress),flush=True)
    def selected_open(*args,**kwargs):
        path=args[0] if args else kwargs.get('name')
        if path is not None and Path(path).resolve()==archive.resolve():return TargetHeaders()
        return tar_open(*args,**kwargs)
    original_write=original.write
    def write(path,value):
        if Path(path).name=='semantic_RGB_LiDAR_pose_pair_inventory_v1.json':
            path=report/'semantic_RGB_LiDAR_pose_pair_inventory_v3.json'
            all_available={t['sequence'] for t in targets}|set(prior['completed_RGB_sequences'])
            value.update(complete=not limited and complete_ranges==all_available,
                completed_RGB_sequences=sorted(complete_ranges),bounded_partial=limited,
                certified_requested_ranges=True,reused_prior_pairs=len(prior['pairs']),new_headers_read=new_headers,
                resume_amendment_sha256=sha(report/'semantic_resume_amendment_v3.json'))
            for row in value['matrix']:
                if row['sequence'] in all_available and row['sequence'] not in complete_ranges:
                    row.update(status='RGB_header_certification_pending',common_complete=False,
                        missing_RGB_not_inferred_from_partial_scan=True)
        result=original_write(path,value)
        if Path(path).name=='semantic_RGB_LiDAR_pose_pair_inventory_v3.json' and Path(path).stat().st_size>20*1024**2:
            raise OSError('new_resume_receipt_above20MiB_preserved')
        return result
    tarfile.open=selected_open;original.write=write
    try:original.main()
    finally:tarfile.open=tar_open;original.write=original_write


if __name__=='__main__':main()
