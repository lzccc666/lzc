"""Bounded, read-only header walk of already indexed target RGB directories."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time

from conceptgraph.completion import kitti_semantic_pair_inventory_20261125 as original


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=original.ROOT;report=original.REPORT
    full_path=report/'depth_tar_header_inventory_v1.json';full=json.loads(full_path.read_text())
    mapping_path=report/'official_odometry_raw_mapping_receipt_v1.json';mapping=json.loads(mapping_path.read_text())
    archive=Path(full['archive']);assert archive.stat().st_size==full['archive_bytes'] and archive.stat().st_mtime_ns==full['archive_mtime_ns']
    targets=[]
    for row in mapping['rows']:
        drive=row['raw_drive']+'_sync'
        candidates=[r for r in full['metadata'] if f'/{drive}/image_02/timestamps.txt' in r['name'] and '/raw/' in r['name']]
        if full['drives'].get(drive,{}).get('RGB2',0):
            assert len(candidates)==1
            targets.append(dict(**row,metadata=candidates[0],expected_headers=full['drives'][drive]['RGB2']))
        else:assert not candidates
    targets.sort(key=lambda r:r['metadata']['offset_data'])
    started=time.monotonic();finished=set();counts={};deadline=started+600.;limited=False
    source_paths=[Path(__file__),root/'scripts/kitti_semantic_pair_offsets_20261125_v2.py',Path(original.__file__)]
    amendment=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in source_paths},
        source_inventory_sha256=sha(full_path),official_mapping_sha256=sha(mapping_path),
        operation='seek to previously certified image02 timestamps tar headers, parse target directory headers and require counts equal full prior inventory',
        source_data_unchanged=True,maximum_header_wall_s=600,model_calls=0,GPU_calls=0,
        original_partial_preserved=True)
    original.write(report/'semantic_pair_offset_amendment_v2.json',amendment)
    tar_open=tarfile.open
    class TargetHeaders:
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def __iter__(self):
            nonlocal limited
            for target in targets:
                if time.monotonic()>deadline:limited=True;return
                meta=target['metadata'];prefix=meta['name'].rsplit('/',1)[0]+'/'
                with archive.open('rb') as file:
                    file.seek(meta['offset_data']-512)
                    with tar_open(fileobj=file,mode='r:') as reader:
                        first=next(iter(reader))
                        if first.name!=meta['name'] or first.size!=meta['bytes'] or first.offset_data!=meta['offset_data']:
                            raise AssertionError('certified_target_header_mismatch')
                        count=0
                        for member in reader:
                            if member.name==first.name:continue
                            if not member.name.startswith(prefix):break
                            if member.isfile() and '/data/' in member.name and member.name.endswith('.png'):
                                count+=1;counts[target['sequence']]=count
                                yield member
                            if time.monotonic()>deadline:limited=True;return
                if count!=target['expected_headers']:raise AssertionError('target_directory_not_complete_or_contiguous')
                finished.add(target['sequence'])
                progress=dict(utc=datetime.now(timezone.utc).isoformat(),finished=sorted(finished),headers_seen=counts,
                    elapsed_s=time.monotonic()-started)
                with (report/'semantic_pair_offset_progress_v2.json').open('w') as stream:json.dump(progress,stream,indent=2)
                print(json.dumps(progress),flush=True)
    def selected_open(*args,**kwargs):
        path=args[0] if args else kwargs.get('name')
        if path is not None and Path(path).resolve()==archive.resolve():return TargetHeaders()
        return tar_open(*args,**kwargs)
    original_write=original.write
    def write(path,value):
        if Path(path).name=='semantic_RGB_LiDAR_pose_pair_inventory_v1.json':
            path=report/'semantic_RGB_LiDAR_pose_pair_inventory_v2.json'
            value.update(complete=not limited and len(finished)==len(targets),
                header_strategy='prior certified directory offsets; all parsed header counts agree before declaring directory complete',
                completed_RGB_sequences=sorted(finished),header_counts_seen=counts,
                bounded_partial=limited,offset_amendment_sha256=sha(report/'semantic_pair_offset_amendment_v2.json'))
            for row in value['matrix']:
                if row['sequence'] in {r['sequence'] for r in targets} and row['sequence'] not in finished:
                    row.update(status='RGB_header_certification_pending',common_complete=False,
                        missing_RGB_not_inferred_from_partial_scan=True)
        return original_write(path,value)
    tarfile.open=selected_open;original.write=write
    try:original.main()
    finally:tarfile.open=tar_open;original.write=original_write


if __name__=='__main__':main()
