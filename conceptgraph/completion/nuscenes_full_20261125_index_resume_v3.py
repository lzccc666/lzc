"""复用旧deadline后已正常CRC/export的索引；不重建C索引、不覆盖失败。"""
from __future__ import annotations
import argparse,gzip,json,os,resource,tarfile,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125 as base


def resume(number,pid,deadline):
    archive=base.ARCHIVES/f'v1.0-trainval{number:02d}_blobs.tgz';name=f'archive{number:02d}';index=base.DATA/'archives_v2'/(name+'.gzidx');oldlog=base.REPORT/(name+'_v2.log')
    process=Path(f'/proc/{pid}/stat')
    assert not process.exists() or process.read_text().split()[2]=='Z','predecessor still alive; do not read unfinished index'
    owner=json.loads((base.DATA/'phaseA_resource_coordination_v3.json').read_text());assert owner['existing_PIDs'][str(number)]==pid
    text=oldlog.read_text();oldsource=base.ROOT/'conceptgraph/completion/nuscenes_full_20261125_index_v2.py';lines=oldsource.read_text().splitlines()
    export_line=next(i for i,line in enumerate(lines,1) if 'source.build_full_index();source.export_index' in line)
    after_export_timeout=(f'line {export_line}, in index' in text and 'base.guard(deadline)' in text and 'TimeoutError: phaseA deadline' in text)
    after_members_timeout=('"stage": "seek_index_complete"' in text and 'TimeoutError: phaseA deadline' in text)
    assert after_export_timeout or after_members_timeout,'not an authenticated post-CRC/export deadline failure'
    frozen=json.loads((base.DATA/'index_technical_revision_v2.json').read_text())
    for p,h in frozen['source_sha256'].items():assert base.sha(p)==h,p
    expected=next(a for a in json.loads((base.DATA/'manifest.json').read_text())['archives'] if Path(a['path']).name==archive.name);stat=archive.stat()
    assert stat.st_size==expected['bytes'] and stat.st_mtime_ns==expected['mtime_ns']
    assert index.exists() and index.stat().st_size>35
    destination=base.DATA/'archives_v3';destination.mkdir(exist_ok=True);output=destination/(name+'_members.jsonl.gz')
    prereg=dict(created_utc=base.now(),archive=number,predecessor_PID=pid,predecessor_log=str(oldlog),predecessor_log_sha256=base.sha(oldlog),
        original_index_source_sha256=base.sha(oldsource),post_export_deadline_trace_verified=True,original_CRC_build_normal_return_inferred_from_authenticated_export_control_flow=True,
        reused_index=str(index),reused_index_sha256=base.sha(index),reused_index_bytes=index.stat().st_size,
        source_archive=str(archive),source_archive_stat=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),source_sha256={str(Path(__file__).resolve()):base.sha(__file__),str(base.ROOT/'scripts/nuscenes_full_20261125_index_resume_v3.py'):base.sha(base.ROOT/'scripts/nuscenes_full_20261125_index_resume_v3.py')},
        resource_only=True,deadline_unix=deadline,old_prefix_preserved=True,new_CRC_pass=False,CRC_evidence='predecessor fixed skip_crc_check=False build_full_index returned before export and guard; original traceback and index frozen',
        procedure='validate full index import/end; full r| member scan/EOF plus compressed-source SHA; never claim import reranCRC',GPU_calls=0)
    base.dump(destination/(name+'_resume_preregistration.json'),prereg)
    run=Path(json.loads((base.DATA/'manifest.json').read_text())['run'])
    for p in [Path(__file__).resolve(),base.ROOT/'scripts/nuscenes_full_20261125_index_resume_v3.py']:
        q=run/'source_snapshot'/p.relative_to(base.ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():assert q.read_bytes()==p.read_bytes()
        else:q.write_bytes(p.read_bytes())
    ig=base.indexed_module();counts=Counter();started=time.monotonic();last=started
    with ThreadPoolExecutor(max_workers=1) as pool:
        digest=pool.submit(base.sha,archive)
        with ig.IndexedGzipFile(filename=str(archive),index_file=str(index),drop_handles=False,readbuf_size=2**20) as stream:
            end=stream.seek(0,os.SEEK_END);assert stream.read(1)==b'' and end>archive.stat().st_size;stream.seek(0)
            with gzip.open(output,'xt',compresslevel=6) as out,tarfile.open(fileobj=stream,mode='r|',bufsize=2**20) as tf:
                for member in tf:
                    counts['tar_headers']+=1
                    if member.isfile():
                        clean=member.name.removeprefix('./')
                        if (clean.startswith('samples/CAM_') and clean.endswith('.jpg')) or ('/LIDAR_TOP/' in clean and clean.endswith('.bin')):
                            out.write(json.dumps([clean,member.offset_data,member.size],separators=(',',':'))+'\n');counts['retained_members']+=1;counts[clean.split('/')[1]]+=1
                    tf.members.clear()
                    if time.monotonic()-last>30:
                        base.guard(deadline);last=time.monotonic();print(json.dumps(dict(archive=number,stage='resume_tar_members',counts=dict(counts),uncompressed_position=stream.tell(),elapsed_s=last-started,utc=base.now())),flush=True)
            while stream.read(2**20):pass
        fingerprint=digest.result()
    assert stat.st_size==archive.stat().st_size and stat.st_mtime_ns==archive.stat().st_mtime_ns and base.sha(index)==prereg['reused_index_sha256']
    result=dict(complete=True,technical_revision='resume_exported_crc_index_v3',archive=str(archive),archive_bytes=stat.st_size,archive_mtime_ns=stat.st_mtime_ns,archive_sha256=fingerprint,
        members=str(output),members_sha256=base.sha(output),seek_index=str(index),seek_index_sha256=base.sha(index),seek_index_bytes=index.stat().st_size,spacing=base.SPACING,
        counts=dict(counts),elapsed_s=time.monotonic()-started,ended_utc=base.now(),gzip_CRC_validation=True,new_CRC_validation=False,
        CRC_validation_provenance=str(destination/(name+'_resume_preregistration.json')),imported_index_CRC_disabled=True,full_import_end_bytes=end,
        maximum_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_calls=0)
    base.dump(base.DATA/'archives'/(name+'_complete.json'),result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=int,required=True);p.add_argument('--predecessor-pid',type=int,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();resume(a.archive,a.predecessor_pid,a.deadline)

if __name__=='__main__':main()
