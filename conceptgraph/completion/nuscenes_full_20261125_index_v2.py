"""资源修订：顺序构造seek索引+顺序tar枚举；保留v1随机seek慢前缀。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip,json,resource,tarfile,time
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125 as base

DATA=base.DATA;REPORT=base.REPORT
REV=DATA/'archives_v2'


def freeze():
    REV.mkdir(exist_ok=True)
    path=DATA/'index_technical_revision_v2.json'
    sources=[Path(__file__).resolve(),base.ROOT/'scripts/nuscenes_full_20261125_index_v2.py']
    hashes={str(p):base.sha(p) for p in sources}
    if path.exists(): assert json.loads(path.read_text())['source_sha256']==hashes
    else:
        record=dict(created_utc=base.now(),source_sha256=hashes,resource_only=True,
            reason='tarfile r: seeks for every member; avoid redecoding gzip checkpoint for each small header',
            procedure='one sequential C gzip full-index/CRC pass and one sequential tar r| membership pass; same sources and requests',
            v1_prefix_preserved=str(DATA/'archives/archive01_members.jsonl.gz'),spacing=base.SPACING,parallel_archives=2,
            index_budget_bytes=2**30,metadata_budget_bytes=128*2**20,GPU_calls=0,deadline_unix=1795609920)
        base.dump(path,record)
        run=Path(json.loads((DATA/'manifest.json').read_text())['run'])
        for p in sources:
            q=run/'source_snapshot'/p.relative_to(base.ROOT);q.parent.mkdir(parents=True,exist_ok=True)
            if q.exists():raise FileExistsError(q)
            q.write_bytes(p.read_bytes())
        manifest=json.loads((DATA/'manifest.json').read_text())
        overlay={s['scene']:dict(original_role=s['role']['original_role'],access_role='previous_mapping_seen_label_access_pending',
            reason='existing797frame20scene mapping/converted cache; do not treat as new unseen') for s in manifest['scenes'] if s['role']['original_role']=='sealed_test'}
        base.dump(DATA/'scene_role_access_overlay_v2.json',dict(created_utc=base.now(),scenes=overlay,source_manifest_sha256=base.sha(DATA/'manifest.json')))


def index(number,deadline):
    freeze();ig=base.indexed_module();archive=base.ARCHIVES/f'v1.0-trainval{number:02d}_blobs.tgz';name=f'archive{number:02d}'
    done=DATA/'archives'/(name+'_complete.json')
    if done.exists():
        result=json.loads(done.read_text());assert base.sha(result['members'])==result['members_sha256'];assert base.sha(result['seek_index'])==result['seek_index_sha256'];return result
    outpath=REV/(name+'_members.jsonl.gz');indexpath=REV/(name+'.gzidx')
    if outpath.exists() or indexpath.exists():raise FileExistsError('technical v2 prefix retained; refuse overwrite')
    started=time.monotonic();stat=archive.stat();counter=Counter();last=started
    base.guard(deadline)
    with ThreadPoolExecutor(max_workers=1) as pool:
        digest=pool.submit(base.sha,archive)
        print(json.dumps(dict(archive=number,stage='sequential_full_index_start',utc=base.now())),flush=True)
        with ig.IndexedGzipFile(filename=str(archive),spacing=base.SPACING,window_size=32768,readbuf_size=2**20,drop_handles=False,skip_crc_check=False) as source:
            source.build_full_index();source.export_index(filename=str(indexpath));base.guard(deadline)
            print(json.dumps(dict(archive=number,stage='seek_index_complete',index_bytes=indexpath.stat().st_size,elapsed_s=time.monotonic()-started,utc=base.now())),flush=True)
            source.seek(0)
            with gzip.open(outpath,'xt',compresslevel=6) as output,tarfile.open(fileobj=source,mode='r|',bufsize=2**20) as tf:
                for member in tf:
                    counter['tar_headers']+=1
                    if member.isfile():
                        clean=member.name.removeprefix('./')
                        if (clean.startswith('samples/CAM_') and clean.endswith('.jpg')) or ('/LIDAR_TOP/' in clean and clean.endswith('.bin')):
                            output.write(json.dumps([clean,member.offset_data,member.size],separators=(',',':'))+'\n')
                            counter['retained_members']+=1;counter[clean.split('/')[1]]+=1
                    tf.members.clear()
                    if time.monotonic()-last>30:
                        base.guard(deadline);last=time.monotonic()
                        progress=dict(archive=number,stage='sequential_tar_members',counts=dict(counter),uncompressed_position=source.tell(),elapsed_s=time.monotonic()-started,utc=base.now())
                        base.dump(REPORT/(name+'_v2_progress.json'),progress,False);print(json.dumps(progress),flush=True)
            # Consume any final padding/footer to ensure this scan also ends normally.
            while source.read(2**20):pass
        fingerprint=digest.result()
    assert archive.stat().st_size==stat.st_size and archive.stat().st_mtime_ns==stat.st_mtime_ns
    total=sum(p.stat().st_size for p in REV.glob('*.gzidx'))
    if total>2**30:raise OSError('complete index total exceeds1GiB')
    result=dict(complete=True,technical_revision='sequential_index_and_tar_v2',archive=str(archive),archive_bytes=stat.st_size,archive_mtime_ns=stat.st_mtime_ns,archive_sha256=fingerprint,
        members=str(outpath),members_sha256=base.sha(outpath),seek_index=str(indexpath),seek_index_sha256=base.sha(indexpath),seek_index_bytes=indexpath.stat().st_size,
        spacing=base.SPACING,counts=dict(counter),elapsed_s=time.monotonic()-started,ended_utc=base.now(),gzip_CRC_validation=True,
        maximum_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_calls=0)
    base.dump(done,result);print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();index(a.archive,a.deadline)

if __name__=='__main__':main()
