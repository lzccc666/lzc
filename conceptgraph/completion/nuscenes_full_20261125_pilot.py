"""固定8场景源读取资源小试；不运行模型、不查看性能或调整方法。"""
from __future__ import annotations
from collections import Counter
import gzip,hashlib,json,shutil,time
from pathlib import Path
from PIL import Image
from conceptgraph.completion import nuscenes_full_20261125 as base

TMP=base.ROOT/'dataset/processed/nuscenes_full_20261125_tmp'


def main():
    done=json.loads((base.DATA/'archives/archive01_complete.json').read_text())
    members={name:(offset,size) for name,offset,size in base.rows(done['members'])}
    manifest=json.loads((base.DATA/'manifest.json').read_text());selected=[]
    for scene in manifest['scenes']:
        req=next(base.rows(scene['request']))
        if req['sensors'] and all(v['filename'] in members for v in req['sensors'].values()):
            selected.append((scene,req))
            if len(selected)==8:break
    assert len(selected)==8
    prereg=dict(created_utc=base.now(),selection='first8lexicographic scenes with all requested original members in archive01; no score access',
        scenes=[s for s,_ in selected],source_archive_complete_sha256=base.sha(base.DATA/'archives/archive01_complete.json'),
        source_sha256=base.sha(__file__),GPU_calls=0,maximum_single_scene_temporary_bytes=512*2**20,
        temporary_root=str(TMP),cleanup='only newly created own scene temp after all source hashes and JPEG/bin validation saved',
        no_models_or_methods_run=True)
    base.dump(base.DATA/'resource_pilot_preregistration.json',prereg)
    ig=base.indexed_module();started=time.monotonic();stats=[];TMP.mkdir(exist_ok=True)
    with ig.IndexedGzipFile(filename=done['archive'],index_file=done['seek_index'],drop_handles=False) as source:
        for scene,req in selected:
            base.guard(manifest['deadline_unix']);folder=TMP/scene['scene'];folder.mkdir(exist_ok=False)
            total=sum(members[v['filename']][1] for v in req['sensors'].values())
            if total>512*2**20:raise OSError('single scene source temporary exceeds512MiB; preserve source request')
            tick=time.monotonic();counts=Counter();evidence=base.DATA/(scene['scene']+'_resource_pilot_members.jsonl.gz')
            with gzip.open(evidence,'xt',compresslevel=6) as out:
                # Sorted offsets limit unnecessary backward decompression.
                for token,sensor in sorted(req['sensors'].items(),key=lambda p:members[p[1]['filename']][0]):
                    name=sensor['filename'];offset,size=members[name];source.seek(offset);raw=source.read(size)
                    assert len(raw)==size
                    path=folder/name;assert folder.resolve() in path.resolve().parents
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as f:f.write(raw)
                    fingerprint=hashlib.sha256(raw).hexdigest();assert base.sha(path)==fingerprint
                    record=dict(token=token,source=name,archive=done['archive'],archive_sha256=done['archive_sha256'],offset=offset,bytes=size,sha256=fingerprint)
                    if name.endswith('.jpg'):
                        with Image.open(path) as image:
                            assert image.size==(sensor['width'],sensor['height']);image.verify()
                        counts['JPEGs']+=1
                    else:
                        assert size%20==0;counts['LiDAR_scans']+=1;record['points']=size//20
                    counts['bytes']+=size;out.write(json.dumps(record,separators=(',',':'))+'\n')
            record=dict(scene=scene['scene'],counts=dict(counts),elapsed_s=time.monotonic()-tick,
                evidence=str(evidence),evidence_sha256=base.sha(evidence),temporary_path=str(folder),cleanup_authorized_by_scope=True)
            assert folder.parent.resolve()==TMP.resolve() and counts['bytes']==total
            shutil.rmtree(folder);record['temporary_removed_after_verification']=not folder.exists();stats.append(record)
            base.dump(base.REPORT/'resource_pilot_progress.json',dict(completed=len(stats),scenes=stats),False)
            print(json.dumps(record),flush=True)
    result=dict(passed=True,elapsed_s=time.monotonic()-started,scenes=stats,scene_count=8,GPU_calls=0,
        peak_scene_temporary_bytes=max(s['counts']['bytes'] for s in stats),all_temporary_cleaned=all(s['temporary_removed_after_verification'] for s in stats),
        configuration_or_scores_accessed=False,source_archive_unchanged=base.ARCHIVES.joinpath('v1.0-trainval01_blobs.tgz').stat().st_size==done['archive_bytes'])
    base.dump(base.REPORT/'resource_pilot_summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
