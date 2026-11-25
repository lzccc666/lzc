"""单reader双场景CPU真实字节合同，不调用模型。"""
from pathlib import Path
import sys,json,shutil,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion import nuscenes_full_20261125_prefetch as p
from conceptgraph.project_paths import experiment_directory
s=p.source

def main():
    run=experiment_directory('nuscenes','validation','nuscenes_full_20261125_prefetch_contract_v1',create=True);scenes=json.loads((p.pilot.DATA/'manifest.json').read_text())['scenes'][:2]
    s.dump(run/'preregistration.json',dict(created_utc=s.now(),scenes=scenes,source_sha256={str(Path(__file__).resolve()):s.sha(__file__),str(Path(p.__file__).resolve()):s.sha(p.__file__)},GPU_calls=0,GT=False,queue=1,RAM_bytes=p.RAM_CAP,physical_temp_bytes=512*2**20,selection='same first2 of frozen8 resource scenes; no quality access'))
    store=p.PrefetchStore(scenes,run,partial=True);store.schedule(scenes[0]['scene']);store.wait_ready();records=[];folders=[]
    try:
        for scene in scenes:
            req=next(s.rows(scene['request']));folder,receipts=store.extract(req,'prefetch_contract_v1');folders.append(folder)
            if scene is scenes[0]:store.next_during_GPU(req)
            old={r['token']:r for r in s.rows(p.pilot.FULL_DATA/(scene['scene']+'_resource_pilot_members.jsonl.gz'))};assert set(old)==set(receipts)
            for token,r in receipts.items():
                assert r['sha256']==old[token]['sha256'] and r['offset']==old[token]['offset'] and r['bytes']==old[token]['bytes'] and s.sha(r['path'])==r['sha256']
            assert p.base.snapshot_size(p.base.TMP)<=512*2**20
            records.append(dict(scene=scene['scene'],members=len(receipts),all_source_bytes_exact=True,member_sha256={t:r['sha256'] for t,r in receipts.items()}))
    finally:store.close()
    s.dump(run/'summary.json',dict(passed=True,records=records,peak_raw_RAM_allocation_bytes=store.peak_RAM,queue_capacity=1,new_GPU_calls=0,actual_original_sensor_bytes_match=True,physical_temp_peak_bound_bytes=512*2**20,source_read_timings=store.measurements))
    for folder in folders:
        assert folder.parent.resolve()==p.base.TMP.resolve();shutil.rmtree(folder)
    print(json.dumps(dict(passed=True,members=sum(r['members'] for r in records),peak_RAM_bytes=store.peak_RAM,GPU_calls=0)),flush=True)
if __name__=='__main__':main()
