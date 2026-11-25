"""单reader、单RAM槽原始字节预取。无模型/阈值/几何/关联变化。"""
from __future__ import annotations
import hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125_resource_runtime_v6 as runtime
pilot=runtime.pilot;base=runtime.base;source=runtime.source
RAM_CAP=128*2**20
RAW_CAP=RAM_CAP-2*2**20

def allocation(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict):size+=sum(allocation(k,seen)+allocation(v,seen) for k,v in value.items())
    elif isinstance(value,(list,tuple)):size+=sum(allocation(v,seen) for v in value)
    return size

class PrefetchStore(pilot._ORIGINAL_STORE):
    def __init__(self,scenes,run,partial=False):
        if partial:
            original=pilot.PilotSourceStore();self.__dict__.update(original.__dict__)
        else:super().__init__()
        self.scenes=scenes;self.run=Path(run);self.by_name={s['scene']:s for s in scenes};self.order=[s['scene'] for s in scenes]
        self.reader=ThreadPoolExecutor(max_workers=1,thread_name_prefix='certified_sensor_reader');self.future=None;self.scheduled=None;self.peak_RAM=0;self.measurements=[]
    def schedule(self,name):
        if self.scheduled is not None:
            assert self.scheduled==name,'single raw slot cannot be overwritten';return
        self.scheduled=name;request=next(source.rows(self.by_name[name]['request']));total=sum(r['bytes'] for r in self.by_scene[name].values())
        if total<=RAW_CAP:self.future=self.reader.submit(self.read_bytes,request)
        else:self.future=None
    def wait_ready(self):
        if self.future is not None:self.future.result()
    def read_bytes(self,request):
        tick=time.monotonic();values={}
        for token,b in sorted(self.by_scene[request['scene']].items(),key=lambda p:(p[1]['archive'],p[1]['offset'])):
            n=b['archive'];a=self.archives[n]
            if n not in self.handles:
                assert Path(a['archive']).stat().st_size==a['archive_bytes'] and Path(a['archive']).stat().st_mtime_ns==a['archive_mtime_ns']
                self.handles[n]=self.ig.IndexedGzipFile(filename=a['archive'],index_file=a['seek_index'],drop_handles=False)
            handle=self.handles[n];handle.seek(b['offset']);raw=handle.read(b['bytes']);assert len(raw)==b['bytes']
            values[token]=(b,raw,hashlib.sha256(raw).hexdigest())
        memory=allocation(values);assert memory<=RAM_CAP,'raw RAM slot128MiB guard'
        self.peak_RAM=max(self.peak_RAM,memory)
        return dict(values=values,source_read_wall_s=time.monotonic()-tick,actual_slot_bytes=memory,source_scene=request['scene'])
    def extract(self,request,attempt='initial'):
        if self.scheduled is None:self.schedule(request['scene'])
        assert self.scheduled==request['scene'];tick=time.monotonic()
        if self.future is None:
            # Oversize scene never enters RAM slot; same original sequential
            # reader thread streams only after the original512MiB disk guard.
            folder,receipts=self.reader.submit(pilot._ORIGINAL_STORE.extract,self,request,attempt).result();measurement=dict(scene=request['scene'],mode='original_streaming_fallback',source_read_and_materialize_s=time.monotonic()-tick,RAM_payload_bytes=0)
        else:
            future=self.future;result=future.result();folder=base.TMP/(request['scene']+'__'+attempt);folder.mkdir(parents=True,exist_ok=False);receipts={}
            for token,(binding,raw,fingerprint) in result['values'].items():
                assert hashlib.sha256(raw).hexdigest()==fingerprint;target=folder/binding['filename'];assert folder.resolve() in target.resolve().parents
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(raw)
                assert source.sha(target)==fingerprint
                receipts[token]=dict(**binding,sha256=fingerprint,path=str(target),archive_sha256=self.archives[binding['archive']]['archive_sha256'])
            measurement=dict(scene=request['scene'],mode='single_RAM_slot',source_reader_wall_s=result['source_read_wall_s'],wait_and_materialize_s=time.monotonic()-tick,RAM_payload_bytes=result['actual_slot_bytes'],raw_sensor_bytes=sum(v['bytes'] for v in receipts.values()))
            del result;del future
        self.future=None;self.scheduled=None;self.measurements.append(measurement)
        directory=self.run/'source_prefetch_receipts';directory.mkdir(exist_ok=True)
        source.dump(directory/(request['scene']+'_'+attempt+'.json'),dict(measurement,UTC=source.now(),queue_capacity=1,payload_RAM_cap_bytes=RAM_CAP,temporary_disk_guard_bytes=512*2**20,GT_in_method=False,source_hashes_verified=True))
        return folder,receipts
    def next_during_GPU(self,request):
        index=self.order.index(request['scene'])
        if index+1<len(self.order):self.schedule(self.order[index+1])
    def close(self):
        self.reader.shutdown(wait=True)
        for handle in self.handles.values():handle.close()
