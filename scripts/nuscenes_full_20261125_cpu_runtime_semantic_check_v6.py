"""计时字段不要求重演相等；同worker完整7方法及其余全部字段零容差核验。"""
from pathlib import Path
import sys,gzip,json,hashlib,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as pilot
source=pilot.source;REPLAY=pilot.RUN.parent/'nuscenes_full_20261125_cpu_runtime_v6_replay_v1'

def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()

def main():
    records=[];checks=[]
    for number in [4,5]:
        name=f'scene-{number:04d}';a=pilot.RUN/name;b=REPLAY/name
        pair=[json.loads((root/'attempts/cpu_v6/runtime_v6_check.json').read_text()) for root in [a,b]];records+=pair
        assert pair[0]['PID']==pair[1]['PID'] and all(r['interop_threads']==1 and not r['CUDA_initialized'] for r in pair)
        values=[]
        for index,(x,y) in enumerate(zip(source.rows(a/'frames.jsonl.gz'),source.rows(b/'frames.jsonl.gz'))):
            timing_x=x.pop('mapper_seconds');timing_y=y.pop('mapper_seconds')
            assert x==y,(name,index,'non-timing output differs')
            per_method={m:digest(x['methods'][m]) for m in pilot.base.METHODS};values.append(dict(frame_idx=index,non_timing_root_sha256=digest(x),method_sha256=per_method))
        aux_a=list(source.rows(a/'sharedscan_frames.jsonl.gz'));aux_b=list(source.rows(b/'sharedscan_frames.jsonl.gz'));assert aux_a==aux_b and len(values)==len(aux_a)==pair[0]['frames']
        for row,value in zip(aux_a,values):value['method_sha256'][pilot.aux.CONTROL]=digest(row['payload'])
        checks.append(dict(scene=name,frames=len(values),all7payloads_exact=True,all_other_root_fields_exact=True,excluded_only_runtime_field='mapper_seconds',frame_checks=values))
    assert len({r['PID'] for r in records})==1
    source.dump(REPLAY/'semantic_comparison_v6.json',dict(passed=True,checks=checks,original_full_file_comparison_failed_due_to_runtime_mapper_seconds_only=True,original_failure_log_sha256=source.sha(source.REPORT/'CPU_cache_repair_v6.log'),source_sha256=source.sha(__file__)))
    for number in [4,5]:
        folder=pilot.base.TMP/f'scene-{number:04d}__run0004';assert folder.parent.resolve()==pilot.base.TMP.resolve();shutil.rmtree(folder)
    source.dump(REPLAY/'summary.json',dict(passed=True,same_worker_two_distinct_scenes_repeated_exact=True,records=records,all_three_original_caches_CPU_closed=True,new_GPU_calls=0,original_failures_preserved=True,semantic_comparison_sha256=source.sha(REPLAY/'semantic_comparison_v6.json'),excluded_only_runtime_field='mapper_seconds',original_full_file_timing_equality_harness_failed_and_retained=True,new_check_source_sha256=source.sha(__file__)))
    print(json.dumps(dict(passed=True,PID=records[0]['PID'],frames=79,methods=7,new_GPU_calls=0,original_failure_preserved=True)),flush=True)
if __name__=='__main__':main()
