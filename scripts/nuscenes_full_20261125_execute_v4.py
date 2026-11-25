"""资源调度v3：8-only先试，850正式仍全证书；已试8与GPU历史原样复用。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,signal,subprocess,time
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as resource
PYTHON='/root/miniconda3/envs/conceptgraph/bin/python'

def step(name,args):
    with (source.REPORT/(name+'.log')).open('x') as out:
        proc=subprocess.run([PYTHON,'-u',*args],cwd=source.ROOT,env=dict(os.environ,LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(name+' failed; prefixes retained')

def main():
    old=24761;p=Path(f'/proc/{old}/cmdline');assert p.exists() and b'nuscenes_full_20261125_execute_v3.py' in p.read_bytes()
    assert not (source.DATA/'source_certification.json').exists();os.kill(old,signal.SIGTERM)
    source.dump(source.DATA/'execution_controller_v4.json',dict(created_utc=source.now(),old_PID=old,takeover='SIGTERM waiting coordinator only; no child models/index stopped',source_sha256=source.sha(__file__),resource_only8_separate_certificate=True,full850_gate_unchanged=True,previous_controller_source_preserved=True,budgets_unchanged=True))
    while not (source.DATA/'source_certification.json').exists() or not (source.REPORT/'GPU_resource_pilot_summary.json').exists():
        source.dump(source.REPORT/'execution_wait_v3.json',dict(utc=source.now(),source_certified=(source.DATA/'source_certification.json').exists(),GPU_pilot_finished=(source.REPORT/'GPU_resource_pilot_summary.json').exists(),full850_never_reduced=True),False);time.sleep(20)
    pilot=json.loads((source.REPORT/'GPU_resource_pilot_summary.json').read_text());assert pilot['passed']
    names=[s['scene'] for s in json.loads((resource.DATA/'manifest.json').read_text())['scenes']];records=[];output_bytes=0
    for name in names:
        folder=resource.RUN/name;record=json.loads((folder/'complete.json').read_text());assert record['auxiliary_sharedscan']['complete'];records.append(record)
        output_bytes+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    frames=sum(r['counts']['frames'] for r in records)
    projected=resource.footprint()-output_bytes+output_bytes/frames*34149+512*2**20
    projection=pilot['projected_full_GPU_occupied_hours'];within=projection<=12 and projected<=4.5*2**30
    source.dump(source.REPORT/'GPU_resource_auxiliary_summary.json',dict(complete=True,auxiliary_scenes=8,frames=frames,extra_GPU_calls=0,auxiliary_CPU_wall_s=sum(r['auxiliary_sharedscan']['CPU_wall_s'] for r in records),physical_persistent_bytes=resource.footprint(),projected_total_persistent_bytes_with512MiB_evaluation_reserve=projected,all_source_indexes_included=True))
    source.dump(source.REPORT/'GPU_resource_budget_gate.json',dict(utc=source.now(),within_budget=within,predicted_full_GPU_occupied_hours=projection,predicted_persistent_bytes=projected,GPU_budget_hours=12,scope850_retained=True,resource_inputs='exact frozen8; brief source IO leases recorded, all GPU occupied time including modelpaired extra counted'))
    if not within:raise RuntimeError('actual8scene resource forecast exceeds budget; preserve850 and report resource-only amendment need')
    entry=str(source.ROOT/'scripts/nuscenes_full_20261125_resource_runtime_v2.py')
    step('phaseB_prepare_v3',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan.py'),'prepare'])
    step('phaseB_promote_resource8_v3',[entry,'promote'])
    step('main_full850_v3',[entry,'fullrun'])
    step('main_evaluate_v3',[str(source.ROOT/'scripts/nuscenes_full_20261125_evaluate.py'),'evaluate'])
    step('main_statistics_v3',[str(source.ROOT/'scripts/nuscenes_full_20261125_statistics.py'),'run'])
    step('sharedscan_evaluate_v3',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan_evaluate.py')])
    source.dump(source.REPORT/'execution_complete.json',dict(complete=True,utc=source.now(),all850=True,source_sha256=source.sha(__file__),configuration_count=1,GT_in_method=False))
if __name__=='__main__':main()
