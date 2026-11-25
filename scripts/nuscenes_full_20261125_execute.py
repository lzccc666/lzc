"""已授权全量执行调度：认证与GPU释放双门、8scene资源试、全850与隔离评分。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,subprocess,time
from conceptgraph.completion import nuscenes_full_20261125 as source
NOTICE=source.DATA/'KITTI_GPU_released_notice.json'
PYTHON='/root/miniconda3/envs/conceptgraph/bin/python'


def step(name,argv):
    path=source.REPORT/(name+'.log');env=dict(os.environ,LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',PYTHONDONTWRITEBYTECODE='1')
    with path.open('x') as log:
        process=subprocess.run([PYTHON,'-u',*argv],cwd=source.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'{name} failed; all source/log/output prefixes retained at {path}')


def main():
    source.dump(source.DATA/'execution_controller_v1.json',dict(created_utc=source.now(),source_sha256=source.sha(__file__),
        authorization='root full850existing-data execution; source8 and complete source certification, same prediction regression and explicitKITTI release beforefullrun',
        GPU_max_occupied_s=12*3600,nu_persistent_bytes=int(4.5*2**30),temporary_bytes=512*2**20,minimum_free_bytes=20*2**30,
        stages=['wait_all10_source_certification_and_KITTI_explicit_release','freeze_sources_configs_weights_roles','fixed8_GPU_resource_pilot','resource_budget_gate','all850same_single_configuration','evaluation_only_after_all_predictions','predeclared_log_statistics'],
        no_parameter_search=True,scope_never_reduced=True,no_source_or_old_data_deleted=True))
    while not (NOTICE.exists() and (source.DATA/'source_certification.json').exists()):
        source.guard(1795613520)
        source.dump(source.REPORT/'execution_wait.json',dict(utc=source.now(),waiting_source_certification=not (source.DATA/'source_certification.json').exists(),waiting_explicit_GPU_release=not NOTICE.exists(),GPU_calls=0),False)
        time.sleep(20)
    assert json.loads(NOTICE.read_text())['explicit_release']
    contract=source.ROOT/'outputs/reports/local_scope_audit_20261125/nuscenes_CPU_contract_independent_v1.json'
    assert json.loads(contract.read_text())['passed']
    step('phaseB_prepare',[str(source.ROOT/'scripts/nuscenes_full_20261125_run.py'),'prepare'])
    step('GPU_resource_pilot',[str(source.ROOT/'scripts/nuscenes_full_20261125_run.py'),'run','--resource-pilot-only','--kitti-release-reference',str(NOTICE)])
    pilot=json.loads((source.REPORT/'GPU_resource_pilot_summary.json').read_text());assert pilot['passed']
    projection=pilot['projected_full_GPU_occupied_hours'];budget_ok=projection<=12
    source.dump(source.REPORT/'GPU_resource_budget_gate.json',dict(utc=source.now(),predicted_full_GPU_occupied_hours=projection,budget_hours=12,
        within_budget=budget_ok,all_scenes_retained=True,action='proceed_fixed_configuration' if budget_ok else 'hold_before_large_run_for_resource_budget_amendment_only; no method/scene change'))
    if not budget_ok:raise RuntimeError('actual fixed8scene profile exceeds approved12hGPUoccupancy; report actual need, preserve all850 scope')
    step('main_full850',[str(source.ROOT/'scripts/nuscenes_full_20261125_run.py'),'run','--kitti-release-reference',str(NOTICE)])
    step('main_evaluate',[str(source.ROOT/'scripts/nuscenes_full_20261125_evaluate.py'),'evaluate'])
    step('main_statistics',[str(source.ROOT/'scripts/nuscenes_full_20261125_statistics.py'),'run'])
    source.dump(source.REPORT/'execution_complete.json',dict(complete=True,utc=source.now(),all850=True,source_sha256=source.sha(__file__),no_parameter_search=True,GT_in_method=False))

if __name__=='__main__':main()
