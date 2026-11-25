"""已授权全量执行调度：认证与GPU释放双门、8scene资源试、全850与隔离评分。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,signal,subprocess,time
from conceptgraph.completion import nuscenes_full_20261125 as source
NOTICE=source.DATA/'KITTI_GPU_released_notice.json'
PYTHON='/root/miniconda3/envs/conceptgraph/bin/python'


def step(name,argv):
    path=source.REPORT/(name+'.log');env=dict(os.environ,LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',PYTHONDONTWRITEBYTECODE='1')
    with path.open('x') as log:
        process=subprocess.run([PYTHON,'-u',*argv],cwd=source.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'{name} failed; all source/log/output prefixes retained at {path}')


def main():
    old_PID=21953
    old_path=Path(f'/proc/{old_PID}/cmdline')
    assert old_path.exists() and b'nuscenes_full_20261125_execute.py' in old_path.read_bytes()
    assert not (source.DATA/'source_certification.json').exists(),'old controller might have left wait stage; inspect before takeover'
    source.dump(source.DATA/'execution_controller_takeover_v2.json',dict(created_utc=source.now(),old_PID=old_PID,signal='SIGTERM',old_source_preserved=True,reason='add authorized7th sharedscan auxiliary without changing65 frozen6condition sources',old_GPU_calls=0))
    os.kill(old_PID,signal.SIGTERM)
    source.dump(source.DATA/'execution_controller_v2.json',dict(created_utc=source.now(),source_sha256=source.sha(__file__),
        authorization='root full850existing-data execution; source8 and complete source certification, same prediction regression and explicitKITTI release beforefullrun',
        GPU_max_occupied_s=12*3600,nu_persistent_bytes=int(4.5*2**30),temporary_bytes=512*2**20,minimum_free_bytes=20*2**30,
        stages=['wait_all10_source_certification_and_KITTI_explicit_release','freeze_sources_configs_weights_roles','fixed8_GPU_resource_pilot_all7conditions','resource_budget_gate','all850same_single_configuration','evaluation_only_after_all_predictions','predeclared_log_statistics'],
        no_parameter_search=True,scope_never_reduced=True,no_source_or_old_data_deleted=True))
    while not (NOTICE.exists() and (source.DATA/'source_certification.json').exists()):
        source.guard(1795613520)
        source.dump(source.REPORT/'execution_wait.json',dict(utc=source.now(),waiting_source_certification=not (source.DATA/'source_certification.json').exists(),waiting_explicit_GPU_release=not NOTICE.exists(),GPU_calls=0),False)
        time.sleep(20)
    assert json.loads(NOTICE.read_text())['explicit_release']
    contract=source.ROOT/'outputs/reports/local_scope_audit_20261125/nuscenes_CPU_contract_independent_v1.json'
    assert json.loads(contract.read_text())['passed']
    contract7=source.ROOT/'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_sharedscan_contract_v1/summary.json'
    assert json.loads(contract7.read_text())['passed']
    step('phaseB_prepare',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan.py'),'prepare'])
    step('GPU_resource_pilot',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan.py'),'run','--resource-pilot-only','--kitti-release-reference',str(NOTICE)])
    pilot=json.loads((source.REPORT/'GPU_resource_pilot_summary.json').read_text());assert pilot['passed']
    main_run=source.ROOT/'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_main_v1'
    pilot_scenes=json.loads((source.DATA/'resource_pilot_preregistration.json').read_text())['scenes'];auxiliary=[];output_bytes=0
    for scene in pilot_scenes:
        folder=main_run/scene['scene'];receipt=json.loads((folder/'complete.json').read_text());assert receipt['auxiliary_sharedscan']['complete']
        auxiliary.append(receipt['auxiliary_sharedscan']);output_bytes+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    from conceptgraph.completion.nuscenes_full_20261125_run import footprint
    projected_total=footprint()-output_bytes+output_bytes/sum(s['frames'] for s in pilot_scenes)*34149+512*2**20
    source.dump(source.REPORT/'GPU_resource_auxiliary_summary.json',dict(complete=True,auxiliary_scenes=8,auxiliary_frames=sum(r['counts']['frames'] for r in auxiliary),auxiliary_CPU_wall_s=sum(r['CPU_wall_s'] for r in auxiliary),
        projected_total_persistent_bytes_with512MiB_evaluation_reserve=projected_total,persistent_cap_bytes=int(4.5*2**30),extra_GPU_calls=0))
    projection=pilot['projected_full_GPU_occupied_hours'];budget_ok=projection<=12 and projected_total<=4.5*2**30
    source.dump(source.REPORT/'GPU_resource_budget_gate.json',dict(utc=source.now(),predicted_full_GPU_occupied_hours=projection,predicted_persistent_bytes=projected_total,budget_hours=12,
        within_budget=budget_ok,all_scenes_retained=True,action='proceed_fixed_configuration' if budget_ok else 'hold_before_large_run_for_resource_budget_amendment_only; no method/scene change'))
    if not budget_ok:raise RuntimeError('actual fixed8scene7condition profile exceeds approved GPU/storage budget; report need, preserve all850 scope')
    step('main_full850',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan.py'),'run','--kitti-release-reference',str(NOTICE)])
    step('main_evaluate',[str(source.ROOT/'scripts/nuscenes_full_20261125_evaluate.py'),'evaluate'])
    step('main_statistics',[str(source.ROOT/'scripts/nuscenes_full_20261125_statistics.py'),'run'])
    step('sharedscan_evaluate',[str(source.ROOT/'scripts/nuscenes_full_20261125_sharedscan_evaluate.py')])
    source.dump(source.REPORT/'execution_complete.json',dict(complete=True,utc=source.now(),all850=True,source_sha256=source.sha(__file__),no_parameter_search=True,GT_in_method=False))

if __name__=='__main__':main()
