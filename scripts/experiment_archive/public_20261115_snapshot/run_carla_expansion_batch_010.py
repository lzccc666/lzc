"""执行冻结完整任务单；普通失败保留并继续，资源/执行异常停止并提供恢复位置。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['primary','methods','ablations'],default='primary');args=parser.parse_args()
    bankfile=R/'carla_official/expansion_registration_010.json';bank=json.loads(bankfile.read_text())
    assert all(digest(p)==h for p,h in bank['source_sha256'].items())
    out=R/'json'/('expansion_jobs_010_'+args.phase);out.mkdir(exist_ok=False)
    sequence=bank['execution_phases'][args.phase]
    dump(out/'started.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_jobs=len(sequence),registration_sha256=digest(bankfile),PID=os.getpid(),goal_complete=False))
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0',
        'PYTHONPATH':'/autodl-fs/data/third_party/carla_0.9.16/python_site:/root/autodl-tmp/lizichong'}
    preflight_log=out/'spawn_preflight.log'
    with preflight_log.open('x') as stream:
        preflight=subprocess.run([sys.executable,str(R/'source/preflight_carla_expansion_010.py')],env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=stream,stderr=subprocess.STDOUT,check=False,timeout=1200)
    if preflight.returncode:raise RuntimeError('生成位置/独立真值预检未通过，不启动任何图像或模型导航回合')
    results=[]
    for number,job in enumerate(sequence,1):
        policy=json.loads((R/'json/resource_policy_002.json').read_text())
        used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
        if (used+512*2**20>=policy['new_project_artifact_budget_bytes'] or shutil.disk_usage(R).free<20*2**30 or shutil.disk_usage(Path(bank['bulk_sensor_root']).parent).free<20*2**30):
            dump(out/'stopped_storage.json',dict(next_job_number=number,next_job=job,project_bytes=used,completed_jobs=len(results),reason='保留全部既有文件；达到冻结配额/剩余空间门限，不启动新回合',goal_complete=False));return
        dest=R/'carla_official/expansion_navigation_010'/job['episode']/f"seed{job['seed']}"/job['mode']/job['condition']
        if dest.exists():
            raise FileExistsError('已有同一回合，禁止覆盖或静默当新运行：'+str(dest))
        command=[sys.executable,str(R/'source/carla_expansion_episode_010.py')]+[part for k in ['episode','seed','mode','condition'] for part in ['--'+k,str(job[k])]]
        log=out/f'episode_{number:03d}.log';started=time.monotonic()
        with log.open('x') as stream:
            completed=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False,timeout=bank['maximum_wall_s']+360)
        row=dict(number=number,**job,exit_code=completed.returncode,wall_s=time.monotonic()-started,episode_directory=str(dest),log_sha256=digest(log),actual_physical_world_started=(dest/'release.json').exists(),audits=[])
        if (dest/'summary.json').exists():
            for auditor in ['audit_carla_expansion_009.py','audit_carla_guarded_execution_002.py']:
                auditlog=out/f'episode_{number:03d}_{auditor}.log'
                with auditlog.open('x') as stream:
                    audit=subprocess.run([sys.executable,str(R/'source'/auditor),'--episode',str(dest.relative_to(R/'carla_official'))],env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=stream,stderr=subprocess.STDOUT,check=False,timeout=1800)
                row['audits'].append(dict(source=auditor,exit_code=audit.returncode,log_sha256=digest(auditlog)))
        results.append(row);dump(out/f'job_{number:03d}.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
        if completed.returncode or len(row['audits'])!=2 or any(a['exit_code'] for a in row['audits']):
            dump(out/'stopped_after_execution_or_audit_error.json',dict(job=row,next_job_number=number+1,goal_complete=False))
            raise RuntimeError('真实执行/审计异常，保留前缀并诊断；任务超时/碰撞等普通负结果不触发本异常')
    dump(out/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=args.phase,planned_jobs=len(sequence),actual_launched_episodes=len(results),all_execution_and_audits_closed=True,jobs=results,success_requires_reading_independent_reports=True,whole_project_sealed_test=False,goal_complete=False))


if __name__=='__main__':main()
