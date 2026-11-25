"""主实验闭合后执行同任务方法/消融，含完整历史维护移除；不覆盖既有回合。"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    registry = R/'carla_official/remaining_comparison_registration_024.json'
    plan = json.loads(registry.read_text())
    for p, sha in plan['source_sha256'].items():
        assert digest(p) == sha, p
    for p, sha in plan['registration_sha256'].items():
        assert digest(p) == sha, p
    closed = json.loads((R/'json/expansion_jobs_011_primary/report.json').read_text())
    assert closed['all_execution_and_audits_closed'] and closed['actual_launched_episodes'] == 150
    assert all(j['exit_code'] == 0 and len(j['audits']) == 2
               and all(a['exit_code'] == 0 for a in j['audits']) for j in closed['jobs'])
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'CUDA_VISIBLE_DEVICES': '0',
           'PYTHONPATH': '/autodl-fs/data/third_party/carla_0.9.16/python_site:/root/autodl-tmp/lizichong'}
    cpu = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    root = R/'json/remaining_comparison_jobs_024'
    root.mkdir(exist_ok=False)
    save(root/'started.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         PID=os.getpid(), registered_jobs=210, registration_sha256=digest(registry), goal_complete=False))
    with (root/'spawn_preflight.log').open('x') as stream:
        process = subprocess.run([sys.executable, str(R/'source/preflight_carla_remaining_014.py')],
                                 env=cpu, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    assert process.returncode == 0, '新服务器同步布局检查失败，未开始方法/消融'
    results = []
    for phase in ['methods', 'ablations', 'history_maintenance']:
        directory = R/('json/expansion_jobs_012_history_maintenance' if phase == 'history_maintenance'
                       else 'json/expansion_jobs_011_'+phase)
        directory.mkdir(exist_ok=False)
        jobs = plan['execution_phases'][phase]
        save(directory/'started.json', dict(phase=phase, planned_jobs=len(jobs),
             registration_sha256=digest(registry), goal_complete=False))
        phase_results = []
        for number, job in enumerate(jobs, 1):
            history = phase == 'history_maintenance'
            bank = json.loads((R/('carla_official/expansion_registration_012.json' if history
                                 else 'carla_official/expansion_registration_011.json')).read_text())
            policy = json.loads((R/'json/resource_policy_002.json').read_text())
            used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
            if (used+512*2**20 >= policy['new_project_artifact_budget_bytes']
                    or shutil.disk_usage(R).free < 20*2**30
                    or shutil.disk_usage(Path(bank['bulk_sensor_root']).parent).free < 20*2**30):
                save(root/'stopped_storage.json', dict(next_phase=phase, next_job_number=number,
                     next_job=job, project_bytes=used, closed_jobs=len(results), goal_complete=False))
                raise RuntimeError('资源保护停止；既有结果不移动/删除')
            output = 'expansion_history_navigation_012' if history else 'expansion_navigation_010'
            dest = R/'carla_official'/output/job['episode']/f"seed{job['seed']}"/job['mode']/job['condition']
            assert not dest.exists(), '禁止覆盖已有回合：'+str(dest)
            runtime = 'carla_history_episode_012.py' if history else 'carla_expansion_episode_010.py'
            command = [sys.executable, str(R/'source'/runtime)] + [
                x for k in ['episode', 'seed', 'mode', 'condition'] for x in ['--'+k, str(job[k])]]
            log = directory/f'episode_{number:03d}.log'
            began = time.monotonic()
            with log.open('x') as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         check=False, timeout=bank['maximum_wall_s']+360)
            row = dict(number=number, **job, exit_code=process.returncode,
                       wall_s=time.monotonic()-began, episode_directory=str(dest),
                       actual_physical_world_started=(dest/'release.json').exists(),
                       log_sha256=digest(log), audits=[])
            auditors = ['audit_carla_expansion_009.py', 'audit_carla_guarded_execution_002.py']
            if history:
                auditors.append('audit_carla_history_execution_023.py')
            if (dest/'summary.json').exists():
                for name in auditors:
                    logfile = directory/f'episode_{number:03d}_{name}.log'
                    with logfile.open('x') as stream:
                        checked = subprocess.run([sys.executable, str(R/'source'/name), '--episode',
                            str(dest.relative_to(R/'carla_official'))], env=cpu, stdout=stream,
                            stderr=subprocess.STDOUT, check=False, timeout=1800)
                    row['audits'].append(dict(source=name, exit_code=checked.returncode,
                                              log_sha256=digest(logfile)))
            save(directory/f'job_{number:03d}.json', row)
            phase_results.append(row)
            results.append(dict(phase=phase, **row))
            print(json.dumps(dict(phase=phase, **row), ensure_ascii=False), flush=True)
            if row['exit_code'] or len(row['audits']) != len(auditors) or any(a['exit_code'] for a in row['audits']):
                save(root/'stopped_after_execution_or_audit_error.json', dict(job=row,
                     phase=phase, next_job_number=number+1, goal_complete=False))
                raise RuntimeError('执行/独立审计异常；普通任务超时仍作为真实负结果保留后继续')
        save(directory/'report.json', dict(phase=phase, planned_jobs=len(jobs),
             actual_launched_episodes=len(phase_results), all_execution_and_audits_closed=True,
             jobs=phase_results, goal_complete=False))
    save(root/'report.json', dict(actual_launched_episodes=len(results),
         all_execution_and_audits_closed=True, jobs=results,
         actual_arrival_requires_independent_reports=True, goal_complete=False))


if __name__ == '__main__':
    main()
