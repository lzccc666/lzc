"""冻结的原厂Prius全传感器三任务开发验证，失败不换目标，逐例独立审计。"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
O = R / 'json/prius_navigation_jobs_005'


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main():
    from conceptgraph.completion.stage9 import digest
    bankfile = R / 'carla_official/prius_navigation_registration_005.json'
    bank = json.loads(bankfile.read_text())
    assert all(digest(p) == h for p, h in bank['source_sha256'].items())
    O.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0',
               PYTHONPATH='/autodl-fs/data/third_party/carla_0.9.16/python_site:/root/autodl-tmp/lizichong')
    rows = []
    for episode in bank['episodes']:
        policy = json.loads((R/'json/resource_policy_002.json').read_text())
        used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
        assert used + bank['maximum_reserve_per_episode_bytes'] < policy['new_project_artifact_budget_bytes']
        assert shutil.disk_usage(R).free >= policy['minimum_physical_free_GiB']*2**30
        name = episode['episode']
        log = O / f'{name}.log'
        tick = time.monotonic()
        with log.open('x') as stream:
            result = subprocess.run([
                sys.executable, str(R/'source/carla_prius_navigation_episode_005.py'),
                '--episode', name, '--condition', 'full_rule_geometry',
            ], stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
        row = dict(episode=name, exit_code=result.returncode, wall_s=time.monotonic()-tick,
                   log=str(log), log_sha256=digest(log))
        dest = R/'carla_official/prius_controller_navigation_005'/name/'full_rule_geometry'
        if (dest/'summary.json').exists():
            auditlog = O / f'{name}_independent_audit.log'
            with auditlog.open('x') as stream:
                audit = subprocess.run([
                    sys.executable, str(R/'source/audit_carla_arrival_003.py'), '--episode',
                    f'prius_controller_navigation_005/{name}/full_rule_geometry',
                ], stdout=stream, stderr=subprocess.STDOUT,
                    env={**env, 'CUDA_VISIBLE_DEVICES':''}, check=False)
            row['audit_exit_code'] = audit.returncode
        rows.append(row)
        save(O/f'job_{len(rows):02d}.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result.returncode:
            body = (dest/'failure.json').read_text() if (dest/'failure.json').exists() else log.read_text()
            # Preserve real prefixes and stop resource/write errors, rather than silently retry.
            if any(word in body for word in ['space', 'GiB', 'No space', 'FileExistsError']):
                break
    save(O/'report.json', dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=rows,
        actual_started_navigation_episodes=len(rows), planned_episodes=3,
        subprocesses_and_audits_complete=len(rows)==3 and all(
            x['exit_code']==0 and x.get('audit_exit_code')==0 for x in rows),
        strict_arrival_must_read_independent_reports=True, source_release_sha256=digest(bankfile),
        whole_project_sealed_test=False, goal_complete=False,
    ))
    if len(rows)!=3 or any(x['exit_code'] or x.get('audit_exit_code',1) for x in rows):
        raise RuntimeError('三任务未全部关闭：保留前缀，不授予完整运行信用')


if __name__ == '__main__':
    main()
