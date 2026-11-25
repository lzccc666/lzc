"""运行冻结008的三种真实方法，逐项保存独立审计及所有失败。"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]
O = R/'json/guarded_complex_jobs_008'


def main():
    bankfile = R/'carla_official/guarded_complex_registration_008.json'
    bank = json.loads(bankfile.read_text())
    assert all(digest(p) == h for p, h in bank['source_sha256'].items())
    O.mkdir(exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':'0',
        'PYTHONPATH':'/autodl-fs/data/third_party/carla_0.9.16/python_site:/root/autodl-tmp/lizichong'}
    rows = []
    for task in bank['episodes']:
        for mode in bank['modes']:
            policy = json.loads((R/'json/resource_policy_002.json').read_text())
            used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
            assert used + bank['maximum_reserve_per_episode_bytes'] < policy['new_project_artifact_budget_bytes']
            assert shutil.disk_usage(R).free >= policy['minimum_physical_free_GiB']*2**30
            log = O/f"{task['episode']}_{mode}.log"
            started = time.monotonic()
            with log.open('x') as stream:
                result = subprocess.run([sys.executable, str(R/'source/carla_guarded_complex_episode_008.py'),
                    '--episode', task['episode'], '--mode', mode], env=env, stdout=stream,
                    stderr=subprocess.STDOUT, check=False)
            dest = R/'carla_official/guarded_complex_navigation_008'/task['episode']/mode
            row = dict(episode=task['episode'], mode=mode, exit_code=result.returncode,
                actual_world_started=(dest/'release.json').exists(),
                wall_s=time.monotonic()-started, log=str(log), log_sha256=digest(log), audits=[])
            if (dest/'summary.json').exists():
                for audit in ['audit_carla_arrival_003.py', 'audit_carla_complex_choices_008.py']:
                    auditlog = O/f"{mode}_{audit}.log"
                    with auditlog.open('x') as stream:
                        completed = subprocess.run([sys.executable, str(R/'source'/audit), '--episode',
                            str(dest.relative_to(R/'carla_official'))], env={**env,'CUDA_VISIBLE_DEVICES':''},
                            stdout=stream, stderr=subprocess.STDOUT, check=False)
                    row['audits'].append(dict(source=audit, exit_code=completed.returncode,
                        log=str(auditlog), log_sha256=digest(auditlog)))
            rows.append(row)
            dump(O/f'job_{len(rows):02d}.json', row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result.returncode:
                dump(O/'stopped_after_execution_failure.json', dict(job=row,
                    reason='原始证据前缀保留；执行异常后先诊断，不自动重跑其他模式', goal_complete=False))
                raise RuntimeError('真实008导航执行异常，未自动重试')
    complete = len(rows) == 3 and all(x['exit_code'] == 0 and len(x['audits']) == 2
        and all(a['exit_code'] == 0 for a in x['audits']) for x in rows)
    dump(O/'report.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        jobs=rows, planned_physical_episodes=3, actual_launched_processes=len(rows),
        actual_started_navigation_episodes=sum(r['actual_world_started'] for r in rows),
        subprocesses_and_audits_complete=complete, registration_sha256=digest(bankfile),
        all_modes_common_controller_guard_intent_protocol=True,
        closed_loop_inputs_can_diverge_after_different_actions=True,
        strict_safe_arrival_requires_independent_report=True,
        whole_project_sealed_test=False, goal_complete=False))
    if not complete:
        raise RuntimeError('008独立审计未全部通过；不是成功率判断')


if __name__ == '__main__':
    main()
