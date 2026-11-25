"""等待既有008句柄结束后全量回读；不启动服务器、不重复任何GPU实验。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]
O = R/'json/guarded_motion_audit_jobs_001'


def start_tick(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    O.mkdir(exist_ok=False)
    watched = start_tick(args.wait_pid)
    source = R/'source/audit_carla_guarded_execution_001.py'
    frozen = {str(p): digest(p) for p in [Path(__file__), source,
        R/'source/carla_motion_guard_001.py', R/'source/carla_guarded_control_001.py']}
    dump(O/'registration.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        watched_PID=args.wait_pid, watched_start_tick=watched, source_sha256=frozen,
        modes=['rule', 'base', 'adapter'], actual_new_GPU_calls=0,
        actual_new_physical_episodes=0, goal_complete=False))
    begun = time.monotonic()
    while watched is not None and start_tick(args.wait_pid) == watched:
        if time.monotonic()-begun > 10800:
            raise TimeoutError('等待实际008句柄超过3h，未杀死或重启该任务')
        time.sleep(20)
    assert all(digest(p) == h for p, h in frozen.items())
    results = []
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'PYTHONDONTWRITEBYTECODE':'1'}
    for mode in ['rule', 'base', 'adapter']:
        episode = R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20'/mode
        if not (episode/'summary.json').exists():
            results.append(dict(mode=mode, completed_episode=False, audit_started=False,
                reason='没有正式终止summary，不伪造完整回合'))
            continue
        log = O/(mode+'.log')
        with log.open('x') as stream:
            result = subprocess.run([sys.executable, str(source), '--episode',
                str(episode.relative_to(R/'carla_official'))], stdout=stream,
                stderr=subprocess.STDOUT, env=env, check=False, timeout=1800)
        summary = json.loads((episode/'summary.json').read_text())
        arrival_path = episode/'independent_audit_003/report.json'
        arrival = json.loads(arrival_path.read_text()) if arrival_path.exists() else {}
        callsfile = episode/'semantic_backend/calls.jsonl'
        calls = list(map(json.loads, callsfile.open())) if callsfile.exists() else []
        row = dict(mode=mode, completed_episode=True, audit_started=True,
            audit_exit_code=result.returncode, audit_log_sha256=digest(log),
            actual_physics_ticks=summary['frames'], actual_frontend_calls=summary['perception_calls'],
            status=summary['status'], correct_first_proposal=summary['correct_object_selected'],
            correctly_locked=summary['correctly_locked'], route_generated=summary['route_generated'],
            strict_safe_arrival=arrival.get('strict_safe_arrival'),
            actual_final_GPU_calls=len(calls), schema_valid_GPU_calls=sum(c['schema_valid'] for c in calls))
        results.append(row)
        dump(O/(mode+'_result.json'), row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    dump(O/'report.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        results=results, all_three_complete_and_guard_audited=all(
            r.get('audit_exit_code') == 0 for r in results) and len(results) == 3,
        actual_new_GPU_calls=0, actual_new_physical_episodes=0,
        whole_project_sealed_test=False, goal_complete=False))


if __name__ == '__main__':
    main()
