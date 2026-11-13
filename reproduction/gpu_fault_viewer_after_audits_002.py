"""完整故障的三个独立审计全部结束后生成并核验只读查看器。"""

import json
import os
from pathlib import Path
import subprocess
import time

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def live(pid, tick):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == tick
    except FileNotFoundError:
        return False


def main():
    root = OUTPUT / 'visualizations/gpu_fault_viewer_queue_002'
    release = json.loads((root / 'release.json').read_text())
    started = time.monotonic()
    try:
        while any(live(int(pid), tick) for pid, tick in release['prerequisite_processes'].items()):
            assert time.monotonic() - started < 86400, '等待超出24小时预算'
            time.sleep(15)
        for relative in release['required_complete_files']:
            assert (OUTPUT / relative).is_file(), relative
        for path, expected in release['source_sha256'].items():
            assert digest(path) == expected, path
        summary = json.loads((OUTPUT / 'interactive_simulation/realtime_faults_gpu_full_001/summary.json').read_text())
        assert summary['attempted'] == summary['expected'] == 180
        source = OUTPUT / 'source_bundle'
        commands = [
            [str(source / 'gpu_fault_evidence_viewer_004.py'), '--input-run', 'realtime_faults_gpu_full_001',
             '--run-id', 'gpu_fault_evidence_full_002', '--event-audit', 'faults/fault_event_full_independent_001',
             '--numeric-audit', 'faults/fault_sensor_numeric_full_001', '--execution-audit',
             'vehicle_execution/fault_execution_full_independent_001'],
            [str(source / 'gpu_fault_viewer_browser_001.py'), '--run-id', 'gpu_fault_evidence_full_002',
             '--output-id', 'gpu_fault_evidence_full_browser_002'],
        ]
        for index, command in enumerate(commands):
            with (root / f'{index:02d}.log').open('x') as log:
                child = subprocess.run([os.sys.executable] + command, env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT, timeout=2400)
            dump(root / f'{index:02d}_exit.json', dict(returncode=child.returncode, goal_complete=False))
            assert child.returncode == 0, '原失败保留，不自动重跑'
        qa = OUTPUT / 'visualizations/gpu_fault_evidence_full_browser_002/report.json'
        report = json.loads(qa.read_text())
        assert report['passed'] and report['cases'] == 180
        dump(root / 'complete.json', dict(cases=180, browser_report_sha256=digest(qa),
            actual_model_calls=0, new_arrivals=0, goal_complete=False))
    except BaseException as error:
        dump(root / 'interrupted.json', dict(error=repr(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
