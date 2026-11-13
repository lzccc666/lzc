from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
ROOT = OUTPUT / 'faults/fault_sensor_final_audit_queue_001'
SOURCE = OUTPUT / 'source_bundle/fault_sensor_numeric_audit_001.py'

def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)

release = json.loads((ROOT / 'release.json').read_text())
begun = time.monotonic()
try:
    while True:
        try:
            fields = Path('/proc/455071/stat').read_text().rsplit(')', 1)[1].split()
            alive = fields[0] != 'Z' and fields[19] == release['supervisor_tick']
        except FileNotFoundError:
            alive = False
        if not alive:
            break
        assert time.monotonic() - begun < 86400
        time.sleep(15)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == release['audit_sha256']
    summary = OUTPUT / 'interactive_simulation/realtime_faults_gpu_full_001/summary.json'
    assert summary.exists(), '完整故障summary缺失，不能将前缀当作完整审计'
    record = json.loads(summary.read_text())
    assert record['attempted'] == record['expected'] == 180
    with (ROOT / 'worker.log').open('x') as log:
        result = subprocess.run([os.sys.executable, str(SOURCE), '--input-run',
            'realtime_faults_gpu_full_001', '--run-id', 'fault_sensor_numeric_full_001'],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=''),
            stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    save('exit.json', dict(returncode=result.returncode, goal_complete=False))
    assert result.returncode == 0
    report = json.loads((OUTPUT / 'faults/fault_sensor_numeric_full_001/report.json').read_text())
    assert report['passed'] and report['complete_cohort'] and report['cases'] == 180
    save('complete.json', dict(audited=180, actual_model_calls=0, goal_complete=False))
except BaseException as error:
    save('interrupted.json', dict(error=repr(error), goal_complete=False))
    raise
