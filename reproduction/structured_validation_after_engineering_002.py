"""回归和保护检查退出后实际执行全范围结构校验，不将前置失败推断为成功。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
ROOT = OUTPUT / 'source_bundle/structured_validation_queue_002'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def live(pid, tick):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == tick
    except FileNotFoundError:
        return False


def main():
    release = json.loads((ROOT / 'release.json').read_text())
    begun = time.monotonic()
    try:
        while live(release['predecessor_pid'], release['predecessor_start_tick']):
            assert time.monotonic() - begun < 86400, '前置工程核验等待超过24小时预算'
            time.sleep(15)
        for path, expected in release['source_sha256'].items():
            assert sha(path) == expected, path
        predecessor = OUTPUT / 'source_bundle/final_engineering_checks_002'
        save('preceding_terminal.json', dict(process_exited=True,
            evidence={name: dict(exists=(predecessor / name).is_file(),
                sha256=sha(predecessor / name) if (predecessor / name).is_file() else None)
                for name in ['complete.json', 'incomplete.json', 'interrupted.json']},
            success_not_assumed=True, goal_complete=False))
        checker = OUTPUT / 'source_bundle/structured_artifact_validation_002.py'
        with (ROOT / 'ruff.log').open('x') as log:
            result = subprocess.run([os.sys.executable, '-m', 'ruff', 'check', str(checker), __file__],
                stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
        save('ruff_exit.json', dict(returncode=result.returncode, goal_complete=False))
        assert result.returncode == 0
        with (ROOT / 'worker.log').open('x') as log:
            result = subprocess.run([os.sys.executable, str(checker), '--run-id',
                'structured_artifact_validation_final_002'], stdout=log,
                stderr=subprocess.STDOUT, timeout=3900, check=False)
        save('worker_exit.json', dict(returncode=result.returncode, goal_complete=False))
        assert result.returncode == 0
        report_path = OUTPUT / 'source_bundle/structured_artifact_validation_final_002/report.json'
        report = json.loads(report_path.read_text())
        assert report['passed'] and report['full_project_output_scope']
        save('complete.json', dict(structured_artifact_check_passed=True,
            report_sha256=sha(report_path), files_checked=report['files_checked'],
            JSONL_records_checked=report['JSONL_records_checked'],
            actual_model_calls=0, final_delivery_completed=False, goal_complete=False))
    except BaseException as error:
        save('interrupted.json', dict(error=repr(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
