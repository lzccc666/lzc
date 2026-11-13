"""机器队列退出后执行普通回归、业务Ruff和全历史保护核验；不改旧保护报告。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'
DESTINATION = OUTPUT / 'source_bundle/final_engineering_checks_002'


def save(name, value):
    with (DESTINATION / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            if deadline is not None:
                assert time.monotonic() < deadline, '保护核验超出冻结预算'
            value.update(block)
    return value.hexdigest()


def live(pid, tick):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == tick
    except FileNotFoundError:
        return False


def main():
    release = json.loads((DESTINATION / 'release.json').read_text())
    begun = time.monotonic()
    try:
        while any(live(int(pid), tick) for pid, tick in release['wait_processes'].items()):
            assert time.monotonic() - begun < 86400, '等待超过24小时预算'
            time.sleep(15)
        for path, expected in release['source_sha256'].items():
            assert sha(path) == expected, path
        checkpoint = json.loads((OUTPUT / 'source_bundle/source_checkpoint_037/checkpoint.json').read_text())
        source_before = {name: sha(PROJECT / name) for name in checkpoint['sources']}
        assert source_before == checkpoint['sources'], '冻结业务源码已变化，应另行验收而非复用旧签名'
        assert sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
        preceding = {}
        for relative in ['source_bundle/session_recovery_001/finished.json',
                         'visualizations/gpu_fault_viewer_queue_002/complete.json']:
            path = OUTPUT / relative
            preceding[relative] = dict(exists=path.is_file(), sha256=sha(path) if path.is_file() else None)
        save('preceding_terminal.json', dict(prerequisite_processes_exited=True,
            results=preceding, preceding_success_not_assumed=True, goal_complete=False))
        execution_started = time.monotonic()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(PROJECT),
            LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib')
        jobs = []

        def run(label, command, timeout):
            with (DESTINATION / (label + '.log')).open('x') as log:
                try:
                    process = subprocess.run(command, cwd=PROJECT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            record = dict(label=label, command=command, returncode=code)
            jobs.append(record)
            save(label + '_exit.json', record)

        run('00_pytest', [sys.executable, '-m', 'pytest', '-q', 'tests',
            '--basetemp', str(DESTINATION / 'pytest_tmp'), '-o',
            'cache_dir=' + str(DESTINATION / 'pytest_cache')], 2400)
        run('01_ruff_business', [sys.executable, '-m', 'ruff', 'check',
            'conceptgraph/completion', 'conceptgraph/navigation', 'tests'], 120)
        run('02_ruff_new_helpers', [sys.executable, '-m', 'ruff', 'check'] +
            release['maintained_helpers'], 120)
        run('03_git_diff_check', ['git', 'diff', '--check'], 120)
        # 不调用原baseline_guard verify；它会覆盖旧baseline_verification.json。
        manifest_path = OUTPUT / 'source_bundle/protected_manifest.jsonl'
        baseline = json.loads((OUTPUT / 'baseline_before.json').read_text())
        assert sha(manifest_path) == baseline['protection_manifest_sha256']
        errors, unstable, count, protected_bytes = [], [], 0, 0
        deadline = time.monotonic() + 3600
        with manifest_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                path = PROJECT / record['path']
                count += 1
                protected_bytes += record['size']
                if not path.is_file():
                    errors.append(dict(path=record['path'], reason='missing'))
                    continue
                before = path.stat()
                actual = sha(path, deadline)
                after = path.stat()
                if before.st_size != record['size'] or actual != record['sha256']:
                    errors.append(dict(path=record['path'], reason='content_changed'))
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    unstable.append(record['path'])
                if count % 20000 == 0:
                    print(json.dumps(dict(protected_files_checked=count, errors=len(errors),
                        unstable=len(unstable), goal_complete=False)), flush=True)
        assert count == baseline['protected_files'] and protected_bytes == baseline['protected_bytes']
        protection = dict(checked_utc=datetime.now(timezone.utc).isoformat(), verified_files=count,
            protected_bytes=protected_bytes, changed_or_missing=errors, unstable_during_read=unstable,
            protection_manifest_sha256=sha(manifest_path), passed=not errors and not unstable,
            original_baseline_verification_untouched=(sha(OUTPUT / 'baseline_verification.json')
                == release['baseline_verification_sha256']), goal_complete=False)
        save('protection_report.json', protection)
        source_after = {name: sha(PROJECT / name) for name in checkpoint['sources']}
        source_unchanged = source_before == source_after
        index_unchanged = sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
        success = (all(j['returncode'] == 0 for j in jobs) and protection['passed']
            and protection['original_baseline_verification_untouched'] and source_unchanged and index_unchanged)
        save('complete.json' if success else 'incomplete.json', dict(
            machine_engineering_checks_passed=success, jobs=jobs,
            original_source_files_verified=len(source_after), original_source_files_unchanged=source_unchanged,
            original_index_unchanged=index_unchanged, protected_files_verified=count,
            protection_passed=protection['passed'], actual_model_calls=0,
            standard_test_scope='pytest tests；不扫描第三方和输出目录以凑测试数量',
            ruff_scope='completion/navigation/tests及明确列出的维护辅助代码；历史冻结脚本和未改动上游不获全库Ruff信用',
            structured_artifact_validation_completed=False, final_delivery_completed=False,
            experiment_success_not_implied=True, human_labels_not_created=True,
            new_test_opened=False, manuscript_text_written=False, goal_complete=False,
            execution_wall_s=time.monotonic() - execution_started))
        if not success:
            raise RuntimeError('普通回归或保护检查未通过，原证据保留')
    except BaseException as error:
        save('interrupted.json', dict(error=repr(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
