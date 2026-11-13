"""新环境冻结复演成功后，应用已验证的测试夹具修正并运行全量工程检查。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'
SOURCE = OUTPUT / 'source_bundle'
DEST = SOURCE / 'final_tests_003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (DEST / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def live():
    try:
        fields = Path('/proc/577264/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == '1387244111'
    except FileNotFoundError:
        return False


def main():
    begun = time.monotonic()
    try:
        while live():
            assert time.monotonic()-begun < 2400
            time.sleep(15)
        environment = OUTPUT / 'reproduction/isolated_environment_after_GPU_005/complete.json'
        assert environment.is_file(), '先保留冻结源码直至新环境复演成功'
        assert json.loads(environment.read_text())['complete']
        checkpoint = json.loads((SOURCE / 'source_checkpoint_037/checkpoint.json').read_text())
        assert all(sha(PROJECT / name) == value for name, value in checkpoint['sources'].items())
        assert sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
        repair = SOURCE / 'test_path_fixture_repair_001'
        prepared = json.loads((repair / 'prepared.json').read_text())
        target = PROJECT / prepared['source']
        assert sha(target) == prepared['before_sha256']
        assert sha(repair / 'candidate.py') == prepared['candidate_sha256']
        target.write_bytes((repair / 'candidate.py').read_bytes())
        save('applied.json', dict(source=prepared['source'], before_sha256=prepared['before_sha256'],
            after_sha256=sha(target), business_runtime_code_changed=False,
            previous_451_pass_1_failure_preserved=True, goal_complete=False))
        jobs = []
        commands = [
            ('pytest', [os.sys.executable, '-m', 'pytest', '-q', 'tests', '--basetemp',
                        str(DEST / 'pytest_tmp'), '-o', 'cache_dir='+str(DEST / 'pytest_cache')], 2400),
            ('ruff', [os.sys.executable, '-m', 'ruff', 'check', 'conceptgraph/completion',
                      'conceptgraph/navigation', 'tests'], 120),
            ('diff_check', ['git', 'diff', '--check'], 120),
        ]
        for name, command, timeout in commands:
            with (DEST / (name+'.log')).open('x') as log:
                result = subprocess.run(command, cwd=PROJECT, env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            jobs.append(dict(name=name, returncode=result.returncode, log_sha256=sha(DEST/(name+'.log'))))
        after = {name: sha(PROJECT/name) for name in checkpoint['sources']}
        changed = [name for name, value in after.items() if value != checkpoint['sources'][name]]
        assert changed == [prepared['source']]
        assert sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
        passed = all(j['returncode'] == 0 for j in jobs)
        save('complete.json' if passed else 'incomplete.json', dict(passed=passed,
            created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, changed_sources=changed,
            sources=after, original_index_sha256=checkpoint['original_index_sha256'],
            root_index_unchanged=True, runtime_code_unchanged=True,
            new_environment_reproduction_sha256=sha(environment),
            actual_model_calls=0, goal_complete=False))
        assert passed
    except BaseException as error:
        save('interrupted.json', dict(error=repr(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
