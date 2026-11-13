"""在最终复现与测试之后核验全部原受保护文件；不覆盖历史保护报告。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'
DEST = Path(__file__).parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert json.loads((OUTPUT/'source_bundle/final_tests_003/complete.json').read_text())['passed']
    baseline = json.loads((OUTPUT/'baseline_before.json').read_text())
    previous = sha(OUTPUT/'baseline_verification.json')
    manifest = OUTPUT/'source_bundle/protected_manifest.jsonl'
    assert sha(manifest) == baseline['protection_manifest_sha256']
    count, total, errors = 0, 0, []
    for line in manifest.open():
        row = json.loads(line)
        path = PROJECT/row['path']
        count += 1
        total += row['size']
        if not path.is_file():
            errors.append(dict(path=row['path'], reason='missing'))
            continue
        before = path.stat()
        checksum = sha(path)
        after = path.stat()
        if checksum != row['sha256'] or before.st_size != row['size']:
            errors.append(dict(path=row['path'], reason='changed'))
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(dict(path=row['path'], reason='unstable'))
        if count % 50000 == 0:
            print(json.dumps(dict(checked=count, errors=len(errors))), flush=True)
    assert count == baseline['protected_files'] and total == baseline['protected_bytes']
    assert previous == sha(OUTPUT/'baseline_verification.json')
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), passed=not errors,
        verified_files=count, protected_bytes=total, changed_or_missing=errors,
        original_baseline_verification_untouched=True, baseline_verification_sha256=previous,
        manifest_sha256=sha(manifest), source_sha256=sha(Path(__file__)), goal_complete=False)
    with (DEST/'report.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    assert not errors
    print(json.dumps(dict(passed=True, files=count)), flush=True)


if __name__ == '__main__':
    main()
