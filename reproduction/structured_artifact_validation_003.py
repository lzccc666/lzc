"""严格校验项目实验JSON/JSONL及冻结源码中的YAML；原始失败载荷单列。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import yaml


PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def constant(value):
    raise ValueError('非标准JSON数值：' + value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--input-root', action='append', default=[])
    args = parser.parse_args()
    destination = OUTPUT / 'source_bundle' / args.run_id
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    roots = [OUTPUT / name for name in args.input_root] if args.input_root else [OUTPUT]
    assert all(p.exists() and p.resolve().is_relative_to(OUTPUT.resolve()) for p in roots)
    excluded = [destination,
        OUTPUT / 'reproduction/isolated_cpu_environment_002/environment',
        OUTPUT / 'reproduction/isolated_cpu_environment_003/environment',
        OUTPUT / 'source_bundle/final_tests_003/pytest_tmp',
        OUTPUT / 'source_bundle/final_tests_003/pytest_cache',
        OUTPUT / 'source_bundle/final_engineering_checks_001/pytest_tmp',
        OUTPUT / 'source_bundle/final_engineering_checks_001/pytest_cache',
        OUTPUT / 'source_bundle/final_engineering_checks_002/pytest_tmp',
        OUTPUT / 'source_bundle/final_engineering_checks_002/pytest_cache']
    registries = [OUTPUT / 'thesis/citation_http_payload_type_audit_002/report.json',
                  OUTPUT / 'source_bundle/failed_json_payload_audit_002/report.json']
    known = {}
    for index, registry in enumerate(registries):
        records = json.loads(registry.read_text())['records']
        for record in records:
            original = OUTPUT / record['original_path']
            assert original.stat().st_size == record['bytes'] and sha(original) == record['sha256']
            if index == 0:
                assert record['http_status'] == 429 and not record['retrieval_success']
                kind = 'registered_unsuccessful_HTTP_payload'
            else:
                manifest = OUTPUT / record['manifest']
                assert sha(manifest) == record['manifest_sha256']
                proof = json.loads(manifest.read_text())
                assert proof['status'] == record['status'] == 'FAILED_PARTIAL_JSON_WRITE'
                assert proof['payload_sha256'] == record['sha256']
                try:
                    json.loads(original.read_bytes(), parse_constant=constant)
                except json.JSONDecodeError:
                    pass
                else:
                    raise AssertionError('已登记的部分失败载荷意外变为有效JSON')
                kind = 'registered_failed_partial_JSON'
            known[(record['bytes'], record['sha256'])] = dict(kind=kind,
                original_path=record['original_path'], registry=str(registry.relative_to(OUTPUT)))

    def excluded_path(path):
        return any(path == root or root in path.parents for root in excluded) or '.git' in path.parts

    def candidates():
        result = set()
        for root in roots:
            if root.is_file():
                if root.suffix in {'.json', '.jsonl'} and not excluded_path(root):
                    result.add(root)
                continue
            for folder, dirs, names in os.walk(root, followlinks=False):
                parent = Path(folder)
                dirs[:] = sorted(n for n in dirs if not excluded_path(parent / n))
                for name in names:
                    path = parent / name
                    if path.suffix in {'.json', '.jsonl'} and not excluded_path(path):
                        result.add(path)
        return result

    initial = candidates()
    errors, unstable, archived_failures = [], [], []
    lines_checked = 0
    with (destination / 'files.jsonl').open('x') as ledger:
        for index, path in enumerate(sorted(initial)):
            assert time.monotonic() - started < 3600, '结构校验超出一小时预算'
            before = path.stat()
            checksum = sha(path)
            registered = known.get((before.st_size, checksum))
            line_count = 0
            number = None
            state = 'parsed'
            if registered:
                state = 'archived_failure_not_valid_result'
                archived_failures.append(dict(path=str(path.relative_to(OUTPUT)), sha256=checksum, **registered))
            else:
                try:
                    if path.suffix == '.json':
                        json.loads(path.read_bytes(), parse_constant=constant)
                    else:
                        with path.open('rb') as stream:
                            for number, line in enumerate(stream, 1):
                                if line.strip():
                                    json.loads(line, parse_constant=constant)
                                    line_count += 1
                        lines_checked += line_count
                except (ValueError, UnicodeError, OSError) as error:
                    state = 'parse_error'
                    errors.append(dict(path=str(path.relative_to(OUTPUT)), error=str(error),
                        failing_line=number if path.suffix == '.jsonl' else None))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                unstable.append(str(path.relative_to(OUTPUT)))
            ledger.write(json.dumps(dict(path=str(path.relative_to(OUTPUT)), sha256=checksum,
                bytes=before.st_size, state=state, parsed_JSONL_records=line_count)) + '\n')
            if index and index % 20000 == 0:
                ledger.flush()
                print(json.dumps(dict(files_checked=index, errors=len(errors), goal_complete=False)), flush=True)
    yaml_rows = []
    checkpoint = json.loads((OUTPUT / 'source_bundle/source_checkpoint_037/checkpoint.json').read_text())
    for relative, expected in checkpoint['sources'].items():
        if Path(relative).suffix not in {'.yaml', '.yml'}:
            continue
        path = PROJECT / relative
        assert sha(path) == expected, relative
        try:
            yaml.safe_load(path.read_text())
            yaml_rows.append(dict(path=relative, sha256=expected, parsed=True))
        except (yaml.YAMLError, UnicodeError) as error:
            yaml_rows.append(dict(path=relative, sha256=expected, parsed=False, error=str(error)))
    final = candidates()
    added = sorted(str(p.relative_to(OUTPUT)) for p in final - initial)
    removed = sorted(str(p.relative_to(OUTPUT)) for p in initial - final)
    passed = not errors and not unstable and not added and not removed and all(r['parsed'] for r in yaml_rows)
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), passed=passed,
        full_project_output_scope=not args.input_root, input_roots=[str(p.relative_to(OUTPUT)) for p in roots],
        files_checked=len(initial), JSONL_records_checked=lines_checked, errors=errors,
        unstable_during_read=unstable, added_during_scan=added, removed_during_scan=removed,
        archived_failed_payloads=archived_failures, failure_payloads_not_counted_as_valid_results=True,
        archived_copy_policy='仅原始登记SHA和尺寸完全一致的载荷及其逐字节副本可单列；未登记错误必须失败',
        excluded_subtrees=[str(p.relative_to(OUTPUT)) for p in excluded],
        exclusion_reasons=['本检查器自身输出', '隔离第三方Python依赖及其测试夹具不是项目实验结果',
                           '普通测试故意构造的临时输入与pytest缓存', 'Git内部元数据'],
        yaml_configs=yaml_rows, yaml_scope='checkpoint037中全部YAML配置，safe_load仅验证语法，不宣称所有运行参数有效',
        source_sha256=sha(__file__), registry_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in registries},
        file_ledger_sha256=sha(destination / 'files.jsonl'), actual_model_calls=0,
        final_delivery_completed=False, goal_complete=False, elapsed_s=time.monotonic() - started)
    with (destination / 'report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=passed, files=len(initial), JSONL_records=lines_checked,
                         YAML_configs=len(yaml_rows), archived_failures=len(archived_failures), errors=len(errors))))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
