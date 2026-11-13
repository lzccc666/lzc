"""不改冻结任务；独立核验源集合唯一附加文件，等原试验结束后版本化汇总。"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
ROOT = OUT/'ablations/semantic_navigation_chain_004'
MAIN = OUT/'interactive_simulation/realtime_three_model_full_001'
MINUS = OUT/'interactive_simulation/realtime_semantic_minus_full_001'
EXTRA = 'conceptgraph/navigation/reference_guided_vehicle.py'
VARIANTS = ['rule', 'new_full', 'new_minus_parse', 'new_minus_attributes', 'new_minus_rerank']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while data := handle.read(1048576):
            value.update(data)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def ticks(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def compatibility():
    first, second = read(MAIN/'release.json'), read(MINUS/'release.json')
    a, b = first['source_sha256'], second['source_sha256']
    assert len(a) == 66 and len(b) == 67 and set(b)-set(a) == {EXTRA} and not set(a)-set(b)
    assert all(b[name] == value == sha(PROJECT/name) for name, value in a.items())
    assert sha(PROJECT/EXTRA) == b[EXTRA]
    for field in ['protocol', 'task_release_sha256', 'adapter_weights_sha256', 'model_selection_sha256']:
        assert first[field] == second[field], field
    references, dynamic = [], []
    for name in a:
        text = (PROJECT/name).read_text()
        assert 'reference_guided' not in text, name
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            function = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
            if (isinstance(node.func, ast.Name) and function in ['exec', 'eval', '__import__']) or function in ['import_module', 'spec_from_file_location', 'run_module', 'run_path']:
                assert function == '__import__' and node.args and isinstance(node.args[0], ast.Constant)
                assert node.args[0].value == 'hashlib'
                dynamic.append(dict(source=name, line=node.lineno, function=function, module=node.args[0].value))
    return first, second, dict(passed=True, exact_source_set_equality=False,
        common_source_files=66, extra_source_files=[EXTRA], common_source_SHA_unchanged=True,
        explicit_extra_module_references=references, dynamic_loading_calls=dynamic,
        same_task_weights_selection_protocol=True,
        limit='静态调用检查和入口导入探针不等于完整运行时模块跟踪；源集合不等仍明确保留。')


def pool(root, release, expected, *, allow_partial=False):
    raw = (root/'trial_records.jsonl').read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith('\n') and allow_partial:
        lines.pop()
    rows = [json.loads(line) for line in lines if line.strip()]
    plans = {row['run_id']: row for row in release['trials']}
    assert len(plans) == expected and len(rows) == len({row['run_id'] for row in rows})
    assert len(rows) <= expected if allow_partial else len(rows) == expected
    for row in rows:
        assert all(row[field] == value for field, value in plans[row['run_id']].items())
        if row['result'] is not None:
            path = Path(row['raw_case'])/'result.json'
            assert read(path) == row['result'] and sha(path) == row['result_sha256']
            assert not row['result']['gt_in_agent'] and row['result']['real_clock_async']
        else:
            assert row['result_sha256'] is None
    if not allow_partial:
        summary = read(root/'summary.json')
        assert summary['attempted'] == summary['expected'] == expected
        assert sha(root/'trial_records.jsonl') == summary['trial_records_sha256']
        assert {row['run_id'] for row in rows} == set(plans)
    return rows


def prepare():
    ROOT.mkdir(exist_ok=False)
    first, second, evidence = compatibility()
    rows = pool(MAIN, first, 450)
    partial = pool(MINUS, second, 450, allow_partial=True)
    probe = ROOT/'entry_import_probe.py'
    probe.write_text("""import importlib,json,sys
importlib.import_module('conceptgraph.completion.realtime_navigation')
importlib.import_module('conceptgraph.completion.realtime_matrix')
names=sorted(name for name in sys.modules if name.startswith('conceptgraph.'))
assert 'conceptgraph.navigation.reference_guided_vehicle' not in sys.modules
torch=sys.modules.get('torch')
initialized=bool(torch and torch.cuda.is_initialized())
assert not initialized
print(json.dumps(dict(passed=True,project_modules=names,extra_module_imported=False,CUDA_initialized=initialized,entry_import_only=True,actual_model_calls=0)))
""")
    result = subprocess.run([sys.executable, str(probe)], cwd=PROJECT,
        env=dict(os.environ, PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES=''),
        capture_output=True, text=True, timeout=120)
    (ROOT/'entry_import_probe_stdout.log').write_text(result.stdout)
    (ROOT/'entry_import_probe_stderr.log').write_text(result.stderr)
    assert result.returncode == 0, '入口导入探针失败；不能创建桥接回执'
    imported = json.loads(result.stdout.splitlines()[-1])
    assert imported['passed'] and not imported['extra_module_imported'] and not imported['CUDA_initialized']
    rehearsal = OUT/'statistics/semantic_compatibility_audit_rehearsal_main_001/report.json'
    assert read(rehearsal)['passed'] and read(rehearsal)['trials'] == 450
    evidence.update(actual_entry_import_probe=imported, checked_main_trials=len(rows),
        checked_partial_minus_trials=len(partial), preflight_is_not_complete_ablation=True,
        audit_rehearsal_sha256=sha(rehearsal), new_model_calls=0, new_executions=0, goal_complete=False)
    save(ROOT/'preflight.json', evidence)
    names = set(second['source_sha256']) | {
        str(Path(__file__).resolve().relative_to(PROJECT)),
        'conceptgraph/completion/realtime_matrix_statistics.py',
        'conceptgraph/completion/interactive_statistics.py', 'conceptgraph/completion/vehicle_statistics.py',
        str(probe.relative_to(PROJECT)),
        'outputs/project_v2/thesis_completion_v1/source_bundle/semantic_compatibility_evidence_check_003.py',
        'outputs/project_v2/thesis_completion_v1/task_protocol/semantic_source_compatibility_003.md'}
    parents = {str(pid): ticks(pid) for pid in [38569, 268029]}
    assert all(parents.values()), '预检必须针对目前仍存活的原任务，不创建替代模型作业'
    save(ROOT/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(PROJECT/name) for name in sorted(names)},
        main_release_sha256=sha(MAIN/'release.json'), minus_release_sha256=sha(MINUS/'release.json'),
        preflight_sha256=sha(ROOT/'preflight.json'), original_process_start_ticks=parents,
        variants=VARIANTS, wait_budget_s=90000, statistics_budget_s=1800, audit_budget_s=900,
        metadata_budget_bytes=536870912, minimum_free_bytes=6442450944,
        completion_bridge='ablations/semantic_navigation_chain_001/complete.json',
        original_source_set_assertion_will_remain_failed=True,
        existing_model_trials_not_restarted=True, new_test_opened=False, goal_complete=False))
    print(json.dumps(dict(preflight_passed=True, main_trials=450, partial_minus_trials=len(partial),
        source_sets_equal=False, source_common_unchanged=True, actual_model_calls=0, goal_complete=False)))


def verify_release(release):
    assert sha(MAIN/'release.json') == release['main_release_sha256']
    assert sha(MINUS/'release.json') == release['minus_release_sha256']
    for name, expected in release['source_sha256'].items():
        assert sha(PROJECT/name) == expected, name
    assert shutil.disk_usage(OUT).free >= release['minimum_free_bytes']


def resume():
    release = read(ROOT/'release.json')
    verify_release(release)
    save(ROOT/'worker_started.json', dict(pid=os.getpid(), start_ticks=ticks(os.getpid()),
        created_utc=datetime.now(timezone.utc).isoformat(), actual_model_calls=0, goal_complete=False))
    begun = time.monotonic()
    try:
        parents = release['original_process_start_ticks']
        while any(ticks(int(pid)) == tick for pid, tick in parents.items()):
            if time.monotonic()-begun > release['wait_budget_s']:
                raise TimeoutError('原任务未在补充等待预算内退出；不重启、不新增完成桥接')
            time.sleep(5)
        verify_release(release)
        first, second, evidence = compatibility()
        main_rows, minus_rows = pool(MAIN, first, 450), pool(MINUS, second, 450)
        original_log = OUT/'source_bundle/semantic_ablation_after_main_001.log'
        text = original_log.read_text()
        assert not text.strip(), '原父进程在源集合断言前终止；当前恢复依据是396例及原进程缺失'
        shutil.copy2(original_log, ROOT/'original_parent_failure.log')
        save(ROOT/'original_failure.json', dict(original_log_sha256=sha(original_log),
            original_parent_exit_code_observed=None, original_parent_and_child_no_longer_live=True,
            original_source_set_equality_passed=False, expected_extra_file=EXTRA,
            original_failure_preserved=True, original_interruption_before_assertion=True, assertion_failure_observed=False, compatibility_evidence=evidence, goal_complete=False))
        combined = OUT/'interactive_simulation/semantic_ablation_combined_004'
        combined.mkdir(exist_ok=False)
        plans = [row for data in [first, second] for row in data['trials'] if row['variant'] in VARIANTS]
        rows = [row for data in [main_rows, minus_rows] for row in data if row['variant'] in VARIANTS]
        assert len(plans) == len(rows) == len({row['run_id'] for row in rows}) == 750
        assert all(sum(row['variant'] == variant for row in rows) == 150 for variant in VARIANTS)
        save(combined/'release.json', dict(preset='semantic-ablation-compatible-source-union-003',
            trials=plans, source_sha256=second['source_sha256'], task_release_sha256=first['task_release_sha256'],
            source_runs={root.name: {name: sha(root/name) for name in ['release.json', 'summary.json', 'trial_records.jsonl']}
                         for root in [MAIN, MINUS]}, source_set_equality_passed=False,
            compatibility_preflight_sha256=release['preflight_sha256'],
            design='5条件×50任务×3种子；300原记录复用+450新增；额外导航文件未在共同冻结调用链引用。',
            independent_test=False, gt_in_agent=False, goal_complete=False))
        with (combined/'trial_records.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        save(combined/'summary.json', dict(attempted=750, expected=750,
            trial_records_sha256=sha(combined/'trial_records.jsonl'), derived_evidence_union=True,
            new_trial_calls=450, reused_trial_records=300, new_calls_by_this_worker=0, goal_complete=False))
        commands = [
            ('statistics.log', [sys.executable, '-m', 'conceptgraph.completion.realtime_matrix_statistics',
                '--input-run', combined.name, '--run-id', combined.name], release['statistics_budget_s']),
            ('independent_audit.log', [sys.executable, str(OUT/'source_bundle/semantic_compatibility_evidence_check_003.py'), '--input-run', 'semantic_ablation_combined_004', '--run-id', 'semantic_compatibility_evidence_check_004'],
                release['audit_budget_s'])]
        for filename, command, timeout in commands:
            with (ROOT/filename).open('x') as handle:
                process = subprocess.run(command, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                    stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            assert process.returncode == 0, filename
        audit = OUT/'statistics/semantic_compatibility_evidence_check_004/report.json'
        assert read(audit)['passed'] and read(audit)['trials'] == 750
        verify_release(release)
        assert sum(path.stat().st_size for root in [ROOT, combined] for path in root.rglob('*') if path.is_file()) < release['metadata_budget_bytes']
        complete = dict(combined_run=combined.name, new_trials=450, combined_trial_records=750,
            reused_main_records=300, new_model_calls_by_this_worker=0, source_set_equality_passed=False,
            common_inference_source_SHA_unchanged=True, unused_extra_file=EXTRA,
            original_failure_sha256=sha(ROOT/'original_failure.json'),
            statistics_report_sha256=sha(OUT/'statistics'/combined.name/'report.json'),
            independent_audit_sha256=sha(audit), elapsed_s_including_wait=time.monotonic()-begun,
            original_version_failed_source_set_equality=False, source_set_inequality_verified=True, original_parent_interrupted=True, explicit_compatibility_version=4,
            independent_test=False, new_test_opened=False, stage12_complete=False, goal_complete=False)
        save(ROOT/'complete.json', complete)
        bridge = dict(complete, compatibility_completion_bridge=True,
            original_script_success=False, versioned_complete='ablations/semantic_navigation_chain_004/complete.json',
            versioned_complete_sha256=sha(ROOT/'complete.json'))
        target = OUT/release['completion_bridge']
        temporary = target.parent/'.complete.compatibility_004.tmp'
        with temporary.open('x') as handle:
            json.dump(bridge, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        temporary.unlink()
        save(ROOT/'bridge_written.json', dict(path=release['completion_bridge'], sha256=sha(target),
            original_script_success=False, goal_complete=False))
        print(json.dumps(dict(combined_records=750, independent_audit_passed=True,
            original_script_success=False, bridge_written=True, goal_complete=False)))
    except BaseException as exc:
        save(ROOT/'interrupted.json', dict(error=repr(exc), bridge_not_authorized_by_failure=True,
            actual_model_calls=0, new_test_opened=False, goal_complete=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare-only', action='store_true')
    choice.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_only else resume()


if __name__ == '__main__':
    main()
