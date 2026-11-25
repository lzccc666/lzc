"""完成027被入口索引格式暴露的收尾；不重写已生成状态文件。"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from close_continuation_delivery_011 import old_sources
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]
PROJECT = R.parents[3]


def main():
    old = old_sources()
    for version in ['008', '009', '010']:
        old.update({x['path']: x['sha256'] for x in json.loads(
            (R/f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    for path, sha in old.items():
        assert digest(path) == sha, path
    index = R/'source_bundle/module_index_013'; index.mkdir(exist_ok=False)
    with (R/'source_bundle/module_index_011/modules.csv').open() as stream:
        entries = list(csv.DictReader(stream))
    additions = [
        ('carla_history_lifecycle_017.py', '完整时间生命周期消融模块定义', 'carla_official/history_runtime_derivation_012.json', '不等同于仅关闭短期过期；不含闭环性能信用'),
        ('replay_carla_history_module_021.py', '真实传感器历史维护机制回放', 'carla_official/history_module_recorded_diagnostic_021/report.json', '只做机制核验，不改车辆控制，不授予到达'),
        ('audit_carla_history_execution_023.py', '历史维护关闭独立重核', 'carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023/report.json', '逐帧检查生命周期字段和跨帧累积'),
        ('run_carla_remaining_comparison_024.py', '主队列后方法与消融执行器', 'carla_official/remaining_comparison_registration_024.json', '等待主队列结束，运行前仍需独立审计'),
        ('run_official_remaining_025.py', '后续官方CARLA服务器等待器', 'json/official_remaining_execution_025_retry/registration.json', '不与现行服务器并行；检查GPU、端口和资源门限'),
        ('prepare_remaining_carla_queue_026.py', '后续210回合注册生成器', 'carla_official/remaining_comparison_registration_024.json', '只注册不运行，不使用封存测试结果调参'),
        ('close_progress_027.py', '阶段状态与源码封存器', 'json/continuation_handoff_016.json', '若索引格式失败保留失败日志，状态文件仍有效'),
        ('finish_close_028.py', '阶段封存收尾索引器', 'source_bundle/closing_supplement_012/receipt.json', '只新增索引和哈希，不复制原始传感器')]
    for filename, module, evidence, boundary in additions:
        source = R/'source'/filename
        tree = ast.parse(source.read_text())
        main_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
        entry = 'main' if main_node else 'module'
        line = main_node.lineno if main_node else 1
        entries.append(dict(module=module, source=str(source), entry=entry, line=line,
                            meaning=module, evidence=str(R/evidence), evidence_exists_at_index_generation=True,
                            boundary=boundary, source_sha256=digest(source)))
    with (index/'modules.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    (index/'README.md').open('x').write('013补充历史维护消融、后续官方队列和当前闭合分母；论文正文尚未写。\n')
    paths = [R/'json/continuation_handoff_016.json', R/'交付补充说明012.md', index/'modules.csv', index/'README.md',
             R/'json/history_compact_replay_derivation_020.json', R/'json/history_lifecycle_tests_022.log',
             R/'json/official_remaining_execution_025_retry/registration.json',
             R/'json/expansion_jobs_011_primary/job_003.json', R/'json/expansion_jobs_011_primary/job_004.json']
    paths += [R/'source'/name for name in [x[0] for x in additions]]
    paths += [p for p in (R/'carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023').rglob('*') if p.is_file()]
    paths += [p for p in (R/'carla_official/expansion_progress_033').rglob('*') if p.is_file()]
    archive_dir = R/'source_bundle/closing_supplement_012'; archive_dir.mkdir(exist_ok=False)
    members = [dict(path=str(p), archive_path=str(p.relative_to(PROJECT)), bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(set(paths))]
    archive = archive_dir/'source_delta.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for member in members: tar.add(member['path'], arcname=member['archive_path'], recursive=False)
    with tarfile.open(archive) as tar:
        for member in members:
            assert hashlib.sha256(tar.extractfile(member['archive_path']).read()).hexdigest() == member['sha256']
    with (archive_dir/'files.json').open('x') as stream:
        json.dump(members, stream, ensure_ascii=False, indent=2); stream.write('\n')
    receipt = dict(protected_files_rehashed_unchanged=len(old), archived_new_files=len(members),
                   archive_sha256=digest(archive), all_members_reopened_SHA_verified=True,
                   raw_sensor_files_not_duplicated_or_moved=True, close_progress_027_index_attempt_failed=True,
                   failure_reason='模块索引假定所有源码都有main，已保留状态文件；013收尾支持module入口', goal_complete=False)
    with (archive_dir/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
