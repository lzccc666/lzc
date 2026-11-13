"""便携实际复现和增量消融审核的交付快照；冻结主表与论文原稿保持只读。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import zipfile

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def tick(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except OSError:
        return None


def main():
    now = datetime.now(timezone.utc).isoformat()
    portable = OUTPUT/'reproduction/portable_executed_001'
    manifest = read(portable/'manifest.json')
    case = Path(manifest['isolated_root'])/'outputs/project_v2/thesis_completion_v1/interactive_simulation/live_navigation_sessions_001'/manifest['session']
    actual = read(case/'result.json')
    independent = read(portable/'independent_audit_001/report.json')
    comparison = read(OUTPUT/'reproduction/portable_business_comparison_002/report.json')
    ledger_root = OUTPUT/'vehicle_execution/semantic_minus_execution_ledger_prefix_001'
    ledger = read(ledger_root/'report.json')
    assert independent['passed'] and independent['portable_entry_actually_executed']
    assert actual['actual_full_perception_arrival'] and ledger['passed']
    previous = read(OUTPUT/'source_bundle/source_checkpoint_034/checkpoint.json')
    export = OUTPUT/'source_bundle/github_sync_001/checkout'
    assert digest(PROJECT/'.git/index') == previous['original_index_sha256']
    for name, expected in previous['sources'].items():
        assert digest(PROJECT/name) == digest(export/name) == expected
        assert (PROJECT/name).stat().st_mode & 0o777 == previous['source_modes'][name]
    for rel, count in previous['frozen_sources_unchanged'].items():
        rows = read(OUTPUT/rel/'release.json')['source_sha256']
        assert len(rows) == count
        assert all(digest(PROJECT/name) == expected for name, expected in rows.items())
    for name, expected in read(OUTPUT/'mapping_evaluation/kitti_tracking_frontend_full_001/release.json')['source_sha256'].items():
        assert digest(PROJECT/name) == expected
    source = OUTPUT/'vehicle_execution/semantic_minus_execution_full_001'
    for name, expected in read(source/'release.json')['source_sha256'].items():
        assert digest(PROJECT/name) == expected
    worker = read(source/'worker_process.json')
    assert tick(worker['pid']) == worker['start_ticks']
    current = read(source/'progress.json')
    minus = sum(1 for _ in (OUTPUT/'interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl').open())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    human = {name:json.load(opener.open('http://127.0.0.1:8768/api/'+name, timeout=10))
        for name in ['review_status', 'natural_collection_status']}
    continuation = read(OUTPUT/'source_bundle/continuation_checkpoint_050.json')
    processes = continuation['process_snapshot']
    for row in processes:
        assert tick(row['pid']) == row['start_ticks']
    processes.append(dict(pid=worker['pid'], start_ticks=worker['start_ticks'], alive=True, role='incremental_semantic_minus_execution_audit'))
    evidence = ['reproduction/portable_executed_001/independent_audit_001/report.json',
        'reproduction/portable_business_comparison_002/report.json',
        'vehicle_execution/semantic_minus_execution_ledger_prefix_001/report.json',
        'vehicle_execution/semantic_minus_execution_full_001/release.json',
        'source_bundle/portable_execution_process_probe_001.json',
        'reproduction/portable_business_comparison_001/report.json',
        'reproduction/portable_business_comparison_002/all_detector_containers.json']
    check = read(OUTPUT/'实验完成度与交付清单_013.json')
    check.update(updated_utc=now, source_checkpoint='source_bundle/source_checkpoint_035',
        semantic_minus_snapshot=minus, human_snapshot=human,
        portable_entry_actually_executed=True, portable_business_comparison_passed=comparison['passed'],
        semantic_minus_independent_execution_audit_active=True,
        semantic_minus_independently_verified_prefix_cases=ledger['audited_prefix_cases'],
        available_disk_bytes=shutil.disk_usage(OUTPUT).free, goal_complete=False)
    for item in check['items']:
        if item['name'] == '语义消融':
            item['implemented'] = f'原GPU450当前{minus}/450；独立执行审核跟随，固定前缀{ledger["audited_prefix_cases"]}项已核对，完整750组合尚待结束。'
        if item['name'] == '隔离源码完整导航复演与便携准备':
            item['status'] = '便携入口实际完整执行及独立审核通过'
            item['implemented'] += f' 本次从无Git/无历史outputs的纯源码目录，通过便携准备器实际运行{actual["actual_frontend_calls"]}次前端/{actual["saved_control_intervals"]}控制并正确到达；完整逻辑业务比较passed={comparison["passed"]}。'
            item['evidence'].extend(evidence[:2])
    check['items'].append(dict(stage=12, name='缺一450独立车身方向与保持审核', status='增量执行中，已有固定前缀账目通过',
        implemented=f'{ledger["audited_prefix_cases"]}项/{ledger["control_intervals"]}控制区间，原到达{ledger["counts"]["reported_arrivals"]}全部有综合证据；严格观测初态{ledger["strict_observed_initials"]}、显式重建{ledger["reconstructed_initials"]}；静止单列。',
        remaining='完整450尚未结束，部分前缀不作方法比较或总体结论；关联仍是粗类别/XY门限而非人工实例真值。', evidence=evidence[2:4]))
    check['evidence_sha256'].update({name:digest(OUTPUT/name) for name in evidence})
    dump(OUTPUT/'实验完成度与交付清单_014.json', check)
    status = '逐项完全相同' if comparison['passed'] else '存在业务差异，见differences.json，未删除差异字段或采用数值容差强制一致'
    note = f'''# 实验进度与复现交付（第14版）

更新：{now}。继续实验、代码和交付，不新增论文正文、章节或框架；Goal和阶段9—12均未全部完成，新封存未开启。

便携入口已真正完成导航：从没有.git和历史outputs的390文件目录出发，准备器校验源码/权限、任务、模型/资产SHA，再独立启动实际渲染和CPU YOLO/MobileSAM/CLIP。实际{actual['actual_frontend_calls']}次前端、{actual['saved_control_intervals']}控制，正确对象、区域、航向、低速及连续2秒保持通过独立审核。与前版相同任务/种子，独立场景增量0，主表信用0。原001未到达和资源超限结果完整保留。

完整逻辑业务比较：{status}。首版001严格压缩文件SHA比较失败已保留；002逐个确认38个检测pickle解压后全部字节完全相同，压缩文件差异只在gzip头4字节mtime，原容器SHA也逐项核对并保留。比较初态、全部感知/节点/关系/候选/路径/图像SHA、全部控制和终态；只规范化源码/case路径前缀及列明的实测耗时字段，不丢弃业务字段。两次均未调用Qwen，不代表完整三任务Qwen闭环缓存门槛完成。使用声明的同一Python和图形依赖，不冒称新OS或全新虚拟环境。

缺一语义消融GPU当前{minus}/450，原进程未重启。新增独立CPU审核逐项跟随，只读完整终态记录；当前实时审核{current['audited_cases']}项。另已冻结并独立复核前{ledger['audited_prefix_cases']}项/{ledger['control_intervals']}控制区间/{ledger['moving_intervals']}运动区间和{ledger['SHA_files']}个输入/证据SHA。严格保存初态{ledger['strict_observed_initials']}项、显式种子重建{ledger['reconstructed_initials']}项；已发生的{ledger['counts']['reported_arrivals']}次原到达均有整车、方向与保守全区间保持证据，未新增到达。静止{ledger['counts'].get('direction_not_exercised_stationary', 0)}项单列，不给行驶安全信用。全450审核未完成，不把前缀混成最终消融对比。

已完成的开发实验仍见第13版：主450规则/base/新Adapter各90正条件正确到达50/39/4，新Adapter负结果保留；KITTI Object64和Tracking90已实际执行并独立核验，Tracking所有74主类别GT仅在一个窗口/4实例，不能宣称跨场景显著性、官方AP或KITTI导航到达。

待完成：GPU缺一450和兼容003完整750组合；其后原队列GPU故障与独占性能；真实人工审核与合格自然指令；至少20新场景各20有效指令及正式新封存对比；完整三任务固定逻辑时钟闭环缓存验收；最终全量保护、整合交付和Git同步。人工审核{human['review_status']['human_submissions']}、人工表达{human['natural_collection_status']['human_authored_submissions']}、合格指令{human['natural_collection_status']['qualified_test_queries']}。不得用机器标签或模板补记人工数据。

当前390业务文件不变，原452回归/导出450通过2数据skip及辅助6项边界测试仍是最近有效验证；本次增加实际便携集成和消融审核前缀，相关Ruff通过，不因仅辅助文件新增重复加载所有测试。11份历史冻结依赖、新KITTI冻结依赖、根Git索引和本次活跃审核依赖均核对。完整历史树最终保护仍待GPU结束后重做。

交付入口：`reproduction/standalone_portable_release_001/README.md`，业务比较`reproduction/portable_business_comparison_002/report.json`，消融前缀`vehicle_execution/semantic_minus_execution_ledger_prefix_001/report.json`。检查点035；当前代码未推送，最后公开仍365源码commit f75725f8fd422ec619775bfa0407dc6733312488。新增量包不是整个Goal最终完成证明。
'''
    (OUTPUT/'实验完成度与交付清单_014.md').write_text(note)
    (OUTPUT/'便携复现与消融审核_交接_001.md').write_text(note)
    for name in ['工作交接_代码与实验.md', 'GPT_Work_写作交接.md', 'reproducible_commands.md', 'remaining_issues.md', 'experiment_report.md']:
        path = OUTPUT/name
        path.write_text(note+'\n\n---\n\n以下保留此前记录；当前状态以第14版及对应原始证据为准。\n\n'+path.read_text())
    validation = read(OUTPUT/'validation_report.json')
    validation.update(updated_utc=now, goal_complete=False, manuscript_work_deferred_by_user=True)
    validation['statuses']['便携入口真实完整复现'] = dict(status='实际执行及独立审核通过', evidence=evidence[:2],
        remaining='现有环境依赖复用；同任务不增加独立场景；三任务Qwen闭环缓存仍待。')
    validation['statuses']['缺一语义消融独立执行审核'] = dict(status=f'固定前缀{ledger["audited_prefix_cases"]}已核对，正式增量仍运行', evidence=evidence[2:4],
        remaining='等待450全部实际结束及最终输入账目；保持严格/重建、运动/静止分列。')
    write(OUTPUT/'validation_report.json', validation)
    checkpoint = OUTPUT/'source_bundle/source_checkpoint_035'
    checkpoint.mkdir(exist_ok=False)
    old = OUTPUT/'source_bundle/source_checkpoint_034'
    shutil.copy2(old/'source_current.tar.gz', checkpoint/'source_current.tar.gz')
    shutil.copytree(old/'reproduction_sources', checkpoint/'reproduction_sources')
    auxiliary = ['semantic_minus_execution_audit_001.py', 'semantic_minus_execution_ledger_001.py',
        'portable_execution_independent_audit_001.py', 'portable_business_reproduction_compare_001.py',
        'portable_gzip_metadata_audit_001.py', 'finalize_portable_and_minus_audit_001.py']
    for name in auxiliary:
        shutil.copy2(OUTPUT/'source_bundle'/name, checkpoint/'reproduction_sources'/name)
    previous.update(created_utc=now, source_archive_reused_from='source_checkpoint_034',
        source_archive_sha256=digest(checkpoint/'source_current.tar.gz'),
        auxiliary_sources={p.name:digest(p) for p in (checkpoint/'reproduction_sources').iterdir() if p.is_file()},
        checklist='实验完成度与交付清单_014.md', semantic_minus_snapshot=minus, human_snapshot=human,
        semantic_minus_audit_worker=worker, semantic_minus_verified_prefix=ledger['audited_prefix_cases'],
        portable_actual_execution_evidence={name:digest(OUTPUT/name) for name in evidence},
        current_sources_published=False, manuscript_text_written=False, goal_complete=False)
    dump(checkpoint/'checkpoint.json', previous)
    small = {
        'doc/thesis_completion/experiment_delivery_status_014.md':'实验完成度与交付清单_014.md',
        'doc/thesis_completion/experiment_delivery_status_014.json':'实验完成度与交付清单_014.json',
        'doc/thesis_completion/portable_execution_and_minus_audit.md':'便携复现与消融审核_交接_001.md',
        'doc/thesis_completion/validation_snapshot.json':'validation_report.json',
    }
    for name in evidence:
        small['experiments/portable_and_minus_audit/'+name.replace('/', '__')] = name
    for name in auxiliary:
        small['reproduction/'+name] = 'source_bundle/'+name
    small['experiments/portable_and_minus_audit/protocol.md'] = 'task_protocol/semantic_minus_execution_audit_001.md'
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for name, source_name in small.items():
        target = export/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT/source_name, target)
        assert not secret.search(target.read_bytes()), name
    prior_export = read(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_020.json')
    prior_export.update(created_utc=now, current_sources_published=False,
        portable_entry_executed=True, current_increment_actual_CPU_frontend_calls=actual['actual_frontend_calls'],
        current_increment_actual_Qwen_calls=0, current_increment_navigation_runs=1,
        current_increment_frozen_main_arrival_credit=0, current_increment_new_independent_tasks=0,
        portable_and_minus_audit_evidence={name:dict(source=value, sha256=digest(export/name)) for name, value in small.items()})
    dump(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_021.json', prior_export)
    continuation.update(updated_utc=now, source_checkpoint='source_bundle/source_checkpoint_035',
        semantic_minus_recorded=minus, human=human, process_snapshot=processes,
        semantic_minus_verified_prefix_cases=ledger['audited_prefix_cases'], portable_actual_execution=True,
        portable_business_comparison_passed=comparison['passed'], current_sources_published=False,
        classification='progress', goal_complete=False)
    dump(OUTPUT/'source_bundle/continuation_checkpoint_051.json', continuation)
    subprocess.run(['git', 'diff', '--check'], cwd=PROJECT, check=True)
    subprocess.run(['git', 'diff', '--check'], cwd=export, check=True)
    receipt = dict(passed=True, updated_utc=now, source_files=390, original_index_unchanged=True,
        source_checkpoint='source_bundle/source_checkpoint_035', live_processes=processes,
        portable_arrival_independently_certified=True, portable_business_comparison_passed=comparison['passed'],
        semantic_minus_verified_prefix_cases=ledger['audited_prefix_cases'], full_semantic_minus_audit_complete=False,
        evidence_sha256={name:digest(OUTPUT/name) for name in evidence}, copied_export_sha256={name:digest(export/name) for name in small},
        main_table_credit=0, new_independent_tasks=0, current_sources_published=False,
        whole_historical_tree_not_rechecked=True, manuscript_text_written=False, goal_complete=False)
    receipt_path = OUTPUT/'source_bundle/portable_and_minus_delivery_receipt_001.json'
    dump(receipt_path, receipt)
    paths = {OUTPUT/name for name in small.values()}
    paths.update({checkpoint/'checkpoint.json', checkpoint/'source_current.tar.gz', receipt_path,
        OUTPUT/'source_bundle/continuation_checkpoint_051.json', portable/'manifest.json', portable/'worker.log',
        portable/'launcher_exit.json', OUTPUT/'source_bundle/portable_execution_process_probe_001.json'})
    paths.update(p for p in case.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
    for folder in [portable/'independent_audit_001', OUTPUT/'reproduction/portable_business_comparison_001', OUTPUT/'reproduction/portable_business_comparison_002', ledger_root]:
        paths.update(p for p in folder.iterdir() if p.is_file())
    archive_path = OUTPUT/'便携复现与消融审核_实验增量包_001.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=str(path.relative_to(OUTPUT)))
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(paths)
        assert archive.testzip() is None
    dump(OUTPUT/'source_bundle/portable_and_minus_bundle_001.json', dict(path=str(archive_path),
        sha256=digest(archive_path), bytes=archive_path.stat().st_size, files=len(paths),
        raw_portable_execution_sensor_evidence_included=True, models_or_third_party_asset_included=False,
        entire_semantic_minus450_evidence_included=False, only_verified_prefix_ledger_included=True,
        project_final_delivery=False, goal_complete=False))
    print(json.dumps(dict(passed=True, portable_actual_arrival=True, business_equal=comparison['passed'],
        semantic_minus_snapshot=minus, verified_prefix_cases=ledger['audited_prefix_cases'],
        bundle_files=len(paths), bundle_bytes=archive_path.stat().st_size, goal_complete=False)))


if __name__ == '__main__':
    main()
