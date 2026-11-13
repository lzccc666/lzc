"""交付002：保留001的只新增状态写入失败；使用新版快照和显式状态替换，不重跑实验。"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def current_process(row):
    try:
        data = (Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        actual = data[19] if data[0] != 'Z' else None
    except OSError:
        actual = None
    return dict(pid=row['pid'], start_ticks=row['start_ticks'], alive=actual == row['start_ticks'],
        actual_start_ticks=actual, role=row.get('role'))


def main():
    base = OUTPUT/'reproduction/three_task_fixedlogical_001'
    audit = read(base/'independent_audit_001/report.json')
    assert audit['passed'] and audit['full_three_task_fixedlogical_small_task_gate']
    executed = read(base/'execution_finished.json')['results']
    assert executed['cold']['actual_Qwen_backend_calls'] == 3 and executed['warm']['actual_Qwen_backend_calls'] == 0
    release = read(base/'experiment_release.json')
    manifest = read(base/'manifest.json')
    isolated = Path(manifest['isolated_root'])
    cases = {phase:isolated/'outputs/project_v2/thesis_completion_v1/interactive_simulation/three_task_fixedlogical_001'/phase
        for phase in ['cold', 'warm']}
    previous = read(OUTPUT/'source_bundle/source_checkpoint_035/checkpoint.json')
    export = OUTPUT/'source_bundle/github_sync_001/checkout'
    for name, expected in previous['sources'].items():
        assert digest(PROJECT/name) == digest(export/name) == expected
        assert (PROJECT/name).stat().st_mode & 0o777 == previous['source_modes'][name]
    assert digest(PROJECT/'.git/index') == previous['original_index_sha256']
    frozen_counts = {}
    for name, count in previous['frozen_sources_unchanged'].items():
        frozen = read(OUTPUT/name/'release.json')['source_sha256']
        assert len(frozen) == count
        assert all(digest(PROJECT/p) == expected for p, expected in frozen.items())
        frozen_counts[name] = count
    for name in ['mapping_evaluation/kitti_tracking_frontend_full_001', 'vehicle_execution/semantic_minus_execution_full_001']:
        frozen = read(OUTPUT/name/'release.json')['source_sha256']
        assert all(digest(PROJECT/p) == expected for p, expected in frozen.items())
        frozen_counts[name] = len(frozen)
    for name, expected in release['auxiliary_sha256'].items():
        assert digest(name) == expected
    assert digest(OUTPUT/'项目实验与复现交接包_v14.zip') == '9915110b691b1ce6ab01e47cab935c8e1bb52b977c66c8960d28880072c01700'
    assert '5 passed' in (OUTPUT/'source_bundle/tests_fixedlogical_semantic_queue_001.log').read_text()
    now = datetime.now(timezone.utc).isoformat()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    human = {key:json.load(opener.open('http://127.0.0.1:8768/api/'+key, timeout=10))
        for key in ['review_status', 'natural_collection_status']}
    minus = read(OUTPUT/'vehicle_execution/semantic_minus_execution_full_001/progress.json')
    raw_minus = sum(1 for _ in (OUTPUT/'interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl').open())
    processes = [current_process(row) for row in read(OUTPUT/'source_bundle/continuation_checkpoint_051.json')['process_snapshot']]
    evidence = [
        'reproduction/three_task_fixedlogical_001/experiment_release.json',
        'reproduction/three_task_fixedlogical_001/execution_finished.json',
        'reproduction/three_task_fixedlogical_001/cold_live_process_proof_001.json',
        'reproduction/three_task_fixedlogical_001/independent_audit_001/report.json',
        'reproduction/three_task_fixedlogical_001/independent_audit_001/cold/report.json',
        'reproduction/three_task_fixedlogical_001/independent_audit_001/warm/report.json',
        'source_bundle/tests_fixedlogical_semantic_queue_001.log',
        'task_protocol/three_task_fixedlogical_cache_001.md',
    ]
    outcomes = {phase:{key:value for key, value in row.items() if key in
        ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival']}
        for phase, row in executed.items()}
    state = deepcopy(read(OUTPUT/'实验完成度与交付清单_014.json'))
    state.update(updated_utc=now, goal_complete=False, manuscript_work_deferred_by_user=True,
        current_source_files=390, source_checkpoint='source_bundle/source_checkpoint_036',
        current_sources_published=False, tests_original_last_passed=452, tests_export_last_passed=450,
        tests_export_data_skipped=2, human_snapshot=human, human_service_snapshot=human,
        semantic_minus_recorded_snapshot=raw_minus, semantic_minus_snapshot=raw_minus,
        semantic_minus_current_incremental_audit=minus['audited_cases'],
        full_three_task_fixedlogical_small_task_gate=True, fixedlogical_actual_cold_CPU_Qwen_calls=3,
        fixedlogical_actual_warm_Qwen_calls=0, fixedlogical_four_outcomes=outcomes,
        process_snapshot=processes, available_disk_bytes=shutil.disk_usage(OUTPUT).free,
        tests_not_repeated_for_auxiliary_only_changes=True)
    for item in state['items']:
        if item['name'] == '语义消融':
            item['implemented'] = f'原GPU进程顺序执行，当前{raw_minus}/450；独立执行审核已覆盖{minus["audited_cases"]}项。'
        if item['name'] == '统一实验与交付入口':
            item['status'] = '门户016桌面手机及失败降级验收通过'
        if item['name'] == '缺一450独立车身方向与保持审核':
            item['implemented'] = f'实时审核{minus["audited_cases"]}/450；固定125前缀的独立账目已经通过，完整450仍待结束。'
    state['items'].append(dict(stage=12, name='三任务固定逻辑时钟完整闭环缓存', status='固定小任务独立验收通过',
        implemented='真实冷CPU Qwen三调用；热运行真实重新渲染/检测/建图/控制，三任务全部命中且后端零调用，完整业务逐项精确一致。',
        remaining='固定开发任务、单任务首次请求预算、同步逻辑投递；不是实时异步、多场景准确率、新封存或GPU性能结论。', evidence=evidence))
    state['evidence_sha256'].update({name:digest(OUTPUT/name) for name in evidence})
    dump(OUTPUT/'实验完成度与交付清单_016.json', state)
    dump(OUTPUT/'source_bundle/three_task_cache_bookkeeping_correction_002.json', dict(
        original_failure='postrun_001/01.log: validation_report.json已存在，dump使用open(x)拒绝覆盖',
        original_postrun_error=read(base/'postrun_001/interrupted.json'),
        original_helper_sha256=digest(OUTPUT/'source_bundle/finalize_three_task_cache_delivery_001.py'),
        partial_015_preserved=True, actual_model_calls_added=0, actual_navigation_runs_added=0, goal_complete=False))
    cold, warm = executed['cold'], executed['warm']
    text = f'''# 实验完成度与交付准备（第16版）

更新时间：{now}。当前只完善实验、代码、复现与交接，不新增论文正文、章节或框架。阶段9—12均未全部验收，goal_complete=false，新封存未开启。

| 阶段 | 已实现和实际核验 | 尚待完成 |
|---|---|---|
| 9 任务、对象与场景图 | 独立目标和终点协议；38开发场景因果建图、关系、身份及上限诊断；KITTI Object64和Tracking90实际实验 | 颜色/细属性/歧义真实人工审核；新封存集完整质量评价 |
| 10 语义模型 | 2398训练/603验证、三个正式种子Adapter训练、公平三任务真实推理与冻结请求缓存复跑；主450开发对照 | 缺一语义消融450及完整750组合统计；可靠真实标签覆盖；新封存对照 |
| 11 车辆与交互闭环 | 自行车控制、整车连续碰撞与方向核验；50冻结任务×3种子；三模型450尝试；可交互实际感知导航；最低正确到达功能已满足 | 正式GPU故障停止/恢复及消融的完整独立执行核验；最终统一验收 |
| 12 完整实验与交付 | 配置/log聚类统计、主450及KITTI浏览器、390文件独立源码实际复演、便携准备；本次三任务完整逻辑缓存闭环 | 至少20新场景各20有效自然指令、冻结后正式新测试；GPU故障与独占性能；最终保护/打包/Git同步 |

主450尝试449完整加1超时均保留分母。各模型90个正任务条件，规则/base/新Adapter的路线/曾选对/曾正确锁定/实际到达分别为83/84/82/50、66/72/66/39、59/33/32/4。新Adapter下游负结果保留，不以格式或调用次数代替改进。

本次固定config00-positive0、seed17、同步逻辑计算和下一感知帧投递。冷运行实际CPU Qwen三任务各1次，{cold['actual_frontend_calls']}次真实前端、{cold['saved_control_intervals']}控制；热运行{warm['actual_frontend_calls']}次真实前端、{warm['saved_control_intervals']}控制，三任务命中缓存、实际后端调用0、没有加载Qwen。两次初态、实际传感器、完整检测pickle内容、对象/关系/候选/路径、模型原文与规范输出、全部控制、终态和四结果精确一致。只规范化角色路径、实测耗时、单列缓存状态和已核验压缩容器元数据，不加业务数值容差。独立连续整车/方向/正确身份及终点保持核验见本次报告；四结果：{json.dumps(outcomes, ensure_ascii=False)}。同一开发任务重跑不增加独立任务或主表成绩，不代表实时异步或GPU性能。

本次两次均未正确到达：到达字段为false，系统在自己的感知接近点停车后失去正确身份；缓存一致不能获得到达信用。001交付整理因既有状态文件的只新增写入保护退出，失败日志和已写第15版快照保留；本版只修复交付，不重跑实验。

语义消融当前{raw_minus}/450，独立执行审核{minus['audited_cases']}/450；原进程及后续兼容003、GPU故障、独占性能队列保持。人工审核{human['review_status']['human_submissions']}，人工表达{human['natural_collection_status']['human_authored_submissions']}，合格新测试指令{human['natural_collection_status']['qualified_test_queries']}。不能用自动标签和模板补齐人工要求。

KITTI Tracking90的74个主类别GT观测只在一个窗口/4个实例；原融合5/74、改进63/74是该开发诊断结果，不称官方AP/HOTA、跨场景显著性或导航到达。负结果与未知项照常保留。

普通回归最近有效452通过，纯代码导出450通过、2真实数据跳过；本次逻辑队列5项边界测试通过，辅助Ruff通过。390业务源码及根Git索引未变；历史冻结依赖保持，完整历史文件最终保护待GPU作业结束后重做。当前390文件未推送，最后公开365文件commit f75725f8fd422ec619775bfa0407dc6733312488。按用户授权，实验收尾核验后再同步GitHub。

复现入口：`reproduction/standalone_portable_release_001/README.md`。本次证据：`reproduction/three_task_fixedlogical_001/independent_audit_001/report.json`。逐项证据与哈希见同名JSON。论文由后续GPT Work根据完成后的实验交接组织写作。
'''
    (OUTPUT/'实验完成度与交付清单_016.md').write_text(text)
    (OUTPUT/'三任务逻辑缓存_实验交接_002.md').write_text(text.replace('# 实验完成度与交付准备（第16版）', '# 三任务逻辑缓存与当前实验交接'))
    before = OUTPUT/'source_bundle/status_aliases_before_016'
    before.mkdir(exist_ok=False)
    aliases = ['工作交接_代码与实验.md', 'GPT_Work_写作交接.md', 'reproducible_commands.md', 'remaining_issues.md', 'experiment_report.md', 'validation_report.json']
    for name in aliases:
        shutil.copy2(OUTPUT/name, before/name)
        if name.endswith('.md'):
            path = OUTPUT/name
            path.write_text(text+'\n\n---\n\n以下为此前保留快照，当前状态以上述第16版及对应原始证据为准。\n\n'+path.read_text())
    validation = read(OUTPUT/'validation_report.json')
    validation.update(updated_utc=now, goal_complete=False, manuscript_work_deferred_by_user=True)
    statuses = validation['statuses']
    statuses['任务定义']['remaining'] = '独立评价及真实主450开发闭环已完成；真实标签和新封存的完整验收待完成。'
    statuses['独立消融']['remaining'] = f'当前实际{raw_minus}/450；完整750组合与独立统计尚未完成，兼容003保留原失败后显式审核。'
    statuses['缓存'] = dict(status='冻结请求及固定小任务三任务闭环均完成，范围受限', evidence=evidence,
        remaining='五模型冻结请求复跑和本次CPU三任务逻辑闭环分列；不能外推真实时钟相同轨迹、实时性能或多场景准确率。')
    statuses['测试'] = dict(status='原机452通过、纯代码450通过2真实数据跳过',
        evidence=['source_bundle/tests_full_051.log', 'source_bundle/github_sync_001/tests_export_020.log',
            'source_bundle/ruff_065.log', 'source_bundle/tests_fixedlogical_semantic_queue_001.log'],
        remaining='本次只增加辅助缓存工具和真实集成核验；最终业务变更后按影响复核，不重复加载模型。')
    statuses['实验交接最新清单'] = dict(status='第16版实验状态和交付准备，Goal未完成',
        evidence=['实验完成度与交付清单_016.md', '实验完成度与交付清单_016.json'],
        remaining='GPU队列、真实标签、新封存及最终打包/Git同步继续；当前不写论文正文。')
    statuses['实验复现交付最新包'] = dict(status='三任务缓存增量交接；整个Goal最终包未完成',
        evidence=['source_bundle/three_task_cache_bundle_002.json', 'source_bundle/three_task_cache_delivery_receipt_002.json'],
        remaining='增量含本次传感器和便携入口前置；外部模型、资产、Python/OSMesa不内置。v14及此前各包保留，最终整合待全部必需实验结束。')
    statuses['缺一语义消融独立执行审核']['status'] = f'实时已覆盖{minus["audited_cases"]}项，固定125前缀独立账目通过'
    statuses['便携入口真实完整复现']['remaining'] = '现有环境依赖复用，同任务不增加独立场景；本次另完成三任务固定逻辑缓存小任务验收。'
    statuses['三任务完整逻辑缓存闭环'] = dict(status='固定小任务验收通过', evidence=evidence,
        remaining='同步逻辑协议、共享CPU、单任务三请求；不当实时异步/独占性能/多场景准确率或新封存。')
    temporary = OUTPUT/'validation_report.cache_delivery_002.tmp'
    temporary.write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(OUTPUT/'validation_report.json')
    checkpoint = OUTPUT/'source_bundle/source_checkpoint_036'
    checkpoint.mkdir(exist_ok=False)
    prior_dir = OUTPUT/'source_bundle/source_checkpoint_035'
    shutil.copy2(prior_dir/'source_current.tar.gz', checkpoint/'source_current.tar.gz')
    shutil.copytree(prior_dir/'reproduction_sources', checkpoint/'reproduction_sources')
    new_helpers = ['fixedlogical_semantic_bridge_001.py', 'three_task_fixedlogical_worker_001.py',
        'run_three_task_fixedlogical_001.py', 'three_task_fixedlogical_audit_001.py',
        'test_fixedlogical_semantic_queue_001.py', 'finalize_three_task_cache_delivery_001.py',
        'three_task_cache_postrun_001.py', 'finalize_three_task_cache_delivery_002.py']
    for name in new_helpers:
        shutil.copy2(OUTPUT/'source_bundle'/name, checkpoint/'reproduction_sources'/name)
    entry = deepcopy(previous)
    entry.update(created_utc=now, source_archive_reused_from='source_checkpoint_035',
        auxiliary_sources={p.name:digest(p) for p in (checkpoint/'reproduction_sources').iterdir() if p.is_file()},
        checklist='实验完成度与交付清单_016.md', fixedlogical_cache_evidence_sha256={name:digest(OUTPUT/name) for name in evidence},
        semantic_minus_snapshot=raw_minus, human_snapshot=human, tests_original_last_passed=452,
        tests_export_last_passed=450, tests_export_data_skipped=2,
        tests_original_log_sha256=digest(OUTPUT/'source_bundle/tests_full_051.log'),
        tests_export_log_sha256=digest(OUTPUT/'source_bundle/github_sync_001/tests_export_020.log'),
        unified_portal_browser_QA_sha256=digest(OUTPUT/'visualizations/thesis_portal_browser_qa_019/report.json'), goal_complete=False)
    dump(checkpoint/'checkpoint.json', entry)
    with tarfile.open(checkpoint/'source_current.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(previous['sources'])
        for member in members:
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == previous['sources'][member.name]
            assert member.mode & 0o777 == previous['source_modes'][member.name]
    derived = {'doc/thesis_completion/experiment_delivery_status_016.md':'实验完成度与交付清单_016.md',
        'doc/thesis_completion/experiment_delivery_status_016.json':'实验完成度与交付清单_016.json',
        'doc/thesis_completion/three_task_cache_handoff.md':'三任务逻辑缓存_实验交接_002.md',
        'doc/thesis_completion/validation_snapshot_016.json':'validation_report.json'}
    for name in evidence:
        if Path(name).name != 'report.json':
            derived['experiments/three_task_fixedlogical/'+str(Path(name).name)] = name
    derived['experiments/three_task_fixedlogical/report.json'] = 'reproduction/three_task_fixedlogical_001/independent_audit_001/report.json'
    for phase in ['cold', 'warm']:
        derived[f'experiments/three_task_fixedlogical/{phase}/report.json'] = f'reproduction/three_task_fixedlogical_001/independent_audit_001/{phase}/report.json'
        derived[f'experiments/three_task_fixedlogical/{phase}/result.json'] = str((cases[phase]/'result.json').relative_to(OUTPUT))
    for name in new_helpers:
        derived['reproduction/'+name] = 'source_bundle/'+name
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for target, source in derived.items():
        path = export/target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT/source, path)
        assert not secret.search(path.read_bytes()), target
    export_state = read(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_021.json')
    export_state.update(created_utc=now, current_sources_published=False,
        current_increment_actual_CPU_frontend_calls=sum(r['actual_frontend_calls'] for r in executed.values()),
        current_increment_actual_Qwen_calls=3, current_increment_navigation_runs=2,
        current_increment_frozen_main_arrival_credit=0, current_increment_new_independent_tasks=0,
        fixedlogical_cache_evidence={target:dict(source=source, sha256=digest(export/target)) for target, source in derived.items()})
    dump(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_022.json', export_state)
    subprocess.run(['git', 'diff', '--check'], cwd=PROJECT, check=True)
    subprocess.run(['git', 'diff', '--check'], cwd=export, check=True)
    assert digest(PROJECT/'.git/index') == previous['original_index_sha256']
    continuation = dict(updated_utc=now, main_recorded=450, main_completed=449, main_failed=1,
        main_outcomes=state['main_outcomes'], semantic_minus_recorded=raw_minus, semantic_minus_expected=450,
        semantic_minus_incremental_audit=minus, process_snapshot=processes, human=human,
        source_checkpoint='source_bundle/source_checkpoint_036', current_source_files=390,
        current_sources_published=False, last_public_commit='f75725f8fd422ec619775bfa0407dc6733312488',
        tests_original_last_passed=452, tests_export_last_passed=450, tests_export_data_skipped=2,
        full_three_task_fixedlogical_small_task_gate=True, fixedlogical_four_outcomes=outcomes,
        manuscript_work_deferred_by_user=True, new_test_opened=False, classification='progress',
        stage9_complete=False, stage10_complete=False, stage11_complete=False, stage12_complete=False, goal_complete=False)
    dump(OUTPUT/'source_bundle/continuation_checkpoint_052.json', continuation)
    receipt = dict(passed=True, updated_utc=now, source_files=390, original_index_unchanged=True,
        frozen_source_sets_checked=frozen_counts, full_historical_tree_not_rechecked=True,
        cold_calls=3, warm_calls=0, cold_frontend=cold['actual_frontend_calls'], warm_frontend=warm['actual_frontend_calls'],
        fixedlogical_four_outcomes=outcomes, new_independent_tasks=0, frozen_main_arrival_credit=0,
        source_checkpoint='source_bundle/source_checkpoint_036', current_sources_published=False,
        evidence_sha256={name:digest(OUTPUT/name) for name in evidence},
        copied_export_sha256={name:digest(export/name) for name in derived},
        manuscript_text_written=False, new_test_opened=False, goal_complete=False)
    dump(OUTPUT/'source_bundle/three_task_cache_delivery_receipt_002.json', receipt)
    paths = {OUTPUT/name for name in evidence+list(derived.values())}
    paths.update({checkpoint/'checkpoint.json', checkpoint/'source_current.tar.gz',
        OUTPUT/'source_bundle/continuation_checkpoint_052.json', OUTPUT/'source_bundle/three_task_cache_delivery_receipt_002.json',
        base/'manifest.json', base/'launcher_process.json', base/'cold_exit.json', base/'warm_exit.json',
        base/'cold_worker.log', base/'warm_worker.log', OUTPUT/'source_bundle/ruff_three_task_fixedlogical_delivery_002.log'})
    paths.update({OUTPUT/'source_bundle/three_task_cache_bookkeeping_correction_002.json',
        base/'warm_live_process_proof_001.json', base/'prefix_comparison_001.json'})
    paths.update(p for p in (base/'postrun_001').iterdir() if p.is_file())
    for directory in [base/'independent_audit_001', *cases.values(), OUTPUT/'reproduction/standalone_portable_release_001']:
        paths.update(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
    cache = isolated/'outputs/project_v2/thesis_completion_v1/cache_audit/three_task_fixedlogical_001/cache'
    paths.update(cache.rglob('*.json'))
    archive_path = OUTPUT/'三任务逻辑缓存_实验增量包_002.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=str(path.relative_to(OUTPUT)))
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(paths)
        assert archive.testzip() is None
    dump(OUTPUT/'source_bundle/three_task_cache_bundle_002.json', dict(path=str(archive_path),
        sha256=digest(archive_path), bytes=archive_path.stat().st_size, files=len(paths),
        raw_actual_sensor_evidence=True, portable_release_prerequisites_included=True,
        model_weights_included=False, third_party_asset_included=False,
        Python_and_OSMesa_external=True, final_project_delivery=False, goal_complete=False))
    print(json.dumps(dict(passed=True, source_files=390, minus=raw_minus, zip_files=len(paths), goal_complete=False)))


if __name__ == '__main__':
    main()
