"""归档新代码与已闭合的CPU/标定证据；正在运行的008回合不纳入完成声明。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from conceptgraph.completion.stage9 import PROJECT, digest, dump

R = Path(__file__).resolve().parents[1]


def live_jobs():
    names = {'run_official_guarded_complex_016.py', 'run_guarded_complex_batch_008.py',
        'carla_guarded_complex_episode_008.py', 'run_guarded_motion_audits_001.py'}
    jobs = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            if len(args) < 2 or Path(args[1]).name not in names:
                continue
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] != 'Z':
                jobs.append(dict(PID=int(proc.name), start_tick=stat[19], command=args))
        except (OSError, UnicodeDecodeError):
            pass
    return jobs


def main():
    out = R/'source_bundle/closing_supplement_006'
    out.mkdir(exist_ok=False)
    old = {x['path']: x['sha256'] for x in json.loads(
        (R/'source_bundle/reproduction_003/source_manifest.json').read_text())}
    old.update({x['path']: x['sha256'] for x in json.loads(
        (R/'source_bundle/closing_supplement_001/receipt.json').read_text())['supplemental_files']})
    old.update({str(PROJECT/p): h for p, h in json.loads(
        (R/'source_bundle/closing_supplement_002/files.json').read_text()).items()})
    for version in ['003', '004', '005']:
        old.update({x['path']: x['sha256'] for x in json.loads(
            (R/f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    assert all(digest(p) == h for p, h in old.items()), '历史源码或已封装交付发生变化'
    qa = R/'json/software_validation_continuation_008'
    qa.mkdir(exist_ok=False)
    sources = sorted((R/'source').glob('*.py'))
    for source in sources:
        ast.parse(source.read_text())
    checks = []
    for name, command in [('ruff_functional', ['/root/miniconda3/envs/conceptgraph/bin/ruff',
            'check', '--no-cache', '--select', 'E9,F63,F7,F82', str(R/'source')]),
            ('git_diff_check', ['git', 'diff', '--check'])]:
        completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT)
        log = qa/(name+'.log')
        log.write_text(completed.stdout+completed.stderr)
        checks.append(dict(check=name, exit_code=completed.returncode, log_sha256=digest(log)))
        assert completed.returncode == 0
    testroot = R/'json/carla_persistent_intent_validation_003'
    for filename, count in [('unittest.log', 25), ('guarded_control_unittest.log', 4)]:
        content = (testroot/filename).read_text()
        assert f'Ran {count} tests' in content and content.rstrip().endswith('OK')
    dump(testroot/'report.json', dict(persistent_bridge_tests_passed=25,
        original_async_bridge_cases_included=13, new_always_guard_control_tests_passed=4,
        injected_backend_only=True, actual_GPU_or_physical_credit=0,
        source_sha256={str(p): digest(p) for p in [R/'source/carla_persistent_intent_bridge_003.py',
            R/'source/test_carla_persistent_intent_bridge_003.py', R/'source/carla_guarded_control_001.py',
            R/'source/test_carla_guarded_control_001.py', testroot/'unittest.log',
            testroot/'guarded_control_unittest.log']}, goal_complete=False))
    dump(qa/'report.json', dict(syntax_checked_source_files=len(sources), checks=checks,
        previous_source_and_delivery_files_rehashed_unchanged=len(old),
        original_business_sources_unchanged=True, original_452_business_tests_not_repeated=True,
        reason='本轮仅新增实验与核验源码；原业务文件与全部已封装文件逐项SHA一致',
        new_software_tests_report=str(testroot/'report.json'), goal_complete=False))
    idx = R/'source_bundle/module_index_007'
    idx.mkdir(exist_ok=False)
    rows = list(csv.DictReader((R/'source_bundle/module_index_006/modules.csv').open()))
    for row in rows:
        if row['module'] == '新CPU依赖完整导航':
            row.update(meaning='独立CPU依赖已完成561物理时刻/281前端，严格安全到达；同种子轨迹与原回合完全一致，不能重复计算泛化样本',
                evidence=str(R/'reproduction/clean_cpu_navigation_readback_003/report.json'),
                evidence_exists_at_index_generation=True, boundary='仅规则单任务复现，未验证复杂Qwen、实时或不同配置')
        elif row['module'] == '停车立即切油的物理标定':
            row.update(meaning='两组官方物理标定均通过，1020命令重算一致；两组停车距离约2.86/2.66m',
                evidence=str(R/'carla_official/prius_cutoff_readback_003/report.json'),
                evidence_exists_at_index_generation=True, boundary='标定没有目标选择/锁定/导航到达信用，不能代表其他道路')
    additions = [
        ('持续核验已锁定对象', 'carla_persistent_intent_bridge_003.py', 'PersistentIntentBridge',
            'json/carla_persistent_intent_validation_003/report.json', '25项软件回归通过；初次模型TTL不变，锁定后每帧新证据与路线再核验，不冒充新模型调用'),
        ('每物理时刻统一保护', 'carla_guarded_control_001.py', 'guarded_control',
            'carla_official/guarded_motion_prefix_001/rule/independent_motion_guard_audit_001/report.json', '覆盖原控制无路线提前返回；20真实时刻前缀回读通过，完整回合及物理安全另验'),
        ('整圈雷达与控制回读', 'audit_carla_guarded_execution_001.py', 'audit',
            'carla_official/guarded_motion_prefix_001/rule/independent_motion_guard_audit_001/report.json', '重建原LiDAR障碍和两种名义预测，核验下一控制；此项不是实际到达证明'),
        ('共同修复后的三方法回合', 'carla_guarded_complex_episode_008.py', 'main',
            'carla_official/guarded_complex_registration_008.json', '三方法同一完整传感器与控制/身份策略；正在运行，不能提前记成功或改善'),
        ('独立CPU导航最终回读', 'readback_clean_cpu_navigation_003.py', 'main',
            'reproduction/clean_cpu_navigation_readback_003/report.json', '19依赖均来自新venv/overlay，依赖运行前后不变；1个同种子复现正例'),
        ('立即切油逐条物理回读', 'readback_prius_cutoff_003.py', 'main',
            'carla_official/prius_cutoff_readback_003/report.json', '1020原始记录与命令重算；名义停车距离仅与两条实际曲线比较，不保证泛化'),
    ]
    for label, filename, entry, evidence, boundary in additions:
        source = R/'source'/filename
        node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == entry)
        rows.append(dict(module=label, source=str(source), entry=entry, line=node.lineno,
            meaning=boundary, evidence=str(R/evidence),
            evidence_exists_at_index_generation=(R/evidence).exists(),
            boundary=boundary, source_sha256=digest(source)))
    with (idx/'modules.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (idx/'README.md').open('x').write('模块索引007：CPU和标定终止状态已更新；008导航仍以真实终止报告为准。旧版本保留。\n')
    jobs = live_jobs()
    used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    handoff = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn_classification='progress：20个真实物理时刻前缀已逐项回读、全量审计作业已排队；中间修复人工审核入口说明',
        this_goal_turn_classification='progress：已闭合证据归档、源码保护重查、模块索引与交付更新',
        authoritative_closed_reports=['reproduction/clean_cpu_navigation_readback_003/report.json',
            'carla_official/prius_cutoff_readback_003/report.json',
            'carla_official/guarded_motion_prefix_001/rule/independent_motion_guard_audit_001/report.json'],
        fresh_CPU_navigation=dict(actual_episodes=1, physical_ticks=561, frontend_calls=281,
            strict_safe_arrival=True, isolated_modules=19, independent_generalization_sample=False),
        immediate_cutoff_calibration=dict(actual_cases=2, physics_ticks=1020, both_constraints_pass=True,
            actual_stopping_distances_m=[2.8555436756822123, 2.6648124539969946], navigation_credit=False),
        new_software_tests=dict(persistent_intent=25, always_guard=4),
        actual_guard_prefix_ticks=20, full_guard_audit_pending=True,
        current_live_jobs=jobs, current_008_navigation_completion_not_claimed=True,
        current_008_registration='carla_official/guarded_complex_registration_008.json',
        new_project_artifact_GiB=used/2**30, physical_free_GiB=shutil.disk_usage(R).free/2**30,
        artifact_budget_GiB=25, new_source_archive='source_bundle/closing_supplement_006',
        latest_module_index='source_bundle/module_index_007/modules.csv',
        human_review_pending=True, human_review_correct_offline_package='annotation/quick_human_review_002/人工审核_车辆行人20张_打开即用.zip',
        human_review_local_port_forward_not_proven=True,
        next_actions=['续读实际launcher与同一exec session20764；不因观察超时重复启动服务器',
            'launcher结束后104005作业逐回合全量重建LiDAR/保护/命令，续读session62947',
            '用新008终止及独立到达/选择/保护报告诊断是否有改善，不用软件测试替代实际到达',
            '继续10配置50任务3种子、至少5配置10个不同正任务正确安全到达、同任务闭环消融；新封存20场景≥400有效指令与真实审核未完成',
            '大范围前端日志与原始传感器需先冻结无损存储与预算，不能删除旧数据腾空间'],
        whole_project_sealed_test_opened=False, original_sources_and_archives_unchanged=True,
        thesis_prose_written=False, git_committed_or_pushed=False, goal_complete=False)
    handoff_path = R/'json/continuation_handoff_011.json'
    dump(handoff_path, handoff)
    doc = R/'交付补充说明007.md'
    doc.open('x').write('''# 实验续跑交付补充007

本版更新006中的运行状态，旧说明与旧结果保留。独立安装CPU环境已完成一次官方CARLA完整传感器导航，561物理时刻、281次真实检测/分割/建图；选对、锁定、进入终区、停车保持2秒，独立严格安全到达通过，所有控制限值零违规。19项依赖路径确实属于新venv或显式新overlay，运行前后依赖不变。该回合与旧seed17同任务轨迹完全一致，是复现证据，不增加独立任务或泛化分母。CPU约1552秒完成56秒仿真，不证明实时。证据：reproduction/clean_cpu_navigation_readback_003/。

原厂Prius立即切油008标定已实际完成两组，每组510时刻；最大加速约0.7125，最大减速绝对值约1.4862，均未超限且停住，无碰撞。1020条实际命令与冻结控制器重算一致。两组切油后的实际停车距离约2.856/2.665米；名义模型与190个停车状态比较，最小余量约-0.00123米，2厘米容差内覆盖，仅是两条实际曲线核查。未据此宣称其他坡度、路面、速度或完整导航安全。见carla_official/prius_cutoff_readback_003/。

持续对象身份核验有25项软件回归、原控制返回后始终保护有4项软件测试通过。初次模型判断仍保留原20秒真实时钟TTL及完整上下文绑定；真实锁定后持续核验同一对象的新证据与安全路线，不能把沿用身份当作新Qwen输出。008三方法共同采用该逻辑、立即切油和每物理时刻整圈LiDAR保护。当前真实回合仍在运行，全部结果以终止和独立到达/选择/保护报告为准。前20个真实时刻已经重建原LiDAR及保护预测，含6个无路线时刻，输入与实际下一命令一致；它不是全回合或到达证明。旧007三项失败均保留，多项共同修复的差值也不能当单因素消融。

审核旧入口001中有10张道路、1张树木而表单选项不齐，属于准备错误。检测类别road本来就在原项目中，未为审核新增检测器。新版002只有11汽车、4公交、3卡车、1行人、1摩托车。使用annotation/quick_human_review_002/人工审核_车辆行人20张_打开即用.zip，解压双击HTML即可离线审核并导出JSON；本机临时转发端口的可达性尚未证实，不能把服务器HTTP200当作用户电脑已能访问。没有代填或伪造人工记录。

源码增量和哈希见source_bundle/closing_supplement_006/，最新模块行号/入口索引见source_bundle/module_index_007/modules.csv；源码保护、语法与功能性检查见json/software_validation_continuation_008/。原始传感器与仍写入的日志不封装为完成证据。恢复先看json/continuation_handoff_011.json中的实际PID，不能重新启动正在运行的批次。

仍需原任务规定的10配置50任务3种子与至少5配置10个不同正任务安全到达、完整同任务消融、新封存至少20场景400有效指令、真人审核和最终交付验收。KITTI离线连续建图不能记实际闭环到达，规则复现也不能代表复杂Qwen。没有撰写论文正文或推送Git，goal_complete=false。
''')
    paths = [p for p in sources if str(p) not in old]
    paths += [doc, handoff_path, idx/'modules.csv', idx/'README.md', qa/'report.json',
        testroot/'report.json', testroot/'unittest.log', testroot/'guarded_control_unittest.log',
        R/'reproduction/clean_cpu_navigation_readback_003/report.json',
        R/'carla_official/prius_cutoff_readback_003/report.json',
        R/'carla_official/guarded_complex_registration_008.json',
        R/'carla_official/guarded_motion_prefix_001/rule/prefix_provenance.json',
        R/'carla_official/guarded_motion_prefix_001/rule/independent_motion_guard_audit_001/report.json',
        R/'annotation/quick_review_version_correction_003.json',
        R/'annotation/review_access_diagnosis_004.json']
    paths += list((R/'carla_official/prius_cutoff_readback_003').glob('*stopping_distances.jsonl'))
    members = [dict(path=str(p), archive_path=str(p.relative_to(PROJECT)),
        bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(set(paths))]
    archive = out/'source_delta.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for row in members:
            tar.add(row['path'], arcname=row['archive_path'], recursive=False)
    with tarfile.open(archive) as tar:
        for row in members:
            assert hashlib.sha256(tar.extractfile(row['archive_path']).read()).hexdigest() == row['sha256']
    dump(out/'files.json', members)
    dump(out/'receipt.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_source_and_delivery_files_rehashed_unchanged=len(old), new_archived_files=len(members),
        archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
        all_archive_members_reopened_SHA_verified=True,
        running_sensor_data_and_logs_not_archived_as_complete=True,
        live_jobs=jobs, goal_complete=False))
    print(json.dumps(dict(old_unchanged=len(old), archived_new_files=len(members),
        live_jobs=jobs, project_GiB=used/2**30, goal_complete=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
