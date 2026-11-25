"""封存规划/存储实测与道路只读清单；008模型终止状态逐项披露。"""
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
from close_continuation_delivery_009 import live_jobs

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'source_bundle/closing_supplement_007'
    out.mkdir(exist_ok=False)
    old = {x['path']: x['sha256'] for x in json.loads((R/'source_bundle/reproduction_003/source_manifest.json').read_text())}
    old.update({x['path']: x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_001/receipt.json').read_text())['supplemental_files']})
    old.update({str(PROJECT/p): h for p, h in json.loads((R/'source_bundle/closing_supplement_002/files.json').read_text()).items()})
    for version in ['003', '004', '005', '006']:
        old.update({x['path']: x['sha256'] for x in json.loads((R/f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    assert all(digest(p) == h for p, h in old.items())
    qa = R/'json/software_validation_continuation_009'
    qa.mkdir(exist_ok=False)
    sources = sorted((R/'source').glob('*.py'))
    for p in sources:
        ast.parse(p.read_text())
    checks = []
    for name, command in [('ruff_functional', ['/root/miniconda3/envs/conceptgraph/bin/ruff', 'check',
            '--no-cache', '--select', 'E9,F63,F7,F82', str(R/'source')]),
            ('git_diff_check', ['git', 'diff', '--check'])]:
        result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT)
        log = qa/(name+'.log')
        log.write_text(result.stdout+result.stderr)
        assert result.returncode == 0
        checks.append(dict(check=name, exit_code=0, log_sha256=digest(log)))
    testroot = R/'json/aggregated_obstacles_validation_004'
    log = testroot/'unittest.log'
    assert 'Ran 4 tests' in log.read_text() and log.read_text().rstrip().endswith('OK')
    dump(testroot/'report.json', dict(geometry_safety_cases_passed=4, injected_planner_only=True,
        original_obstacle_audit_can_reject_mock_planner_success=True,
        source_sha256={str(p):digest(p) for p in [log, R/'source/carla_aggregated_obstacle_bridge_004.py',
            R/'source/test_carla_aggregated_obstacles_004.py']}, new_GPU_or_physical_credit=0, goal_complete=False))
    dump(qa/'report.json', dict(syntax_checked_files=len(sources), checks=checks,
        old_source_and_delivery_rehashed_unchanged=len(old), original_business_sources_unchanged=True,
        original_452_business_tests_not_repeated=True, new_safety_tests=4, goal_complete=False))
    index = R/'source_bundle/module_index_008'
    index.mkdir(exist_ok=False)
    rows = list(csv.DictReader((R/'source_bundle/module_index_007/modules.csv').open()))
    additions = [
        ('原障碍并集的计算诊断', 'benchmark_obstacle_union_001.py', 'main', 'carla_official/obstacle_union_benchmark_001/report.json',
            '6个真实观测/路线同输出，含并集开销中位加速3.10倍；30s仅诊断预算，不是正式规划或物理到达增益'),
        ('保留原障碍复核的规划加速候选', 'carla_aggregated_obstacle_bridge_004.py', 'AggregatedObstacleBridge',
            'json/aggregated_obstacles_validation_004/report.json', '4安全测试通过；并集规划后再用原逐个障碍审计，未接入正在运行的008'),
        ('逐记录flush无损日志压缩', 'probe_streaming_jsonl_003.py', 'main', 'storage_audit/streaming_jsonl_probe_003/report.json',
            '6日志2316记录逐字节一致，147.12MiB到60.08MiB；不修改旧文件，不证明RGB或LiDAR压缩率'),
        ('官方地图只读扩展清单', 'inspect_official_task_geometry_001.py', 'main', 'carla_official/task_geometry_inventory_001/report.json',
            '实际读取5起点，形成10配置/50任务候选，未执行或冻结正式性能任务；不推进仿真时钟'),
        ('008规则失败分层诊断', 'summarize_guarded_rule_009.py', 'main', 'carla_official/guarded_rule_diagnosis_009/report.json',
            '493物理时刻247前端；选对/锁定但墙钟超时，55次规划预算失败，不能记到达或归因新增保护过严'),
    ]
    for title, filename, entry, evidence, boundary in additions:
        p = R/'source'/filename
        node = next(n for n in ast.parse(p.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == entry)
        rows.append(dict(module=title, source=str(p), entry=entry, line=node.lineno, meaning=boundary,
            evidence=str(R/evidence), evidence_exists_at_index_generation=True, boundary=boundary,
            source_sha256=digest(p)))
    with (index/'modules.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (index/'README.md').open('x').write('008为最新模块入口索引；候选清单、软件通过和计算加速都不能代替正式物理到达或封存测试。\n')
    rule = json.loads((R/'carla_official/guarded_rule_diagnosis_009/report.json').read_text())
    base_root = R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/base'
    base = json.loads((base_root/'independent_choice_audit_008/report.json').read_text())
    for p, h in base['source_sha256'].items():
        assert digest(p) == h
    jobs = live_jobs()
    size = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    handoff = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn_classification='progress：664旧文件重查，32新文件已归档；中间用户问答只解释人工审核依据',
        this_goal_turn_classification='progress：真实规则493时刻和基础模型45调用终止回读、6帧计算对照、2316日志无损回读、5官方起点只读清单',
        source_archive='source_bundle/closing_supplement_007', module_index='source_bundle/module_index_008/modules.csv',
        rule008=dict(complete=True, status=rule['final_status'], actual_physics_ticks=493,
            actual_frontend_calls=247, correct_proposal=True, correct_lock=True, wrong_lock=False,
            strict_safe_arrival=False, collisions=0, lane_violations=0, planning_timeouts=55,
            report='carla_official/guarded_rule_diagnosis_009/report.json'),
        base008=dict(complete=True, status=base['final_status'], actual_GPU_calls=base['actual_GPU_model_calls'],
            valid_outputs=base['valid_model_outputs'], correct_lock=False, strict_safe_arrival=False,
            no_target_route=True, metric_warning='原地等待无碰撞不能称为任务安全成功；无路线仍失败',
            report=str(base_root/'independent_choice_audit_008/report.json')),
        adapter008_completion_not_claimed=True, current_live_jobs=jobs,
        full_motion_guard_audits_still_queued=True, audit_session=62947, official_batch_session=20764,
        new_computation_diagnostic=dict(observed_frames=6, median_speedup=3.104226578605868,
            exact_outputs=True, production_004_not_physically_used_yet=True),
        lossless_log_probe=dict(files=6, real_records=2316, original_MiB=147.12459182739258,
            compressed_MiB=60.0785026550293, old_logs_modified=False),
        expanded_CARLA_candidate_geometry=dict(actual_spawn_inventories=5, configurations=10,
            task_candidates=50, future_seed_candidates=[17,29,43], future_episode_candidates=150,
            actual_new_episodes_from_inventory=0, positive_task_geometry_not_navigation_proof=True,
            truck_blueprint_confirmed='vehicle.carlamotors.carlacola',
            report='carla_official/task_geometry_inventory_001/report.json'),
        project_artifact_GiB=size/2**30, physical_free_GiB=shutil.disk_usage(R).free/2**30,
        project_budget_GiB=25, human_review_pending=True,
        next_actions=['续读adapter真实PID、launcher101895和session20764，不重启现有任务；全部结束后104005会执行逐物理时刻全量审计',
            '按原始逐条模型输出、提议/授权/锁定/实际到达分层关闭008三方法，保留所有失败',
            '将004障碍并集加速接入新的共同冻结协议前，保留原逐障碍复核；六帧诊断速度不能直接充当正式性能或到达增益',
            '实现并冻结10配置50任务3种子：现已确认5起点道路、2天气和原厂truck蓝图；不可达须实际横向阻挡的独立几何断连证明，无匹配须披露原生未标注世界道具限制',
            '后续正式配置应包含简单及复杂关系任务、匹配闭环模块消融，不能让去关系/证据开关实际未生效而当正式因果消融',
            '新封存至少20场景400有效指令、真人审核、至少5配置10个不同正任务正确安全到达和完整交付仍未完成'],
        whole_project_sealed_test_opened=False, historical_sources_unchanged=True,
        thesis_prose_written=False, git_committed_or_pushed=False, goal_complete=False)
    handoff_path = R/'json/continuation_handoff_012.json'
    dump(handoff_path, handoff)
    doc = R/'交付补充说明008.md'
    doc.open('x').write('''# 实验续跑交付补充008

008三方法的规则和基础Qwen已实际终止，Adapter仍在运行。规则保存493个物理时刻、247次真实前端，选对且稳定锁定，无错误锁定、碰撞、越界或控制超限，但900秒墙钟只推进49.2秒仿真，未到达，最终WALL_TIMEOUT。247次前端/建图/规划合计约711.90秒，有55次10秒规划预算失败，19帧当前证据不足、40物理时刻没有路线；新增名义运动保护stop请求为0。不能把本次低速停滞解释为新增保护太保守，也不能把混合链路耗时都写成检测器耗时。见carla_official/guarded_rule_diagnosis_009/。

基础Qwen实际完成45次GPU调用，其中最后一次在关闭期间结束，严格格式有效0。没有目标路线、选对或锁定，原地等到墙钟超时；无碰撞不能作为任务成功或“100%安全”证明。原始模型输出、队列迟到/作废及失败均保留。Adapter及全量逐物理时刻LiDAR/控制回读以实际终止报告为准，不提前算完成。

对6个真实观测和既有路线做计算诊断，原逐个障碍与几何并集返回状态、原因完全相同；返回路线再按原障碍逐项扫掠通过。包含并集开销的中位加速约3.10倍。诊断预算统一30秒，只用于避免时间截断遮蔽一致性，不修改正在运行的生产预算。004新候选模块保留原障碍最终审计，4项安全软件测试通过，尚未用于物理导航；不能把这6帧结果称为Hybrid规划整体加速、实时系统或到达改善。见carla_official/obstacle_union_benchmark_001/。

6个已结束真实日志、2316记录进行了逐记录flush的gzip无损试验，147.12MiB压到60.08MiB，约省59.16%；解压后的每一字节与原日志SHA一致，原文件未改变。只是未来日志存储候选，不代表图像/深度/点云同等压缩率，当前008格式不变，也未验证SIGKILL尾部恢复。见storage_audit/streaming_jsonl_probe_003/。

只读官方Town10地图，实际读取预先指定的0—4号五个起点及道路，未推进时钟、重载世界、生成演员或改变当前天气。五个起点前40米地面高度范围为0，18/20/22米候选终区之前的中心线整车道路几何检查均通过。形成两天气、10配置、50任务、3种子的候选清单，共150候选回合，实际执行数仍为0。原厂Prius、Model3、carlacola卡车蓝图均确认存在。不可达标签仍需放置真实阻挡后的独立几何证明；无匹配还须明确原生世界未标注道具限制。道路检查不保证感知、曲率、控制或可达，不能把候选数算完成任务。见carla_official/task_geometry_inventory_001/。

最新模块索引source_bundle/module_index_008/modules.csv，新增源码与闭合报告source_bundle/closing_supplement_007/，恢复状态json/continuation_handoff_012.json。当前预算仍为25GiB，既有产物未删改。真实人工审核、新封存、10配置50任务多种子、至少5配置10不同正任务安全到达、完整闭环消融和最终交付仍未完成。未撰写论文正文或推送Git，goal_complete=false。
''')
    paths = [p for p in sources if str(p) not in old]
    paths += [doc, handoff_path, index/'modules.csv', index/'README.md', qa/'report.json',
        testroot/'report.json', testroot/'unittest.log', R/'carla_official/guarded_rule_diagnosis_009/report.json',
        R/'carla_official/guarded_rule_diagnosis_009/frame_causes.jsonl',
        R/'storage_audit/streaming_jsonl_probe_003/registration.json',
        R/'storage_audit/streaming_jsonl_probe_003/report.json']
    paths += list((R/'carla_official/obstacle_union_benchmark_001').glob('*.json'))
    paths += list((R/'carla_official/task_geometry_inventory_001').glob('*.json'))
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
        old_source_and_delivery_files_rehashed_unchanged=len(old), archived_new_files=len(members),
        archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
        all_members_reopened_SHA_verified=True, live_logs_not_archived_as_complete=True,
        goal_complete=False))
    print(json.dumps(dict(old_unchanged=len(old), new_archived=len(members),
        live_jobs=jobs, project_GiB=size/2**30, goal_complete=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
