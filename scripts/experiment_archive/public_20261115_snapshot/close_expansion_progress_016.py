"""保存第二回合真实负结果与只读诊断；现有运行和历史文件均保持原样。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from close_continuation_delivery_011 import old_sources, live_jobs
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]
PROJECT = R.parents[3]


def write(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def profiler_jobs():
    output = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv = [x.decode(errors='replace') for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
            if not argv or not any(a.endswith('/source/profile_carla_expansion_live_001.py') for a in argv[1:3]):
                continue
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z':
                output.append(dict(PID=int(p.name), start_tick=fields[19], argv=argv))
        except OSError:
            pass
    return output


def main():
    old = old_sources()
    for version in ['008', '009']:
        old.update({x['path']: x['sha256'] for x in json.loads(
            (R / f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    for path, expected in old.items():
        assert digest(path) == expected, path
    sources = [R / 'source' / name for name in [
        'summarize_carla_expansion_013.py', 'test_carla_expansion_summary_013.py',
        'diagnose_carla_closed_trace_014.py', 'audit_sealed_inventory_015.py',
        'close_expansion_progress_016.py']]
    qa = R / 'json/software_validation_continuation_012'
    qa.mkdir(exist_ok=False)
    for p in sources:
        ast.parse(p.read_text())
    # 三个新汇总测试已通过；在同一次交付执行中留存可复核日志。
    checks = []
    for name, command in [
        ('summary_tests', ['/root/miniconda3/envs/conceptgraph/bin/python', '-B', str(sources[1])]),
        ('ruff', ['/root/miniconda3/envs/conceptgraph/bin/ruff', 'check', '--no-cache',
                  '--select', 'E9,F63,F7,F82', *map(str, sources)]),
        ('git_diff_check', ['git', 'diff', '--check'])]:
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        log = qa / (name + '.log')
        with log.open('x') as stream:
            stream.write(result.stdout + result.stderr)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        checks.append(dict(check=name, exit_code=result.returncode, log_sha256=digest(log)))
    write(qa / 'report.json', dict(source_files_AST_checked=len(sources), checks=checks,
                                   new_summary_tests_passed=3,
                                   protected_files_rehashed_unchanged=len(old),
                                   business_code_tests_not_repeated=True, goal_complete=False))
    progress = json.loads((R / 'carla_official/expansion_progress_013/report.json').read_text())
    failure = json.loads((R / 'carla_official/expansion_failure_trace_014/report.json').read_text())
    sealed = json.loads((R / 'annotation/sealed_inventory_readback_015/report.json').read_text())
    jobs = live_jobs()
    profiler = profiler_jobs()
    state = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 previous_goal_turn_classification='progress：第一新起点严格安全到达，75同步布局与参考路径审计完成',
                 this_goal_turn_classification='progress：第二实际回合900物理时刻闭合并保留超时；逐帧定位保护/规划余量停滞；330队列有效分母汇总和封存材料来源复核',
                 prior_snapshot='json/continuation_handoff_014.json',
                 registered_total_episodes=progress['registered_episodes'],
                 planned_primary_episodes=150, snapshot_primary_closed=2,
                 snapshot_primary_strict_safe_arrivals=1,
                 all_closed_physics_ticks=progress['physical_ticks'],
                 all_closed_frontend_calls=progress['frontend_calls'],
                 no_Qwen_improvement_credit_for_simple_rule_branch=True,
                 second_episode=dict(name='town10_spawn002_ClearNoon_car18', seed=17,
                     physics_ticks=900, frontend_calls=450, status='TIMEOUT',
                     correct_proposal=True, correct_lock=True, route_generated=True,
                     actually_arrived=False, collisions=0, lane_violation=False,
                     motion_guard_stop_ticks=failure['motion_guard_stop_ticks'],
                     invalid_initial_body_plan_ticks=failure['plan_failure_ticks']['INVALID_INITIAL_BODY'],
                     trace='carla_official/expansion_failure_trace_014/trace.csv'),
                 sealed_inventory={k: sealed[k] for k in ['scenes', 'logs', 'frames',
                     'original_natural_commands', 'template_main_queries', 'template_challenge_queries',
                     'preselected_template_replay_queries', 'original_natural_scenes_already_at_least20']},
                 sealed_inventory_does_not_certify_label_quality_or_model_results=True,
                 live_jobs=jobs, live_profiler=profiler, official_launcher_session=43408,
                 performance_session=54577,
                 primary_not_complete=True, matched_method_and_ablation_phases_not_launched=True,
                 full_history_maintenance_ablation_pending=True,
                 positive_arrivals_minimum5configs10tasks_not_met=True,
                 human_attribute_review_images=20, human_3D_identity_reviews=0,
                 new_human_natural_instruction_reviews=0,
                 source_archive='source_bundle/closing_supplement_010',
                 module_index='source_bundle/module_index_011/modules.csv',
                 future_sensor_bulk_authorized=True,
                 project_artifact_GiB=sum(p.stat().st_size for p in R.rglob('*')
                                           if p.is_file() and not p.is_symlink()) / 2**30,
                 project_budget_GiB=25, physical_free_GiB=shutil.disk_usage(R).free / 2**30,
                 new_sealed_test_opened=False, thesis_prose_written=False,
                 git_committed_or_pushed=False, goal_complete=False)
    status = R / 'json/continuation_handoff_015.json'
    write(status, state)
    doc = R / '交付补充说明011.md'
    with doc.open('x') as stream:
        stream.write('''# 实验续跑交付补充011

当前扩展主系统150回合仍在执行。快照已有2回合完成实际执行和两项独立审计，共1431物理时刻、716次前端。第1项spawn001/car20严格安全到达；第2项spawn002/car18正确提议、正确锁定并生成过目标路线，但900物理时刻后TIMEOUT，实际到达为false。没有碰撞、越界或运动限值违规不等于完成任务。两项都是简单类别指令，由规则分支处理，真实Qwen调用0，不将成绩归因于Adapter。不能把已闭合的1/2当整个批次的最终成功率。

第二回合逐帧只读诊断：17.4秒首次因滑行预测越出道路要求停车；37.2秒开始当前名义车身加0.05m安全余量越出原道路，后续无法重新规划。614物理时刻要求保护停车，434物理时刻对应最近前端的INVALID_INITIAL_BODY。独立实际车道审计使用原车身与外扩0.05m道路，未越界；两种口径不同。日志只能证明观察到的过程，不能证明取消保护就能成功。未改动冻结的规划、控制和安全余量。逐帧诊断在carla_official/expansion_failure_trace_014/。

330回合完整计划索引在carla_official/expansion_progress_013/episodes.csv，含150主实验、60额外方法、120额外消融。未运行、执行中、待审计和已闭合明确分开；只有双审计通过的真实回合进入当前描述性指标；普通超时作为负结果进入。不可达任务和无匹配任务不混入正目标到达分母。额外方法和消融阶段尚未启动，完整历史维护消融仍待补齐。每次刷新使用新run-id运行source/summarize_carla_expansion_013.py，禁止覆盖快照。

封存材料数量复核在annotation/sealed_inventory_readback_015/：30场景、14日志、1196帧；已有600条预先构造的模板主查询、278条挑战查询、203条事前选定回放查询，每场景模板主查询20条且回放至少6条。另有439条原始Talk2Car人工自然表达，只有3场景本身达到20条；不能将这439条或模板混称为已完成人工核验的自然指令。20张车辆/行人人工属性审核不覆盖这批封存指令。此次只核对材料元数据，未打开方法测试成绩，也未新增或修改标签。

性能采样器持续独立运行，真实CPU进程高水位与GPU占用采样保存performance/carla_expansion_live_profile_001/。当前汇总保留当时采样前缀，GPU峰值是采样下界，第一回合未采样，不补造。此前handoff014的空live_profiler字段是命令参数定位漏检，采样器没有中断；独立更正回执performance/carla_expansion_live_profile_process_verification_003.json。

状态json/continuation_handoff_015.json记录实际PID和start_tick，继续轮询原launcher/session43408和采样session54577，不重启同一回合。额外60方法、120消融、完整历史维护消融、新封存测试、最少5配置10不同正任务实际安全到达、最终统计交付仍未完成。新增传感器大文件沿用用户已授权共享卷，原文件不移动、不删除。未写论文正文，未提交或推送Git；goal_complete=false。
''')
    index = R / 'source_bundle/module_index_011'
    index.mkdir(exist_ok=False)
    with (R / 'source_bundle/module_index_010/modules.csv').open() as stream:
        entries = list(csv.DictReader(stream))
    for source, name, evidence, boundary in [
        (sources[0], '真实扩展回合有效分母汇总', 'carla_official/expansion_progress_013/report.json', '未闭合不算失败；普通超时保留；非到达任务分母单列'),
        (sources[2], '闭合回合运动保护与规划失败定位', 'carla_official/expansion_failure_trace_014/report.json', '只读日志事件关联，不授予新到达或宣称唯一因果'),
        (sources[3], '封存查询来源与每场景容量核验', 'annotation/sealed_inventory_readback_015/report.json', '人工原始表达与模板明确区分，数量不等于审核或封存结果')]:
        entry = next(n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'main')
        entries.append(dict(module=name, source=str(source), entry='main', line=entry.lineno,
                            meaning=name, evidence=str(R / evidence),
                            evidence_exists_at_index_generation=True,
                            boundary=boundary, source_sha256=digest(source)))
    with (index / 'modules.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    with (index / 'README.md').open('x') as stream:
        stream.write('011补充真实闭合分母、第二回合失败定位和封存材料来源核验；论文正文未撰写。\n')
    paths = sources + [status, doc, *qa.glob('*'), *index.glob('*'),
                       R / 'performance/carla_expansion_live_profile_process_verification_003.json',
                       R / 'json/expansion_jobs_011_primary/job_002.json']
    for folder in ['carla_official/expansion_progress_013',
                   'carla_official/expansion_failure_trace_014',
                   'annotation/sealed_inventory_readback_015']:
        paths.extend(p for p in (R / folder).rglob('*') if p.is_file())
    episode = R / 'carla_official/expansion_navigation_010/town10_spawn002_ClearNoon_car18/seed17/adapter/full_rule_geometry'
    paths.extend(episode / name for name in ['summary.json', 'release.json', 'sensor_files_manifest.json',
        'evaluation_only/truth.json', 'semantic_queue_final.json',
        'independent_expansion_audit_009/report.json', 'independent_expansion_audit_009/frames.jsonl',
        'independent_expansion_audit_009/choices.jsonl',
        'independent_motion_guard_audit_002/report.json', 'independent_motion_guard_audit_002/ticks.jsonl'])
    archive_dir = R / 'source_bundle/closing_supplement_010'
    archive_dir.mkdir(exist_ok=False)
    members = [dict(path=str(p), archive_path=str(p.relative_to(PROJECT)),
                    bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(set(paths))]
    archive = archive_dir / 'source_delta.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for member in members:
            tar.add(member['path'], arcname=member['archive_path'], recursive=False)
    with tarfile.open(archive) as tar:
        for member in members:
            assert hashlib.sha256(tar.extractfile(member['archive_path']).read()).hexdigest() == member['sha256']
    write(archive_dir / 'files.json', members)
    write(archive_dir / 'receipt.json', dict(protected_files_rehashed_unchanged=len(old),
        archived_new_files=len(members), archive_sha256=digest(archive),
        all_members_reopened_SHA_verified=True, raw_sensors_not_duplicated_moved_or_deleted=True,
        goal_complete=False))
    print(json.dumps(dict(protected_unchanged=len(old), archived_new_files=len(members),
                         live_jobs=jobs, live_profiler=profiler, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
