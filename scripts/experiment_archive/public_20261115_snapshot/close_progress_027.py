"""记录当前扩展、完整历史维护诊断和后续队列状态；不复制原始传感器。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from close_continuation_delivery_011 import old_sources, live_jobs
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]
PROJECT = R.parents[3]


def dump(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    old = old_sources()
    for version in ['008', '009', '010']:
        old.update({x['path']: x['sha256'] for x in json.loads(
            (R/f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    for path, sha in old.items():
        assert digest(path) == sha, path
    progress = json.loads((R/'carla_official/expansion_progress_033/report.json').read_text())
    history = json.loads((R/'carla_official/history_module_recorded_diagnostic_021/report.json').read_text())
    history_audit = json.loads((R/'carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023/report.json').read_text())
    sealed = json.loads((R/'annotation/sealed_inventory_readback_015/report.json').read_text())
    remaining = json.loads((R/'carla_official/remaining_comparison_registration_024.json').read_text())
    jobs = live_jobs()
    profile = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv = [x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if any(a.endswith('/source/profile_carla_expansion_live_001.py') for a in argv[1:3]):
                fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] != 'Z':
                    profile.append(dict(PID=int(p.name), start_tick=fields[19]))
        except OSError:
            pass
    state = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 previous='json/continuation_handoff_015.json',
                 classification='progress：4个官方主回合双审计闭合；完整时间生命周期诊断2148帧核验；后续210回合注册并等待主队列释放',
                 primary_registered=progress['registered_episodes'],
                 primary_closed=progress['closed_episodes'],
                 primary_completion_counts=progress['completion_counts'],
                 primary_positive_safe_arrivals=progress['primary_positive_safe_arrivals'],
                 primary_physics_ticks=progress['physical_ticks'],
                 primary_actual_Qwen_calls=progress['actual_GPU_Qwen_calls'],
                 second_and_blocked_failures_kept=True,
                 history_lifecycle=dict(recorded_sensor_frames=history['unique_sensor_frames'],
                     actual_mapper_updates=history['actual_CPU_mapping_updates'],
                     configs=3, conditions=['full', 'without_short_term_expiry', 'without_history_maintenance'],
                     independent_recheck=history_audit['actual_recorded_CPU_updates_rechecked'],
                     actual_new_physics_or_GPU_calls=0, closed_loop_credit=0,
                     report='carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023/report.json'),
                 sealed_inventory=dict(scenes=sealed['scenes'], logs=sealed['logs'], frames=sealed['frames'],
                     template_queries=sealed['template_main_queries']+sealed['template_challenge_queries'],
                     replay_queries=sealed['preselected_template_replay_queries'],
                     original_natural_commands=sealed['original_natural_commands'],
                     natural_scenes_at_least20=sealed['original_natural_scenes_already_at_least20'],
                     method_test_opened=False),
                 remaining_registered=remaining['registered_additional_episodes'],
                 remaining_phases={k: len(v) for k, v in remaining['execution_phases'].items()},
                 remaining_waiter='json/official_remaining_execution_025_retry',
                 live_jobs=jobs, live_profiler=profile,
                 project_artifact_GiB=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())/2**30,
                 project_budget_GiB=25, physical_free_GiB=shutil.disk_usage(R).free/2**30,
                 output_sensor_bulk_authorized=True, history_noncompact_overrun_preserved=True,
                 no_full_closed_loop_history_ablation_yet=True, no_sealed_method_result_yet=True,
                 thesis_prose_written=False, git_committed_or_pushed=False, goal_complete=False)
    status = R/'json/continuation_handoff_016.json'
    dump(status, state)
    doc = R/'交付补充说明012.md'
    with doc.open('x') as stream:
        stream.write('''# 实验续跑交付补充012

官方 CARLA 主队列当前已完成4个双独立审计回合，计划仍为150个。第1个简单车辆任务严格安全到达；第2个正确提议/锁定但超时；第3个阻断卡车任务没有错误锁定，独立证书判定不可达；第4个无公交车任务保持拒答。4个回合累计2344物理时刻、1157次前端，Qwen实际调用0；这些结果是当前闭合前缀，不能当150回合总成功率。主队列第5回合正在运行，审计完成后自动继续。

完整历史维护的定义已补齐并用真实保存传感器完成机制核验：完整维护、只关短期过期、关闭整个时间生命周期三种配置在两条真实回合上共2148次CPU映射更新。关闭后仍保留一对一跨帧累积，运动历史字段不再生成，动态/静止证据门限按设计拒绝未知对象。该证据只证明消融开关和输入链路生效，没有授予闭环性能信用；真正的30个同任务官方闭环回合已注册，等待主队列和GPU释放。

先前一次非紧凑诊断因逐帧重复保存历史属性超过32MiB保护而主动停止，现场和超限记录保留；紧凑诊断在4.7MiB左右完成，未触碰官方回合。这个过程记录在carla_official/history_module_recorded_diagnostic_021/interrupted_resource_audit_001.json，不删除任何结果。

后续注册表remaining_comparison_registration_024.json包含60个方法比较、120个原语义/闭环消融和30个完整历史维护消融；官方等待器在json/official_remaining_execution_025_retry/，当前只等待主队列PID112329，确认GPU和端口空闲后才启动。传感器原始大文件继续放用户授权共享卷，项目内保存压缩执行记录、摘要和哈希。

封存材料来源复核：30场景、14日志、1196帧，模板主/挑战查询878条、预选回放203条；原始Talk2Car自然表达439条只有3场景达到20条，不能把模板查询冒充自然语言人工审核。方法尚未打开这套封存测试，颜色/细属性人工审核20图也不等同于三维身份或自然指令审核。

状态见json/continuation_handoff_016.json；最新实际分母见carla_official/expansion_progress_033/，历史维护审计见carla_official/history_module_recorded_diagnostic_021/，新增注册和源码待后续封存。论文正文、最终图表、完整方法/消融统计和Git同步仍未完成，goal_complete=false。
''')
    index = R/'source_bundle/module_index_012'
    index.mkdir(exist_ok=False)
    with (R/'source_bundle/module_index_011/modules.csv').open() as stream:
        entries = list(csv.DictReader(stream))
    additions = [
        (R/'source/carla_history_lifecycle_017.py', '完整时间生命周期消融模块定义', 'carla_official/history_runtime_derivation_012.json', '不等同于仅关闭短期过期；不含闭环性能信用'),
        (R/'source/replay_carla_history_module_021.py', '真实传感器历史维护机制回放', 'carla_official/history_module_recorded_diagnostic_021/report.json', '只做机制核验，不改车辆控制，不授予到达'),
        (R/'source/audit_carla_history_execution_023.py', '历史维护关闭独立重核', 'carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023/report.json', '逐帧检查生命周期字段和跨帧累积'),
        (R/'source/run_carla_remaining_comparison_024.py', '主队列后方法与消融执行器', 'carla_official/remaining_comparison_registration_024.json', '等待主队列结束，运行前仍需独立审计'),
        (R/'source/run_official_remaining_025.py', '后续官方CARLA服务器等待器', 'json/official_remaining_execution_025_retry/registration.json', '不与现行服务器并行；检查GPU、端口和资源门限')]
    for source, module, evidence, boundary in additions:
        entry = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        entries.append(dict(module=module, source=str(source), entry='main', line=entry.lineno,
                            meaning=module, evidence=str(R/evidence), evidence_exists_at_index_generation=True,
                            boundary=boundary, source_sha256=digest(source)))
    with (index/'modules.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    (index/'README.md').open('x').write('012补充历史维护消融、后续官方队列和当前闭合分母；论文正文尚未写。\n')
    paths = [status, doc, index/'modules.csv', index/'README.md',
             R/'json/history_compact_replay_derivation_020.json',
             R/'json/history_lifecycle_tests_022.log', R/'json/official_remaining_execution_025_retry/registration.json',
             R/'json/expansion_jobs_011_primary/job_003.json', R/'json/expansion_jobs_011_primary/job_004.json']
    paths += [R/'source'/name for name in ['carla_history_lifecycle_017.py', 'build_carla_history_runtime_018.py',
                                           'replay_carla_history_module_021.py', 'test_carla_history_lifecycle_022.py',
                                           'audit_carla_history_execution_023.py', 'run_carla_remaining_comparison_024.py',
                                           'run_official_remaining_025.py', 'prepare_remaining_carla_queue_026.py', 'close_progress_027.py']]
    paths += [p for p in (R/'carla_official/history_module_recorded_diagnostic_021/independent_lifecycle_audit_023').rglob('*') if p.is_file()]
    paths += [p for p in (R/'carla_official/expansion_progress_033').rglob('*') if p.is_file()]
    archive_dir = R/'source_bundle/closing_supplement_011'; archive_dir.mkdir(exist_ok=False)
    members = [dict(path=str(p), archive_path=str(p.relative_to(PROJECT)), bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(set(paths))]
    archive = archive_dir/'source_delta.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for member in members: tar.add(member['path'], arcname=member['archive_path'], recursive=False)
    with tarfile.open(archive) as tar:
        for member in members:
            assert hashlib.sha256(tar.extractfile(member['archive_path']).read()).hexdigest() == member['sha256']
    dump(archive_dir/'files.json', members)
    dump(archive_dir/'receipt.json', dict(protected_files_rehashed_unchanged=len(old), archived_new_files=len(members),
         archive_sha256=digest(archive), all_members_reopened_SHA_verified=True,
         raw_sensor_files_not_duplicated_or_moved=True, goal_complete=False))
    print(json.dumps(dict(closed=progress['closed_episodes'], archived_new_files=len(members), live_jobs=jobs,
                          goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
