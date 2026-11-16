"""保存本轮诊断交付、旧运行源码核验与恢复边界；不改旧报告、不重启CARLA。"""
import ast
import datetime
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from diagnose_carla_ten_bottlenecks_043 import sha

R = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    out = R / 'carla_official/diagnosis_delivery_048'
    out.mkdir(exist_ok=False)
    protected = {}
    reports = []
    for i in range(1, 11):
        jobpath = R / f'json/expansion_jobs_011_primary/job_{i:03d}.json'
        job = json.loads(jobpath.read_text())
        d = Path(job['episode_directory'])
        release = json.loads((d / 'release.json').read_text())
        protected.update(release['source_sha256'])
        old = json.loads((d / 'independent_expansion_audit_009/report.json').read_text())
        protected.update(old['inputs_sha256'])
        reports += [jobpath, d / 'independent_expansion_audit_009/report.json']
    verification = dict(checked_files=len(protected), mismatches=[], missing=[])
    for name, expected in protected.items():
        p = Path(name)
        if not p.exists():
            verification['missing'].append(name)
        elif sha(p) != expected:
            verification['mismatches'].append(name)
    verification['scope'] = '首批10回合发布清单内的源码与输入结果；不是重新核验整个阶段1—8'
    write(out / 'frozen_runs_verification.json', verification)
    sources = [R / 'source' / name for name in [
        'diagnose_carla_ten_bottlenecks_043.py', 'carla_current_plan_handoff_044.py',
        'test_carla_current_plan_handoff_044.py', 'audit_carla_arrival_windows_045.py',
        'audit_carla_identity_geometry_046.py', 'carla_reference_readiness_047.py',
        'test_carla_reference_readiness_047.py', 'replay_reference_readiness_047.py',
        'finalize_carla_diagnosis_048.py']]
    for p in sources:
        ast.parse(p.read_text(), filename=str(p))
    checks = subprocess.run(['git', 'diff', '--check'], cwd=R.parents[3],
        text=True, capture_output=True, check=False)
    active_modules = [p for p in sources if p.name != 'diagnose_carla_ten_bottlenecks_043.py']
    lint = subprocess.run(['/root/miniconda3/envs/conceptgraph/bin/python', '-B', '-m', 'ruff',
        'check', '--no-cache', *map(str, active_modules)], text=True, capture_output=True, check=False)
    with (out / 'ruff.log').open('x') as f:
        f.write(lint.stdout + lint.stderr)
    with (out / 'git_diff_check.log').open('x') as f:
        f.write(checks.stdout + checks.stderr)
    log44 = R / 'json/plan_handoff_validation_044.log'
    log47 = R / 'json/reference_readiness_validation_047.log'
    tests_passed = '\nOK\n' in log44.read_text() and '\nOK\n' in log47.read_text()
    validation = dict(new_sources_ast_parsed=len(sources),
        ruff_exit_code=lint.returncode, git_diff_check_exit_code=checks.returncode,
        unittest_cases_passed=21 if tests_passed else None,
        tests_are_injected_backend_CPU_tests=True,
        previous_test_attempt=dict(test='test_finite_valid_goal_does_not_authorize_nan_trajectory',
            failure='构造NaN状态时被VehicleState构造器提前拒绝；已改为显式模拟传递后损坏，再检查门闸拒绝',
            initial_successful_tests=12, initial_errors=1),
        environment_warning='回放和047测试出现libgomp OMP_NUM_THREADS值无效警告，命令仍exit0；不将该计时当正式性能基准。后续运行显式OMP_NUM_THREADS=1。',
        whole_project_test_suite_rerun=False, changed_old_runtime_sources=False,
        frozen_runs_verification=verification, goal_complete=False)
    write(out / 'validation.json', validation)
    diagnoses = [R / 'carla_official' / name for name in [
        'ten_bottlenecks_043', 'arrival_windows_045', 'identity_geometry_audit_046',
        'reference_readiness_replay_047', 'expansion_failure_trace_042_job05',
        'expansion_failure_trace_042_job06', 'expansion_failure_trace_042_job10']]
    collected = sources + [log44, log47, R / 'json/reference_readiness_protocol_047.json']
    for directory in diagnoses:
        collected += [p for p in directory.rglob('*') if p.is_file()]
    archive_dir = R / 'source_bundle/carla_diagnosis_048'
    archive_dir.mkdir(exist_ok=False)
    archive = archive_dir / 'diagnosis_and_code.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(collected)):
            z.write(p, str(p.relative_to(R)))
    manifest = {str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(collected))}
    write(archive_dir / 'manifest.json', dict(files=manifest, zip_sha256=sha(archive),
        module_status={'044': '未部署防御候选；旧路线假设已被本批日志排除，不计为实测修复',
            '047': '新参照物观察门闸；450帧CPU重放与8项测试通过，尚未CARLA闭环验证'},
        goal_complete=False))
    note = '''# CARLA十回合补充诊断与评价口径纠正

CARLA仍只完成首批10个回合；所有相关运行已停止。本轮只做保存记录上的CPU诊断，没有新开回合、训练或调用GPU。

## 已核实的卡点

* 第2/10回合：已选对和锁定目标，但没有进入目标区；停车保护与车身初始状态复核失效持续出现。不能据此认定检测模型识别错误。
* 第5回合：几何上进入区域57个物理帧，其中39帧身份也满足旧评价条件；全部仍在运动，最低速度0.2024m/s，高于0.08m/s。没有完成停车保持。
* 第6回合：几何上进入区域109帧、停车19帧，单看几何/方向/停车的连续保持也仅1.8s，未达到2s；部分时刻身份无法确认。仍记真实失败。
* 第7回合：卡车参照物的保守0—4m高外形柱在全部450帧中投影均处于前相机左侧之外。画面右界最接近时仍为u=-12.11像素，图像有效范围从u=0开始。不能将该回合说成“画面内卡车漏检”或“Qwen看到了完整证据却判断错”。

## 纠正旧表的名称与解释

旧correct_first_proposal实际统计“曾经至少一次授权后的首位候选正确”，并不是第一次提议；新表将首次与曾经分开。此次五个成功提议的正任务，首次也都正确，但其他任务不能沿用这个假定。

旧wrong_lock同时包含“关联到其他物体”和“没有任何可靠关联”。第6回合这里只有1个前端帧无法关联，没有确定关联到其他车辆的帧。此前的“曾错误锁定，需要排查身份切换”应改为“曾无法核验锁定身份，需要排查定位与关联误差”；不能声称已经证明换锁到别车。

旧inside_independent_region还要求当前对象身份与区域对应。新表独立重算几何进入，所以进入帧数可以增加，但它绝不自动带来正确到达信用。严格安全到达仍为6个正任务中的1个，所有旧报告均保留。

## 代码与实测边界

最初怀疑新计划被丢弃，但逐帧检查发现本批所有已锁定目标的有效新计划都已在同帧安装。这个怀疑已排除。044只作为未部署的防御候选保存，13项软件测试通过，不作为本批问题修复或性能提升。

047新增了规则/base/Adapter共用的参照物观察门闸：参照物不足时不提交新Qwen请求、不授权目标驾驶，不把未观察到解释为整个范围内不存在；既有独立安全搜索路线仍由原规划和控制保护检查。真实450快照重放全部判为缺参照；旧65次CUDA调用的输入也全缺参照。8项测试通过。尚未启动新闭环，不能声称它已提升到达率、模型准确率或实际节省了多少GPU时间。

## 入口与后续

* ../identity_geometry_audit_046/episodes.csv：新的首次/曾经、已证选错/关联不足、几何/身份条件进入统计。
* ../arrival_windows_045/episodes.csv：原评价条件下到达窗口分解；几何统计以046为准。
* ../ten_bottlenecks_043/reference_projection.csv：卡车视野覆盖诊断；官方真值只在隔离诊断中读取。
* ../reference_readiness_replay_047/report.json：新观察门闸的真实输入离线重放。
* validation.json：21项新软件测试、源码核验、定向Ruff与git diff --check；不是全项目测试已重跑的声明。
* ../../source_bundle/carla_diagnosis_048/：本轮代码和诊断副本、逐项哈希。

下一步优先核查实际切油/制动与规划停止距离是否一致、转向和道路安全余量的一致性，以及目标几何误差导致的身份关联不足；使用已记录的官方标定数据继续离线分析。不增加CARLA回合，不放宽目标区、速度、保持时间或安全门限。新的闭环版本需要另行冻结并保留旧负结果。

KITTI/nuScenes/Qwen公平比较、封存主测试和总交付的其他未完成项继续沿用上轮交接，goal_complete=false；本说明不是整项课题完成验收。
'''
    with (out / '交付说明.md').open('x') as f:
        f.write(note)
    write(R / 'json/continuation_handoff_018.json', dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_handoff=str(R / 'json/continuation_handoff_017.json'),
        latest_diagnosis=str(out / '交付说明.md'), CARLA_limit=10,
        do_not_restart_CARLA_batch=True, new_CARLA_calls_this_turn=0,
        old_wrong_lock_claim_corrected='job6仅1帧关联不足，未证实选中其他物体',
        rejected_hypothesis='本批不存在有效目标计划未同帧安装的情况',
        source_candidates={'044': 'not_deployed_not_actual_root_cause',
            '047': 'CPU_replay_and_8_tests_passed_closed_loop_pending'},
        archive=str(archive), next_work='用既有官方标定及首批记录核查实际制动/转向与规划预测的差异',
        global_requirements_not_satisfied=True, goal_complete=False))
    disk = shutil.disk_usage(R)
    write(out / 'storage.json', dict(project_free_bytes=disk.free,
        new_delivery_archive_bytes=archive.stat().st_size,
        no_deletions=True, no_existing_files_moved=True, user_project_budget_GiB=25))
    assert tests_passed and checks.returncode == 0 and lint.returncode == 0
    assert not verification['missing'] and not verification['mismatches']
    print(json.dumps(dict(protected_files_verified=verification['checked_files'],
        tests=21, archive_bytes=archive.stat().st_size, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
