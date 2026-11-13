"""等待实际故障观察终止，核验输入并交付完整范围及负结果；不推送、不写论文。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def ticks(pid):
    try:
        row = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return row[19] if row[0] != 'Z' else None
    except OSError:
        return None


def replace_json(path, value):
    temporary = path.with_suffix(path.suffix+'.semantic_fault.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def main():
    base = OUTPUT/'faults/semantic_async_cpu_003'
    destination = base/'delivery_001'
    destination.mkdir(exist_ok=False)
    producer = read(base/'worker_process.json')
    assert ticks(producer['pid']) == producer['start_ticks']
    auxiliaries = sorted(set([p for p in (OUTPUT/'source_bundle').glob('semantic_async_fault*.py')]))
    frozen = {str(p.relative_to(PROJECT)):digest(p) for p in auxiliaries}
    dump(destination/'release.json', dict(producer=producer, source_sha256=frozen,
        wait_budget_s=2100, work_budget_s=300, actual_model_calls=0, no_retry=True, goal_complete=False))
    pid = os.getpid()
    dump(destination/'worker_process.json', dict(pid=pid, start_ticks=ticks(pid), source_sha256=digest(__file__)))
    begun = time.monotonic()
    try:
        while ticks(producer['pid']) == producer['start_ticks']:
            assert time.monotonic()-begun < 2100, '原故障作业仍活跃；不重启'
            time.sleep(5)
        assert (base/'complete.json').exists() and not (base/'interrupted.json').exists()
        assert all(digest(PROJECT/p) == expected for p, expected in frozen.items())
        results = read(base/'complete.json')
        assert results['execution_complete']
        selected = [('semantic_async_cpu_001', 'baseline'), ('semantic_async_cpu_001', 'backend_exception')]
        selected += [('semantic_async_cpu_003', record['condition']) for record in results['reports']]
        negative = ('semantic_async_cpu_002', 'expired_before_delivery')
        inputs = {}
        env = dict(os.environ, PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES='',
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib')
        for run, condition in selected+[negative]:
            case = OUTPUT/'faults'/run/condition
            report = case/'input_audit_001/report.json'
            if not report.exists():
                with (destination/f'{run}_{condition}_input.log').open('x') as log:
                    child = subprocess.run([sys.executable, str(OUTPUT/'source_bundle/semantic_async_fault_input_audit_001.py'),
                        '--run-id', run, '--condition', condition], cwd=PROJECT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False)
                assert child.returncode == 0, '输入审核失败；不重跑模型'
            assert read(report)['passed']
            inputs[f'{run}/{condition}'] = digest(report)
        reports = [read(OUTPUT/'faults'/run/condition/'independent_audit_004/report.json') for run, condition in selected]
        old_negative = read(OUTPUT/'faults'/negative[0]/negative[1]/'independent_audit_004/report.json')
        assert old_negative['passed'] is False and not old_negative['fault_exposure_satisfied']
        assert len(reports) == 8 and len(set(record['condition'] for record in reports)) == 8
        now = datetime.now(timezone.utc).isoformat()
        summary = dict(updated_utc=now, execution_complete=True,
            all_eight_condition_exposures_passed=all(record['passed'] for record in reports), reports=reports,
            previous_unexposed_negative=old_negative,
            counted_new_CPU_Qwen_calls=sum(record['actual_Qwen_calls'] for record in reports)+old_negative['actual_Qwen_calls'],
            counted_actual_frontend_calls=sum(record['actual_frontend_calls'] for record in reports)+old_negative['actual_frontend_calls'],
            counted_control_intervals=sum(record['actual_control_intervals'] for record in reports)+old_negative['actual_control_intervals'],
            input_audit_sha256=inputs, conditions_not_independent_tasks=True, independent_tasks=1,
            distinct_task_seed_pairs=1, new_independent_tasks_added=0,
            protocol_versions_separate=True, actual_GPU_calls=0, formal_GPU_fault_gate=False,
            accuracy_comparison_credit=False, real_time_performance_credit=False,
            manuscript_text_written=False, new_test_opened=False, goal_complete=False)
        dump(destination/'summary.json', summary)
        previous = read(OUTPUT/'source_bundle/source_checkpoint_036/checkpoint.json')
        assert len(previous['sources']) == 390
        export = OUTPUT/'source_bundle/github_sync_001/checkout'
        for name, expected in previous['sources'].items():
            assert digest(PROJECT/name) == digest(export/name) == expected
        assert digest(PROJECT/'.git/index') == previous['original_index_sha256']
        frozen_counts = {}
        for name, count in previous['frozen_sources_unchanged'].items():
            values = read(OUTPUT/name/'release.json')['source_sha256']
            assert len(values) == count and all(digest(PROJECT/p) == expected for p, expected in values.items())
            frozen_counts[name] = count
        for name in ['mapping_evaluation/kitti_tracking_frontend_full_001', 'vehicle_execution/semantic_minus_execution_full_001',
            'faults/semantic_async_cpu_001', 'faults/semantic_async_cpu_002', 'faults/semantic_async_cpu_003']:
            values = read(OUTPUT/name/'release.json')['source_sha256']
            assert all(digest(PROJECT/p) == expected for p, expected in values.items())
            frozen_counts[name] = len(values)
        checkpoint = OUTPUT/'source_bundle/source_checkpoint_037'
        checkpoint.mkdir(exist_ok=False)
        old_checkpoint = OUTPUT/'source_bundle/source_checkpoint_036'
        shutil.copy2(old_checkpoint/'source_current.tar.gz', checkpoint/'source_current.tar.gz')
        shutil.copytree(old_checkpoint/'reproduction_sources', checkpoint/'reproduction_sources')
        for path in auxiliaries:
            shutil.copy2(path, checkpoint/'reproduction_sources'/path.name)
        entry = deepcopy(previous)
        entry.update(created_utc=now, source_archive_reused_from='source_checkpoint_036',
            auxiliary_sources={p.name:digest(p) for p in (checkpoint/'reproduction_sources').iterdir() if p.is_file()},
            checklist='实验完成度与交付清单_017.md', semantic_CPU_fault_summary_sha256=digest(destination/'summary.json'),
            source_files=390, current_sources_published=False, goal_complete=False)
        dump(checkpoint/'checkpoint.json', entry)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        human = {key:json.load(opener.open('http://127.0.0.1:8768/api/'+key, timeout=10))
            for key in ['review_status', 'natural_collection_status']}
        minus = read(OUTPUT/'vehicle_execution/semantic_minus_execution_full_001/progress.json')
        accepted = sum(record['passed'] for record in reports)
        text = f'''# 实验完成度与交付准备（第17版）

更新：{now}。继续完善代码、真实实验、复现与交付，不写论文正文、章节或框架；goal_complete=false，阶段9—12均未全部验收。

新增语义故障CPU集成：八个条件实际执行和输入/整车/方向/控制/保持审核结束，实际故障暴露通过{accepted}/8（含基线）。实际CPU Qwen共{summary['counted_new_CPU_Qwen_calls']}调用、真实前端{summary['counted_actual_frontend_calls']}次、控制{summary['counted_control_intervals']}区间；包含保留的002未暴露负条件，另有六条件采用固定180秒观察窗。后台调用、投递、期限丢弃和退出未轮询分别统计。原始模型文本和故障后文本逐项保存；真实crop像素与当前安全候选独立核对，请求无评价GT字段。

这是同一config00-positive0/seed17的开发集成证据，不增加独立场景/任务，不代表GPU故障、独占实时性能或模型准确率。001错误地要求全部响应都投递的审核失败保留；002提前到达、未暴露投递前过期的负结果保留。003仅延长终态后的故障观察至固定预算，原控制/制动/安全守卫不改，生成函数与原版仅两处终止分支差异已逐字核验。各条件所有路线、对象、锁定和到达结果及独立保持证据见summary，不将规则回退到达归于Qwen。

开发主450及先前KITTI实验保持：规则/base/新Adapter各90正条件正确到达50/39/4，新Adapter负结果不改。三任务逻辑缓存小任务已完成，冷/热各61真实前端301控制、热新增后端0；两次未到达。主450、KITTI Object64/Tracking90、便携入口、主450和KITTI浏览器见第16版及各原始证据；不将这些开发诊断当新封存。

GPU缺一消融当前{minus['audited_cases']}/450，独立审核跟随；兼容750组合、正式GPU故障和独占性能仍沿原队列。人工审核{human['review_status']['human_submissions']}、人工表达{human['natural_collection_status']['human_authored_submissions']}、合格新测试指令{human['natural_collection_status']['qualified_test_queries']}。至少20新场景各20有效指令的正式新封存仍缺，不用模板或自动标签补齐人工。

390业务源码/根索引及冻结依赖保持；最近普通回归452通过、纯代码导出450通过2数据跳过。本次辅助Ruff及实际集成分列，最终全量历史保护和整合打包待原GPU队列结束。检查点037；当前版本未推送，按用户授权在实验收尾核验后同步GitHub。未来论文由用户以GPT Work组织写作。

本次实际证据：`faults/semantic_async_cpu_003/delivery_001/summary.json`；逐条件原始输入/输出/传感器和控制在三个版本目录，失败不覆盖。完整既有阶段分类参见第16版。
'''
        (OUTPUT/'实验完成度与交付清单_017.md').write_text(text)
        state = deepcopy(read(OUTPUT/'实验完成度与交付清单_016.json'))
        evidence = ['faults/semantic_async_cpu_003/delivery_001/summary.json']
        state.update(updated_utc=now, source_checkpoint='source_bundle/source_checkpoint_037',
            semantic_CPU_fault_exposures_passed=accepted, semantic_CPU_fault_conditions=8,
            semantic_CPU_fault_actual_Qwen_calls=summary['counted_new_CPU_Qwen_calls'],
            semantic_minus_recorded_snapshot=minus['audited_cases'], human_snapshot=human,
            goal_complete=False, current_sources_published=False, manuscript_work_deferred_by_user=True)
        state['items'].append(dict(stage=12, name='真实语义异步故障CPU集成', status=f'八条件实际执行，暴露验收{accepted}/8',
            implemented='实际模型后注入、真实渲染/感知、控制及全部独立证据齐备，原失败和未暴露条件保留。',
            remaining='共享CPU、同一任务、终态观察版本分列；不代替正式GPU故障、新封存和独占性能。', evidence=evidence))
        state['evidence_sha256'].update({name:digest(OUTPUT/name) for name in evidence})
        dump(OUTPUT/'实验完成度与交付清单_017.json', state)
        aliases = ['experiment_report.md', 'remaining_issues.md', 'reproducible_commands.md', '工作交接_代码与实验.md', 'GPT_Work_写作交接.md', 'validation_report.json']
        backup = destination/'status_before_017'
        backup.mkdir()
        for name in aliases:
            path = OUTPUT/name
            shutil.copy2(path, backup/name)
            if name.endswith('.md'):
                path.write_text(text+'\n\n---\n\n以下为此前保留快照，以第17版及原始证据为准。\n\n'+path.read_text())
        validation = read(OUTPUT/'validation_report.json')
        validation.update(updated_utc=now, goal_complete=False, manuscript_work_deferred_by_user=True)
        validation['statuses']['真实语义异步故障CPU集成'] = dict(status=f'八条件实际执行，暴露验收{accepted}/8', evidence=evidence,
            remaining='CPU同任务及固定观察窗；正式GPU故障、安全停止/恢复、性能和新封存仍待。')
        validation['statuses']['实验交接最新清单'] = dict(status='第17版实验状态和交付准备',
            evidence=['实验完成度与交付清单_017.md', '实验完成度与交付清单_017.json'],
            remaining='GPU队列、人工标签、新封存、最终保护/打包/Git同步待完成，不写论文正文。')
        replace_json(OUTPUT/'validation_report.json', validation)
        copied = {}
        secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
        exports = {'doc/thesis_completion/experiment_delivery_status_017.md':'实验完成度与交付清单_017.md',
            'doc/thesis_completion/experiment_delivery_status_017.json':'实验完成度与交付清单_017.json',
            'doc/thesis_completion/validation_snapshot_017.json':'validation_report.json',
            'experiments/semantic_CPU_faults/summary.json':'faults/semantic_async_cpu_003/delivery_001/summary.json'}
        for path in auxiliaries:
            exports['reproduction/'+path.name] = str(path.relative_to(OUTPUT))
        for target, source in exports.items():
            path = export/target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OUTPUT/source, path)
            assert not secret.search(path.read_bytes())
            copied[target] = dict(source=source, sha256=digest(path))
        export_state = read(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_022.json')
        export_state.update(created_utc=now, current_sources_published=False, semantic_CPU_fault_evidence=copied,
            current_increment_actual_CPU_frontend_calls=summary['counted_actual_frontend_calls'],
            current_increment_actual_Qwen_calls=summary['counted_new_CPU_Qwen_calls'],
            current_increment_new_independent_tasks=0, current_increment_frozen_main_arrival_credit=0)
        dump(OUTPUT/'source_bundle/github_sync_001/incremental_export_manifest_023.json', export_state)
        subprocess.run(['git', 'diff', '--check'], cwd=PROJECT, check=True)
        subprocess.run(['git', 'diff', '--check'], cwd=export, check=True)
        assert digest(PROJECT/'.git/index') == previous['original_index_sha256']
        dump(destination/'complete.json', dict(passed=True, all_eight_condition_exposures_passed=accepted == 8,
            source_checkpoint='source_bundle/source_checkpoint_037', original_index_unchanged=True,
            source_files=390, checked_frozen_sets=frozen_counts, full_historical_tree_not_rechecked=True,
            copied_export_sha256=copied, actual_model_calls_added_by_delivery=0,
            manuscript_text_written=False, current_sources_published=False, new_test_opened=False, goal_complete=False))
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
