"""全体保存执行的CPU方向审核；保留静止/未知，不重跑模型或控制。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

COHORTS = {'rule': ('independent_body_rule_full_001', 150),
           'fault_cpu': ('independent_body_fault_cpu_full_001', 6),
           'main': ('independent_body_main_full_001', 450)}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError('smoke数量必须为正整数')
    started = time.monotonic()
    body_name, expected_count = COHORTS[args.cohort]
    parent = OUTPUT/'vehicle_execution'/body_name
    parent_report = read(parent/'report.json')
    assert parent_report['cases'] == parent_report['expected_cases'] == expected_count
    assert not parent_report['is_smoke']
    plans = [json.loads(line) for line in (parent/'cases.jsonl').open()]
    assert len(plans) == expected_count and len({(c['task'], c['seed'], c['method']) for c in plans}) == expected_count
    tasks_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    tasks = read(tasks_path)
    configs = {c['config_id']: c for c in tasks['configs']}
    root = OUTPUT/'vehicle_execution'/args.run_id
    root.mkdir(exist_ok=False)
    module = PROJECT/'conceptgraph/completion/execution_direction_audit.py'
    protocol = OUTPUT/'task_protocol/execution_direction_audit_001.md'
    frozen = {str(p.relative_to(PROJECT)): digest(p) for p in [Path(__file__).resolve(), module, protocol]}
    dump(root/'release.json', dict(cohort=args.cohort, source_sha256=frozen,
        task_release_sha256=digest(tasks_path), parent_body_report_sha256=digest(parent/'report.json'),
        parent_body_cases_sha256=digest(parent/'cases.jsonl'), expected_cases=expected_count,
        selected_limit=args.limit, budget_wall_s=900, budget_bytes=1024**3, minimum_free_bytes=6*1024**3,
        CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), shared_CPU=True,
        posthoc_development_audit=True, new_test_opened=False, goal_complete=False))
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'

    def event(state):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), event=args.run_id,
                         status=state, shared_CPU=True, CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\n')

    event('start')
    results = []
    try:
        for index, plan in enumerate(plans[:args.limit]):
            assert time.monotonic()-started < 900 and shutil.disk_usage(OUTPUT).free > 6*1024**3
            assert all(digest(PROJECT/p) == value for p, value in frozen.items())
            case = Path(plan['raw_case'])
            key = f"{index:04d}_{plan['task']}_seed{plan['seed']}_{plan['method']}"
            record = {field: plan[field] for field in ['task', 'config_id', 'seed', 'method', 'raw_case']}
            record.update(prior_reported_arrival=plan['prior_reported_arrival'],
                          prior_body_certificate=plan['passed'] is True, source_sha256={})
            for path, value in plan['source_sha256'].items():
                assert digest(path) == value
                record['source_sha256'][path] = value
            execution, perception = case/'execution.jsonl', case/'perception.jsonl'
            if not execution.exists() or not perception.exists() or plan['audited_intervals'] == 0:
                record.update(status='unknown_missing_complete_evidence', passed=None, audited_intervals=0,
                              moving_intervals=0, counts={})
            else:
                # 来源必须已经绑定，不能临时补入另一版本的执行/感知。
                assert str(execution) in record['source_sha256'] and str(perception) in record['source_sha256']
                rows = [json.loads(line) for line in execution.open()]
                initial = json.loads(next(perception.open()))['sensor']['state']
                audited = audit_execution_direction(initial, rows, configs[plan['config_id']]['centerline_xy'], tasks['vehicle'])
                dump(root/key/'interval_audit.json', audited)
                record.update({k: v for k, v in audited.items() if k != 'intervals'})
                record['saved_control_records'] = len(rows)
                if initial['time_s'] == 0:
                    assert audited['audited_intervals'] == len(rows) == plan['audited_intervals']
            record['combined_body_and_moving_direction_certificate'] = bool(
                record['prior_body_certificate'] and record['passed'] is True)
            dump(root/key/'report.json', record)
            results.append(record)
            with (root/'cases.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(dict(case=index+1, status=record['status'], intervals=record['audited_intervals'])), flush=True)
        groups = defaultdict(lambda: Counter())
        for row in results:
            c = groups[row['config_id']]
            c['cases'] += 1
            c['moving_cases'] += row['moving_intervals'] > 0
            c['forward_certified_cases'] += row['passed'] is True
            c['stationary_not_exercised'] += row['status'] == 'not_exercised_stationary'
            c['wrong_way_cases'] += row['status'] == 'wrong_way_witness'
            c['unknown_cases'] += row['status'].startswith('unknown_')
        with (root/'by_config.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['config_id', 'cases', 'moving_cases', 'forward_certified_cases',
                                                       'stationary_not_exercised', 'wrong_way_cases', 'unknown_cases'])
            writer.writeheader()
            writer.writerows(dict(config_id=k, **v) for k, v in sorted(groups.items()))
        total_moving = sum(r['moving_intervals'] > 0 for r in results)
        passed = sum(r['passed'] is True for r in results)
        ci = None
        if len(groups) > 1 and total_moving:
            matrix = np.array([[v['forward_certified_cases'], v['moving_cases']] for _, v in sorted(groups.items())])
            rng = np.random.default_rng(5723)
            values = []
            for _ in range(4000):
                selected = matrix[rng.integers(0, len(matrix), size=len(matrix))].sum(axis=0)
                if selected[1]:
                    values.append(float(selected[0]/selected[1]))
            ci = np.quantile(values, [.025, .975]).tolist() if values else None
        report = dict(cohort=args.cohort, cases=len(results), expected_cases=expected_count,
            is_smoke=args.limit is not None, status_counts=dict(Counter(r['status'] for r in results)),
            audited_intervals=sum(r['audited_intervals'] for r in results),
            moving_intervals=sum(r['moving_intervals'] for r in results), moving_cases=total_moving,
            direction_certified_moving_cases=passed,
            direction_certificate_coverage_among_observed_moving_cases=passed/total_moving if total_moving else None,
            config_cluster_bootstrap_95ci=ci, bootstrap_replicates=4000 if ci is not None else 0,
            cluster_count=len(groups), bootstrap_seed=5723,
            previously_reported_arrivals=sum(r['prior_reported_arrival'] for r in results),
            previous_arrivals_with_body_and_direction=sum(r['prior_reported_arrival'] and r['combined_body_and_moving_direction_certificate'] for r in results),
            previous_arrivals_without_combined_certificate=[dict(task=r['task'], seed=r['seed'], method=r['method'], status=r['status']) for r in results
                if r['prior_reported_arrival'] and not r['combined_body_and_moving_direction_certificate']],
            maximum_arc_endpoint_residual_m=max((r.get('max_endpoint_arc_residual_m') or 0 for r in results), default=None),
            multi_lane_connection_accuracy=None, connection_status='not_exercised_single_directed_lane',
            parent_body_report_sha256=digest(parent/'report.json'), all_original_results_unchanged=True,
            CPU_shared_wall_s=time.monotonic()-started, actual_model_calls=0, new_executions=0, new_arrivals=0,
            limits=['固定保存积分模型的连续速度方向；不是另一套动力学积分、车身碰撞或动态交通证明',
                    '方向证书覆盖率只描述已保存且实际行驶的试验，不能外推未知道路安全概率',
                    '十配置共用三种道路生成族；故障仅一个配置/任务/种子，不给配置置信区间',
                    '缺失或未行驶保留，单向单车道不提供路口/多车道连接正确率'], goal_complete=False)
        dump(root/'report.json', report)
        dump(root/'complete.json', dict(complete_cohort=args.limit is None and len(results) == expected_count,
             cases=len(results), report_sha256=digest(root/'report.json'), goal_complete=False))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        dump(root/'interrupted.json', dict(error=repr(exc), completed_cases=len(results), goal_complete=False))
        raise
    finally:
        event('end')


if __name__ == '__main__':
    main()
