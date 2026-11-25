"""车辆执行全任务统计，按场景配置聚类，保留GT辅助条件的独立口径。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, dump, digest


def clustered_interval(rows, value, denominator, *, repetitions=4000):
    groups = defaultdict(lambda: [0.,0.])
    for r in rows:
        groups[r['config_id']][0] += value(r)
        groups[r['config_id']][1] += denominator(r)
    data = np.asarray(list(groups.values()))
    if not len(data) or not data[:,1].sum():
        return dict(estimate=None,ci95=None,clusters=len(data),denominator=0)
    rng = np.random.default_rng(17)
    samples = data[rng.integers(0,len(data),(repetitions,len(data)))].sum(axis=1)
    valid = samples[:,1]>0
    interval = np.quantile(samples[valid,0]/samples[valid,1],[.025,.975]).tolist()
    return dict(estimate=float(data[:,0].sum()/data[:,1].sum()),ci95=interval,
                clusters=len(data),denominator=int(data[:,1].sum()),numerator=float(data[:,0].sum()))


def summarize(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method']==method]
        def arrival(r):
            return bool(r.get('actual_kinematic_arrival',r.get('actual_full_perception_arrival',False)))
        pos = [r for r in subset if r['kind']=='reachable_positive']
        executions = [r for r in subset if r.get('actual_path_length_m',r.get('path_length_m',0))>.01]
        def audited(r):
            return r.get('independent_execution_audit_passed',r.get('independent_execution_audit',{}).get('passed'))
        seed_reports = {}
        for seed in sorted({r['seed'] for r in subset}):
            s = [r for r in pos if r['seed']==seed]
            seed_reports[str(seed)] = dict(positive_trials=len(s),arrivals=sum(map(arrival,s)),
                statuses=dict(Counter(r['status'] for r in s)))
        result[method] = dict(trials=len(subset),positive_trials=len(pos),
            positive_arrival=clustered_interval(pos,arrival,lambda _:1),
            positive_route=clustered_interval(pos,lambda r:r['route_generated'],lambda _:1),
            statuses_by_kind={kind:dict(Counter(r['status'] for r in subset if r['kind']==kind))
                              for kind in sorted({r['kind'] for r in subset})},
            safety_denominator='实际轨迹长度>0.01m的试验；原地等待/零路径不增加安全执行信用',
            executed_trials=len(executions),independent_audits_passed=sum(bool(audited(r)) for r in executions),
            safety_rate_among_executed=(sum(bool(audited(r)) for r in executions)/len(executions)
                                        if executions else None),
            distinct_positive_tasks_arrived=len({r['task_id'] for r in pos if arrival(r)}),
            configs_with_positive_arrival=len({r['config_id'] for r in pos if arrival(r)}),
            seeds=seed_reports)
    methods = sorted(result)
    differences = {}
    for a in methods:
        for b in methods:
            if a>=b:
                continue
            lookup = {(r['task_id'],r['seed'],r['method']):r for r in rows}
            paired = []
            for r in rows:
                if r['method']!=a or r['kind']!='reachable_positive':
                    continue
                other = lookup.get((r['task_id'],r['seed'],b))
                if other:
                    ar = bool(r.get('actual_kinematic_arrival',r.get('actual_full_perception_arrival',False)))
                    br = bool(other.get('actual_kinematic_arrival',other.get('actual_full_perception_arrival',False)))
                    paired.append(dict(config_id=r['config_id'],difference=int(ar)-int(br)))
            differences[a+' minus '+b] = clustered_interval(paired,lambda r:r['difference'],lambda _:1)
    return result,differences


def add_execution_evidence(rows,source):
    """读取真实积分状态；状态行数不能证明车辆发生运动。"""
    enriched=[]
    for original in rows:
        row=dict(original)
        root=source.parent/row['task_id']/f"seed{row['seed']}"
        choices=[root/row['method']/'execution.jsonl',root/'execution.jsonl']
        files=[p for p in choices if p.exists()]
        if len(files)>1:
            raise ValueError('ambiguous execution evidence')
        if files:
            path=files[0]
            states=[json.loads(line)['state'] for line in path.read_text().splitlines()]
            length=sum(float(np.hypot(b['x']-a['x'],b['y']-a['y'])) for a,b in zip(states,states[1:]))
            row.update(actual_path_length_m=length,actual_state_records=len(states),
                measured_execution_source=str(path.resolve()),measured_execution_sha256=digest(path),
                first_interval_not_in_length='首个保存状态之前的区间不用于运动距离，避免推测起点')
        enriched.append(row)
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    source = Path(args.input)
    rows = add_execution_evidence([json.loads(s) for s in source.read_text().splitlines()],source)
    methods,differences = summarize(rows)
    oracle = any(r.get('gt_assisted_correct_object') for r in rows)
    dest = OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    report = dict(source=str(source.resolve()),source_sha256=digest(source),
        statistics_source_sha256=digest(Path(__file__)),trials=len(rows),methods=methods,
        paired_positive_arrival_differences=differences,
        bootstrap=dict(unit='synthetic scene configuration',repetitions=4000,seed=17),
        gt_assisted_correct_object=oracle,
        full_perception_minimum_arrival_credit=0 if oracle else sum(r.get('actual_full_perception_arrival',False) for r in rows),
        limitations=['10参数配置源自3种共享走廊家族；配置并非独立城市',
            '帧、调用及seed不增加独立场景/任务计数',
            '不给规划失败或没有执行的任务100%安全信用',
            '给定正确对象的无匹配拒绝是GT上限，不能评价实际语义拒绝能力'] if oracle else
            ['简单合成环境；规则与真实时钟异步Qwen需分别报告'],goal_complete=False)
    dump(dest/'report.json',report)
    with (dest/'methods.csv').open('x') as f:
        writer = csv.DictWriter(f,fieldnames=['method','trials','positive_trials','arrivals','arrival_rate',
            'arrival_ci_low','arrival_ci_high','distinct_positive_tasks_arrived','configs_with_positive_arrival'])
        writer.writeheader()
        for method,r in methods.items():
            a = r['positive_arrival']
            writer.writerow(dict(method=method,trials=r['trials'],positive_trials=r['positive_trials'],
                arrivals=a['numerator'],arrival_rate=a['estimate'],arrival_ci_low=a['ci95'][0],
                arrival_ci_high=a['ci95'][1],distinct_positive_tasks_arrived=r['distinct_positive_tasks_arrived'],
                configs_with_positive_arrival=r['configs_with_positive_arrival']))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
