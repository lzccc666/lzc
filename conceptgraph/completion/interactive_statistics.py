"""完整感知导航的四级结果、实际运动与配置聚类统计，保留所有拒绝和失败。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from conceptgraph.completion.stage9 import OUTPUT, dump, digest
from conceptgraph.completion.vehicle_statistics import add_execution_evidence, clustered_interval


def summarize(rows):
    result={}
    fields=['route_generated','selected_object_evaluated_correct','stable_correct_lock','actual_full_perception_arrival']
    for kind in sorted({r['kind'] for r in rows}):
        subset=[r for r in rows if r['kind']==kind]
        moving=[r for r in subset if r.get('actual_path_length_m',0)>.01]
        audits=[r for r in subset if 'independent_execution_audit' in r]
        result[kind]=dict(trials=len(subset),unique_tasks=len({r['task_id'] for r in subset}),
            configs=len({r['config_id'] for r in subset}),statuses=dict(Counter(r['status'] for r in subset)),
            four_outcomes={key:clustered_interval(subset,lambda r:bool(r[key]),lambda _:1) for key in fields},
            any_lock=clustered_interval(subset,lambda r:r['any_lock'],lambda _:1),
            actually_moving_trials=len(moving),
            safety_among_moving=clustered_interval(moving,lambda r:r['independent_execution_audit']['passed'],lambda _:1),
            stationary_audited_trials=sum(r.get('actual_path_length_m',0)<=.01 for r in audits),
            all_audits_passed=sum(r['independent_execution_audit']['passed'] for r in audits),
            independent_audited_trials=len(audits),
            seeds={str(seed):dict(trials=len(s),four_outcomes={key:sum(bool(r[key]) for r in s) for key in fields})
                for seed in sorted({r['seed'] for r in subset})
                for s in [[r for r in subset if r['seed']==seed]]})
    arrived=[r for r in rows if r['kind']=='reachable_positive' and r['actual_full_perception_arrival']]
    tasks=len({r['task_id'] for r in arrived})
    configs=len({r['config_id'] for r in arrived})
    return dict(by_kind=result,functional_evidence=dict(arrivals=len(arrived),distinct_positive_tasks=tasks,
        scene_configs=configs,minimum_10_tasks_5_configs=tasks>=10 and configs>=5),
        safety_denominator='实际保存轨迹长度>0.01m；静止审计单列；零路径不获得100%安全信用',
        no_match_interpretation='无匹配集合内任何锁定均为误锁；证据不足拒绝不等于已证明全道路无目标')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    source=Path(args.input)
    before=digest(source)
    releasepath=source.parent/'release.json'
    release=json.loads(releasepath.read_text())
    if not (source.parent/'summary.json').exists():
        raise ValueError('full run must finish before final statistics')
    raw=[json.loads(line) for line in source.read_text().splitlines()]
    expected={(task,seed) for task in release['task_ids'] for seed in release['seeds']}
    actual=[(r['task_id'],r['seed']) for r in raw]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('missing or duplicate frozen trial')
    rows=add_execution_evidence(raw,source)
    if digest(source)!=before:
        raise ValueError('source changed during evidence audit')
    report=dict(source=str(source.resolve()),source_sha256=before,
        release_sha256=digest(releasepath),statistics_source_sha256=digest(Path(__file__)),
        trials=len(rows),tasks=len({r['task_id'] for r in rows}),configs=len({r['config_id'] for r in rows}),
        summary=summarize(rows),bootstrap=dict(unit='synthetic scene configuration',repetitions=4000,seed=17),
        limitations=['10参数配置源自3种共享走廊家族，不代表10个独立真实城市',
            '固定逻辑0.1s积分与0.5s感知调度；不作真实时钟异步Qwen成绩',
            '真实YOLO/SAM/CLIP、实际RGB和稀疏LiDAR；HSV颜色基线和4.6m车长/车道对齐先验明确声明',
            '每个任务的3种子不增加不同任务计数；失败或拒绝均保留'],
        gt_in_agent=False,stage11_complete=False,goal_complete=False)
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',report)
    with (dest/'audited_trials.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    keys=['task_id','config_id','kind','seed','status','route_generated','selected_object_evaluated_correct',
        'stable_correct_lock','actual_full_perception_arrival','any_lock','actual_path_length_m',
        'actual_state_records','measured_execution_source','measured_execution_sha256']
    with (dest/'trials.csv').open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:r.get(k) for k in keys} for r in rows)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
