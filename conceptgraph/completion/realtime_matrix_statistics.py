"""完整真实时钟矩阵的冻结分母、四项结果及配对配置聚类统计。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.interactive_statistics import summarize
from conceptgraph.completion.realtime_matrix import verified_records
from conceptgraph.completion.stage9 import OUTPUT, digest, dump
from conceptgraph.completion.vehicle_statistics import clustered_interval


FIELDS=['route_generated','selected_object_evaluated_correct','stable_correct_lock','actual_full_perception_arrival']


def enriched_trial(record):
    """未知业务字段保留None；统计成功数时仍将缺结果任务留在冻结分母。"""
    result=record['result']
    row=dict(task_id=record['task'],config_id=record['config_id'],kind=record['kind'],seed=record['seed'],
        method=record['variant'],status=record['status'],result_missing=result is None,
        process_returncode=record.get('returncode'),process_timeout=record.get('process_timeout',False),
        any_lock=False,actual_path_length_m=0.,actual_state_records=0,
        independent_execution_audit=dict(passed=False,status='unknown_missing_complete_result'))
    row.update({key:None for key in FIELDS})
    if result is not None:
        row.update({key:result[key] for key in FIELDS})
        row['independent_execution_audit']=result['independent_execution_audit']
        row['wall_duration_s']=result['wall_duration_s']
        if result['gt_in_agent'] or not result['real_clock_async']:
            raise ValueError('unexpected GT-assisted or logical-clock trial')
        if result['actual_full_perception_arrival'] and (result['status']!='ARRIVED' or
                not result['independent_execution_audit']['passed']):
            raise ValueError('arrival without independent successful execution audit')
    hashes={}
    partial=[]
    root=Path(record['raw_case'])
    for name in ['execution.jsonl','perception.jsonl','semantic_events.json','independent_execution_audit.json']:
        path=root/name
        if not path.exists():
            if result is not None:
                raise ValueError('complete result missing raw evidence: '+str(path))
            continue
        hashes[str(path)]=digest(path)
        if path.suffix!='.jsonl':
            continue
        raw=[]
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                if result is not None or index!=len(lines)-1:
                    raise
                partial.append(dict(path=str(path),line=index+1,reason='incomplete_tail_of_failed_process'))
        if name=='execution.jsonl':
            states=[r['state'] for r in raw]
            row['actual_path_length_m']=sum(float(np.hypot(b['x']-a['x'],b['y']-a['y']))
                for a,b in zip(states,states[1:]))
            row['actual_state_records']=len(states)
        else:
            row['any_lock']=any(frame.get('selected_node_id') is not None for frame in raw)
    row['raw_sha256']=hashes
    row['partial_failed_process_records']=partial
    row['semantic_applied_by_task']=result.get('semantic_applied_by_task',{}) if result else {}
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wait-budget-s',type=float,default=0.)
    args=parser.parse_args()
    started=time.monotonic()
    source_digest=digest(__file__)
    source=OUTPUT/'interactive_simulation'/args.input_run
    while not (source/'summary.json').exists():
        if time.monotonic()-started>args.wait_budget_s:
            raise TimeoutError('full frozen matrix has not finished; no final comparison written')
        time.sleep(5)
    if digest(__file__)!=source_digest:
        raise ValueError('statistics source changed while waiting')
    release=json.loads((source/'release.json').read_text())
    records=verified_records(source,release)
    if len(records)!=len(release['trials']):
        raise ValueError('all frozen trials required, including failures')
    summary=json.loads((source/'summary.json').read_text())
    if digest(source/'trial_records.jsonl')!=summary['trial_records_sha256']:
        raise ValueError('completed trial list changed')
    rows=[enriched_trial(record) for record in records]
    methods=sorted({r['method'] for r in rows})
    reports={method:summarize([r for r in rows if r['method']==method]) for method in methods}
    paired=[]
    lookup={(r['task_id'],r['seed'],r['method']):r for r in rows}
    for a,b in itertools.combinations(methods,2):
        for field in FIELDS:
            differences=[]
            for row in rows:
                if row['method']!=a or row['kind']!='reachable_positive':
                    continue
                other=lookup[(row['task_id'],row['seed'],b)]
                differences.append(dict(config_id=row['config_id'],difference=int(bool(row[field]))-int(bool(other[field]))))
            paired.append(dict(a=a,b=b,metric=field,**clustered_interval(differences,lambda r:r['difference'],lambda _:1)))
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,dest/'generator_source.py')
    with (dest/'audited_trials.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['method','task_id','config_id','kind','seed','status','result_missing','process_returncode','process_timeout',*FIELDS,'any_lock',
            'actual_path_length_m','actual_state_records','wall_duration_s']
    with (dest/'trials.csv').open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:row.get(key) for key in fields} for row in rows)
    dump(dest/'report.json',dict(source_run=args.input_run,source_release_sha256=digest(source/'release.json'),
        source_summary_sha256=digest(source/'summary.json'),source_records_sha256=digest(source/'trial_records.jsonl'),
        statistics_source_sha256=source_digest,audited_trials_sha256=digest(dest/'audited_trials.jsonl'),
        trials=len(rows),methods=reports,paired_positive_differences=paired,
        missing_complete_results=dict(Counter(r['method'] for r in rows if r['result_missing'])),
        unknown_business_fields='缺报告时四项业务字段为null；成功分子无信用，冻结正任务分母仍保留',
        incomplete_execution_safety='失败进程即使有轨迹前缀也不增加完整安全执行信用；未知不等于已证明碰撞',
        semantic_application_counts={m:{task:sum(r['semantic_applied_by_task'].get(task,0) for r in rows if r['method']==m)
            for task in ['query_parse','object_attributes','candidate_rerank']} for m in methods},
        bootstrap=dict(unit='合成场景配置',repetitions=4000,seed=17,paired=True),
        limitations=['开发合成颜色任务，10配置共享3走廊家族，不是自然指令或新封存测试',
            '三种子/帧/调用不增加独立任务数；全部失败和拒绝保留',
            '后台语义真实单调时钟运行，运动为0.1s独立心跳，不宣称硬实时',
            'CPU与其他开发作业共享；墙钟含模型调度与资源影响，不作为独占性能表',
            '此处核验原始证据指纹并读取原独立执行审计，未新增人工实例正确性信用'],
        independent_test=False,stage11_complete=False,stage12_complete=False,goal_complete=False))
    print(json.dumps(dict(trials=len(rows),methods=methods,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':
    main()
