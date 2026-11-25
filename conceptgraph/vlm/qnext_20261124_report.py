"""Produce an auditable report for the fixed Qnext 20261124 comparison."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from conceptgraph.vlm.qnext_20261124 import (
    DATA, EVAL, REPORT, ROOT, TRAIN, canonical, dump, now, prompt, rows, score, sha,
)


def main():
    source=rows(DATA/'validation_full.jsonl')+rows(DATA/'language_seen_regression.jsonl')
    by_id={r['sample_id']:r for r in source}
    manifest=json.loads((DATA/'manifest.json').read_text())
    runtime=json.loads((REPORT/'compatibility_amendment_v1.json').read_text())
    assert sha(ROOT/'conceptgraph/vlm/qnext_20261124.py')==runtime['runtime_source_sha256']
    protected={p:sha(p)==h for p,h in manifest['protected_sha256'].items()}
    assert all(protected.values())
    for name,h in manifest['files_sha256'].items():assert sha(DATA/name)==h
    output={};first_prompts=None
    for variant in ['input_only_rule_v2','base','old_adapter','new_adapter']:
        recs=rows(EVAL/variant/'raw_records.jsonl')
        assert len(recs)==187 and len({r['sample_id'] for r in recs})==187
        assert {r['sample_id'] for r in recs}==set(by_id)
        fingerprints={r['sample_id']:r['prompt_sha256'] for r in recs}
        expected={r['sample_id']:hashlib.sha256(prompt(r).encode()).hexdigest() for r in source}
        assert fingerprints==expected
        if first_prompts is None:first_prompts=fingerprints
        assert first_prompts==fingerprints
        rank=[r for r in recs if r['task']=='candidate_rerank']
        lang=[r for r in recs if r['task']=='language_parse']
        for r in recs:
            rescore=score(by_id[r['sample_id']],r['raw_output'])
            for k,v in rescore.items():assert r[k]==v,(variant,r['sample_id'],k)
        parseable=0;parseable_correct=0;syntax_fail=0;schema_fail=Counter();selected_counts=Counter()
        for r in rank:
            try:p=json.loads(r['raw_output'])
            except (ValueError,TypeError):syntax_fail+=1;continue
            parseable+=1
            if isinstance(p,dict):
                s=p.get('selected_node_id')
                if s is not None:selected_counts[str(s)]+=1
                if type(s) is int and s in by_id[r['sample_id']]['acceptable_node_ids']:parseable_correct+=1
            if not r['schema_valid']:schema_fail[r['validation_error'].splitlines()[0]]+=1
        logs={}
        for log in manifest['validation_logs']:
            rs=[r for r in rank if by_id[r['sample_id']]['log_token']==log]
            pos=[r for r in rs if r['stratum'] in ['ordinary_covered','ordinary_missing']]
            logs[log]=dict(total=len(rs),ordinary_positive=len(pos),ordinary_correct=sum(r['target_correct'] for r in pos),
                           covered=sum(r['stratum']=='ordinary_covered' for r in pos))
        copy_groups={}
        for group,group_names in [('ordinary_positive',{'ordinary_covered','ordinary_missing'}),('true_no_match',{'true_no_match'}),('controlled_fault',{'controlled_fault'})]:
            valid_group=[r for r in rank if r['schema_valid'] and r['stratum'] in group_names]
            first_count=0;exact_count=0;exact_text_count=0
            for record in valid_group:
                original=by_id[record['sample_id']];ids=original['candidate_ids'];parsed=record['normalized_output']
                first_count+=bool(ids and parsed['selected_node_id']==ids[0])
                example=dict(schema_version='1.0.0',selected_node_id=ids[0] if ids else None,ranking=ids,
                             no_match=not bool(ids),explanation='按观测证据判断',confidence=.8)
                exact_count+=parsed==example
                exact_text_count+=record['raw_output']==canonical(example)
            copy_groups[group]=dict(schema_valid=len(valid_group),selected_first_candidate=first_count,exact_example_object_copy=exact_count,exact_example_text_copy=exact_text_count)
        language_diagnostic=dict(json_parseable_after_fence_removal=0,fence_removal_needed=0,action_correct=0,target_description_exact=0)
        for record in lang:
            raw=record['raw_output'];stripped=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip())
            try:parsed=json.loads(stripped)
            except (ValueError,TypeError):continue
            language_diagnostic['json_parseable_after_fence_removal']+=1
            language_diagnostic['fence_removal_needed']+=stripped!=raw.strip()
            if isinstance(parsed,dict):
                expected_language=by_id[record['sample_id']]['response']
                language_diagnostic['action_correct']+=parsed.get('action')==expected_language['action']
                language_diagnostic['target_description_exact']+=parsed.get('target_description')==expected_language['target_description']
        report=json.loads((EVAL/variant/'report.json').read_text())
        lats=[r['latency_s'] for r in rank]
        output[variant]=dict(summary=report['summary'],wall_s=report['wall_s'],peak_gpu_mib=report['peak_gpu_mib'],
            latency_ranking_median_s=statistics.median(lats),latency_ranking_p90_s=sorted(lats)[int(.9*(len(lats)-1))],
            ranking_json_parseable=parseable,ranking_json_syntax_failures=syntax_fail,schema_failure_types=dict(schema_fail),
            diagnostic_parseable_target_correct_without_contract=parseable_correct,
            diagnostic_note='JSON-parsed target selection without schema validity is diagnostic only; never credited in main metric.',
            prompt_example_copy_diagnostic=copy_groups,language_raw_field_diagnostic=language_diagnostic,
            language_semantic_fields_all_correct=sum(all(r['field_correct'].values()) for r in lang),per_log=logs,
            raw_records_sha256=sha(EVAL/variant/'raw_records.jsonl'))
    train=json.loads((TRAIN/'training_report.json').read_text())
    sizes={str(p.relative_to(ROOT)):sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in [DATA,REPORT,TRAIN,EVAL]}
    summary=dict(created_utc=now(),variants=output,training=dict(best_step=train['best_step'],max_steps=40,seed=17,
        training_compute_s=train['training_compute_s'],training_process_gpu_wall_s=train['training_process_gpu_wall_s']),
        official_positive_tasks=95,covered_positive_tasks=36,missing_positive_tasks=59,
        fixed_evaluation=dict(ranking=171,seen_language_regression=16,checkpoint_selection=40,formal_includes_selection_subset=True),
        all_prompt_hashes_equal=True,raw_outputs_rescored_identically=True,protected_files_unchanged=len(protected),
        scope='Historical real predicted graph, controlled same-template across log validation; no unseen-language test, RGB, new three-view graph, navigation or closed-loop claim.',
        size_bytes=sizes,report_source_sha256=sha(Path(__file__)),
        discarded_rule_v1=dict(path=str(EVAL/'rule'),reason='Reads task-family/fault metadata unavailable to models; retained, disqualified from fair comparison. CPU input-only v2 supersedes without selecting by score.'))
    dump(REPORT/'comparison_summary.json',summary)
    lines=['# Qwen首轮受控短训与完整验证', '',
      '这是原真实预测图上的跨log验证，同模板可重复；40条选点验证包含在171完整验证内，因此171也不是独立未见测试。16条用户审核句只做已见语言回归，不参与训练。', '',
      '训练40条（目标可选20、漏候选4、真无目标4、证据故障12），单seed17、40步；按固定40选点选择第20步。无新增场景GT、RGB或GT补节点。三模型均由全新进程加载本地基座和相应adapter，共用提示、严格schema、贪心解码和256 token上限。', '',
      '| 方法 | 普通正任务选对/95 | 目标在候选选对/36 | 真无目标误选/19↓ | 证据故障误选/57↓ | 排序schema有效/171 | 语言schema/16 | 语言全字段正确/16 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels=[('input_only_rule_v2','仅输入规则v2'),('base','原始Qwen'),('old_adapter','旧adapter'),('new_adapter','本轮adapter')]
    for name,label in labels:
        v=output[name];s=v['summary'];r=s['ranking'];l=s['language_seen_regression']
        def rate(n,d):return f'{n}/{d} ({n/d:.1%})'
        lines.append('| '+' | '.join([label,rate(s['ordinary_positive_correct'],95),rate(s['ordinary_covered_correct'],36),
          rate(s['true_no_match_wrong_selection'],19),rate(s['controlled_fault_wrong_selection'],57),rate(r['valid'],171),
          rate(l['schema_valid'],16),rate(v['language_semantic_fields_all_correct'],16)])+' |')
    lines+=['','| 方法 | 真无目标正确拒绝/19 | 真无目标可确认误选/19 | 真无目标格式无效/19 | 故障正确拒绝/57 | 故障可确认误选/57 | 故障格式无效/57 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in labels:
        counts=output[name]['summary']['ranking']
        lines.append('| '+' | '.join([label,str(counts.get('true_no_match_null_correct',0)),str(counts.get('true_no_match_selected',0)),
          str(19-counts.get('true_no_match_valid',0)),str(counts.get('controlled_fault_null_correct',0)),
          str(counts.get('controlled_fault_selected',0)),str(57-counts.get('controlled_fault_valid',0))])+' |')
    lines += ['',
      '普通正任务的95条中59条目标不在候选中，候选覆盖上限36/95=37.9%。这些任务继续计入全正任务分母，不改作真无目标负例。格式失败按失败计，不给回退模型信用；JSON可解析但schema不合格时的目标选择仅另存诊断。', '',
      '表中误选是schema有效输出中可确认的误选；无效格式单列，不当作正确拒绝。原规则v1使用了任务类别/故障标志，已撤销公平比较资格且保留原输出；此表仅用从输入文字解析任务的规则v2。语言规则采用预置动作/属性词典和标准目标描述，没有逐句或sample_id查表；16/16只表示已见回归覆盖，不能宣称自然语言泛化。', '',
      '语言回归反映对已审核文字的解析，不代表场景中找对停车区域、识别车位空闲或颜色。这些需要图像/官方实例和预测图绑定后的独立证据。关系任务采用所有方法统一的“任一可见同类参照满足方向及12m限制，再按距离最近”协议；不与旧提示历史结果直接相减归因训练收益。', '',
      '本轮40条只更新候选重排参数，完全不含16条语言审核句。16条严格schema不过关表示结构化接口尚不可用，不能解释为语言理解能力为零；去围栏后的动作等字段只在JSON诊断中单列，不转为主分。', '',
      f"训练纯更新{train['training_compute_s']:.1f}秒，含两次40条选点推理的GPU进程墙钟{train['training_process_gpu_wall_s']:.1f}秒。首启SDPA不兼容在加载前失败，随后统一eager；失败日志、原冻结快照与兼容修复记录均保留。未扩seed或长训。", '',
      '原答、逐请求schema/格式错误、候选与来源分别保存在对应run和冻结dataset。模型输入只包含query_text/candidate_nodes/relation_metadata；官方实例和答案在独立标签侧。', '',
      '下一轮只优先纠正示例锚定：冻结没有真实首候选答案暗示的统一输出契约提示，先在固定调试集验证严格schema和正确拒选，未通过前不追加长训。本轮不启动该实验。', '',
      '完整数字与保护SHA复核见comparison_summary.json；不要将旧adapter/新adapter标签或本轮结果替换原论文成绩。']
    p=REPORT/'首轮Qwen完整验证.md'
    with p.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(canonical(dict(report=str(p),comparison=str(REPORT/'comparison_summary.json'),variants=output)))


if __name__=='__main__':main()
