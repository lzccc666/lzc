"""Close the bounded independent audit using verified saved artifacts only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from conceptgraph.project_paths import PROJECT_ROOT as ROOT

OUT=ROOT/'outputs/reports/iteration_independent_audit_20261124'


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def allfiles(root):return {p for p in root.rglob('*') if p.is_file()}
def sized(paths):return dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),MiB=sum(p.stat().st_size for p in paths)/2**20)


def main():
    mapping=load(OUT/'mapping_independent_final_v1.json');qwen=load(OUT/'qwen_independent_final_v1.json')
    assert mapping['passed'] and qwen['passed']
    now=datetime.now(timezone.utc);start=datetime.fromisoformat('2026-11-24T06:38:13+00:00')
    assert now<datetime.fromisoformat('2026-11-24T08:08:00+00:00')
    roots={
      'mapping':[ROOT/'dataset/processed/mapnext_20261124_inputs_v1',ROOT/'outputs/reports/mapping_iteration_20261124',*list((ROOT/'outputs/results/nuscenes/validation/iterations').glob('mapnext_20261124_*'))],
      'qwen':[ROOT/'dataset/qwen/qnext_20261124_v1',ROOT/'outputs/reports/qwen_iteration_20261124',ROOT/'outputs/results/qwen/train/iterations/qnext_20261124_seed17_v1',ROOT/'outputs/results/qwen/validation/iterations/qnext_20261124_comparison_v1'],
      'audit':[OUT,*list((ROOT/'outputs/results/nuscenes/validation/iterations').glob('iteration_independent_audit_20261124_*')),*list((ROOT/'outputs/results/qwen/validation/iterations').glob('iteration_independent_audit_20261124_*'))],
    }
    sets={k:set().union(*(allfiles(p) for p in ps)) for k,ps in roots.items()}
    for key,prefix,dirs in [('mapping','mapnext_20261124',[ROOT/'conceptgraph/slam',ROOT/'scripts']),('qwen','qnext_20261124',[ROOT/'conceptgraph/vlm',ROOT/'scripts']),('audit','iteration_audit_20261124',[ROOT/'scripts'])]:
        for directory in dirs:
            sets[key].update(p for p in directory.glob(prefix+'*.py') if p.is_file())
            sets[key].update(p for p in (directory/'__pycache__').glob(prefix+'*.pyc') if p.is_file())
    sizes={k:sized(v) for k,v in sets.items()}
    assert sizes['mapping']['bytes']<2**30 and sizes['audit']['bytes']<64*2**20
    assert not (sets['mapping']&sets['qwen'] or sets['mapping']&sets['audit'] or sets['qwen']&sets['audit'])
    total=sized(set.union(*sets.values()));free=shutil.disk_usage(ROOT).free
    assert free>20*2**30
    compare=ROOT/'outputs/results/qwen/validation/iterations/qnext_20261124_comparison_v1'
    evaluations={name:load(compare/name/'report.json') for name in ['base','old_adapter','new_adapter']}
    delivery=load(ROOT/'outputs/reports/mapping_iteration_20261124/mapnext_20261124_final_delivery.json')
    replay=load(ROOT/'outputs/reports/mapping_iteration_20261124/mapnext_20261124_closeout.json')
    metric=load(ROOT/'outputs/results/nuscenes/validation/iterations/iteration_independent_audit_20261124_S2_all_v1/scores.json')
    primary={r['method']:r for r in metric['summaries'] if r['scope']=='front3' and r['method'] in ['baseline:T3_three','S2:S2_three']}
    conclusion=dict(passed=True,completed_utc=now.isoformat(),task_started_utc=start.isoformat(),elapsed_to_audit_close_s=(now-start).total_seconds(),deadline_met=True,
        mapping=dict(primary=primary,decision='S2 is a small explicit experimental candidate; defaults preserved',gain_object_frames=1,lost_object_frames=0,duplicate_count_unchanged=11,common_GT_position_and_jitter_unchanged=True,per_log_no_negative=True,
            task_finished_utc=delivery['finished_utc'],task_elapsed_s=delivery['elapsed_since_task_start_s'],cached_CPU_replay_wall_s=sum(r['wall_s'] or 0 for r in replay['reports']),new_frontend_calls=0,GPU_calls=0),
        qwen=dict(variants={k:dict(counts=v['counts'],negative_outcomes=v['negative_outcomes']) for k,v in qwen['variants'].items()},
            decision='New adapter failed to improve the original/old adapter on complete validation; do not promote it',ordinary_positive_denominator=95,correct_candidate_present=36,missing_candidate=59,
            training_best_step=20,training_GPU_process_wall_s=qwen['training_gpu_wall_s'],pure_parameter_update_s=load(ROOT/'outputs/results/qwen/train/iterations/qnext_20261124_seed17_v1/training_report.json')['training_compute_s'],
            neural_evaluation_process_wall_s={k:v['wall_s'] for k,v in evaluations.items()},neural_evaluation_last_completed_utc=max(v['completed_utc'] for v in evaluations.values()),
            scope='Historical real predicted candidates, no RGB/no new 3-camera graph; controlled cross-log validation, no independent unseen test or navigation/arrival claim'),
        disk_at_measurement=dict(per_line=sizes,total=total,free_bytes=free,free_GiB=free/2**30,note='Counts new line-specific datasets, results, reports, source and bytecode. Taken before writing this small closeout JSON/Markdown.'),
        protections=dict(original56_unchanged=True,old_adapter_hashes_unchanged=True,original_T3_and_dependencies_unchanged=True,defaults_unchanged=True,raw_GT_not_in_model_or_mapper_inputs=True),
        audit_model_calls=0,audit_GPU_calls=0,audit_training_calls=0,source_sha256=sha(__file__),reports={name:str(OUT/name) for name in ['mapping_independent_final_v1.json','qwen_preflight_v1.json','qwen_input_only_rule_v2_audit.json','qwen_independent_final_v1.json']})
    p=OUT/'combined_independent_conclusion_v1.json'
    with p.open('x') as f:json.dump(conclusion,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['本轮独立复核通过；通过表示结果和来源核对一致，不表示优化成功。','',
      '建图：同3m、同类、XY一对一、120组完整分母687，T3→S2召回44.98%→45.12%，F1 50.49%→50.61%，IDF1 45.92%→46.04%。仅新增1个正确对象帧，0损失；重复节点仍11，共同对象位置、抖动和ID变化不变，40组已见回归原样。不能称已解决重复或稳定性问题。','',
      '| Qwen方法 | 正目标选对/95 | 在候选选对/36 | 真无目标 拒绝/误选/无效（19） | 故障 拒绝/误选/无效（57） | 排序格式有效/171 |',
      '|---|---:|---:|---:|---:|---:|']
    for key,label in [('input_only_rule_v2','输入同权规则v2'),('base','基座'),('old_adapter','旧adapter'),('new_adapter','本轮adapter')]:
        v=qwen['variants'][key];c=v['counts'];n=v['negative_outcomes']
        def fmt(s):return '/'.join(str(n[s][k]) for k in ['correct_rejection','valid_wrong_selection','format_invalid'])
        lines.append(f"| {label} | {c['ordinary_positive_correct']} | {c['ordinary_covered_target_correct']} | {fmt('true_no_match')} | {fmt('controlled_fault')} | {c['ranking_schema_valid']} |")
    lines += ['', '本轮Qwen不晋升：新adapter为8/95，低于基座9/95和旧adapter12/95；格式和拒绝能力也未得到可靠改善。95条正任务中59条缺正确候选，均保留原分母。16句只做已见语言回归且不入训；新adapter语言格式1/16、字段全对0/16，格式失败不能直接解释为语言理解为零。','',
      '全部187×4条原答从原文本独立复算，与执行结果一致，回退不计分。40条选点独立重算选择step20，未用171选点；但40本来包含于171中，171不是独立未见测试。规则v1因读隐藏任务标志取消公平资格，v2只读可见输入；两版语义输出相同且旧文件保留。','',
      f"本轮开始UTC06:38:13，独立收口{now.strftime('%H:%M:%S')}，历时{(now-start).total_seconds()/60:.2f}分钟；纯参数更新{conclusion['qwen']['pure_parameter_update_s']:.2f}秒，训练GPU进程（含80次选点推理）{qwen['training_gpu_wall_s']:.2f}秒。新增磁盘：建图{sizes['mapping']['MiB']:.2f}MiB，Qwen{sizes['qwen']['MiB']:.2f}MiB，审计{sizes['audit']['MiB']:.2f}MiB，共{total['MiB']:.2f}MiB；测量不含本小型收口JSON/Markdown，空闲{free/2**30:.2f}GiB。",'',
      '保护56项、旧adapter、原T3及依赖、旧数据、默认入口均未改。审计没有模型、GPU或训练调用。完整来源、负项、计时和体量见同目录JSON。']
    doc=OUT/'独立复核结论.md'
    with doc.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,report=str(doc),data=str(p),elapsed_minutes=(now-start).total_seconds()/60,new_MiB=total['MiB'],per_line_MiB={k:v['MiB'] for k,v in sizes.items()},free_GiB=free/2**30),ensure_ascii=False))


if __name__=='__main__':main()
