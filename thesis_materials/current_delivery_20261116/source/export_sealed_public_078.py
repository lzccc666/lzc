"""仅在五模型和冻结评价全部完成后，从实际评价文件生成完整图表。"""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'main_experiments/sealed_execution_075'
OUT = ROOT/'figures/sealed_public_078'
METRICS = ['object_recall', 'object_precision', 'IDF1', 'center_error_m',
           'relation_f1', 'duplicate_fraction']
LABELS = {
    'full': '完整方案', 'without_depth_quality': '去深度质量',
    'without_one_to_one': '去一对一关联', 'without_dynamic_maintenance': '去动态维护',
    'without_short_term_expiry': '去短期过期', 'without_relations': '去关系边',
    'original_fusion': '原融合（架构诊断）', 'single_frame': '单帧（架构诊断）',
    'one_to_one_only': '仅一对一（架构诊断）',
    'one_to_one_depth_quality': '一对一加深度（架构诊断）',
    'base': '基础Qwen', 'old_adapter': '旧Adapter',
    'real_graph_seed17': '新Adapter seed17', 'real_graph_seed29': '新Adapter seed29',
    'real_graph_seed43': '新Adapter seed43', 'category_nearest': '类别加最近距离',
    'complete_geometry_rule': '完整几何规则', 'cooperation_seed17': '完整合作 seed17',
    'without_rule_query_resolution': '去规则解析', 'without_attribute_check': '去属性检查',
    'without_candidate_reranking': '去候选重排', 'without_evidence_check': '去证据检查',
    'clip_current_crop': '当前帧CLIP', 'controlled_sparse_clip_grid': '稀疏语义格'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return float(a/b) if b else None


def metric(counts, name):
    c = counts
    definitions = {
        'object_recall': (c.get('true_positive', 0), c.get('truth', 0)),
        'object_precision': (c.get('true_positive', 0), c.get('predictions', 0)),
        'IDF1': (2*c.get('IDTP', 0), 2*c.get('IDTP', 0)+c.get('IDFN', 0)+c.get('IDFP', 0)),
        'center_error_m': (c.get('center_error_sum_m', 0), c.get('true_positive', 0)),
        'relation_f1': (2*c.get('relation_true_positive', 0),
            c.get('relation_predicted_resolved', 0)+c.get('relation_predicted_unresolved', 0)+c.get('relation_truth', 0)),
        'duplicate_fraction': (c.get('duplicate_nodes', 0), c.get('predictions', 0))}
    return ratio(*definitions[name])


def totals(rows):
    output = Counter()
    for row in rows:
        output.update(row['totals'])
        output.update(row['identity'])
    return output


def summarize_mapping(rows, conditions, logs, draws):
    summary, paired = [], []
    full_boot = {}
    for condition in conditions:
        rr = [r for r in rows if r['condition'] == condition]
        assert len(rr) == 30 and sum(r['frames'] for r in rr) == 1196
        by_log = [totals([r for r in rr if r['log'] == log]) for log in logs]
        for name in METRICS:
            value = metric(totals(rr), name)
            boot = []
            for draw in draws:
                count = Counter()
                for i in draw:
                    count.update(by_log[i])
                v = metric(count, name)
                boot.append(np.nan if v is None else v)
            boot = np.asarray(boot)
            finite = boot[np.isfinite(boot)]
            lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (None, None)
            summary.append(dict(condition=condition, label=LABELS[condition], metric=name,
                value=value, ci95_low=float(lo) if lo is not None else None,
                ci95_high=float(hi) if hi is not None else None,
                frames=1196, scenes=30, logs=len(logs), valid_bootstrap_draws=len(finite)))
            if condition == 'full':
                full_boot[name] = boot
            else:
                # 差值原方向保留，误差/重复比例应为负才更好。
                delta = full_boot[name]-boot
                finite_delta = delta[np.isfinite(delta)]
                low, high = np.percentile(finite_delta, [2.5, 97.5]) if len(finite_delta) else (None, None)
                reference = next(r['value'] for r in summary if r['condition'] == 'full' and r['metric'] == name)
                paired.append(dict(reference='full', comparison=condition, metric=name,
                    full_minus_comparison=reference-value if reference is not None and value is not None else None,
                    ci95_low=float(low) if low is not None else None,
                    ci95_high=float(high) if high is not None else None,
                    desirable_direction='negative' if name in {'center_error_m', 'duplicate_fraction'} else 'positive',
                    paired_log_draws=True))
    return summary, paired


def write_csv(name, rows):
    with (OUT/name).open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_retrieval(records, methods, logs, draws):
    output = []
    for cohort in ['distribution_preserving', 'challenge']:
        for name, positive, field in [('task_correct', None, 'correct'),
                ('positive_target_correct', True, 'correct'),
                ('no_match_false_selection', False, 'no_match_false_selection')]:
            selected = [r for r in records if r['cohort'] == cohort
                        and (positive is None or r['positive'] == positive)]
            indexed = {m: {r['query_id']: r for r in selected if r['method'] == m} for m in methods}
            base = indexed['base']
            denominator = len(base)
            assert denominator > 0
            base_counts = np.asarray([[sum(r[field] for r in base.values() if r['log'] == log),
                sum(r['log'] == log for r in base.values())] for log in logs])
            sampled_base = base_counts[draws].sum(1)
            for method in methods:
                if method == 'base':
                    continue
                rows = indexed[method]
                assert rows.keys() == base.keys()
                assert all(rows[q]['log'] == base[q]['log'] for q in rows)
                counts = np.asarray([[sum(r[field] for r in rows.values() if r['log'] == log),
                    sum(r['log'] == log for r in rows.values())] for log in logs])
                sampled = counts[draws].sum(1)
                assert np.array_equal(sampled[:, 1], sampled_base[:, 1])
                delta = np.divide(sampled[:, 0]-sampled_base[:, 0], sampled[:, 1],
                    out=np.full(len(draws), np.nan), where=sampled[:, 1] > 0)
                finite = delta[np.isfinite(delta)]
                low, high = np.percentile(finite, [2.5, 97.5])
                output.append(dict(cohort=cohort, method=method, reference='base', metric=name,
                    method_minus_base=float((counts[:, 0].sum()-base_counts[:, 0].sum())/denominator),
                    ci95_low=float(low), ci95_high=float(high), denominator_per_method=denominator,
                    desirable_direction='negative' if name == 'no_match_false_selection' else 'positive',
                    paired_log_draws=True, valid_bootstrap_draws=len(finite)))
    return output


def main():
    contract = json.loads((OUT/'figure_contract.json').read_text())
    assert contract['backend'] == 'Python, matplotlib only'
    release = json.loads((INPUT/'release.json').read_text())
    evaluation = INPUT/'evaluation'
    report = json.loads((evaluation/'report.json').read_text())
    assert report['conditions'] == 10 and report['queries_each'] == 878
    assert report['actual_query_method_results'] == 12292
    assert report['actual_lock_credit'] is False and report['actual_arrival_credit'] is False
    sources = {str(OUT/'figure_contract.json'): sha(OUT/'figure_contract.json'),
               str(INPUT/'release.json'): sha(INPUT/'release.json'),
               str(evaluation/'report.json'): sha(evaluation/'report.json'),
               str(Path(__file__)): sha(__file__)}
    for name, expected in report['files_sha256'].items():
        path = evaluation/name
        assert sha(path) == expected
        sources[str(path)] = expected
    model_reports = []
    for variant in release['variants']:
        path = INPUT/'models'/variant/'report.json'
        item = json.loads(path.read_text())
        assert item['requests'] == 878 and item['gt_received'] is False
        sources[str(path)] = sha(path)
        model_reports.append(item)
    rows = json.loads((evaluation/'mapping_summary.json').read_text())
    assert len(rows) == 300
    logs = sorted({r['log'] for r in rows})
    assert len(logs) == 14
    draws = np.random.default_rng(17).integers(0, len(logs), (4000, len(logs)))
    mapping, delta = summarize_mapping(rows, release['conditions'], logs, draws)
    retrieval = json.loads((evaluation/'retrieval_summary.json').read_text())
    assert len(retrieval) == 84
    methods = release['retrieval_methods']
    assert len(methods) == 14
    # 保留全部原始主指标，不重新筛选任务或用总正确率代替正目标。
    write_csv('mapping_metrics.csv', mapping)
    write_csv('mapping_paired_differences.csv', delta)
    write_csv('retrieval_all_metrics.csv', retrieval)
    with gzip.open(evaluation/'query_metrics.jsonl.gz', 'rt') as f:
        decisions = list(map(json.loads, f))
    assert len(decisions) == 12292
    retrieval_delta = paired_retrieval(decisions, methods, logs, draws)
    assert len(retrieval_delta) == 78
    write_csv('retrieval_paired_differences.csv', retrieval_delta)
    seed_summary = []
    for cohort in ['distribution_preserving', 'challenge']:
        for name in ['task_correct', 'positive_target_correct', 'no_match_false_selection']:
            per_seed = [r for r in retrieval if r['cohort'] == cohort and r['metric'] == name
                        and r['method'].startswith('real_graph_seed')]
            assert len(per_seed) == 3 and len({r['denominator'] for r in per_seed}) == 1
            values = np.asarray([r['value'] for r in per_seed])
            seed_summary.append(dict(cohort=cohort, metric=name, seeds=3,
                denominator_per_seed=per_seed[0]['denominator'],
                mean_over_three_seeds=float(values.mean()), sample_std_over_three_seeds=float(values.std(ddof=1)),
                minimum_seed_value=float(values.min()), maximum_seed_value=float(values.max()),
                seeds_not_independent_scene_samples=True))
    write_csv('adapter_three_seed_summary.csv', seed_summary)
    model_stats = []
    for variant, item in zip(release['variants'], model_reports):
        path = INPUT/'models'/variant/'outputs.jsonl.gz'
        assert sha(path) == item['output_sha256']
        sources[str(path)] = sha(path)
        with gzip.open(path, 'rt') as f:
            data = list(map(json.loads, f))
        latencies = [r['GPU_latency_s'] for r in data if r['actual_GPU_call']]
        assert len(latencies) == item['actual_GPU_calls'] and len(data) == 878
        model_stats.append(dict(variant=variant, requests=878, actual_GPU_calls=len(latencies),
            schema_valid=sum(r['schema_valid'] for r in data),
            mean_GPU_latency_s=float(np.mean(latencies)) if latencies else None,
            p50_GPU_latency_s=float(np.percentile(latencies, 50)) if latencies else None,
            p95_GPU_latency_s=float(np.percentile(latencies, 95)) if latencies else None,
            peak_GPU_MiB=item['peak_GPU_MiB'], wall_s=item['wall_s'], load_s=item['load_s'],
            realtime_end_to_end_credit=False))
    write_csv('model_execution_statistics.csv', model_stats)
    from openpyxl import Workbook, load_workbook
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, data in [('建图指标', mapping), ('建图配对差值', delta),
                        ('全部目标选择指标', retrieval), ('目标选择配对差值', retrieval_delta),
                        ('新Adapter三种子', seed_summary), ('模型推理统计', model_stats)]:
        sheet = workbook.create_sheet(title)
        sheet.append(list(data[0]))
        for row in data:
            sheet.append(list(row.values()))
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(42, max(16,
                max(len(str(c.value or '')) for c in column)+2))
    workbook.save(OUT/'封存公开数据结果表.xlsx')
    verification = load_workbook(OUT/'封存公开数据结果表.xlsx', read_only=True)
    assert len(verification.sheetnames) == 6
    verification.close()
    draw_figures(mapping, retrieval, release, sources)


def draw_figures(mapping, retrieval, release, sources):
    import plot_public_evidence_004 as plot
    plot.O = OUT
    plot.SOURCES = dict(sources)
    plot.EXPORTS = []
    plot.QA = []
    plt, np = plot.plt, plot.np
    plots = [dict(name='mapping_sealed', group=release['conditions'],
        panels=[('object_recall', '对象召回率 ↑'), ('IDF1', '跨帧身份 IDF1 ↑'),
                ('center_error_m', '中心误差（米）↓'), ('relation_f1', '关系 F1 ↑')], cohort=None)]
    for cohort, title in [('distribution_preserving', '模板主集'), ('challenge', '模板挑战集')]:
        plots.append(dict(name='retrieval_'+cohort, group=release['retrieval_methods'],
            panels=[('positive_target_correct', title+'：正目标选对率 ↑'),
                    ('no_match_false_selection', title+'：无匹配误选率 ↓')], cohort=cohort))
    for spec in plots:
        count = len(spec['panels'])
        fig, axes = plt.subplots(2 if count == 4 else 1, 2,
            figsize=(183/25.4, 155/25.4))
        axes = np.asarray(axes).ravel()
        y = np.arange(len(spec['group']))
        for i, ((key, title), ax) in enumerate(zip(spec['panels'], axes)):
            data = [next(r for r in (mapping if spec['cohort'] is None else retrieval)
                if (r['condition'] if spec['cohort'] is None else r['method']) == method
                and r['metric'] == key and (spec['cohort'] is None or r['cohort'] == spec['cohort']))
                for method in spec['group']]
            values = np.array([np.nan if r['value'] is None else r['value'] for r in data])
            valid = np.isfinite(values)
            colors = ['#507C9E' if m == 'full' or m.startswith('cooperation') else
                      '#758F89' if m.startswith('real_graph') else '#A7ADB3' for m in spec['group']]
            ax.barh(y[valid], values[valid], height=.57, color=np.asarray(colors)[valid])
            for j, row in enumerate(data):
                if row['value'] is not None and row['ci95_low'] is not None:
                    value = row['value']
                    ax.errorbar(value, j, xerr=[[max(0, value-row['ci95_low'])],
                        [max(0, row['ci95_high']-value)]], fmt='none', color='#343A40',
                        linewidth=.65, capsize=1.8)
            cap = max([1.]+[float(r['ci95_high']) for r in data if r['ci95_high'] is not None])*1.15 if key == 'center_error_m' else 1.
            ax.set_xlim(0, cap)
            ax.set_yticks(y, [LABELS.get(m, m) for m in spec['group']])
            ax.invert_yaxis()
            ax.set_title(chr(97+i)+'  '+title, loc='left', fontweight='bold', fontsize=7.5)
            ax.grid(axis='x', color='#E6E8EA', linewidth=.5)
            ax.set_axisbelow(True)
            for j, row in enumerate(data):
                label = 'NA' if row['value'] is None else f"{row['value']:.3f}"
                ax.text(cap*.985, j, label, ha='right', va='center', fontsize=6,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.3))
            if count == 4:
                ax.axhline(5.5, color='#8B939B', linewidth=.5, linestyle='--')
        foot = ('30个封存场景 / 14个日志，每配置1196帧；虚线下方为架构诊断，非单模块消融。' if count == 4 else
                '30个封存场景 / 14个日志；全部十四方法固定顺序，三种子分别显示。')
        if spec['cohort']:
            nums = {r['metric']: r['denominator'] for r in retrieval
                    if r['cohort'] == spec['cohort'] and r['method'] == release['retrieval_methods'][0]}
            foot += f"\n正目标分母{nums['positive_target_correct']}，无匹配分母{nums['no_match_false_selection']}。模板查询，不是人工自然指令。"
        foot += '\n误差线为日志聚类95%区间，4000次重采样；仅离线评价，无实际锁定或到达信用。'
        fig.tight_layout(rect=(0, .12, 1, .99), h_pad=2., w_pad=1.3)
        fig.text(.03, .025, foot, va='bottom', fontsize=6)
        plot.export(fig, spec['name'])
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), figures=plot.QA,
        exports=plot.EXPORTS, sources_sha256=plot.SOURCES,
        CSV_rows=dict(mapping=60, mapping_paired=54, retrieval=84,
                      retrieval_paired=78, adapter_three_seed=6, models=5),
        source_files_still_match=all(sha(p) == h for p, h in sources.items()),
        visual_inspection_pending=True, goal_complete=False)
    assert receipt['source_files_still_match']
    with (OUT/'automated_qa.json').open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
