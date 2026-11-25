"""未知动态节点过期修复的五方法、七比较验证汇总；不覆盖旧三比较。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.summarize_association_refinement import (  # noqa: E402
    CLASSES, DEFAULT_RUNS, REPORT_ROOT, RUN_ROOT, describe_predictions, exact_numbers,
    read, sha, threshold_summary, validate_scope, write, write_csv,
)
from conceptgraph.completion.config_comparison import ratios, summarize  # noqa: E402
from conceptgraph.completion.metric_acceptance import paired_metric_uncertainty  # noqa: E402
from conceptgraph.completion.quality import object_frame_metrics  # noqa: E402

RUNS = dict(DEFAULT_RUNS,
    R3_unknown_ttl='association_R3_unknown_ttl_full_20261123_v1',
    R1_unknown_ttl='association_R1_unknown_ttl_full_20261123_v1')
FAMILY = [
    ['R1', 'R3'], ['R1', 'R3_single_stage'], ['R3_single_stage', 'R3'],
    ['R1', 'R3_unknown_ttl'], ['R3', 'R3_unknown_ttl'],
    ['R1', 'R1_unknown_ttl'], ['R1_unknown_ttl', 'R3_unknown_ttl'],
]
ADAPTIVE_CAVEAT = ('未知节点TTL修复是在看到R3失败后追加，属于自适应验证分析；'
    '统一七比较Bonferroni校正不能消除方法选择偏差，不能称新的独立测试。')


def validate_plan(plan):
    expected = dict(registered_comparisons=FAMILY, primary_metric='IDF1',
                    bootstrap_seed=20261123, bootstrap_replicates=10000, adaptive_selection=True,
                    comparisons_registered=7, ttl_s=2., ttl_parameter_changed=False,
                    confirmation_changed=False, default_changed=False, threshold_m=3.,
                    test_accessed=False, ground_truth_in_inference=False)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('追加计划必须声明七比较、固定主指标/种子及自适应选择')


def family_intervals(baseline, candidate, *, seed=20261123, replicates=10000):
    """重用原日志抽样口径；仅在新报告统一采用实际七项比较的分位数。"""
    original = paired_metric_uncertainty(baseline, candidate, seed=seed, replicates=replicates)
    logs = original['logs']
    keys = ('true_positive', 'truth', 'predictions', 'IDTP', 'IDFP', 'IDFN',
            'duplicate_nodes', 'center_error_sum_m')
    draws = np.random.default_rng(seed).integers(0, len(logs), (replicates, len(logs)))

    def sampled(rows):
        grouped = np.array([[sum(r['result']['counts'][k] for r in rows if r['log'] == log)
                            for k in keys] for log in logs], dtype=float)
        c = grouped[draws].sum(axis=1)
        return dict(recall=100*c[:, 0]/np.maximum(c[:, 1], 1),
            precision=100*c[:, 0]/np.maximum(c[:, 2], 1),
            F1=200*c[:, 0]/np.maximum(c[:, 1]+c[:, 2], 1),
            IDF1=200*c[:, 3]/np.maximum(2*c[:, 3]+c[:, 4]+c[:, 5], 1),
            duplicate_nodes=c[:, 6], center_error_m=np.divide(c[:, 7], c[:, 0],
                out=np.full(replicates, np.nan), where=c[:, 0] > 0))

    left, right = sampled(baseline), sampled(candidate)
    intervals = {}
    for metric in left:
        delta = right[metric]-left[metric]
        undefined = int(np.count_nonzero(~np.isfinite(delta)))
        intervals[metric] = dict(units=original['intervals'][metric]['units'],
            undefined_replicates=undefined,
            ci95=np.quantile(delta, [.025, .975]).tolist() if not undefined else None,
            ci_bonferroni7=np.quantile(delta, [.05/7/2, 1-.05/7/2]).tolist() if not undefined else None)
        if not undefined:
            np.testing.assert_allclose(intervals[metric]['ci95'], original['intervals'][metric]['ci95'], atol=1e-10)
    return dict(logs=logs, seed=seed, replicates=replicates, comparisons_in_family=7,
        primary_metric='IDF1', corrected_confidence_level=1-.05/7, intervals=intervals,
        adaptive_selection=True, selection_caveat=ADAPTIVE_CAVEAT,
        supplementary_interpretation='附指标为描述性区间，不构成六指标联合家族错误率控制；中心误差匹配样本组成不同。')


def load_checked_run(path, method, freeze, guard):
    report, release = read(path/'report.json'), read(path/'release.json')
    scope = validate_scope(report, release, method)
    for filename in ('report.json', 'release.json', 'input_fingerprints.json'):
        freeze(path/filename)
    for name in ('quality.py', 'association.py', 'config_comparison.py', 'temporal_identity_evaluation.py'):
        current = PROJECT/'conceptgraph/completion'/name
        frozen = path/'source_snapshot'/current.relative_to(PROJECT)
        freeze(frozen, release['sources_sha256'][str(current)])
        if sha(current) != sha(frozen):
            raise ValueError('当前匹配/身份实现与冻结源码不同：'+str(current))
    for source, digest in release['sources_sha256'].items():
        freeze(path/'source_snapshot'/Path(source).relative_to(PROJECT), digest)
    frames, scenes = {}, []
    for scene, log in scope:
        guard()
        saved = read(path/scene/'report.json')
        freeze(path/scene/'report.json')
        freeze(path/scene/'frames.jsonl.gz', saved['records_sha256'])
        with gzip.open(path/scene/'frames.jsonl.gz', 'rt') as stream:
            rows = list(map(json.loads, stream))
        if (len(rows) != saved['frames'] or len({r['frame_idx'] for r in rows}) != len(rows)
                or saved['scene'] != scene or saved['log'] != log):
            raise ValueError('场景/帧数/日志范围错误')
        for row in rows:
            if row['gt_in_inference'] is not False or row['scene'] != scene:
                raise ValueError('保存的推理隔离标志或场景错误')
            predictions, truth = row['predictions'][method], row['evaluation_only_official']
            if any(obj['category'] not in CLASSES for obj in predictions+truth):
                raise ValueError('原六类之外对象不能擅自过滤')
            if object_frame_metrics(predictions, truth, threshold_m=3.) != row['metrics'][method]:
                raise ValueError(f'原3米逐帧匹配未复现：{method}/{scene}/{row["frame_idx"]}')
        recomputed = summarize(rows, method)
        for field in ('counts', 'metrics'):
            exact_numbers(recomputed[field], saved['result'][field], f'{method}/{scene}/{field}')
        if not math.isclose(sum(r['mapping_s'] for r in rows), saved['mapping_s'], abs_tol=1e-8):
            raise ValueError('逐场景建图耗时不一致')
        frames[scene] = rows
        scenes.append(saved)
    counts = {k: sum(row['result']['counts'][k] for row in scenes) for k in scenes[0]['result']['counts']}
    exact_numbers(counts, report['result']['counts'], method+'/完整计数')
    exact_numbers(ratios(counts), report['result']['metrics'], method+'/完整指标')
    if sum(row['frames'] for row in scenes) != 1508:
        raise ValueError('完整帧范围错误')
    return report, release, scenes, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=REPORT_ROOT/'unknown_expiry_plan.json')
    parser.add_argument('--output', type=Path, default=REPORT_ROOT/'unknown_expiry_comparison_v1')
    parser.add_argument('--wall-budget-s', type=float, default=900)
    args = parser.parse_args()
    begun = time.monotonic()
    plan = read(args.plan)
    validate_plan(plan)
    # 部分运行不创建统计目录，也不进入bootstrap。
    scope = fingerprints_ref = release_ref = None
    for method, run in RUNS.items():
        path = RUN_ROOT/run
        if not (path/'report.json').is_file() or (path/'interrupted.json').exists():
            raise ValueError(f'{method}尚未完整结束：{path}')
        report, release = read(path/'report.json'), read(path/'release.json')
        current_scope = validate_scope(report, release, method)
        if method.endswith('_unknown_ttl'):
            expected = dict(comparisons_registered=7, bootstrap_seed=20261123,
                paired_bootstrap_replicates=10000, primary_metric='IDF1',
                comparison_family='association_with_adaptive_unknown_expiry_20261123',
                adaptive_after_R3_failure=True)
            if any(release.get(k) != v for k, v in expected.items()):
                raise ValueError('新修复候选release未对应追加七比较计划')
            settings = release['algorithm_settings']
            if (settings.get('unknown_potential_dynamic_expiry') is not True
                    or settings.get('original_dynamic_ttl_s') != 2.):
                raise ValueError('修复开关或原TTL数值不对应')
        current_inputs = read(path/'input_fingerprints.json')
        if len(current_inputs) != 4600:
            raise ValueError('4600条输入清单不完整')
        if scope is None:
            scope, fingerprints_ref, release_ref = current_scope, current_inputs, release
        elif current_scope != scope or current_inputs != fingerprints_ref:
            raise ValueError('五方法场景、日志或输入清单不一致')
        for key in ('inputs_sha256', 'seed', 'threshold_m', 'camera_shape_wh', 'config'):
            if release[key] != release_ref[key]:
                raise ValueError('原评价、寿命数值或基础配置不同：'+key)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT/'outputs/reports'):
        raise ValueError('报告必须位于outputs/reports')
    if shutil.disk_usage(PROJECT).free < 20*1024**3:
        raise OSError('磁盘低于20GiB')
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def freeze(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('原始来源SHA不一致：'+str(path))
        sources[str(Path(path).resolve())] = digest
        return digest

    def guard():
        if time.monotonic()-begun > args.wall_budget_s:
            raise TimeoutError('达到汇总墙钟预算')
        if shutil.disk_usage(PROJECT).free < 20*1024**3:
            raise OSError('低于20GiB停止新增文件')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 63*1024**2:
            raise OSError('达到64MiB汇总存储保护线')

    try:
        freeze(args.plan)
        freeze(REPORT_ROOT/'start_manifest.json')
        freeze(REPORT_ROOT/'comparison_v1/comparison.json')
        source_paths = [Path(__file__).resolve(), PROJECT/'scripts/summarize_association_refinement.py']
        source_paths += [PROJECT/'conceptgraph/completion'/name for name in
            ('quality.py', 'association.py', 'config_comparison.py', 'metric_acceptance.py',
             'temporal_identity_evaluation.py')]
        for source in source_paths:
            freeze(source)
            target = output/'source_snapshot'/source.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        reports, releases, scene_reports, data = {}, {}, {}, {}
        for method, run in RUNS.items():
            reports[method], releases[method], scene_reports[method], data[method] = load_checked_run(
                RUN_ROOT/run, method, freeze, guard)
        groups, sensitivity, audit_counts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        changes = {f'{right}-{left}': Counter() for left, right in FAMILY}
        gain_loss = {key: [] for key in changes}
        for scene, _ in scope:
            guard()
            paths = [Path(p) for p in releases['R1']['inputs_sha256']
                     if Path(p).name == 'frames.jsonl' and Path(p).parent.name == scene]
            if len(paths) != 1:
                raise ValueError('原位姿元数据不唯一')
            freeze(paths[0], releases['R1']['inputs_sha256'][str(paths[0])])
            with paths[0].open() as stream:
                originals = list(map(json.loads, stream))
            metadata = {row['frame_idx']: row for row in originals}
            if len(metadata) != len(originals) or len(originals) != len(data['R1'][scene]):
                raise ValueError('元数据帧数不同')
            for grouped in zip(*(data[m][scene] for m in RUNS), strict=True):
                rows = dict(zip(RUNS, grouped, strict=True))
                base = rows['R1']
                idx, truth = base['frame_idx'], base['evaluation_only_official']
                for row in rows.values():
                    for key in ('scene', 'frame_idx', 'camera_timestamp_s', 'evaluation_only_official', 'input'):
                        if row[key] != base[key]:
                            raise ValueError(f'五方法GT/输入不一致：{scene}/{idx}/{key}')
                if truth != metadata[idx]['evaluation_only_official']:
                    raise ValueError('GT与原元数据不同')
                matched = {}
                for method, row in rows.items():
                    matched[method] = {x['truth_index']: x for x in row['metrics'][method]['matches']}
                    audit_counts[method].update(e['stage']+':'+e['reason'] for e in row['audit'])
                    for threshold in (1., 2.):
                        values = object_frame_metrics(row['predictions'][method], truth, threshold_m=threshold)
                        sensitivity[method, threshold].update({k: values[k] for k in
                            ('true_positive', 'truth', 'predictions')})
                pose = np.asarray(metadata[idx]['camera_pose'])
                for index, obj in enumerate(truth):
                    distance = float(np.linalg.norm(np.asarray(obj['center'][:2])-pose[:2, 3]))
                    bucket = '0_20m' if distance < 20 else '20_40m' if distance < 40 else '40m_plus'
                    for method in RUNS:
                        for group in ('category:'+obj['category'], 'range:'+bucket):
                            groups[method, group]['truth'] += 1
                            groups[method, group]['matched'] += index in matched[method]
                    for left, right in FAMILY:
                        a, b = index in matched[left], index in matched[right]
                        kind = 'both' if a and b else 'gain' if b else 'loss' if a else 'neither'
                        key = f'{right}-{left}'
                        changes[key][kind] += 1
                        if kind in {'gain', 'loss'}:
                            gain_loss[key].append(dict(kind=kind, scene=scene, frame_idx=idx,
                                truth_index=index, category=obj['category'], instance=obj['instance_token'],
                                range_m=distance, saved_matches={m: matched[m].get(index) for m in (left, right)},
                                source_records={m: str(RUN_ROOT/RUNS[m]/scene/'frames.jsonl.gz') for m in (left, right)}))
        table = []
        diagnostics = {}
        for method, report in reports.items():
            counts, metrics = report['result']['counts'], report['result']['metrics']
            table.append(dict(method=method, scenes=38, logs=11, frames=1508,
                TP=counts['true_positive'], GT=counts['truth'], predictions=counts['predictions'],
                FP=counts['predictions']-counts['true_positive'], FN=counts['truth']-counts['true_positive'],
                **metrics, duplicates=counts['duplicate_nodes'], mapping_s=report['mapping_s'],
                wall_s=report['wall_s'], peak_rss_mib=report['peak_rss_mib']))
            diagnostics[method] = describe_predictions(data[method], method)
            if diagnostics[method]['output_observations'] != counts['predictions']:
                raise ValueError('输出诊断分母不一致')
        comparisons = {}
        for left, right in FAMILY:
            stats = family_intervals(scene_reports[left], scene_reports[right],
                seed=plan['bootstrap_seed'], replicates=plan['bootstrap_replicates'])
            a, b = reports[left]['result'], reports[right]['result']
            deltas = {k: 100*(b['metrics'][k]-a['metrics'][k]) for k in ('recall', 'precision', 'F1', 'IDF1')}
            ci = stats['intervals']['IDF1']['ci_bonferroni7']
            comparisons[f'{right}-{left}'] = dict(baseline=left, candidate=right, statistics=stats,
                delta_percentage_points=deltas, corrected_primary_IDF1_positive=ci is not None and ci[0] > 0,
                point_metrics_non_decreasing=all(v >= 0 for v in deltas.values()),
                duplicates_non_increasing=b['counts']['duplicate_nodes'] <= a['counts']['duplicate_nodes'],
                duplicate_delta=b['counts']['duplicate_nodes']-a['counts']['duplicate_nodes'],
                center_error_delta_m=b['metrics']['center_error_m']-a['metrics']['center_error_m'],
                paired_observation_changes=dict(changes[f'{right}-{left}']),
                automatic_adoption=False)
        group_rows = [dict(method=m, group=g, matched=c['matched'], truth=c['truth'],
                          recall=c['matched']/c['truth']) for (m, g), c in sorted(groups.items())]
        for method in RUNS:
            for prefix in ('category:', 'range:'):
                if sum(r['truth'] for r in group_rows if r['method'] == method and r['group'].startswith(prefix)) != 5723:
                    raise ValueError('分组未保留完整5723分母')
        supplementary = [dict(method=m, threshold_m=t, **threshold_summary(sensitivity[m, t]))
                         for t in (1., 2.) for m in RUNS]
        write_csv(output/'main_metrics.csv', table)
        write_csv(output/'group_recall.csv', group_rows)
        write_csv(output/'supplementary_1m_2m.csv', supplementary)
        write(output/'prediction_failure_diagnostics.json', diagnostics)
        write(output/'all_gain_loss_cases.json', gain_loss)
        write(output/'audit_event_counts.json', {m: dict(c) for m, c in audit_counts.items()})
        caveats = [ADAPTIVE_CAVEAT, '原三比较comparison_v1与原R1-B1报告不覆写；本报告统一显示实际七比较家族。',
            'IDF1为主，其他指标区间描述性，不构成六指标联合显著；全部退化保留。',
            'unknown为潜在动态未定状态，不代表已证明动态；本次只是显式生命周期策略对照。',
            '原3米为水平XY中心同类一对一匹配，不证明高度、形状或车辆到达。',
            '1米/2米所有方法统一补充评价，不改3米主标准，不从补充表选择参数。',
            '中心误差仅针对各方法已匹配对象，匹配组成不同；历史耗时非随机交错性能基准。',
            '逐帧指标/身份重算通过不等于每组都已另行完成推理确定性复现。',
            'unknown陈旧输出计数为描述性证据，不单独证明全部额外预测的因果来源；真实稀疏表示不按node_id前缀判断。']
        write(output/'comparison.json', dict(primary_metric='IDF1', correction_family_size=7,
            registered_comparisons=FAMILY, adaptive_selection=True, plan=str(args.plan.resolve()),
            scope=dict(scenes=38, logs=11, frames=1508, GT=5723, inputs=4600, main_threshold_m=3),
            runs={m: str(RUN_ROOT/run) for m, run in RUNS.items()}, main_metrics=table,
            comparisons=comparisons, supplementary=supplementary, caveats=caveats,
            original_three_comparison_report=str(REPORT_ROOT/'comparison_v1/comparison.json'),
            default_changed=False, test_accessed=False, new_inference=False))
        lines = ['# 未知动态节点过期修复的追加验证', '', ADAPTIVE_CAVEAT, '',
            '五组全部38验证/调试场景、11日志、1508帧、5723GT及4600输入清单一致；原3米逐帧匹配、场景身份和总计数复现。', '',
            '| 方法 | Precision | Recall | F1 | IDF1 | 重复节点观测 | 匹配中心误差(m) | 建图(s) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in table:
            lines.append(f'| {r["method"]} | {r["precision"]:.2%} | {r["recall"]:.2%} | '
                f'{r["F1"]:.2%} | {r["IDF1"]:.2%} | {r["duplicates"]} | {r["center_error_m"]:.4f} | {r["mapping_s"]:.2f} |')
        lines += ['', '七项实际比较均按11日志配对bootstrap 10000次、种子20261123、Bonferroni7重新报告。', '']
        for key, value in comparisons.items():
            lo, hi = value['statistics']['intervals']['IDF1']['ci_bonferroni7']
            lines.append(f'- {key}：IDF1增量{value["delta_percentage_points"]["IDF1"]:+.3f}个百分点；'
                f'校正区间[{lo:+.3f},{hi:+.3f}]；四项点估计均不降={value["point_metrics_non_decreasing"]}；'
                f'重复节点不增={value["duplicates_non_increasing"]}。')
        lines += ['', '单次观测、unknown、原测量时间戳落后当前超过2秒的输出：', '']
        for method, value in diagnostics.items():
            aged = value['single_unknown_timestamp_over_2s_by_source_and_status']
            total = sum(sum(c.values()) for c in aged.values())
            unmatched = sum(v for c in aged.values() for status, v in c.items() if status != 'matched')
            lines.append(f'- {method}：{total}次，其中未匹配{unmatched}次。')
        lines += ['', '全部类别/距离分母、1/2米补充表、七比较全量得失案例、audit和预测诊断见同目录。', '']
        lines += ['- '+text for text in caveats]
        (output/'未知节点过期修复对照.md').write_text('\n'.join(lines)+'\n')
        guard()
        changed = [path for path, value in sources.items() if sha(path) != value]
        if changed:
            raise ValueError('汇总期间输入或源码变化：'+repr(changed))
        write(output/'source_manifest.json', sources)
        write(output/'complete.json', dict(complete=True, comparisons=7,
            comparison_sha256=sha(output/'comparison.json'), wall_s=time.monotonic()-begun,
            storage_bytes_before_completion=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            default_changed=False, test_accessed=False, new_inference=False))
        print(json.dumps(dict(output=str(output), primary_delta_pp={
            k: v['delta_percentage_points']['IDF1'] for k, v in comparisons.items()}), ensure_ascii=False), flush=True)
    except BaseException as exc:
        write(output/'interrupted.json', dict(complete=False, error=repr(exc),
            traceback=traceback.format_exc(), wall_s=time.monotonic()-begun))
        raise


if __name__ == '__main__':
    main()
