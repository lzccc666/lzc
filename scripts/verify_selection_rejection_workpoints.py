"""核验固定拒选工作点并计算同工作点的配对log区间，不新增阈值。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval  # noqa: E402
from conceptgraph.completion.stage9 import available_development  # noqa: E402
from diagnose_selection_validation import dump, jsonl, read, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--report-directory', type=Path, required=True)
    args = parser.parse_args()
    source, replay, destination = args.run, args.replay, args.report_directory
    results = read(source/'report.json')
    replays = read(replay/'report.json')
    for key in ('all_shared_workpoints', 'development_selected_workpoints', 'score_sha256', 'decisions_sha256'):
        assert results[key] == replays[key], key
    score_rows = list(jsonl(source/'inference_only_scores.jsonl'))
    decision_rows = list(jsonl(source/'all_workpoint_decisions.jsonl'))
    assert len(score_rows) == 2230 and len(decision_rows) == 13380
    pool = np.array([row['score'] for row in score_rows if row['score'] is not None])
    frozen = read(source/'thresholds_frozen_before_evaluation.json')
    assert np.array_equal(np.quantile(pool, [.1, .3, .5, .7, .9]), frozen['thresholds'])
    original = list(jsonl(PROJECT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/queries.jsonl'))
    originals = {(r['scene'], r['frame_idx'], r['query_index'], r['method']): r for r in original}
    assert len(originals) == 2230
    scores = {(r['scene'], r['frame_idx'], r['query_index'], r['method']): r for r in score_rows}
    counts = {}
    stats_rows = []
    for row in decision_rows:
        key = (row['scene'], row['frame_idx'], row['query_index'], row['method'])
        initial, score = originals[key], scores[key]
        assert row['positive'] == initial['positive']
        assert score['selected_node_id'] == initial['selected_node_id']
        selected = bool(initial['selected'] and (row['threshold'] is None or score['score'] >= row['threshold']))
        assert row['selected'] == selected
        correct = bool(selected and initial['positive'] and initial['selected_correct'])
        false = bool(selected and not initial['positive'])
        assert bool(row['selected_correct']) == correct
        assert bool(row['false_selection_on_no_match']) == false
        counts.setdefault((row['method'], row['point']), Counter()).update(
            queries=1, positive=int(initial['positive']), negative=int(not initial['positive']),
            correct=int(correct), false_selection=int(false), rejected=int(not selected))
        stats_rows.append(dict(initial, method=row['method']+'::'+row['point'],
            selected_correct=correct, false_selection_on_no_match=false))
    for row in results['all_shared_workpoints']:
        assert all(row[k] == v for k, v in counts[(row['method'], row['point'])].items())
    scenes = {r['scene'] for r in original}
    scene_logs = {r['scene']: r['log_token'] for r in available_development() if r['scene'] in scenes}
    methods, logs, clusters = aggregate(stats_rows, scene_logs)
    totals, estimates, rates = bootstrap(clusters, repetitions=10000, seed=17)
    differences = []
    for point in ['origin', 'q10', 'q30', 'q50', 'q70', 'q90']:
        old = methods.index('old_map_original_rule::'+point)
        new = methods.index('R1_map_original_rule::'+point)
        for i, metric in enumerate(['positive_selection', 'negative_false_selection']):
            differences.append(dict(point=point, metric=metric,
                difference=float(estimates[new, i]-estimates[old, i]),
                **interval(rates[:, new, i]-rates[:, old, i])))
    changed = []
    for directory in (source, replay):
        release = read(directory/'release.json')
        for path, expected in {**release['source_sha256'], **release['input_frames_sha256']}.items():
            # 首次执行源码保存在快照；之后只作CSV语句拆行并在replay完整重验。
            actual = directory/'source_snapshot'/Path(path).relative_to(PROJECT) if path.endswith('.py') else Path(path)
            if sha(actual) != expected:
                changed.append(str(actual))
    protected = read(PROJECT/'outputs/results/qwen/validation/iterations/selection_diagnosis_20261123_074804/start_manifest.json')['protected_sha256']
    changed += [p for p, expected in protected.items() if sha(PROJECT/p) != expected]
    assert not changed, changed
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'map_and_workpoint', 'correct', 'positive', 'false_selection', 'negative'])
        for i, log in enumerate(logs):
            for j, method in enumerate(methods):
                writer.writerow([log, method, *clusters[i, j].tolist()])
    extra_sources = [Path(__file__), PROJECT/'scripts/audit_r1_selection_interface.py',
        PROJECT/'scripts/diagnose_selection_validation.py', PROJECT/'conceptgraph/completion/retrieval_statistics.py']
    for path in extra_sources:
        target = destination/'source_snapshot'/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump(destination/'validation_report.json', dict(
        independent_record_recalculation=True, original_points_reproduced=True,
        fixed_shared_thresholds_reproduced=True, all_denominators_retained=True,
        replay_scores_and_all_13380_decisions_identical=True,
        score_sha256=results['score_sha256'], decisions_sha256=results['decisions_sha256'],
        paired_differences=differences, logs=len(logs), scenes=len(scenes),
        bootstrap=dict(unit='acquisition_log', paired=True, repetitions=10000, seed=17,
            interpretation='六个工作点区间仅作开发诊断；非多重比较校正、非独立测试或泛化保证'),
        changed_protected_inputs=[],
        extra_source_sha256={str(p): sha(p) for p in extra_sources},
        code_style_initial_failure='Ruff E702两处CSV同一行多语句；原执行快照保留，拆行后通过并完整replay',
        verification_first_failure='核验脚本首次有多余右括号，运行前SyntaxError；已修正，失败日志保留'))
    print(json.dumps(dict(validation='passed', logs=len(logs), paired_differences=differences), ensure_ascii=False))


if __name__ == '__main__':
    main()
