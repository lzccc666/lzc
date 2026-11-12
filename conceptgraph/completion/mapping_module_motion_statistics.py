"""动态模块消融的官方运动属性分组；GT只在全部建图结束后进入评价。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wait-budget-s', type=float, default=0.)
    args = parser.parse_args()
    source = OUTPUT/'ablations'/args.input_run
    begun = time.monotonic()
    source_sha = digest(__file__)
    while not (source/'summary.json').exists():
        if (source/'interrupted.json').exists():
            raise RuntimeError('mapping ablation interrupted')
        if time.monotonic()-begun > args.wait_budget_s:
            raise TimeoutError('full mapping module experiment not ready')
        time.sleep(5)
    if digest(__file__) != source_sha:
        raise ValueError('motion statistics source changed while waiting')
    label_path = OUTPUT/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
    labels = {(r['scene'], r['frame_idx']): r['objects'] for r in map(json.loads, label_path.open())}
    release = json.loads((source/'release.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if digest(source/'cases.jsonl') != summary['records_sha256']:
        raise ValueError('mapping records changed')
    for filename, sha in release['source_frames_sha256'].items():
        path = Path(filename)
        if digest(path) != sha:
            raise ValueError('original evaluation frame changed')
        for frame in map(json.loads, path.open()):
            official = labels[(frame['scene'], frame['frame_idx'])]
            if [o['annotation_token'] for o in official] != [o['annotation_token'] for o in frame['evaluation_only_official']]:
                raise ValueError('motion annotations have different identities or ordering')
    cases = [json.loads(s) for s in (source/'cases.jsonl').open()]
    expected = {(s, c) for s in release['scene_logs'] for c in release['conditions']}
    if len(cases) != len(expected) or {(c['scene'], c['condition']) for c in cases} != expected:
        raise ValueError('all frozen module cases required')
    conditions = list(release['conditions'])
    logs = sorted(set(release['scene_logs'].values()))
    groups = ['static', 'dynamic', 'unknown']
    totals = np.zeros((len(logs), len(conditions), len(groups), 4), dtype=int)
    confusion = defaultdict(Counter)
    raw_sha = {}
    for case in cases:
        path = Path(case['report_path'])
        report = json.loads(path.read_text())
        frames_path = path.parent/'frames.jsonl'
        if digest(path) != case['report_sha256'] or digest(frames_path) != report['records_sha256']:
            raise ValueError('raw module report/frames changed')
        raw_sha[str(frames_path)] = digest(frames_path)
        for frame in map(json.loads, frames_path.open()):
            truth = labels[(case['scene'], frame['frame_idx'])]
            if len(truth) != frame['metrics']['truth']:
                raise ValueError('motion group labels disagree with original evaluation universe')
            matched = {m['truth_index']: frame['predictions'][m['prediction_index']] for m in frame['metrics']['matches']}
            for index, obj in enumerate(truth):
                group = obj['official_motion']
                prediction = matched.get(index)
                state = prediction['motion_state'] if prediction else 'unmatched'
                values = [1, prediction is not None, state == group and group != 'unknown', state == 'unknown']
                totals[logs.index(case['log_token']), conditions.index(case['condition']), groups.index(group)] += values
                confusion[case['condition']][group+'→'+state] += 1
    rng = np.random.default_rng(17)
    samples = totals[rng.integers(0, len(logs), (4000, len(logs)))].sum(axis=1)
    points = totals.sum(axis=0)
    results, paired = {}, []
    for j, condition in enumerate(conditions):
        results[condition] = {}
        for g, group in enumerate(groups):
            result = dict(zip(['truth', 'matched', 'correct_motion', 'unknown_prediction'], points[j, g].tolist()))
            for name, numerator, denominator in [('recall', 1, 0), ('matched_motion_accuracy', 2, 1),
                                                  ('unknown_among_matched', 3, 1)]:
                if name == 'matched_motion_accuracy' and group == 'unknown':
                    result[name] = None
                    continue
                den = samples[:, j, g, denominator]
                rate = np.divide(samples[:, j, g, numerator], den, out=np.full(len(samples), np.nan), where=den > 0)
                valid = rate[np.isfinite(rate)]
                result[name] = dict(estimate=float(points[j, g, numerator]/points[j, g, denominator])
                    if points[j, g, denominator] else None,
                    ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None)
                if j:
                    base_den = samples[:, 0, g, denominator]
                    base = np.divide(samples[:, 0, g, numerator], base_den,
                        out=np.full(len(samples), np.nan), where=base_den > 0)
                    difference = rate-base
                    difference = difference[np.isfinite(difference)]
                    point_difference = (result[name]['estimate']-points[0, g, numerator]/points[0, g, denominator]
                        if result[name]['estimate'] is not None and points[0, g, denominator] else None)
                    paired.append(dict(condition=condition, group=group, metric=name, difference=point_difference,
                        ci95=np.quantile(difference, [.025, .975]).tolist() if len(difference) else None))
            results[condition][group] = result
    destination = OUTPUT/'statistics'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'official_motion', 'truth', 'matched', 'correct_motion', 'unknown_prediction'])
        for i, log in enumerate(logs):
            for j, condition in enumerate(conditions):
                for g, group in enumerate(groups):
                    writer.writerow([log, condition, group, *totals[i, j, g].tolist()])
    dump(destination/'report.json', dict(source_release_sha256=digest(source/'release.json'),
        source_cases_sha256=digest(source/'cases.jsonl'), labels_sha256=digest(label_path),
        raw_frames_sha256=raw_sha, statistics_source_sha256=source_sha, conditions=results,
        paired_differences=paired, confusion={c: dict(v) for c, v in confusion.items()},
        scenes=len(release['scene_logs']), logs=len(logs), bootstrap=dict(unit='采集log', paired=True, seed=17, repetitions=4000),
        scope='官方属性分组只供推理结束后的独立评估；unknown不自动当静止，运动准确率仅条件于已匹配对象',
        independent_test=False, arrival_evaluated=False, goal_complete=False))


if __name__ == '__main__':
    main()
