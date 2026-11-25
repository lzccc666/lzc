"""完整日志级配对对照和提前停止证据；不改变任何推理输出。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.config_comparison import paired_intervals
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def load_run(path):
    report = json.loads((path / 'report.json').read_text())
    release = json.loads((path / 'release.json').read_text())
    rows = [json.loads((path / s['scene'] / 'report.json').read_text()) for s in release['scenes']]
    if (not report['full_validation_scope'] or report['scenes'] != 38 or report['frames'] != 1508
            or report['logs'] != 11 or report['result']['counts']['truth'] != 5723):
        raise ValueError('部分场景或分母变化不能进行完整效果判断')
    for s in rows:
        if digest(path / s['scene'] / 'frames.jsonl.gz') != s['records_sha256']:
            raise ValueError('逐帧输出指纹变化')
    return report, release, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = PROJECT / 'outputs/results/nuscenes/validation/iterations'
    base_path, new_path = root / args.baseline_run, root / args.candidate_run
    base, base_release, base_scenes = load_run(base_path)
    new, new_release, new_scenes = load_run(new_path)
    if base['condition'] != 'B0' or not base['baseline_counts_reproduced']:
        raise ValueError('必须使用已复现的真实B0基线')
    for item in ['inputs_sha256', 'seed', 'threshold_m', 'camera_shape_wh']:
        if base_release[item] != new_release[item]:
            raise ValueError('输入或评价口径不同：' + item)
    input_hashes = json.loads((base_path / 'input_fingerprints.json').read_text())
    if input_hashes != json.loads((new_path / 'input_fingerprints.json').read_text()):
        raise ValueError('逐帧RGB/深度/检测的输入清单不同')
    if any(digest(Path(p)) != sha for p, sha in input_hashes.items()):
        raise ValueError('原始输入发生变化')
    if any(digest(Path(p)) != sha for p, sha in base_release['protected_sha256'].items()):
        raise ValueError('关键保护文件发生变化')
    ci = paired_intervals(base_scenes, new_scenes)
    b, n = base['result']['metrics'], new['result']['metrics']
    statistical_pass = ci['ci_bonferroni']['recall'][0] > 0
    quality_pass = n['F1'] >= b['F1'] and n['IDF1'] >= b['IDF1']
    groups, reason_counts = defaultdict(Counter), {'B0': Counter(), new['condition']: Counter()}
    diagnostics = Counter()
    examples = {'gains': [], 'losses': []}
    for scene in base_release['scenes']:
        name = scene['scene']
        frame_path = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
        with frame_path.open() as stream:
            poses = {r['frame_idx']: np.asarray(r['camera_pose']) for r in map(json.loads, stream)}
        with gzip.open(base_path / name / 'frames.jsonl.gz', 'rt') as a, gzip.open(new_path / name / 'frames.jsonl.gz', 'rt') as bstream:
            left, right = list(map(json.loads, a)), list(map(json.loads, bstream))
        if len(left) != len(right):
            raise ValueError('配对逐帧范围不同')
        for x, y in zip(left, right):
            if x['frame_idx'] != y['frame_idx'] or x['evaluation_only_official'] != y['evaluation_only_official']:
                raise ValueError('配对真值或帧顺序不同')
            truth = x['evaluation_only_official']
            matched = {}
            for row, method in [(x, 'B0'), (y, new['condition'])]:
                matched[method] = {m['truth_index'] for m in row['metrics'][method]['matches']}
                for event in row['audit']:
                    reason_counts[method][event['stage'] + ':' + event['reason']] += 1
            for i, obj in enumerate(truth):
                distance = float(np.linalg.norm(np.asarray(obj['center'][:2]) - poses[x['frame_idx']][:2, 3]))
                bucket = '0_20m' if distance < 20 else '20_40m' if distance < 40 else '40m_plus'
                for method in ['B0', new['condition']]:
                    for group in ['category:' + obj['category'], 'range:' + bucket]:
                        groups[(method, group)]['truth'] += 1
                        groups[(method, group)]['matched'] += i in matched[method]
                old_ok, new_ok = i in matched['B0'], i in matched[new['condition']]
                diagnostic = 'both' if old_ok and new_ok else 'gain' if new_ok else 'loss' if old_ok else 'neither'
                diagnostics[diagnostic] += 1
                kind = 'gains' if diagnostic == 'gain' else 'losses' if diagnostic == 'loss' else None
                if kind and len(examples[kind]) < 12:
                    examples[kind].append(dict(scene=name, frame_idx=x['frame_idx'],
                        instance=obj['instance_token'], category=obj['category'], distance_m=distance,
                        rgb=x['input']['rgb'], baseline_predictions=x['predictions']['B0'],
                        candidate_predictions=y['predictions'][new['condition']], official_center=obj['center']))
    output = Path(args.output).resolve()
    if not output.is_relative_to(PROJECT / 'outputs/reports'):
        raise ValueError('跨实验报告必须放outputs/reports')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'group_recall.csv').open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'group', 'matched', 'truth', 'recall'])
        for (method, group), c in sorted(groups.items()):
            writer.writerow([method, group, c['matched'], c['truth'], c['matched'] / c['truth']])
    dump(output / 'examples.json', examples)
    dump(output / 'comparison.json', dict(
        baseline=str(base_path), candidate=str(new_path), baseline_report=base, candidate_report=new,
        baseline_report_sha256=digest(base_path / 'report.json'),
        candidate_report_sha256=digest(new_path / 'report.json'),
        statistics=ci, statistical_recall_pass=statistical_pass, point_quality_pass=quality_pass,
        numerical_early_stop_gate=statistical_pass and quality_pass,
        early_stop_gate_caveat='还必须核验相关测试、静态检查和证据完整性后才能最终交付',
        delta_percentage_points={k: 100 * (new['result']['metrics'][k] - base['result']['metrics'][k])
                                 for k in ('recall', 'precision', 'F1', 'IDF1')},
        paired_observation_changes=dict(diagnostics),
        audit_event_counts={k: dict(v) for k, v in reason_counts.items()},
        inputs_equal=True, protected_fingerprints_unchanged=True,
        default_changed=False, test_accessed=False,
        limitation='验证/调试集同输入算法比较，不是独立测试泛化结论或车辆闭环；自动案例非人工审核'))
    print(json.dumps(dict(condition=new['condition'], statistical_pass=statistical_pass,
        quality_pass=quality_pass, delta_recall_ci_corrected=ci['ci_bonferroni']['recall'])), flush=True)


if __name__ == '__main__':
    main()
