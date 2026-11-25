"""只读上一轮真实入口证据，定位B1丢失的对象出现；不参与推理。"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.stage9 import PROJECT, digest, dump


def main():
    root = PROJECT / 'outputs/results/nuscenes/validation/iterations'
    baseline = root / 'config_quality_B0_full_20261120_v1'
    candidate = root / 'config_quality_B1_full_20261120_v1'
    scenes = json.loads((baseline / 'release.json').read_text())['scenes']
    output = PROJECT / 'outputs/reports/recall_refinement_20261120_v2'
    counts, cases = defaultdict(Counter), []
    for scene in scenes:
        name = scene['scene']
        with gzip.open(baseline / name / 'frames.jsonl.gz', 'rt') as a, gzip.open(candidate / name / 'frames.jsonl.gz', 'rt') as b:
            for old, new in zip(map(json.loads, a), map(json.loads, b)):
                old_ok = {m['truth_index'] for m in old['metrics']['B0']['matches']}
                new_ok = {m['truth_index'] for m in new['metrics']['B1']['matches']}
                old_obs = {e['detection_index']: e for e in old['audit']
                           if e['stage'] == 'geometry' and e['reason'] == 'accepted'}
                new_obs = {e['detection_index']: e for e in new['audit']
                           if e['stage'] == 'geometry' and e['reason'] == 'accepted'}
                for i in sorted(old_ok - new_ok):
                    truth = old['evaluation_only_official'][i]
                    category, center = truth['category'], np.asarray(truth['center'])
                    old_near = [e for e in old_obs.values() if e['category'] == category
                                and np.linalg.norm((np.asarray(e['observation_center']) - center)[:2]) <= 3]
                    new_near = [e for e in new_obs.values() if e['category'] == category
                                and np.linalg.norm((np.asarray(e['observation_center']) - center)[:2]) <= 3]
                    reason = 'current_dense_observation_near_but_output_unmatched' if new_near else 'no_current_dense_observation_near'
                    for key in ['all', category]:
                        counts[key]['losses'] += 1
                        counts[key][reason] += 1
                        counts[key]['baseline_current_dense_near'] += bool(old_near)
                    shifts = []
                    for e in old_near:
                        other = new_obs.get(e['detection_index'])
                        if other:
                            displacement = float(np.linalg.norm((np.asarray(other['observation_center']) - e['observation_center'])[:2]))
                            shifts.append(displacement)
                    if shifts and min(shifts) > 1:
                        counts[category]['same_detection_geometry_shift_gt_1m'] += 1
                    if len(cases) < 40:
                        cases.append(dict(scene=name, frame_idx=old['frame_idx'], category=category,
                            instance=truth['instance_token'], reason=reason,
                            baseline_near_observations=old_near, candidate_near_observations=new_near,
                            same_detection_geometry_shifts_m=shifts,
                            candidate_predictions=new['predictions']['B1'], truth_center=truth['center']))
    dump(output / 'regression_diagnosis.json', dict(counts={k: dict(v) for k, v in counts.items()},
        caveat='真实当前稠密观测与已保存输出的诊断，未覆盖稀疏或过去证据，不是因果分解或理论上界',
        baseline=str(baseline), candidate=str(candidate), source_sha256=digest(Path(__file__).resolve())))
    dump(output / 'regression_cases.json', cases)
    print(json.dumps({k: dict(v) for k, v in counts.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
