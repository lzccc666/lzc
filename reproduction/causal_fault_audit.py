"""读取全部实际扰动帧，以独立代价矩阵重算3m同类XY关联和log统计；不跑模型。"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

out = Path(__file__).resolve().parents[1]
source = out/'faults/causal_fault_full_001'
destination = source/'audit_001'
destination.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
release, summary = [load(source/name) for name in ['release.json', 'summary.json']]
statistics = load(out/'statistics/causal_fault_full_001/report.json')
conditions = sorted(release['conditions'])
logs = sorted(set(release['scene_logs'].values()))
frames = {}
files = {}
for name, digest in release['source_frames'].items():
    assert sha(name) == digest
    rows = [json.loads(line) for line in Path(name).read_text().splitlines()]
    frames[rows[0]['scene']] = rows
    files[name] = digest
expected = {(scene, seed, condition) for scene in release['scene_logs']
    for seed in release['seeds'] for condition in conditions}
observed = set()
counts = np.zeros((len(logs), len(conditions), 7))
frame_count = truth_count = predicted_count = 0
for path in sorted(source.glob('scene-*/seed*/*/report.json')):
    case = load(path)
    scene, seed, condition = (case[k] for k in ['scene', 'seed', 'condition'])
    key = scene, seed, condition
    assert key in expected and key not in observed
    observed.add(key)
    raw = path.parent/'frames.jsonl'
    assert sha(raw) == case['records_sha256']
    assert statistics['case_report_hashes'][str(path.relative_to(source))] == sha(path)
    files[str(path)] = sha(path)
    files[str(raw)] = sha(raw)
    actual = [json.loads(line) for line in raw.read_text().splitlines()]
    originals = frames[scene]
    assert len(actual) == len(originals) == case['frames']
    totals = defaultdict(int)
    error_sum = 0.
    last = {}
    node_instances = defaultdict(set)
    for row, original in zip(actual, originals):
        assert row['scene'] == scene and row['seed'] == seed and row['condition'] == condition
        assert row['frame_idx'] == original['frame_idx']
        assert row['camera_timestamp_s'] == original['camera_timestamp_s']
        assert row['gt_sent_to_mapper'] is False
        predictions, truth = row['predictions'], original['evaluation_only_official']
        m, n = len(predictions), len(truth)
        distances = np.full((m, n), np.inf)
        for i, pred in enumerate(predictions):
            for j, target in enumerate(truth):
                if pred['category'] == target['category']:
                    distances[i, j] = np.hypot(pred['center'][0]-target['center'][0], pred['center'][1]-target['center'][1])
        allowed = distances <= 3.
        # 每多一条合法匹配的奖励大于所有合法边距离之和，先最大匹配数再最小距离。
        costs = np.zeros((m, n+m))
        costs[:, :n] = np.where(allowed, distances-4*(m+1), 1e9)
        pairs = []
        if m and n:
            ii, jj = linear_sum_assignment(costs)
            pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if j < n and allowed[i, j]]
        saved = row['metrics']
        assert set(pairs) == {(p['prediction_index'], p['truth_index']) for p in saved['matches']}
        matched = {i for i, _ in pairs}
        values = dict(predictions=m, truth=n, true_positive=len(pairs),
            unmatched_predictions=m-len(pairs), false_negative=n-len(pairs),
            duplicate_nodes=sum(bool(allowed[i].any()) for i in range(m) if i not in matched))
        for k, v in values.items():
            assert saved[k] == v, (path, row['frame_idx'], k)
            totals[k] += v
        errors = [distances[i, j] for i, j in pairs]
        assert (saved['center_error_m_mean'] is None if not errors else
            np.isclose(saved['center_error_m_mean'], np.mean(errors), atol=1e-12, rtol=1e-12))
        error_sum += sum(errors)
        for i, j in pairs:
            node, instance = predictions[i]['node_id'], truth[j]['instance_token']
            if instance in last and last[instance][0] == row['frame_idx']-1:
                totals['adjacent_matched_transitions'] += 1
                totals['identity_switches'] += last[instance][1] != node
            last[instance] = row['frame_idx'], node
            node_instances[node].add(instance)
        frame_count += 1
        truth_count += n
        predicted_count += m
    totals['nodes_associated_with_multiple_instances'] = sum(len(v)>1 for v in node_instances.values())
    for k, v in totals.items():
        assert case['metrics'][k] == v, (path, k)
    tp = totals['true_positive']
    assert np.isclose(case['metrics']['center_error_m_mean'], error_sum/tp) if tp else case['metrics']['center_error_m_mean'] is None
    counts[logs.index(release['scene_logs'][scene]), conditions.index(condition)] += [tp, totals['truth'],
        totals['predictions'], error_sum, totals['identity_switches'],
        totals['adjacent_matched_transitions'], totals['duplicate_nodes']]
assert observed == expected and summary['cases'] == len(expected) == 684
assert frame_count == 1508*3*6
ratio_columns = {'recall':(0,1), 'precision':(0,2), 'center_error_m_mean':(3,0),
    'identity_switch_rate':(4,5), 'duplicates_per_gt_observation':(6,1)}
def ratios(matrix):
    return np.stack([np.divide(matrix[...,i], matrix[...,j],
        out=np.full(matrix.shape[:-1], np.nan), where=matrix[...,j]>0)
        for i,j in ratio_columns.values()], axis=-1)
points = ratios(counts.sum(axis=0))
draws = np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
samples = ratios(counts[draws].sum(axis=1))
for j, condition in enumerate(conditions):
    for k, metric in enumerate(ratio_columns):
        stat = statistics['conditions'][condition][metric]
        assert np.isclose(points[j,k], stat['estimate'], atol=1e-12)
        assert np.allclose(np.quantile(samples[:,j,k],[.025,.975]), stat['ci95'], atol=1e-12)
for stat in statistics['paired_differences']:
    j, k = conditions.index(stat['condition']), list(ratio_columns).index(stat['metric'])
    base = conditions.index('unperturbed')
    assert np.isclose(points[j,k]-points[base,k],stat['difference'],atol=1e-12)
    assert np.allclose(np.quantile(samples[:,j,k]-samples[:,base,k],[.025,.975]),stat['ci95'],atol=1e-12)
with (destination/'recomputed_counts_by_log.csv').open('x') as handle:
    w=csv.writer(handle);w.writerow(['log_token','condition',*statistics['counts_fields']])
    for i, log in enumerate(logs):
        for j, condition in enumerate(conditions):w.writerow([log,condition,*counts[i,j]])
(destination/'files.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
report=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),cases=len(observed),
    actual_frame_condition_records=frame_count,GT_observations_with_seed_and_condition_repetition=truth_count,
    predicted_observations=predicted_count,source_files_checked=len(files),scenes=38,logs=11,
    all_frame_assignments_independently_recomputed=True,all_case_counts_and_ID_transitions_recomputed=True,
    all_30_estimates_and_intervals_recomputed=True,all_25_paired_differences_and_intervals_recomputed=True,
    files_sha256=sha(destination/'files.json'),statistics_sha256=sha(out/'statistics/causal_fault_full_001/report.json'),
    audit_source_sha256=sha(__file__),new_frontend_or_GPU_calls=0,actual_arrivals=0,new_test_opened=False,
    limitations=['独立重算共享SciPy优化器但不调用原质量评价函数，不是人工身份真值',
        '3m同类XY中心代理评价；样本为38场景11log，三种子和六条件不增加独立场景',
        '官方GT较相机晚33–42ms；离线合成扰动，不证明闭环故障恢复或到达安全'],goal_complete=False)
(destination/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# 过去LiDAR录制扰动开发诊断','',
    '38场景、11个采集log、1508帧，3种子×6条件=684次真实CPU重建。全部失败保留；不是新封存或闭环故障恢复实验。', '',
    '|条件|召回|匹配精确率|匹配中心误差(m)|相对未扰动召回差(pp)及95%CI|','|---|---:|---:|---:|---|']
for condition in conditions:
    s=statistics['conditions'][condition]
    diff=next((v for v in statistics['paired_differences'] if v['condition']==condition and v['metric']=='recall'),None)
    delta=f"{100*diff['difference']:.3f} [{100*diff['ci95'][0]:.3f}, {100*diff['ci95'][1]:.3f}]" if diff else '参考'
    lines.append(f"|{condition}|{100*s['recall']['estimate']:.3f}%|{100*s['precision']['estimate']:.3f}%|{s['center_error_m_mean']['estimate']:.4f}|{delta}|")
lines += ['', '按11个log配对bootstrap 4000次、seed17。中心误差和精确率仍条件于代理匹配；LiDAR丢失降低召回不能解释为提高了安全性。位姿扰动的中心误差差+0.1887m，CI[0.1687,0.2332]m；噪声为预设白噪声/随机缺失，未标定真实传感器误差分布。', '',
    f'全部{frame_count}条帧条件记录、逐帧关联、684汇总及30项估计/25项差异的区间已独立重算。未知与官方GT时间近似保留，不计车辆路线、正确锁定或实际到达。', '', 'goal_complete=false。']
(destination/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False))
