"""已冻结开发检索的地图派生道路分组，不开启测试或增加模型调用。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from conceptgraph.completion.map_constraint_ablation import load_prior
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump


def classify_road(prior, index, pose):
    point = np.asarray(pose)[:2, 3]
    distances, neighbors = index['tree'].query(point, k=min(64, len(index['lane_ids'])))
    candidates = {}
    for distance, neighbor in zip(np.atleast_1d(distances), np.atleast_1d(neighbors)):
        lane_id = int(index['lane_ids'][neighbor])
        if distance > 3. or lane_id in candidates:
            continue
        lane = prior['lanes'][lane_id]
        if Polygon(lane['polygon']).covers(Point(point)):
            candidates[lane_id] = float(distance)
    if not candidates:
        return dict(group='unknown', reason='no_containing_lane_with_arcline_within_3m', lane_token=None)
    lane_id = min(candidates, key=lambda i: (candidates[i], prior['lanes'][i]['token']))
    lane = prior['lanes'][lane_id]
    line = np.asarray(lane['centerline'])
    nearest = int(np.argmin(np.linalg.norm(line[:, :2]-point, axis=1)))
    arc = np.r_[0., np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1).cumsum()]
    local = line[np.abs(arc-arc[nearest]) <= 10., 2]
    heading_change = float(np.ptp(np.unwrap(local))) if len(local) else 0.
    endpoint = min(float(np.linalg.norm(point-lane['start_pose'][:2])),
                   float(np.linalg.norm(point-lane['end_pose'][:2])))
    branches = len(lane['incoming']) > 1 or len(lane['outgoing']) > 1
    group = ('junction_or_connector' if lane['lane_type'] == 'lane_connector' or branches and endpoint <= 15.
             else 'curved_lane' if heading_change >= math.radians(15.) else 'straight_lane')
    return dict(group=group, lane_token=lane['token'], lane_type=lane['lane_type'],
        nearest_arcline_distance_m=candidates[lane_id], local_arcline_heading_span_deg=math.degrees(heading_change),
        endpoint_distance_m=endpoint, incoming=len(lane['incoming']), outgoing=len(lane['outgoing']))


def count(rows):
    return np.array([sum(bool(r['positive']) and bool(r['selected_correct']) for r in rows),
        sum(bool(r['positive']) for r in rows), sum(not r['positive'] and bool(r['selected']) for r in rows),
        sum(not r['positive'] for r in rows)], dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    begun = time.monotonic()
    source = OUTPUT/'main_experiments/retrieval_baselines_dev_full_001'
    release = json.loads((source/'release.json').read_text())
    queries = [json.loads(s) for s in (source/'queries.jsonl').open()]
    wanted = {(r['scene'], r['frame_idx']) for r in queries}
    known = {r['scene']: r for r in available_development()}
    if not {scene for scene, _ in wanted} <= known.keys():
        raise ValueError('only frozen development observations allowed')
    meta = PROJECT/'dataset/nuscenes_trainval_meta/v1.0-trainval'
    scenes = {r['name']: r for r in json.loads((meta/'scene.json').read_text())}
    logs = {r['token']: r for r in json.loads((meta/'log.json').read_text())}
    locations = {s: logs[scenes[s]['log_token']]['location'] for s, _ in wanted}
    destination = OUTPUT/'performance'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination/'release.json', dict(source_query_sha256=digest(source/'queries.jsonl'),
        source_frames_sha256=release['source_frames_sha256'], source_sha256=digest(__file__),
        prior_builder_sha256=digest(PROJECT/'conceptgraph/completion/map_constraint_ablation.py'),
        map_sha256={loc: digest(PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{loc}.json') for loc in sorted(set(locations.values()))},
        definition=dict(nearest_arcline_m=3., candidate_samples=64, must_contain_pose=True,
            local_arc_window_m=10., curve_heading_span_deg=15., branching_endpoint_m=15.),
        wall_budget_s=300, storage_budget_bytes=20000000, grouping_reads_method_success=False,
        independent_test=False, performance_credit=False, goal_complete=False))
    shutil.copy2(__file__, destination/'generator_source.py')
    priors, indices, groups = {}, {}, {}
    for filename, sha in release['source_frames_sha256'].items():
        if digest(filename) != sha:
            raise ValueError('development observation changed')
        for frame in map(json.loads, Path(filename).open()):
            key = (frame['scene'], frame['frame_idx'])
            if key not in wanted:
                continue
            location = locations[frame['scene']]
            if location not in priors:
                priors[location] = load_prior(location)
                lines = [lane['centerline'][:, :2] for lane in priors[location]['lanes']]
                indices[location] = dict(tree=cKDTree(np.concatenate(lines)),
                    lane_ids=np.concatenate([np.full(len(line), i, dtype=int) for i, line in enumerate(lines)]))
            groups[key] = dict(scene=frame['scene'], frame_idx=frame['frame_idx'], location=location,
                **classify_road(priors[location], indices[location], frame['camera_pose']))
            if time.monotonic()-begun > 300:
                raise TimeoutError('road grouping budget')
    if set(groups) != wanted:
        raise ValueError('missing frozen road observation')
    with (destination/'frames.jsonl').open('x') as handle:
        for key in sorted(groups):
            handle.write(json.dumps(groups[key], ensure_ascii=False, allow_nan=False)+'\n')
    methods = sorted({r['method'] for r in queries})
    grouped = defaultdict(list)
    for row in queries:
        grouped[(row['method'], groups[(row['scene'], row['frame_idx'])]['group'], known[row['scene']]['log_token'])].append(row)
    road_groups = sorted({g['group'] for g in groups.values()})
    all_logs = sorted({known[s]['log_token'] for s, _ in wanted})
    rows, results = [], {}
    for group in road_groups:
        result = {}
        data = np.array([[count(grouped[(method, group, log)]) for method in methods] for log in all_logs])
        rng = np.random.default_rng(17)
        boot = data[rng.integers(0, len(all_logs), (4000, len(all_logs)))].sum(axis=1)
        totals = data.sum(axis=0)
        for j, method in enumerate(methods):
            value = dict(zip(['positive_correct', 'positive_queries', 'negative_false', 'negative_queries'], totals[j].tolist()))
            for name, a, b in [('positive_selection', 0, 1), ('no_match_false_selection', 2, 3)]:
                valid = boot[:, j, b] > 0
                rates = boot[valid, j, a]/boot[valid, j, b]
                value[name] = dict(estimate=float(totals[j, a]/totals[j, b]) if totals[j, b] else None,
                    ci95=np.quantile(rates, [.025, .975]).tolist() if len(rates) else None,
                    contributing_logs=int(np.count_nonzero(data[:, j, b])), valid_resamples=len(rates))
            result[method] = value
            for i, log in enumerate(all_logs):
                rows.append([log, group, method, *data[i, j].tolist()])
        results[group] = result
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'road_group', 'method', 'positive_correct', 'positive_queries', 'negative_false', 'negative_queries'])
        writer.writerows(rows)
    baseline = {m: count([r for r in queries if r['method'] == m]).tolist() for m in methods}
    dump(destination/'report.json', dict(road_groups=results, all_development_baseline_counts=baseline,
        original_query_records=len(queries), paired_unique_queries=len({(r['scene'], r['frame_idx'], r['query_index']) for r in queries}),
        scenes=len(locations), logs=len(all_logs), distinct_frames=len(groups),
        frame_groups=dict(Counter(g['group'] for g in groups.values())), frame_records_sha256=digest(destination/'frames.jsonl'),
        elapsed_s=time.monotonic()-begun, bootstrap=dict(unit='采集log', paired_grouped_counts=True, repetitions=4000, seed=17),
        limitations=['地图派生局部道路组是探索代理，不是人工道路标签或独立因果泛化实验',
            '候选中心线64点且必须落在车道多边形并距线≤3m；未满足者unknown，不强制归组',
            '同一采集log可跨组，组内置信区间不能当作组间显著差异检验；贡献log过少的区间尤其有限',
            '只汇总已完成四基线开发检索，不涉及Qwen新调用、控制、正确锁定或到达',
            '未因分组结论修改冻结导航或开启新封存测试'], independent_test=False, performance_credit=False, goal_complete=False))


if __name__ == '__main__':
    main()
