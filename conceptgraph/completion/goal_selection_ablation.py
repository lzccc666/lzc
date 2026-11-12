"""录制开发观测上的联合/首达/去安全接近点消融；只规划，不执行车辆。"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from conceptgraph.completion.map_constraint_ablation import (
    PROJECT, OUTPUT, RESOLUTION, _save_context, audit, build_context, load_prior, plan,
)
from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval
from conceptgraph.completion.selection_ablation import select as semantic_candidates
from conceptgraph.completion.navigation_diagnostics import to_nodes
from conceptgraph.completion.stage9 import available_development, digest, dump
from conceptgraph.navigation.lane_constraints import project_to_legal_lane
from conceptgraph.navigation.planner import OccupancyGrid2D, minimum_path_clearance_m, path_length_m
from conceptgraph.navigation.target_selector import TargetSelector, TargetSelectionConfig, reachable_goal_to_dict
from conceptgraph.navigation.types import ReachableGoalCandidate


CONDITIONS = ('joint_safe_approach', 'first_reachable_same_safe_pool', 'joint_without_safe_approach')


class OfflineSelector(TargetSelector):
    """复用生产候选点与联合效用；规划显式有界且用独立连续车道审计准入。"""

    def __init__(self, context, nodes, start, pose, goal_budget_s, remove_approach=False):
        grid = OccupancyGrid2D(context.obstacle_grid.astype(np.uint8), RESOLUTION,
            context.topology.origin_xy, distance_transform_edt(~context.obstacle_grid)*RESOLUTION)
        super().__init__(grid, nodes, start, set(), TargetSelectionConfig(mode='joint_utility', top_k=3),
            lane_topology=context.topology, start_heading_xy=pose[:2, 2], vehicle_xy=grid.grid_to_world(start))
        self.context, self.goal_budget_s, self.remove_approach = context, goal_budget_s, remove_approach
        self.heading = pose[:2, 2]
        self.audits = {}

    def _generate_goal_candidates(self, node):
        if not self.remove_approach:
            return super()._generate_goal_candidates(node)
        # 离线去掉接近环、对象距离与端点净空保护；预测障碍和完整车道审计仍保留。
        center = np.asarray(node.bbox_center)[:2]
        projected = project_to_legal_lane(self.grid, self.lane_topology, center, max_distance_m=8.)
        if projected is None:
            return [], 'NO_LEGAL_LANE_GOAL', dict(removed_safe_approach=True)
        rc, xy, distance = projected
        extent = np.asarray(node.bbox_extent)[:2]
        bbox_distance = float(np.linalg.norm(np.maximum(np.abs(xy-center)-extent/2, 0)))
        return [ReachableGoalCandidate(goal_xy=xy, goal_rc=rc, desired_goal_xy=center,
            desired_goal_rc=self.grid.world_to_grid(center), distance_to_object_m=bbox_distance,
            clearance_m=self._clearance(rc), connected_to_start=True, fallback_distance_m=distance)], None, dict(
                removed_safe_approach=True, goal_source='predicted_object_center_projected_to_lane')

    def _plan_lane_goal(self, goal):
        raw = plan(self.context, self.start_rc, goal.goal_rc, self.heading,
            'lane_direction_connection', wall_budget_s=self.goal_budget_s)
        goal.debug['bounded_planner'] = {k:v for k,v in raw.items() if k not in ['path', 'planner_lane_trace']}
        if raw['path'] is None:
            goal.rejection_reason = raw['failure']
            return
        checked = audit(self.context, raw['path'], self.heading)
        goal.debug['independent_audit'] = checked
        self.audits[tuple(goal.goal_rc)] = checked
        if not checked['full_lane_legal']:
            goal.rejection_reason = 'INDEPENDENT_LANE_AUDIT_REJECTED'
            return
        goal.path_found = True
        goal.path_cost = raw['cost_grid']*RESOLUTION
        goal.path_grid_raw = list(raw['path'])
        goal.path_grid = list(raw['path'])
        goal.path_length_m = path_length_m(self.grid, raw['path'])
        goal.min_path_clearance_m = minimum_path_clearance_m(self.grid, raw['path'])
        goal.expanded_nodes = raw['expanded']


def choose_first(feasible):
    """仅替换对象选择顺序，与联合条件共享同一批实际规划的安全接近点。"""
    return min(feasible, key=lambda c:(c.semantic_rank, c.node_id)) if feasible else None


def run_inference(query, predictions, pose, frame_index, context, goal_budget_s):
    # 此函数不接收官方对象/类别、可接受集合、可达性或最终融合图。
    ranked = semantic_candidates(query, predictions, pose, frame_index, 'full_joint_rule')[:3]
    nodes = to_nodes(predictions)
    for node in nodes.values():
        node.metadata.update(bbox_center=node.bbox_center.tolist(), bbox_extent=node.bbox_extent.tolist())
    empty = dict(selected=None, evaluated=[], raw_ranked_ids=[c.node_id for c in ranked])
    if not ranked:
        return {condition:dict(empty, failure='NO_SEMANTIC_CANDIDATES') for condition in CONDITIONS}
    free_grid = OccupancyGrid2D(context.obstacle_grid.astype(np.uint8), RESOLUTION, context.topology.origin_xy)
    start_projection = project_to_legal_lane(free_grid, context.topology, pose[:2, 3], max_distance_m=8.)
    if start_projection is None:
        return {condition:dict(empty, failure='COMMON_START_PROJECTION_FAILED') for condition in CONDITIONS}
    start = start_projection[0]
    outputs = {}
    for remove in (False, True):
        selector = OfflineSelector(context, nodes, start, pose, goal_budget_s, remove)
        result = selector.select(copy.deepcopy(ranked))
        feasible = [c for c in result.evaluated_candidates if c.reachable]
        conditions = [('joint_without_safe_approach' if remove else 'joint_safe_approach', result.selected_candidate)]
        if not remove:
            conditions.append(('first_reachable_same_safe_pool', choose_first(feasible)))
        evaluated = [dict(node_id=c.node_id, semantic_rank=c.semantic_rank, reachable=c.reachable,
            score=c.score, utility=c.selection_utility, rejection=c.rejection_reason,
            goals=[reachable_goal_to_dict(g) for g in c.goal_candidates]) for c in result.evaluated_candidates]
        for condition, selected in conditions:
            selected_data = None
            if selected is not None:
                goal = next(g for g in selected.goal_candidates if g.goal_rc==selected.target_grid and g.path_found)
                selected_data = dict(node_id=selected.node_id, semantic_rank=selected.semantic_rank,
                    goal=reachable_goal_to_dict(goal), audit=selector.audits[tuple(goal.goal_rc)])
            outputs[condition] = dict(selected=selected_data, evaluated=evaluated,
                failure=None if selected is not None else result.failure_code,
                common_start_xy=free_grid.grid_to_world(start).tolist(),
                common_start_projection_m=start_projection[2], raw_ranked_ids=empty['raw_ranked_ids'])
        # 仅在同一查询的目标候选之间复用几何边缓存，不携带未来观测。
        context.edge_cache.clear()
    return outputs


def summarize(destination, rows, scene_logs):
    methods, logs, counts = aggregate(rows, scene_logs)
    totals, estimates, rates = bootstrap(counts)
    base = methods.index('joint_safe_approach')
    values, paired = {}, []
    for j, method in enumerate(methods):
        values[method] = {}
        current = [r for r in rows if r['method']==method]
        for k, metric in enumerate(['positive_selection', 'no_match_false_selection']):
            values[method][metric] = dict(estimate=float(estimates[j, k]),
                numerator=int(totals[j, k*2]), denominator=int(totals[j, k*2+1]), **interval(rates[:, j, k]))
            if j!=base:
                paired.append(dict(method=method, baseline=methods[base], metric=metric,
                    difference=float(estimates[j,k]-estimates[base,k]), **interval(rates[:,j,k]-rates[:,base,k])))
        values[method]['selected'] = sum(r['selected'] for r in current)
        for metric in ['route_generated', 'full_road_legal', 'static_body_pose_legal', 'endpoint_2m_from_predicted_bbox']:
            tested = [r[metric] for r in current if r[metric] is not None]
            values[method][metric] = dict(count=sum(tested), denominator=len(tested),
                rate=sum(tested)/len(tested) if tested else None)
        values[method]['failures'] = dict(Counter(r['inference']['failure'] for r in current if r['inference']['failure']))
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'positive_correct', 'positive_queries', 'negative_false', 'negative_queries'])
        for i, log in enumerate(logs):
            for j, method in enumerate(methods):
                writer.writerow([log, method, *counts[i,j].tolist()])
    return dict(conditions=values, paired_differences=paired, scenes=len(scene_logs), logs=len(logs),
        query_cases=len(rows)//3, case_records=len(rows), raw_sha256=digest(destination/'queries.jsonl'),
        first_and_joint_share_actual_safe_goal_pool=True, bootstrap=dict(unit='采集log', repetitions=4000, seed=17),
        limitations=['离线当前观测规划；两种策略不产生新的控制轨迹，实际到达/稳定锁定均未评价',
            '沿用生产接近环与联合效用，但为有界离线诊断替换为0.5m点车道规划，非运行中Hybrid导航成绩',
            '每目标12个接近环候选，Top-K=3；首达条件共享完整条件实际规划池，仅改变对象选择顺序',
            '去安全接近条件只改为预测中心投影并取消距离/净空门限，仍有预测障碍和车道审计；未执行该条件',
            '未匹配预测身份为unknown并保留为正查询失败；预测bbox完整性不足，安全结论不覆盖真实整车',
            '预算耗尽不证明不可达；无路径条件不记100%安全；共享CPU不计独占性能'],
        gt_in_inference=False, independent_test=False, actual_arrival_evaluated=False, goal_complete=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retrieval-run', default='retrieval_baselines_dev_full_001')
    parser.add_argument('--scene')
    parser.add_argument('--goal-budget-s', type=float, default=.5)
    parser.add_argument('--wall-budget-s', type=float, default=3600.)
    args = parser.parse_args()
    begun = time.monotonic()
    source = OUTPUT/'main_experiments'/args.retrieval_run
    release = json.loads((source/'release.json').read_text())
    queries = [r for r in map(json.loads, (source/'queries.jsonl').open())
               if r['method']=='existing_object_joint_rule' and (not args.scene or r['scene']==args.scene)]
    scenes = {r['scene']:r for r in available_development()}
    if not queries or not {q['scene'] for q in queries} <= scenes.keys():
        raise ValueError('only nonempty development queries allowed')
    metadata = PROJECT/'dataset/nuscenes_trainval_meta/v1.0-trainval'
    official_scenes = {r['name']:r for r in json.loads((metadata/'scene.json').read_text())}
    logs = {r['token']:r for r in json.loads((metadata/'log.json').read_text())}
    locations = {q['scene']:logs[official_scenes[q['scene']]['log_token']]['location'] for q in queries}
    dest = OUTPUT/'ablations'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), *[Path(__file__).with_name(n+'.py') for n in
        ['map_constraint_ablation', 'selection_ablation', 'navigation_diagnostics', 'retrieval_statistics', 'stage9']],
        *sorted((PROJECT/'conceptgraph/navigation').glob('*.py')), PROJECT/'conceptgraph/scenegraph/relations.py']
    source_shas = {str(p):digest(p) for p in dependencies}
    dump(dest/'release.json', dict(retrieval_run=args.retrieval_run,source_sha256=source_shas, source_query_sha256=digest(source/'queries.jsonl'),
        source_frames_sha256=release['source_frames_sha256'], query_cases=len(queries), expected_records=len(queries)*3,
        conditions=CONDITIONS, selection='全部冻结开发查询；smoke可按scene过滤，不按规划结果筛选',
        top_k=3, goals_per_object=12, approach_distance_m=2., minimum_clearance_m=.5,
        goal_budget_s=args.goal_budget_s, wall_budget_s=args.wall_budget_s, storage_budget_bytes=500000000,
        map_sha256={loc:digest(PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{loc}.json') for loc in sorted(set(locations.values()))},
        first_reachable_changes_only_object_choice=True, no_vehicle_execution=True, independent_test=False, goal_complete=False))
    shutil.copy2(__file__, dest/'generator_source.py')
    frozen = defaultdict(list)
    for q in queries:
        frozen[(q['scene'],q['frame_idx'])].append(q)
    priors, rows = {}, []
    try:
        with (dest/'queries.jsonl').open('x') as handle:
            for filename, sha in release['source_frames_sha256'].items():
                if Path(filename).parent.name not in locations:
                    continue
                if digest(filename)!=sha:
                    raise ValueError('frozen development frames changed')
                for frame in map(json.loads, Path(filename).open()):
                    wanted = frozen.get((frame['scene'],frame['frame_idx']), [])
                    if not wanted:
                        continue
                    location = locations[frame['scene']]
                    if location not in priors:
                        priors[location] = load_prior(location)
                    context = build_context(priors[location], frame)
                    context_source = _save_context(dest/'contexts'/f"{frame['scene']}_f{frame['frame_idx']}", context, frame)
                    pose = np.asarray(frame['camera_pose'])
                    predictions = frame['predictions']['one_to_one_depth_quality']
                    for q in wanted:
                        outputs = run_inference(q['query_text'], predictions, pose, frame['frame_idx'], context, args.goal_budget_s)
                        # 推理结束才访问官方实例关联与可接受集合。
                        truth = frame['evaluation_only_official']
                        identities = {m['prediction_index']:truth[m['truth_index']]['instance_token']
                            for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                        for method, output in outputs.items():
                            selected = output['selected']
                            identity = identities.get(selected['node_id']) if selected else None
                            positive = bool(q['positive'])
                            goal = selected['goal'] if selected else None
                            checked = selected['audit'] if selected else None
                            row = {k:q[k] for k in ['scene','frame_idx','query_index','query_text','query_type','acceptable_instances']}
                            row.update(method=method,positive=positive,selected=selected is not None,
                                evaluation_identity=identity,selected_correct=identity in q['acceptable_instances'] if positive else None,
                                false_selection_on_no_match=bool(selected) if not positive else None,
                                route_generated=selected is not None, full_road_legal=bool(checked['full_lane_legal'] and checked['continuous_point_drivable']) if checked else None,
                                static_body_pose_legal=checked['body_pose_legal'] if checked else None,
                                endpoint_2m_from_predicted_bbox=goal['distance_to_object_m']>=2.-1e-6 if goal else None,
                                stable_correct_lock=None,actual_arrival=None,context=context_source,inference=output,
                                vehicle_executed=False,gt_in_inference=False)
                            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                            rows.append(row)
                        handle.flush()
                        if time.monotonic()-begun > args.wall_budget_s:
                            raise TimeoutError('offline goal selection wall budget')
                    size = sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
                    if size>500000000:
                        raise RuntimeError('offline goal selection storage budget')
                    print(json.dumps(dict(scene=frame['scene'],frame=frame['frame_idx'],records=len(rows),expected=len(queries)*3)),flush=True)
                    context.edge_cache.clear()
        if len(rows)!=len(queries)*3 or any(digest(p)!=sha for p,sha in source_shas.items()):
            raise ValueError('missing records or inference source changed')
        report = summarize(dest, rows, {s:scenes[s]['log_token'] for s in locations})
        report.update(elapsed_s=time.monotonic()-begun,release_sha256=digest(dest/'release.json'))
        dump(dest/'report.json', report)
        dump(dest/'summary.json', dict(query_cases=len(queries),records=len(rows),expected_records=len(queries)*3,
            elapsed_s=time.monotonic()-begun,report_sha256=digest(dest/'report.json'),goal_complete=False))
    except BaseException as exc:
        dump(dest/'interrupted.json',dict(error=repr(exc),records=len(rows),elapsed_s=time.monotonic()-begun,goal_complete=False))
        raise


if __name__ == '__main__':
    main()
