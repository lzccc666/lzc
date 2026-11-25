"""相同真实观测/路线下比较障碍列表与其几何并集，保留原碰撞与扫掠门限。"""
from dataclasses import asdict
import datetime
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.completion.planning_reuse import revalidate_route
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, audit_executed_states

R = Path(__file__).resolve().parents[1]


def main():
    root = R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule'
    out = R/'carla_official/obstacle_union_benchmark_001'
    out.mkdir(exist_ok=False)
    release = json.loads((root/'release.json').read_text())
    observations = list(map(json.loads, (root/'perception.jsonl').open()))
    execution = {r['tick']: r for r in map(json.loads, (root/'execution.jsonl').open())}
    spec = VehicleSpec(**release['vehicle_spec'])
    road = Polygon(release['prior']['road_polygon_xy'])
    line = LineString(release['prior']['centerline_xy'])
    known_goal, known_yaw = None, None
    available = []
    for obs in observations:
        plan = obs['planning']
        if plan is not None:
            known_goal, known_yaw = plan['goal_xy'], plan['goal_yaw_rad']
        if obs['path_world'] and len(obs['path_world']) > 2 and known_goal is not None:
            available.append((obs, known_goal, known_yaw))
    selected = [available[i] for i in np.linspace(0, len(available)-1, 6, dtype=int)]
    dump(out/'registration.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_source_sha256={str(p): digest(p) for p in [root/'release.json', root/'perception.jsonl',
            root/'execution.jsonl', Path(__file__)]},
        sample_frames=[s[0]['frame_idx'] for s in selected],
        selection='所有有已知目标和非空实际路线的观测，按时间等间隔取6帧，不按正确/通过/快慢选样',
        shared_revalidation_budget_s=30., collision_padding_unchanged=True,
        executed_on_CPU_only=True, current_official_GPU_job_can_run_concurrently=True,
        purpose='相同输出下的计算诊断；时延只作同机探索性比较，不作独立配置性能主表',
        actual_new_GPU_or_physical_credit=0, goal_complete=False))
    reports = []
    for obs, goal, yaw in selected:
        state = VehicleState(**execution[obs['frame_idx']*2]['state'])
        obstacles = [Polygon(p) for p in obs['known_obstacle_polygons']]
        start = time.monotonic()
        union = unary_union(obstacles)
        aggregate_time = time.monotonic()-start
        variants = []
        for name, values in [('original_list', obstacles), ('geometry_union', [union])]:
            started = time.monotonic()
            plan, reason = revalidate_route(state, obs['path_world'], goal, yaw,
                road, values, line, spec, budget_s=30.)
            elapsed = time.monotonic()-started
            audit = audit_executed_states(plan.states, road, obstacles, spec) if plan else None
            variants.append(dict(variant=name, returned_plan=plan is not None, reason=reason,
                wall_s=elapsed, states=[] if plan is None else [asdict(s) for s in plan.states],
                original_individual_obstacles_sweep_audit=audit))
        exact = variants[0]['states'] == variants[1]['states'] and variants[0]['reason'] == variants[1]['reason']
        assert exact, '原障碍与并集输出不一致，禁止采用'
        assert all(v['original_individual_obstacles_sweep_audit'] is None
            or v['original_individual_obstacles_sweep_audit']['passed'] for v in variants)
        # 原始每个障碍与并集的碰撞判断在实际路径采样点邻域再独立比较。
        probes = [Point(p).buffer(.1) for p in obs['path_world']]
        assert all(any(p.intersects(o) for o in obstacles) == p.intersects(union) for p in probes)
        row = dict(frame_idx=obs['frame_idx'], obstacle_count=len(obstacles),
            union_preparation_wall_s=aggregate_time, variants=variants,
            returned_states_and_reasons_exactly_equal=exact, independent_probe_count=len(probes),
            speedup_including_union=variants[0]['wall_s']/(variants[1]['wall_s']+aggregate_time))
        reports.append(row)
        dump(out/f"frame_{obs['frame_idx']:06d}.json", row)
        print(json.dumps({k:v for k,v in row.items() if k != 'variants'}, ensure_ascii=False), flush=True)
    dump(out/'report.json', dict(actual_observed_frames=len(reports),
        exact_outputs_and_original_sweeps_match=all(r['returned_states_and_reasons_exactly_equal'] for r in reports),
        median_speedup_including_union=float(np.median([r['speedup_including_union'] for r in reports])),
        results=[dict(frame_idx=r['frame_idx'], speedup=r['speedup_including_union'],
            obstacle_count=r['obstacle_count']) for r in reports],
        no_collision_margin_or_vehicle_limit_relaxed=True, not_adopted_in_running_008=True,
        current_30s_budget_is_diagnostic_not_changed_production_budget=True,
        actual_new_GPU_calls=0, actual_new_physical_episodes=0, goal_complete=False))


if __name__ == '__main__':
    main()
