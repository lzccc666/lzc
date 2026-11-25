"""六种地图约束的开发集离线诊断；正确对象由隔离评价端给定，不执行车辆。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import unary_union

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump
from conceptgraph.navigation.lane_constraints import LaneGeometry, LaneTopology, project_to_legal_lane
from conceptgraph.navigation.lane_validation import validate_continuous_lane_path
from conceptgraph.navigation.planner import OccupancyGrid2D
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon
from conceptgraph.scripts.build_lane_constrained_bev import _lane_records


CONDITIONS = ('free_space', 'drivable_area', 'lane_area', 'lane_direction',
              'lane_connection', 'lane_direction_connection')
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
RESOLUTION = .5
WINDOW_RADIUS_M = 70.


@dataclass
class Context:
    topology: LaneTopology
    obstacle_grid: np.ndarray
    drivable_mask: np.ndarray
    drivable: object
    obstacles: list
    lane_union: object
    diagnostics: dict = field(default_factory=dict)
    edge_cache: dict = field(default_factory=dict)
    pair_polygons: dict = field(default_factory=dict)

    def inside(self, rc):
        return 0 <= rc[0] < self.obstacle_grid.shape[0] and 0 <= rc[1] < self.obstacle_grid.shape[1]


def polygon_with_holes(record, nodes):
    """官方多边形孔洞保留；无效图元不自动修补为更大可通行区域。"""
    outer = [(nodes[t]['x'], nodes[t]['y']) for t in record['exterior_node_tokens']]
    holes = [[(nodes[t]['x'], nodes[t]['y']) for t in h['node_tokens']] for h in record.get('holes', [])]
    result = Polygon(outer, holes)
    if not result.is_valid or result.is_empty:
        raise ValueError('invalid official drivable polygon')
    return result


def load_prior(location):
    path = PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
    data = json.loads(path.read_text())
    nodes = {n['token']: n for n in data['node']}
    polygons = {p['token']: p for p in data['polygon']}
    tokens = {t for r in data['drivable_area'] for t in r['polygon_tokens']}
    road_polygons = [polygon_with_holes(polygons[t], nodes) for t in sorted(tokens)]
    xy = np.array([[n['x'], n['y']] for n in nodes.values()])
    lanes, diagnostics = _lane_records(data, xy.min(axis=0)-1, xy.max(axis=0)+1, RESOLUTION)
    # Vector priors may cover the whole city. The evaluated window depends only on the current pose.
    return dict(lanes=lanes, drivable=unary_union(road_polygons), diagnostics=diagnostics,
                polygon_holes=sum(len(polygons[t].get('holes', [])) for t in tokens),
                path=str(path), sha256=digest(path))


def build_context(prior, frame):
    pose = np.asarray(frame['camera_pose'])
    origin = pose[:2, 3]-WINDOW_RADIUS_M
    shape = (int(2*WINDOW_RADIUS_M/RESOLUTION)+1,)*2
    end = origin+(np.asarray(shape)[::-1]-1)*RESOLUTION
    window = box(*origin, *end)
    lanes = [r for r in prior['lanes'] if np.all(np.max(r['polygon'], axis=0) >= origin)
             and np.all(np.min(r['polygon'], axis=0) <= end)]
    tokens = [r['token'] for r in lanes]
    token_set = set(tokens)
    geometries = {r['token']: LaneGeometry(**r) for r in lanes}
    topology = LaneTopology(np.full(shape, -1, dtype=np.int32), np.full(shape, np.nan),
        np.full(shape, np.inf), tokens, [r['lane_type'] for r in lanes],
        {r['token']: tuple(t for t in r['outgoing'] if t in token_set) for r in lanes},
        geometries, dict(transition_endpoint_tolerance_m=2.), origin, RESOLUTION)
    rows, cols = np.indices(shape)
    x, y = origin[0]+cols*RESOLUTION, origin[1]+rows*RESOLUTION
    drivable = prior['drivable'].intersection(window)
    drivable_mask = shapely.intersects_xy(drivable, x, y)
    obstacles = []
    # Only already saved current/past predicted objects are visible here. GT is never an obstacle source.
    for prediction in frame['predictions']['one_to_one_depth_quality']:
        center = np.asarray(prediction['center'])[:2]
        half = np.maximum(np.asarray(prediction['extent'])[:2]/2, .1)
        obstacle = box(*(center-half), *(center+half))
        if obstacle.intersects(window):
            obstacles.append(obstacle)
    obstacle_geometry = unary_union(obstacles)
    occupancy = shapely.intersects_xy(obstacle_geometry, x, y)
    lane_union = unary_union([Polygon(g.polygon) for g in geometries.values()])
    return Context(topology, occupancy, drivable_mask, drivable, obstacles, lane_union,
        dict(current_pose_only_window=True, prior_source=prior['path'], prior_sha256=prior['sha256'],
             official_drivable_holes=prior['polygon_holes'], local_lanes=len(lanes),
             predicted_obstacles=len(obstacles), no_future_observation=True, **prior['diagnostics']))


def _movement_allowed(context, source, target, a_rc, b_rc, direction, connection):
    topology = context.topology
    a, b = topology.world(a_rc), topology.world(b_rc)
    if connection and source != target:
        sg = topology.geometries[topology.lane_tokens[source]]
        tg = topology.geometries[topology.lane_tokens[target]]
        if (tg.token not in topology.transitions.get(sg.token, ()) or sg.end_pose is None
                or tg.start_pose is None or np.linalg.norm(a-sg.end_pose[:2]) > 2.+1e-9
                or np.linalg.norm(b-tg.start_pose[:2]) > 2.+1e-9):
            return False
    key = (min(source, target), max(source, target))
    if key not in context.pair_polygons:
        context.pair_polygons[key] = unary_union([
            Polygon(topology.geometries[topology.lane_tokens[i]].polygon) for i in set(key)])
    edge = Point(a) if a_rc == b_rc else LineString([a, b])
    if not context.pair_polygons[key].covers(edge):
        return False
    if direction and a_rc != b_rc:
        vector = (b-a)/np.linalg.norm(b-a)
        for lane, rc in ((source, a_rc), (target, b_rc)):
            heading = topology.headings[lane, rc[0], rc[1]]
            if not math.isfinite(heading) or vector @ [math.cos(heading), math.sin(heading)] < math.cos(math.radians(75))-1e-9:
                return False
    return True


def plan(context, start, goal, heading, condition, *, wall_budget_s=3., max_expansions=30000):
    """相同8邻域、欧氏代价和端点；只改变指定的地图约束。不得用该结果自证不可达。"""
    if condition not in CONDITIONS:
        raise ValueError('unknown map condition')
    begun = time.monotonic()
    topology = context.topology
    lane = condition.startswith('lane_')
    direction = condition in ('lane_direction', 'lane_direction_connection')
    connection = condition in ('lane_connection', 'lane_direction_connection')
    allowed = ~context.obstacle_grid
    if condition == 'drivable_area':
        allowed = allowed & context.drivable_mask
    elif lane:
        allowed = allowed & topology.lane_mask
    if not context.inside(start) or not context.inside(goal) or not allowed[start] or not allowed[goal]:
        return dict(path=None, failure='ENDPOINT_NOT_FREE_IN_CONDITION', expanded=0,
                    elapsed_s=time.monotonic()-begun)
    queue, costs, previous = [], {}, {}
    serial = 0
    for lane_id in topology.lanes_at(start) if lane else [-1]:
        if direction:
            yaw = topology.headings[lane_id, start[0], start[1]]
            if not math.isfinite(yaw) or np.dot(heading, [math.cos(yaw), math.sin(yaw)]) < 0:
                continue
        state = (*start, lane_id)
        costs[state], previous[state] = 0., None
        heapq.heappush(queue, (math.dist(start, goal), serial, 0., state))
        serial += 1
    expanded = 0
    failure = 'EXHAUSTED_SEARCH_NOT_PROOF_OF_UNREACHABILITY'
    while queue:
        if expanded % 64 == 0 and time.monotonic()-begun > wall_budget_s:
            failure = 'PLANNING_WALL_BUDGET'
            break
        _, _, cost, state = heapq.heappop(queue)
        if cost > costs[state]+1e-9:
            continue
        expanded += 1
        if expanded > max_expansions:
            failure = 'PLANNING_EXPANSION_BUDGET'
            break
        row, col, source = state
        if (row, col) == goal:
            states = [state]
            while previous[states[-1]] is not None:
                states.append(previous[states[-1]])
            path, lane_trace = [], []
            for step in reversed(states):
                if not path or path[-1] != step[:2]:
                    path.append(step[:2])
                lane_trace.append(step[2])
            return dict(path=path, planner_lane_trace=lane_trace, cost_grid=cost,
                        expanded=expanded, failure=None, elapsed_s=time.monotonic()-begun)
        for dr, dc in OFFSETS:
            rc = (row+dr, col+dc)
            if not context.inside(rc) or not allowed[rc]:
                continue
            if dr and dc and (not allowed[row, rc[1]] or not allowed[rc[0], col]):
                continue
            for target in topology.lanes_at(rc) if lane else [-1]:
                if dr == dc == 0 and source == target:
                    continue
                # Cache only immutable current-window edge geometry; never share it between frames.
                key = (state, rc, target, condition)
                legal = context.edge_cache.get(key)
                if legal is None:
                    if lane:
                        legal = _movement_allowed(context, source, target, (row, col), rc, direction, connection)
                    elif condition == 'drivable_area':
                        legal = context.drivable.covers(LineString([topology.world((row, col)), topology.world(rc)]))
                    else:
                        legal = True
                    context.edge_cache[key] = legal
                if not legal:
                    continue
                neighbor = (*rc, target)
                candidate = cost+max(math.hypot(dr, dc), 1e-6)
                if candidate+1e-9 < costs.get(neighbor, float('inf')):
                    costs[neighbor], previous[neighbor] = candidate, state
                    serial += 1
                    heapq.heappush(queue, (candidate+math.dist(rc, goal), serial, candidate, neighbor))
    return dict(path=None, failure=failure, expanded=expanded, elapsed_s=time.monotonic()-begun)


def audit(context, path, heading):
    """固定完整HD规则审计全部条件；车身只作静态姿态检查，不虚构控制与运动扫掠。"""
    if path is None:
        return dict(path_generated=False, full_lane_legal=None, body_pose_legal=None,
                    actual_arrival=None, selected_correct_object=None, stable_correct_lock=None)
    lane = validate_continuous_lane_path(path, context.topology, context.obstacle_grid,
                                        start_heading_xy=heading)
    worlds = np.array([context.topology.world(rc) for rc in path])
    point_geometry = LineString(worlds) if len(worlds) > 1 else Point(worlds[0])
    drivable_legal = context.drivable.covers(point_geometry)
    body_violations, yaws = [], []
    initial_yaw = math.atan2(heading[1], heading[0])
    for i, xy in enumerate(worlds):
        delta = worlds[min(i+1, len(worlds)-1)]-worlds[max(i-1, 0)]
        yaw = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-9 else initial_yaw
        yaws.append(yaw)
        body = footprint_polygon(VehicleState(*xy, yaw), VehicleSpec())
        reasons = []
        if not context.drivable.covers(body):
            reasons.append('rectangle_outside_drivable')
        if not context.lane_union.covers(body):
            reasons.append('rectangle_outside_lane_union')
        if any(body.intersects(obstacle) for obstacle in context.obstacles):
            reasons.append('rectangle_intersects_observed_obstacle')
        if reasons:
            body_violations.append(dict(index=i, reasons=reasons))
    initial_body = footprint_polygon(VehicleState(*worlds[0], initial_yaw), VehicleSpec())
    initial_body_legal = bool(context.drivable.covers(initial_body) and context.lane_union.covers(initial_body)
                              and all(not initial_body.intersects(o) for o in context.obstacles))
    differences = np.abs((np.diff(yaws)+math.pi) % (2*math.pi)-math.pi)
    distances = np.linalg.norm(np.diff(worlds, axis=0), axis=1)
    curvature = np.divide(differences, distances, out=np.zeros_like(distances), where=distances > 1e-9)
    return dict(path_generated=True, full_lane_legal=bool(lane['valid']), continuous_point_drivable=bool(drivable_legal),
        independent_lane_audit=lane, body_pose_legal=bool(not body_violations and initial_body_legal),
        body_pose_violations=body_violations, initial_body_pose_legal=initial_body_legal,
        waypoint_tangent_curvature_max_per_m=float(curvature.max()) if len(curvature) else None,
        curvature_exceeds_vehicle_bound=bool(np.any(curvature > 1/VehicleSpec().minimum_turn_radius_m)),
        static_pose_geometry_only=True, continuous_body_sweep_evaluated=False, controller_evaluated=False,
        kinematic_executability=None, actual_arrival=None, selected_correct_object=None, stable_correct_lock=None)


def selected_queries(source, limit_per_scene):
    grouped = {}
    for row in map(json.loads, (source/'queries.jsonl').open()):
        if row['method'] == 'existing_object_joint_rule' and row['positive']:
            grouped.setdefault(row['scene'], []).append(row)
    selected = []
    for scene, queries in sorted(grouped.items()):
        def key(q):
            identity = f"{q['scene']}/{q['frame_idx']}/{q['query_index']}"
            return hashlib.sha256(('map-ablation-v1/'+identity).encode()).hexdigest()
        selected.extend(sorted(queries, key=key)[:limit_per_scene])
    return selected


def _save_context(dest, context, frame):
    dest.mkdir(parents=True, exist_ok=False)
    t = context.topology
    np.savez_compressed(dest/'rasters.npz', obstacle_grid=context.obstacle_grid,
        drivable_mask=context.drivable_mask, lane_mask=t.lane_mask, origin_xy=t.origin_xy, resolution=t.resolution)
    geometries = {}
    for token, g in t.geometries.items():
        geometries[token] = dict(token=token, lane_type=g.lane_type, polygon=g.polygon.tolist(),
            centerline=g.centerline.tolist(), heading=g.heading, start_pose=g.start_pose,
            end_pose=g.end_pose, incoming=g.incoming, outgoing=g.outgoing)
    dump(dest/'context.json', dict(scene=frame['scene'], frame_idx=frame['frame_idx'],
        camera_pose=frame['camera_pose'], lane_geometries=geometries, transitions=t.transitions,
        drivable_geojson=mapping(context.drivable), obstacles_geojson=[mapping(o) for o in context.obstacles],
        diagnostics=context.diagnostics, obstacle_input_source='current predictions.one_to_one_depth_quality'))
    return dict(context_sha256=digest(dest/'context.json'), rasters_sha256=digest(dest/'rasters.npz'),
                path=str(dest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retrieval-run', default='retrieval_baselines_dev_full_001')
    parser.add_argument('--per-scene', type=int, default=6)
    parser.add_argument('--query-limit', type=int)
    parser.add_argument('--wall-budget-s', type=float, default=7200.)
    parser.add_argument('--case-budget-s', type=float, default=3.)
    args = parser.parse_args()
    if not 1 <= args.per_scene <= 6 or args.query_limit is not None and args.query_limit < 1:
        raise ValueError('invalid development query budget')
    begun = time.monotonic()
    source = OUTPUT/'main_experiments'/args.retrieval_run
    source_release = json.loads((source/'release.json').read_text())
    queries = selected_queries(source, args.per_scene)
    if args.query_limit:
        queries = queries[:args.query_limit]
    scenes = {r['scene']: r for r in available_development()}
    if not {q['scene'] for q in queries} <= scenes.keys():
        raise ValueError('only known development scenes allowed')
    metadata = PROJECT/'dataset/nuscenes_trainval_meta/v1.0-trainval'
    official_scenes = {r['name']: r for r in json.loads((metadata/'scene.json').read_text())}
    logs = {r['token']: r for r in json.loads((metadata/'log.json').read_text())}
    locations = {scene: logs[official_scenes[scene]['log_token']]['location'] for scene in scenes}
    modules = [Path(__file__), PROJECT/'conceptgraph/completion/stage9.py',
        PROJECT/'conceptgraph/scripts/build_lane_constrained_bev.py',
        *sorted((PROJECT/'conceptgraph/navigation').glob('*.py'))]
    frozen = {str(p.relative_to(PROJECT)): digest(p) for p in modules}
    dest = OUTPUT/'ablations'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    release = dict(retrieval_run=args.retrieval_run, conditions=CONDITIONS, selected_queries=queries, queries=len(queries),
        expected_cases=len(queries)*len(CONDITIONS), selection='开发正查询逐场景SHA256固定排序取前6，不访问路径结果',
        source_queries_sha256=digest(source/'queries.jsonl'), source_frames_sha256=source_release['source_frames_sha256'],
        source_sha256=frozen, map_sha256={location: digest(PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json')
                                       for location in sorted({locations[q['scene']] for q in queries})},
        metadata_sha256={n: digest(metadata/n) for n in ['scene.json', 'log.json']},
        scene_logs={s: scenes[s]['log_token'] for s in sorted({q['scene'] for q in queries})},
        resolution_m=RESOLUTION, window_radius_m=WINDOW_RADIUS_M, endpoint_projection_bound_m=8.,
        direction_tolerance_deg=75., connection_endpoint_bound_m=2., maximum_expansions=30000,
        path_cost='所有条件欧氏长度，无中心线偏置；同一8邻域和已观测障碍，不同地图约束',
        case_wall_budget_s=args.case_budget_s, wall_budget_s=args.wall_budget_s, storage_budget_bytes=500000000,
        oracle='隔离评价端给定当前正确对象，投影为共同车道参考点；不证明静止、安全接近或可达',
        gt_in_obstacle_mapping=False, future_observation_used=False, independent_test=False,
        vehicle_executed=False, arrival_credit=False, performance_credit=False, goal_complete=False)
    dump(dest/'release.json', release)
    with tarfile.open(dest/'source_snapshot.tar.gz', 'x:gz') as archive:
        for module in modules:
            archive.add(module, arcname=str(module.relative_to(PROJECT)))
    priors, frames, rows = {}, {}, []
    wanted = {(q['scene'], q['frame_idx']) for q in queries}
    for name, sha in source_release['source_frames_sha256'].items():
        if Path(name).parent.name not in {q['scene'] for q in queries}:
            continue
        if digest(name) != sha:
            raise ValueError('frozen input changed')
        for frame in map(json.loads, Path(name).open()):
            if (frame['scene'], frame['frame_idx']) in wanted:
                frames[(frame['scene'], frame['frame_idx'])] = frame
    try:
        with (dest/'cases.jsonl').open('x') as handle:
            for q in queries:
                if time.monotonic()-begun > args.wall_budget_s:
                    raise TimeoutError('overall offline map ablation budget')
                scene, frame_index = q['scene'], q['frame_idx']
                frame = frames[(scene, frame_index)]
                location = locations[scene]
                if location not in priors:
                    priors[location] = load_prior(location)
                context = build_context(priors[location], frame)
                context_path = dest/'contexts'/f'{scene}_f{frame_index}'
                if context_path.exists():
                    context_evidence = dict(path=str(context_path), context_sha256=digest(context_path/'context.json'),
                                            rasters_sha256=digest(context_path/'rasters.npz'))
                else:
                    context_evidence = _save_context(context_path, context, frame)
                pose = np.asarray(frame['camera_pose'])
                heading = pose[:2, 2]/np.linalg.norm(pose[:2, 2])
                accepted = set(q['acceptable_instances'])
                objects = [o for o in frame['evaluation_only_official'] if o['instance_token'] in accepted]
                if not objects:
                    raise ValueError('frozen positive identity missing from isolated evaluation')
                target = min(objects, key=lambda o: (float(np.linalg.norm(np.asarray(o['center'])[:2]-pose[:2, 3])), o['instance_token']))
                grid = OccupancyGrid2D(context.obstacle_grid, RESOLUTION, context.topology.origin_xy)
                start = project_to_legal_lane(grid, context.topology, pose[:2, 3], max_distance_m=8.)
                goal = project_to_legal_lane(grid, context.topology, np.asarray(target['center'])[:2], max_distance_m=8.)
                for condition in CONDITIONS:
                    if start is None or goal is None:
                        result = dict(path=None, failure='COMMON_ENDPOINT_PROJECTION_FAILED', expanded=0, elapsed_s=0.)
                    else:
                        result = plan(context, start[0], goal[0], heading, condition, wall_budget_s=args.case_budget_s)
                    independent = audit(context, result['path'], heading)
                    row = dict(scene=scene, log_token=scenes[scene]['log_token'], frame_idx=frame_index,
                        query_index=q['query_index'], query_text=q['query_text'], query_type=q['query_type'],
                        condition=condition, target_instance=target['instance_token'], target_category=target['category'],
                        current_camera_xy=pose[:2, 3].tolist(), target_center_xy=target['center'][:2],
                        common_start=None if start is None else dict(rc=start[0], xy=start[1].tolist(), distance_m=start[2]),
                        common_goal=None if goal is None else dict(rc=goal[0], xy=goal[1].tolist(), distance_m=goal[2]),
                        context=context_evidence, planner=result, evaluation=independent,
                        actual_arrival_evaluated=False, vehicle_executed=False, diagnostic_oracle=True)
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush()
                    print(json.dumps(dict(scene=scene, frame=frame_index, query=q['query_index'], condition=condition,
                        route=independent['path_generated'], legal=independent['full_lane_legal'], failure=result['failure']),
                        ensure_ascii=False), flush=True)
                # Do not retain the potentially large edge cache across queries.
                del context
                if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > release['storage_budget_bytes']:
                    raise RuntimeError('offline map storage budget')
        if any(digest(PROJECT/p) != sha for p, sha in frozen.items()):
            raise ValueError('frozen source changed during run')
        dump(dest/'summary.json', dict(cases=len(rows), queries=len(queries), expected_cases=release['expected_cases'],
            unique_frames=len(wanted), scenes=len(release['scene_logs']), logs=len(set(release['scene_logs'].values())),
            records_sha256=digest(dest/'cases.jsonl'), elapsed_s=time.monotonic()-begun,
            oracle_diagnostic=True, independent_test=False, arrival_credit=False, goal_complete=False))
    except BaseException as error:
        dump(dest/'interrupted.json', dict(error=repr(error), completed_cases=len(rows), elapsed_s=time.monotonic()-begun,
                                          goal_complete=False))
        raise


if __name__ == '__main__':
    main()
