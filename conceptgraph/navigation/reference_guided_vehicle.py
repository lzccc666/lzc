"""中心线引导的前向运动搜索；每个实际积分步均受固定有向车道约束。"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
import time

from shapely.geometry import LineString, Point

from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, angle_difference, footprint_polygon, step_vehicle, swept_body_polygon,
)


@dataclass
class GuidedVehiclePlan:
    states: list[VehicleState]
    expanded: int
    duration_s: float
    failure: str | None = None
    algorithm: str = 'reference_guided_forward_steering_search_v1'
    diagnostics: dict = field(default_factory=dict)


def reference_guided_vehicle(start, goal_xy, goal_yaw, corridor, reference_line,
                             spec=VehicleSpec(), *, budget_s=3., max_expanded=2000):
    """先沿合法地图中心线提出连续控制，再从已检查前缀进行有界搜索。"""
    begun = time.perf_counter()
    if not math.isfinite(budget_s) or budget_s <= 0 or not isinstance(max_expanded, int) or max_expanded < 1:
        raise ValueError('规划预算必须为正')
    if len(goal_xy) != 2 or not all(math.isfinite(v) for v in [*goal_xy, goal_yaw]):
        raise ValueError('目标位姿无效')
    reference = reference_line if isinstance(reference_line, LineString) else LineString(reference_line)
    if reference.is_empty or reference.length < 1e-6 or not reference.is_valid:
        raise ValueError('参考线无效')
    counts = dict(guided_extensions=0, searched_extensions=0, rejected_steps=0,
                  checked_control_intervals=0, prefix_states=0, topology_in_search_state=True)
    expanded = 0
    deadline = begun+budget_s

    def finish(states, failure=None):
        return GuidedVehiclePlan(states, expanded, time.perf_counter()-begun, failure,
                                 diagnostics=dict(counts))

    initial = frozenset(corridor.initial_stages(start))
    body = footprint_polygon(start, spec, .05)
    if not initial or not corridor.road.covers(body) or any(body.intersects(o) for o in corridor.obstacles):
        return finish([], 'INVALID_INITIAL_BODY_OR_ENTRY_LANE')

    def key(state, stages):
        return (round(state.x/.4), round(state.y/.4), round(state.yaw_rad/math.radians(7.5)),
                round(state.steering_rad/math.radians(8)), round(state.speed_mps/.25), tuple(sorted(stages)))

    def heuristic(state):
        return max(0., math.dist((state.x, state.y), goal_xy)-.65)

    def arrived(state, stages):
        return (heuristic(state) <= 1e-8 and len(corridor.lanes)-1 in stages and
                abs(angle_difference(state.yaw_rad, goal_yaw)) <= math.radians(12))

    def suggestion(state):
        progress = reference.project(Point(state.x, state.y))
        point = reference.interpolate(min(reference.length, progress+2.5))
        dx, dy = point.x-state.x, point.y-state.y
        distance = max(math.hypot(dx, dy), .05)
        alpha = angle_difference(math.atan2(dy, dx), state.yaw_rad)
        return max(-spec.max_steering_rad, min(spec.max_steering_rad,
                   math.atan2(2*spec.wheelbase_m*math.sin(alpha), distance)))

    def extend(state, stages, target, steps):
        primitive = []
        current, active = state, stages
        for _ in range(steps):
            if time.perf_counter() >= deadline:
                return None, None, True
            following = step_vehicle(current, (1.-current.speed_mps)/spec.dt_s, target, spec)
            sweep = swept_body_polygon(current, following, spec, .05)
            if not corridor.road.covers(sweep) or any(sweep.intersects(o) for o in corridor.obstacles):
                counts['rejected_steps'] += 1
                return None, None, False
            checked = corridor.validate_step(current, following, active, spec)
            counts['checked_control_intervals'] += 1
            if not checked['legal']:
                counts['rejected_steps'] += 1
                return None, None, False
            current, active = following, frozenset(checked['next_stages'])
            primitive.append(current)
            if arrived(current, active):
                break
        return primitive, active, False

    sequence = itertools.count()
    first = next(sequence)
    nodes = {first: (start, initial, None, [], 0.)}
    best = {key(start, initial): 0.}
    queue = [(heuristic(start), first)]

    def reconstruct(identity):
        chunks = []
        while nodes[identity][2] is not None:
            chunks.append(nodes[identity][3])
            identity = nodes[identity][2]
        return [start]+[state for chunk in reversed(chunks) for state in chunk]

    def insert(parent, points, stages):
        state, _, _, _, prior_cost = nodes[parent]
        following = points[-1]
        length = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip([state]+points, points))
        deviation = reference.distance(Point(following.x, following.y))
        cost = prior_cost+length+.08*deviation+.05*abs(following.steering_rad-state.steering_rad)
        candidate = key(following, stages)
        if cost >= best.get(candidate, math.inf)-1e-9:
            return None
        identity = next(sequence)
        best[candidate] = cost
        nodes[identity] = following, stages, parent, points, cost
        heapq.heappush(queue, (cost+heuristic(following), identity))
        return identity

    # 所有前导积分也计入扩展与墙钟预算；前缀不等于完整路线。
    identity = first
    while expanded < max_expanded:
        state, stages, _, _, _ = nodes[identity]
        if arrived(state, stages):
            return finish(reconstruct(identity))
        if time.perf_counter() >= deadline:
            return finish([], 'PLANNING_TIME_BUDGET')
        expanded += 1
        counts['guided_extensions'] += 1
        points, active, timed_out = extend(state, stages, suggestion(state), 1)
        if timed_out:
            return finish([], 'PLANNING_TIME_BUDGET')
        if points is None:
            break
        child = insert(identity, points, active)
        # 一个0.1秒步可能仍在同一离散键；前导积分必须保留而不被自身合并。
        if child is None:
            child = next(sequence)
            following = points[-1]
            length = math.dist((state.x, state.y), (following.x, following.y))
            cost = nodes[identity][4]+length
            nodes[child] = following, active, identity, points, cost
            heapq.heappush(queue, (cost+heuristic(following), child))
        identity = child
        counts['prefix_states'] += len(points)
    if expanded >= max_expanded:
        return finish([], 'EXPANSION_BUDGET')
    while queue and expanded < max_expanded:
        if time.perf_counter() >= deadline:
            return finish([], 'PLANNING_TIME_BUDGET')
        _, identity = heapq.heappop(queue)
        state, stages, _, _, cost = nodes[identity]
        if cost > best.get(key(state, stages), math.inf)+1e-8:
            continue
        if arrived(state, stages):
            return finish(reconstruct(identity))
        expanded += 1
        counts['searched_extensions'] += 1
        guide = suggestion(state)
        targets = [guide, guide-math.radians(8), guide+math.radians(8), 0.,
                   -spec.max_steering_rad/2, spec.max_steering_rad/2,
                   -spec.max_steering_rad, spec.max_steering_rad]
        used = set()
        for target in targets:
            target = max(-spec.max_steering_rad, min(spec.max_steering_rad, target))
            if round(target, 10) in used:
                continue
            used.add(round(target, 10))
            points, active, timed_out = extend(state, stages, target, 10)
            if timed_out:
                return finish([], 'PLANNING_TIME_BUDGET')
            if points:
                child = insert(identity, points, active)
                if child is not None and arrived(points[-1], active):
                    return finish(reconstruct(child))
    return finish([], 'EXPANSION_BUDGET' if expanded >= max_expanded else 'NO_FEASIBLE_PRIMITIVE_ROUTE')
