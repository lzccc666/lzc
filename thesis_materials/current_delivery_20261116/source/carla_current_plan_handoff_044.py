"""本帧获授权的计划交接；独立版本，未接入旧冻结CARLA实验。"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from shapely.geometry import Point

from conceptgraph.navigation.vehicle_geometry import (
    VehicleState, angle_difference, audit_executed_states, footprint_polygon,
    swept_body_polygon,
)


@dataclass
class SelectionSnapshot:
    index: int
    state: dict
    candidate_ids: set
    authorized_id: int | None
    evidence_passed: bool
    plans: dict


class RecordingPlanBridge:
    """透明记录select返回的计划，不改变规则、模型、预算或安全门闸。"""

    def __init__(self, delegate):
        self.delegate = delegate
        self.last_selection = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def select(self, agent, candidates, edges, state, index):
        self.last_selection = None
        final, plans, audit = self.delegate.select(agent, candidates, edges, state, index)
        saved = {}
        for identity, plan in plans.items():
            goal, yaw = agent.approach_goal(identity)
            saved[identity] = dict(
                states=tuple(VehicleState(**asdict(s)) for s in plan.states),
                failure=plan.failure, goal=np.asarray(goal, dtype=float).copy(), yaw=float(yaw),
                object_identity=agent.nodes[identity].metadata.get('object_identity'),
                expanded=plan.expanded, duration_s=plan.duration_s,
                algorithm=plan.algorithm)
        self.last_selection = SelectionSnapshot(index, asdict(state),
            {c.node_id for c in final}, audit.get('authorized_node_id'),
            bool(audit.get('current_evidence_passed')), saved)
        return final, plans, audit


def apply_current_plan(agent, bridge, result, state, spec):
    """在agent.perceive之后调用；接收同一物理快照的计划，绝不重置停车保护。"""
    selection = bridge.last_selection
    identity = agent.selected
    old_guard = agent.guard_latched

    def reject(reason):
        if identity is not None or agent.path_purpose == 'target_approach':
            agent.path = []
            agent.plan_target = None
            agent.goal = None
            agent.path_purpose = None
            result.update(path_world=[], path_purpose=None)
        result['current_plan_handoff'] = dict(applied=False, reason=reason,
            guard_latch_changed=False, actual_CARLA_validation_pending=True)
        assert agent.guard_latched == old_guard
        return result

    if identity is None:
        return reject('no_locked_target_search_policy_unchanged')
    if selection is None or selection.index != result['frame_idx'] or selection.state != asdict(state):
        return reject('different_or_missing_physical_snapshot')
    if (not selection.evidence_passed or identity != selection.authorized_id
            or identity not in selection.candidate_ids or identity not in selection.plans):
        return reject('current_target_not_authorized')
    plan = selection.plans[identity]
    node = agent.nodes.get(identity)
    if node is None or not plan['object_identity'] or node.metadata.get('object_identity') != plan['object_identity']:
        return reject('object_identity_changed')
    states = plan['states']
    if not states or plan['failure'] is not None or asdict(states[0]) != asdict(state):
        return reject('plan_does_not_start_at_current_state')
    if not all(math.isfinite(v) for s in states for v in asdict(s).values()):
        return reject('nonfinite_planned_state')
    goal, yaw = agent.approach_goal(identity)
    goal = np.asarray(goal, dtype=float)
    if (goal.shape != (2,) or not np.isfinite(goal).all() or not math.isfinite(yaw)
            or not np.array_equal(goal, plan['goal']) or yaw != plan['yaw']):
        return reject('observed_goal_changed_after_selection')
    end = states[-1]
    if math.dist((end.x, end.y), goal) > .65 or abs(angle_difference(end.yaw_rad, yaw)) > math.radians(12):
        return reject('plan_endpoint_no_longer_matches_observed_goal')
    if len(states) == 1 and state.speed_mps > .15:
        return reject('single_state_plan_cannot_stop_moving_vehicle')
    prior_progress = agent.line.project(Point(state.x, state.y))
    for i, s in enumerate(states):
        progress = agent.line.project(Point(s.x, s.y))
        a = agent.line.interpolate(max(0., progress - .2))
        b = agent.line.interpolate(min(agent.line.length, progress + .2))
        heading = math.atan2(b.y - a.y, b.x - a.x)
        if progress < prior_progress - .02 or abs(angle_difference(s.yaw_rad, heading)) > math.radians(60):
            return reject('directed_lane_progress_or_heading_invalid')
        prior_progress = progress
        shapes = [footprint_polygon(s, spec, .05)]
        if i:
            shapes.append(swept_body_polygon(states[i - 1], s, spec, .05))
        if any(not agent.road.covers(p) or any(p.intersects(o) for o in agent.obstacles) for p in shapes):
            return reject('current_obstacles_or_padded_body_invalid')
    audit = audit_executed_states(states, agent.road, agent.obstacles, spec)
    if not audit['passed']:
        return reject('independent_plan_audit_failed')
    previous_endpoint = agent.path[-1] if agent.path else None
    agent.path = [[s.x, s.y] for s in states]
    agent.goal = goal.copy()
    agent.plan_target = identity
    agent.path_purpose = 'target_approach'
    result.update(path_world=agent.path, path_purpose=agent.path_purpose,
        planning=dict(purpose='target_approach', expanded=plan['expanded'],
            duration_s=plan['duration_s'], failure=None, goal_xy=goal.tolist(),
            goal_yaw_rad=yaw, algorithm=plan['algorithm'],
            goal_source='same_frame_observed_target_approach_goal'))
    result['current_plan_handoff'] = dict(applied=True, reason='same_frame_authorized_plan_applied',
        selected_node_id=identity, previous_endpoint=previous_endpoint,
        new_endpoint=agent.path[-1], same_frame_index=selection.index,
        independent_audit_passed=True, padded_clearance_m=.05,
        guard_latch_changed=False, actual_CARLA_validation_pending=True)
    assert agent.guard_latched == old_guard
    return result
