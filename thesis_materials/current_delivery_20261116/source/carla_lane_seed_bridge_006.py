"""已声明道路只提供候选形状；从当前车辆状态积分并逐障碍审计后才授权。"""
import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from carla_aggregated_obstacle_bridge_005 import AggregatedObstacleBridge
from conceptgraph.completion.planning_reuse import revalidate_route
from conceptgraph.navigation.vehicle_geometry import audit_executed_states


class LaneSeedBridge(AggregatedObstacleBridge):
    def current_plan(self, agent, identity, state):
        if self.planner is not None:
            return super().current_plan(agent, identity, state)
        import carla_agent_isolated_002 as nav
        original = tuple(agent.obstacles)
        if any(not p.is_valid for p in original):
            return super().current_plan(agent, identity, state)
        goal, yaw = agent.approach_goal(identity)
        begin = agent.line.project(Point(state.x, state.y))
        end = agent.line.project(Point(goal))
        reason = 'goal_not_ahead_in_declared_lane'
        if end > begin:
            path = [list(agent.line.interpolate(s).coords[0]) for s in
                    np.linspace(begin, end, max(2, math.ceil((end-begin)/.25)+1))]
            combined = [unary_union(original)] if original else []
            plan, reason = revalidate_route(state, path, goal, yaw, agent.road,
                combined, agent.line, nav.SPEC, budget_s=1.)
            if plan is not None:
                audit = audit_executed_states(plan.states, agent.road, original, nav.SPEC)
                if audit['passed']:
                    plan.algorithm = 'current_state_vehicle_tracking_of_declared_lane_seed'
                    self.plans[identity] = dict(
                        identity=agent.nodes[identity].metadata.get('object_identity'),
                        goal=np.asarray(goal).copy(), yaw=yaw,
                        path=[[s.x, s.y] for s in plan.states])
                    return plan, dict(passed=True, route_reused=False,
                        route_seed='declared_lane_not_target_truth',
                        route_seed_budget_s=1., failure=None,
                        original_individual_obstacles_sweep_audit=audit,
                        original_obstacle_count=len(original),
                        original_agent_obstacles_unchanged=True)
                reason = 'original_obstacle_sweep_rejected_lane_seed'
        plan, check = super().current_plan(agent, identity, state)
        return plan, dict(check, lane_seed_attempt_reason=reason,
                          lane_seed_attempt_budget_s=1.)
