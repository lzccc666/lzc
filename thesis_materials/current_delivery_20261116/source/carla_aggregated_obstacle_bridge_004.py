"""几何并集减少规划重复碰撞查询，返回前仍按原逐个障碍作扫掠审计。"""
from types import SimpleNamespace

from shapely.ops import unary_union
from carla_persistent_intent_bridge_003 import PersistentIntentBridge
from conceptgraph.navigation.vehicle_geometry import audit_executed_states


class AggregatedObstacleBridge(PersistentIntentBridge):
    def current_plan(self, agent, identity, state):
        # 注入后端单元测试的自定义规划器保持原行为，不能冒充真实几何优化测试。
        if self.planner is not None:
            return super().current_plan(agent, identity, state)
        original = tuple(agent.obstacles)
        if any(not obstacle.is_valid for obstacle in original):
            self.plans.pop(identity, None)
            return None, dict(passed=False, reason='invalid_current_obstacle_geometry')
        combined = unary_union(original) if original else None
        proxy = SimpleNamespace(nodes=agent.nodes, road=agent.road, line=agent.line,
            approach_goal=agent.approach_goal, obstacles=[combined] if original else [])
        plan, check = super().current_plan(proxy, identity, state)
        check = dict(check, collision_query_representation='exact_union_of_all_current_obstacles',
            original_obstacle_count=len(original), original_agent_obstacles_unchanged=True,
            collision_padding_and_original_planning_budget_unchanged=True)
        if plan is not None:
            import carla_agent_isolated_002 as nav
            independent = audit_executed_states(plan.states, agent.road, original, nav.SPEC)
            check['original_individual_obstacles_sweep_audit'] = independent
            if not independent['passed']:
                self.plans.pop(identity, None)
                return None, dict(check, passed=False, reason='original_individual_obstacle_audit_failed')
        return plan, check
