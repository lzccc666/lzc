"""仅在新的交付诊断中保持感知接近点停车；不读环境身份或独立终点。"""
from copy import deepcopy
import math

from conceptgraph.completion.interactive_navigation import PerceptionVehicleAgent
from conceptgraph.completion.kinematic_benchmark import SPEC


class PerceivedStopCommitmentAgent(PerceptionVehicleAgent):
    """冻结主方法之外的诊断策略；停止后只能新建任务重置。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_successful_perceived_goal = None
        self.stop_commitment = None

    def perceive(self, *args, **kwargs):
        record = super().perceive(*args, **kwargs)
        plan = record['planning']
        if (plan and plan['purpose'] == 'target_approach' and not plan['failure']
                and len(record['path_world']) >= 2 and self.selected is not None):
            self.last_successful_perceived_goal = dict(node_id=self.selected,
                goal_xy=self.goal.tolist(), frame_idx=record['frame_idx'],
                source='successful plan to actual mapped LiDAR surface and declared vehicle prior')
        record['perceived_stop_commitment'] = deepcopy(self.stop_commitment)
        record['last_successful_perceived_goal'] = deepcopy(self.last_successful_perceived_goal)
        return record

    def control(self, state):
        previous = self.last_successful_perceived_goal
        if (self.stop_commitment is None and previous is not None
                and self.selected == previous['node_id']
                and math.dist([state.x, state.y], previous['goal_xy']) < .5
                and 0 <= state.speed_mps <= .15):
            # .5m来自既有跟踪器接近点停止阈值；.15m/s来自既有低速终止定义。
            # 是否正确到达仍由外部评价器判断，错误接近点也可能被保持。
            self.stop_commitment = dict(**deepcopy(previous), committed_at_sim_s=state.time_s,
                speed_mps_at_commit=state.speed_mps, reset_policy='new task only',
                independent_goal_or_truth_read=False)
        if self.stop_commitment is not None:
            return -SPEC.max_braking_mps2, state.steering_rad, dict(
                reason='perceived_goal_stop_commitment',
                commitment=deepcopy(self.stop_commitment),
                independent_goal_or_truth_read=False)
        return super().control(state)
