"""在原控制器返回之后始终核验当前运动，覆盖无路线时的提前返回。"""
from carla_motion_guard_001 import nominal_coasting_check


def guarded_control(agent, state, spec, acceleration, steering, tracking):
    check = nominal_coasting_check(state, agent.road, agent.obstacles, spec,
        observation=agent.current_lidar_observation, path=agent.path, goal=agent.goal,
        passive_deceleration_mps2=.03, reaction_s=.3)
    # 切油时下一时刻保持当前轮角；同时检查这一条名义滑行轨迹。
    coast = nominal_coasting_check(state, agent.road, agent.obstacles, spec,
        observation=agent.current_lidar_observation, path=None, goal=None,
        passive_deceleration_mps2=.03, reaction_s=.3)
    stop = check['stop_required'] or coast['stop_required']
    details = dict(always_executed=True, route_following_forecast=check,
        current_steering_coasting_forecast=coast, stop_required=stop,
        actual_physical_safety_proven=False)
    tracking = dict(tracking, always_on_motion_guard=details)
    if stop:
        return -spec.max_braking_mps2, state.steering_rad, dict(tracking,
            desired_speed_mps=0., emergency_braking=True,
            current_motion_guard_stop=True)
    return acceleration, steering, tracking
