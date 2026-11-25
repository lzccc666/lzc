"""原厂Prius切油/低速制动的名义接近速度包络；开发候选，未部署旧运行。"""
from dataclasses import dataclass
import math

from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class StockStoppingProfile:
    # .03/.15/.3继承冻结整圈LiDAR滑行保护；.8来自原tracking_control的减速设计。
    # 两条官方标定只支持局部检验，不能把这些数称作车辆物理保证。
    coast_deceleration_mps2: float = .03
    brake_gate_mps: float = .15
    nominal_low_speed_deceleration_mps2: float = .8
    reaction_s: float = .3
    distance_reserve_m: float = .3
    cruise_cap_mps: float = .55

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('positive finite stopping parameters required')

    def distance(self, measured_speed_mps):
        """使用实际速度，不能先按期望巡航速度截断，以免低估停车距离。"""
        v = float(measured_speed_mps)
        if not math.isfinite(v) or v < 0:
            raise ValueError('finite nonnegative measured speed required')
        low = min(v, self.brake_gate_mps)
        coast = max(0., v*v - self.brake_gate_mps**2) / (2*self.coast_deceleration_mps2)
        brake = low*low / (2*self.nominal_low_speed_deceleration_mps2)
        return v*self.reaction_s + coast + brake + self.distance_reserve_m

    def speed_cap(self, remaining_m):
        """反解同一个停车距离函数；不扩大目标区，不修改终点判定速度。"""
        remaining = float(remaining_m)
        if not math.isfinite(remaining):
            raise ValueError('finite remaining distance required')
        if remaining <= self.distance_reserve_m:
            return 0.
        lo, hi = 0., self.cruise_cap_mps
        for _ in range(50):
            middle = (lo + hi) / 2
            if self.distance(middle) <= remaining:
                lo = middle
            else:
                hi = middle
        return lo


def observed_approach_profile(state, path, goal, tracking, profile=StockStoppingProfile()):
    """只给出本帧观测路径的速度建议；不释放停车锁、不改变转向或安全门闸。"""
    if len(path) < 2 or goal is None:
        return dict(tracking, desired_speed_mps=0., stopping_profile_reason='missing_observed_route')
    line = LineString(path)
    if line.is_empty or not line.is_valid or line.length <= 0:
        return dict(tracking, desired_speed_mps=0., stopping_profile_reason='invalid_observed_route')
    progress = line.project(Point(state.x, state.y))
    path_remaining = max(0., line.length - progress)
    remaining = min(path_remaining, math.dist((state.x, state.y), goal))
    cap = profile.speed_cap(remaining)
    original = float(tracking.get('desired_speed_mps', 0.))
    if not math.isfinite(original) or original < 0:
        raise ValueError('finite nonnegative existing speed request required')
    return dict(tracking, desired_speed_mps=min(original, cap),
        stock_stopping_profile=dict(
            remaining_observed_distance_m=remaining,
            nominal_stopping_distance_m=profile.distance(state.speed_mps),
            nominal_speed_cap_mps=cap,
            current_speed_exceeds_nominal_envelope=state.speed_mps > cap,
            parameters=vars(profile), safety_guarantee=False,
            current_frame_forecast_only=True, official_closed_loop_validation_pending=True))
