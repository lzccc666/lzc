"""独立任务真值与终点评价；不依赖规划器自行报告的success。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from shapely.geometry import Point, Polygon


class TaskStatus(str, Enum):
    NOT_YET_OBSERVED = 'NOT_YET_OBSERVED'
    OCCLUDED = 'OCCLUDED'
    NO_MATCH_IN_SCOPE = 'NO_MATCH_IN_SCOPE'
    INACCESSIBLE = 'INACCESSIBLE'
    INSUFFICIENT_EVIDENCE = 'INSUFFICIENT_EVIDENCE'
    WAITING = 'WAITING'
    PLANNING_FAILED = 'PLANNING_FAILED'
    EXECUTION_FAILED = 'EXECUTION_FAILED'
    ARRIVED = 'ARRIVED'
    TIMEOUT = 'TIMEOUT'


@dataclass(frozen=True)
class GoalTruth:
    """独立标注，必须在方法运行前生成；每个区域绑定一个允许的对象身份。"""
    task_id: str
    accepted_instance_ids: tuple[str, ...]
    regions: tuple[dict, ...]
    spatial_scope: str
    time_scope_s: tuple[float, float]
    provenance: str
    reachability: str = 'unknown'
    max_speed_mps: float = 0.15
    max_heading_error_rad: float = math.radians(15)
    hold_s: float = 2.0
    timeout_s: float = 90.0

    def __post_init__(self):
        if self.reachability not in {'reachable', 'inaccessible', 'unknown'}:
            raise ValueError('可达性必须独立声明')
        if not self.provenance or not self.spatial_scope:
            raise ValueError('缺少独立证据或空间范围')
        if self.hold_s <= 0 or self.timeout_s <= self.hold_s:
            raise ValueError('非法到达时间阈值')
        for region in self.regions:
            if region['instance_id'] not in self.accepted_instance_ids:
                raise ValueError('终点区域必须绑定可接受对象')
            polygon = Polygon(region['polygon_xy'])
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError('终点区域无效')


class ArrivalEvaluator:
    """按实际车辆状态和独立身份关联判定；任何不满足均重置保持时间。"""
    def __init__(self, truth: GoalTruth):
        self.truth = truth
        self.started_at = None
        self.last_time = None
        self.hold_started_at = None
        self.safety_violation = False
        self.arrived = False

    def update(self, *, time_s, xy, yaw_rad, speed_mps,
               evaluated_instance_id, body_collision=False, lane_violation=False):
        values = [time_s, *xy, yaw_rad, speed_mps]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('执行状态含非有限值')
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError('执行时钟必须严格递增')
        if self.started_at is None:
            self.started_at = time_s
        self.last_time = time_s
        self.safety_violation |= body_collision or lane_violation
        identity_ok = evaluated_instance_id in self.truth.accepted_instance_ids
        in_region = heading_ok = False
        for region in self.truth.regions:
            if region['instance_id'] != evaluated_instance_id:
                continue
            if Polygon(region['polygon_xy']).covers(Point(*xy)):
                in_region = True
                target_heading = region.get('heading_rad')
                heading_ok |= target_heading is None or abs(math.atan2(
                    math.sin(yaw_rad-target_heading), math.cos(yaw_rad-target_heading)
                )) <= self.truth.max_heading_error_rad
        valid = (identity_ok and in_region and heading_ok
                 and abs(speed_mps) <= self.truth.max_speed_mps
                 and not self.safety_violation
                 and self.truth.reachability == 'reachable')
        if valid:
            if self.hold_started_at is None:
                self.hold_started_at = time_s
        else:
            self.hold_started_at = None
        held_s = 0.0 if self.hold_started_at is None else time_s-self.hold_started_at
        elapsed_s = time_s-self.started_at
        self.arrived = bool(valid and held_s >= self.truth.hold_s
                            and elapsed_s <= self.truth.timeout_s)
        status = (TaskStatus.EXECUTION_FAILED if self.safety_violation
                  else TaskStatus.TIMEOUT if elapsed_s > self.truth.timeout_s
                  else TaskStatus.ARRIVED if self.arrived else TaskStatus.WAITING)
        return dict(status=status.value, correct_identity=identity_ok,
                    inside_independent_region=in_region, heading_ok=heading_ok,
                    stopped=abs(speed_mps) <= self.truth.max_speed_mps,
                    safe_so_far=not self.safety_violation, held_s=held_s,
                    actually_arrived=self.arrived)
