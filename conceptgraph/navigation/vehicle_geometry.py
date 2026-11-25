"""低速前向车辆模型、矩形车身与保守连续扫掠检查，坐标原点为后轴中心。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class VehicleSpec:
    wheelbase_m: float = 2.7
    front_from_rear_axle_m: float = 3.7
    rear_overhang_m: float = .9
    width_m: float = 1.85
    max_speed_mps: float = 2.0
    max_acceleration_mps2: float = 1.0
    max_braking_mps2: float = 2.0
    max_steering_rad: float = math.radians(32)
    max_steering_rate_radps: float = math.radians(25)
    dt_s: float = .1

    def __post_init__(self):
        values=list(vars(self).values())
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('vehicle parameters must be finite and positive')
        if self.max_steering_rad>=math.pi/2 or self.wheelbase_m>=self.front_from_rear_axle_m:
            raise ValueError('invalid vehicle geometry or steering')

    @property
    def minimum_turn_radius_m(self):
        return self.wheelbase_m/math.tan(self.max_steering_rad)

    @property
    def length_m(self):
        return self.front_from_rear_axle_m+self.rear_overhang_m


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw_rad: float
    speed_mps: float = 0.0
    steering_rad: float = 0.0
    time_s: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()) or self.speed_mps<0:
            raise ValueError('invalid vehicle state')


def angle_difference(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi


def step_vehicle(state, acceleration_mps2, steering_command_rad, spec=VehicleSpec()):
    """限幅后的真实状态更新；不读取计划路径或直接设置位置。"""
    if not math.isfinite(acceleration_mps2) or not math.isfinite(steering_command_rad):
        raise ValueError('nonfinite control input')
    if state.speed_mps>spec.max_speed_mps+1e-9 or abs(state.steering_rad)>spec.max_steering_rad+1e-9:
        raise ValueError('initial state exceeds vehicle limits')
    accel=float(np.clip(acceleration_mps2,-spec.max_braking_mps2,spec.max_acceleration_mps2))
    speed=float(np.clip(state.speed_mps+accel*spec.dt_s,0,spec.max_speed_mps))
    target=float(np.clip(steering_command_rad,-spec.max_steering_rad,spec.max_steering_rad))
    steering=state.steering_rad+float(np.clip(target-state.steering_rad,
        -spec.max_steering_rate_radps*spec.dt_s,spec.max_steering_rate_radps*spec.dt_s))
    velocity=(state.speed_mps+speed)/2
    delta=(state.steering_rad+steering)/2
    omega=velocity*math.tan(delta)/spec.wheelbase_m
    change=omega*spec.dt_s
    if abs(omega)<1e-10:
        dx=velocity*spec.dt_s*math.cos(state.yaw_rad)
        dy=velocity*spec.dt_s*math.sin(state.yaw_rad)
    else:
        dx=velocity/omega*(math.sin(state.yaw_rad+change)-math.sin(state.yaw_rad))
        dy=velocity/omega*(-math.cos(state.yaw_rad+change)+math.cos(state.yaw_rad))
    return VehicleState(state.x+dx,state.y+dy,angle_difference(state.yaw_rad+change,0),
        speed,steering,state.time_s+spec.dt_s)


def footprint_vertices(state,spec=VehicleSpec(),padding_m=0):
    if not math.isfinite(padding_m) or padding_m<0:
        raise ValueError('invalid padding')
    rear=-spec.rear_overhang_m-padding_m
    front=spec.front_from_rear_axle_m+padding_m
    half=spec.width_m/2+padding_m
    local=np.asarray([[rear,-half],[front,-half],[front,half],[rear,half]])
    c,s=math.cos(state.yaw_rad),math.sin(state.yaw_rad)
    return local@np.array([[c,s],[-s,c]])+np.array([state.x,state.y])


def footprint_polygon(state,spec=VehicleSpec(),padding_m=0):
    return Polygon(footprint_vertices(state,spec,padding_m))


def swept_body_polygon(before,after,spec=VehicleSpec(),padding_m=0):
    """相邻积分步的车身凸包加圆弧偏离上界，保守覆盖连续车身。"""
    angle=abs(angle_difference(after.yaw_rad,before.yaw_rad))
    if angle>math.pi/2:
        raise ValueError('sweep interval too large; subdivide state history')
    chord=math.hypot(after.x-before.x,after.y-before.y)
    radius=chord/(2*math.sin(angle/2)) if angle>1e-8 else 0
    body_radius=math.hypot(spec.front_from_rear_axle_m+padding_m,spec.width_m/2+padding_m)
    bulge=(radius+body_radius)*(1-math.cos(angle/2))
    return unary_union([footprint_polygon(before,spec,padding_m),
        footprint_polygon(after,spec,padding_m)]).convex_hull.buffer(bulge+1e-8)


def body_is_legal(state,drivable_polygon,obstacles,spec=VehicleSpec(),padding_m=.05):
    body=footprint_polygon(state,spec,padding_m)
    return bool(drivable_polygon.covers(body) and all(not body.intersects(o) for o in obstacles))


def audit_executed_states(states,drivable_polygon,obstacles,spec=VehicleSpec()):
    """独立检查实际控制日志，不读取规划器success、路径碰撞标记或终止状态。"""
    if not states:
        return dict(passed=False,reason='no_executed_states',audited_intervals=0)
    violations=[]
    for i,state in enumerate(states):
        if not body_is_legal(state,drivable_polygon,obstacles,spec,padding_m=0):
            violations.append(dict(index=i,reason='body_collision_or_boundary'))
        if state.speed_mps>spec.max_speed_mps+1e-8 or abs(state.steering_rad)>spec.max_steering_rad+1e-8:
            violations.append(dict(index=i,reason='state_limits'))
        if not i:
            continue
        before=states[i-1]
        dt=state.time_s-before.time_s
        if abs(dt-spec.dt_s)>1e-7:
            violations.append(dict(index=i,reason='time_step'))
            continue
        if abs(state.steering_rad-before.steering_rad)>spec.max_steering_rate_radps*dt+1e-8:
            violations.append(dict(index=i,reason='steering_rate'))
        accel=(state.speed_mps-before.speed_mps)/dt
        if not -spec.max_braking_mps2-1e-8<=accel<=spec.max_acceleration_mps2+1e-8:
            violations.append(dict(index=i,reason='acceleration'))
        displacement=math.hypot(state.x-before.x,state.y-before.y)
        travel=(state.speed_mps+before.speed_mps)/2*dt
        if displacement>travel+1e-6:
            violations.append(dict(index=i,reason='position_jump'))
        dyaw=abs(angle_difference(state.yaw_rad,before.yaw_rad))
        if dyaw>travel/spec.minimum_turn_radius_m+1e-7:
            violations.append(dict(index=i,reason='curvature'))
        signed_yaw=angle_difference(state.yaw_rad,before.yaw_rad)
        expected_yaw=travel*math.tan((state.steering_rad+before.steering_rad)/2)/spec.wheelbase_m
        if abs(signed_yaw-expected_yaw)>1e-7:
            violations.append(dict(index=i,reason='yaw_control_residual'))
        chord_length=travel*(math.sin(signed_yaw/2)/(signed_yaw/2) if abs(signed_yaw)>1e-8 else 1)
        chord_heading=before.yaw_rad+signed_yaw/2
        expected=np.array([math.cos(chord_heading),math.sin(chord_heading)])*chord_length
        observed=np.array([state.x-before.x,state.y-before.y])
        if np.linalg.norm(expected-observed)>1e-6:
            violations.append(dict(index=i,reason='translation_control_residual'))
        swept=swept_body_polygon(before,state,spec)
        if not drivable_polygon.covers(swept) or any(swept.intersects(o) for o in obstacles):
            violations.append(dict(index=i,reason='swept_collision_or_boundary'))
    return dict(passed=not violations,violations=violations,audited_states=len(states),
        audited_intervals=len(states)-1,geometry='rear_axle_rectangular_body_and_conservative_continuous_sweep',
        dynamic_safety_claim=False)
