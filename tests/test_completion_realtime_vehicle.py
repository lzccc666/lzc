import time
from types import SimpleNamespace

from shapely.geometry import box
from shapely import is_prepared
from shapely.prepared import prep

from conceptgraph.completion.realtime_navigation import WallClockVehicle
from conceptgraph.navigation.vehicle_geometry import VehicleState, audit_executed_states


def test_expired_sensor_plan_cannot_accelerate_and_actual_vehicle_stops(tmp_path):
    vehicle = WallClockVehicle(VehicleState(0,0,0,speed_mps=1),tmp_path/'execution.jsonl',sensor_ttl_s=.01)
    road = box(-20,-20,50,20)
    source = SimpleNamespace(path=[[0,0],[30,0]],goal=[30,0],selected=7,
        path_purpose='target_approach',road=road,obstacles=[],guard_latched=False)
    try:
        # 虽然规划刚交付，其来源观测已经过期；不能以交付时间刷新TTL。
        vehicle.publish(source,time.monotonic()-1)
        time.sleep(.6)
    finally:
        vehicle.stop()
    assert vehicle.snapshot().speed_mps == 0
    assert all(row['control']['acceleration_mps2'] < 0 for row in vehicle.records)
    assert all(row['stale_sensor_braking'] for row in vehicle.records)
    assert .15 < vehicle.snapshot().x < .35
    assert audit_executed_states(vehicle.states,road,[])['passed']


def test_new_packet_cannot_clear_executor_safety_latch(tmp_path):
    vehicle = WallClockVehicle(VehicleState(0,0,0),tmp_path/'execution.jsonl',sensor_ttl_s=2.)
    source = SimpleNamespace(path=[[0,0],[30,0]],goal=[30,0],selected=7,
        path_purpose='target_approach',road=box(-.1,-.1,.1,.1),obstacles=[],guard_latched=False)
    try:
        vehicle.publish(source,time.monotonic())
        time.sleep(.15)
        assert vehicle.guard_latched
        source.road = box(-20,-20,50,20)
        vehicle.publish(source,time.monotonic())
        time.sleep(.15)
        assert vehicle.snapshot().speed_mps==0
        assert vehicle.records[-1]['tracking']['reason']=='safety_guard_latched'
    finally:
        vehicle.stop()


def test_executor_owns_geometry_while_planner_prepares_source(tmp_path):
    vehicle = WallClockVehicle(VehicleState(0,0,0),tmp_path/'execution.jsonl',sensor_ttl_s=3.)
    source = SimpleNamespace(path=[[0,0],[30,0]],goal=[30,0],selected=7,
        path_purpose='target_approach',road=box(-20,-20,50,20),
        obstacles=[box(40,8,41,9)],guard_latched=False)
    try:
        vehicle.publish(source,time.monotonic())
        packet = vehicle.packet[0]
        assert packet.road is not source.road
        assert packet.obstacles[0] is not source.obstacles[0]
        assert packet.road.equals_exact(source.road,0)
        assert packet.obstacles[0].equals_exact(source.obstacles[0],0)
        prep(source.road)
        prep(source.obstacles[0])
        assert is_prepared(source.road) and is_prepared(source.obstacles[0])
        assert not is_prepared(packet.road) and not is_prepared(packet.obstacles[0])
        # 真实控制线程继续运行时，规划端重复准备和查询自己的GEOS对象。
        for _ in range(500):
            assert prep(source.road).covers(box(-1,-1,1,1))
        time.sleep(.2)
    finally:
        vehicle.stop()
    assert vehicle.error is None
    assert vehicle.snapshot().x > 0
    assert audit_executed_states(vehicle.states,source.road,source.obstacles)['passed']


def test_fast_perception_waits_for_actual_control_tick_without_faking_timestamp(tmp_path):
    vehicle = WallClockVehicle(VehicleState(0,0,0),tmp_path/'execution.jsonl')
    try:
        state = vehicle.snapshot()
        following = vehicle.snapshot_after(state.time_s,timeout_s=1.)
        assert following is not None and following.time_s > state.time_s
        assert following in vehicle.states
        # 同一控制周期内不得重复返回旧状态，0秒预算不制造新时间戳。
        with vehicle._lock:
            previous = vehicle.state.time_s
        result = vehicle.snapshot_after(previous,timeout_s=0.)
        assert result is None or result.time_s > previous
    finally:
        vehicle.stop()
    assert vehicle.snapshot_after(vehicle.snapshot().time_s,timeout_s=.1) is None
