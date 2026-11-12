"""实际墙钟控制心跳的断流/恢复及阻塞锁存功能实验；不加载感知或Qwen模型。"""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import time

from shapely.geometry import box

from conceptgraph.completion.realtime_navigation import WallClockVehicle
from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.stage9 import OUTPUT,digest,dump
from conceptgraph.navigation.vehicle_geometry import VehicleState,audit_executed_states


def main():
    dest=OUTPUT/'faults/realtime_control_fault_functional_001'
    dest.mkdir(exist_ok=False)
    begun=time.monotonic()
    source=Path(__file__)
    release=dict(source_sha256=digest(source),frozen_driver_sha256=digest(Path(__file__).resolve().parents[4]/'conceptgraph/completion/realtime_navigation.py'),
        maximum_wall_s=30,maximum_saved_bytes=1000000,model_calls=0,CUDA_used=False,
        conditions=['fresh_gap_fresh','blocked_packet_then_removed_obstacle'],sensor_ttl_s=.3,
        scope='已知道路与预置路径的实际控制功能实验；不是完整感知导航、性能或到达证据',
        vehicle_parameters=asdict(SPEC),start_states=['rest','1.6m/s'],arrival_credit=False,goal_complete=False)
    dump(dest/'release.json',release)
    road=box(-5,-5,60,5)
    def agent(obstacles=()):
        return SimpleNamespace(path=[(float(i),0.) for i in range(41)],goal=[40.,0.],
            selected=None,path_purpose='bounded_lane_search',road=road,obstacles=obstacles,guard_latched=False)
    def keep_fresh(driver,duration,packet):
        end=time.monotonic()+duration
        while time.monotonic()<end:
            driver.publish(packet,time.monotonic())
            time.sleep(.025)
    events=[]
    driver=WallClockVehicle(VehicleState(0,0,0),dest/'blackout_execution.jsonl',sensor_ttl_s=.3)
    try:
        keep_fresh(driver,1.8,agent())
        moving=driver.snapshot()
        assert moving.speed_mps>.5 and moving.x>.3
        events.append(dict(event='stop_publishing',wall_s=time.monotonic(),state=asdict(moving)))
        stopped_deadline=time.monotonic()+3.
        while time.monotonic()<stopped_deadline:
            time.sleep(.025)
            if driver.snapshot().speed_mps==0:
                break
        stopped=driver.snapshot()
        assert stopped.speed_mps==0
        with driver._lock:
            before_restore=len(driver.records)
        assert any(r['stale_sensor_braking'] and r['state']['speed_mps']>0 for r in driver.records)
        events.append(dict(event='stale_braking_stopped',wall_s=time.monotonic(),state=asdict(stopped)))
        keep_fresh(driver,1.2,agent())
        resumed=driver.snapshot()
        assert resumed.speed_mps>.5 and resumed.x>stopped.x+.1
        assert any(not r['stale_sensor_braking'] and r['state']['speed_mps']>0 for r in driver.records[before_restore:])
        events.append(dict(event='fresh_packet_motion_resumed',wall_s=time.monotonic(),state=asdict(resumed)))
    finally:
        driver.stop()
    first_audit=audit_executed_states(driver.states,road,[],SPEC)
    assert first_audit['passed']
    # 从已在运动的状态加入近处障碍，仍留足真正制动距离；不是把当前位置放进障碍。
    initial=VehicleState(0,0,0,speed_mps=1.6)
    obstacle=box(3.7+initial.speed_mps**2/4+.2,-3,5.5,3)
    driver2=WallClockVehicle(initial,dest/'blocked_execution.jsonl',sensor_ttl_s=.3)
    try:
        keep_fresh(driver2,1.2,agent([obstacle]))
        stopped_blocked=driver2.snapshot()
        assert driver2.guard_latched and stopped_blocked.speed_mps==0
        events.append(dict(event='blocked_path_latched_and_stopped',wall_s=time.monotonic(),state=asdict(stopped_blocked)))
        keep_fresh(driver2,.5,agent())
        assert driver2.snapshot().speed_mps==0 and driver2.guard_latched
        assert driver2.snapshot().x==stopped_blocked.x
        events.append(dict(event='fresh_packet_cannot_restore_latched_path',wall_s=time.monotonic(),state=asdict(driver2.snapshot())))
    finally:
        driver2.stop()
    second_audit=audit_executed_states(driver2.states,road,[obstacle],SPEC)
    assert second_audit['passed']
    dump(dest/'events.json',events)
    dump(dest/'independent_execution_audits.json',dict(blackout=first_audit,blocked=second_audit))
    assert time.monotonic()-begun<30 and sum(p.stat().st_size for p in dest.iterdir())<1000000
    report=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-begun,
        actual_control_ticks=[len(driver.records),len(driver2.records)],
        blackout_moving_before=moving.speed_mps,blackout_stopped_speed=stopped.speed_mps,
        blackout_resumed_speed=resumed.speed_mps,blocked_guard_latched=True,blocked_stopped_speed=stopped_blocked.speed_mps,
        unsafe_path_not_restored_by_fresh_packet=True,independent_execution_audits_passed=True,
        control_log_sha256={name:digest(dest/name) for name in ['blackout_execution.jsonl','blocked_execution.jsonl']},
        real_clock=True,model_calls=0,full_perception_fault_experiment=False,arrival_credit=False,
        limitations=['已知直道路面、预置搜索路径、静态障碍','TTL .3s是功能试验值，正式感知矩阵1.5s不变',
            '共享主机墙钟事件只验证功能，不是独占实时性能','没有真实感知输入或Qwen，不能作为全链故障验收'],goal_complete=False)
    dump(dest/'report.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
