"""在已关闭的真实LiDAR回合上离线诊断候选保护器；不授予碰撞改善信用。"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import Polygon

from carla_motion_guard_001 import current_full_lidar_obstacles,nominal_coasting_check
from conceptgraph.completion.stage9 import digest,dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState

R=Path(__file__).resolve().parents[1]


def main():
    out=R/'carla_official/motion_guard_replay_001'
    out.mkdir(exist_ok=False)
    test=R/'source/test_carla_motion_guard_001.py'
    spec=importlib.util.spec_from_file_location('motion_guard_tests',test)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    (out/'unittest.log').write_text(stream.getvalue())
    assert result.wasSuccessful()
    allrows=[]
    frozen={str(p):digest(p) for p in [Path(__file__),R/'source/carla_motion_guard_001.py',test]}
    for mode in ['rule','base','adapter']:
        e=R/'carla_official/complex_semantic_navigation_007/complex_second_nearest20'/mode
        release=json.loads((e/'release.json').read_text())
        road=Polygon(release['prior']['road_polygon_xy'])
        vehicle=VehicleSpec(**release['vehicle_spec'])
        execution={r['frame']:r for r in map(json.loads,(e/'execution.jsonl').open())}
        perceptions=list(map(json.loads,(e/'perception.jsonl').open()))
        frozen.update({str(p):digest(p) for p in [e/'release.json',e/'execution.jsonl',e/'perception.jsonl']})
        goal=None
        for i,p in enumerate(perceptions):
            if p['planning'] is not None:goal=p['planning']['goal_xy']
            if i%10 and i!=len(perceptions)-1:continue
            current=execution[p['sensor_frame']]
            state=VehicleState(**current['state'])
            raw=e/'frames'/f"{p['frame_idx']:06d}"/'lidar.bin'
            points=np.fromfile(raw,np.float32).reshape(-1,4)
            frozen[str(raw)]=digest(raw)
            obstacles,observation=current_full_lidar_obstacles(points,p['lidar_pose_carla'],state,
                vehicle,sensor_frame=p['sensor_frame'],snapshot_frame=current['snapshot_frame'],ground_z=0.)
            assert observation['valid']
            guard=nominal_coasting_check(state,road,obstacles,vehicle,observation=observation,
                path=p['path_world'],goal=goal,passive_deceleration_mps2=.03,reaction_s=.3)
            allrows.append(dict(mode=mode,frame_idx=p['frame_idx'],time_s=p['sim_time_s'],
                speed_mps=state.speed_mps,existing_path_points=len(p['path_world']),
                full_LiDAR_points=len(points),observed_obstacle_clusters=len(obstacles),
                nominal_stop_required=guard['stop_required'],reason=guard['reason'],
                predicted_travel_m=guard.get('predicted_travel_m'),
                actual_original_control_reason=current['tracking'].get('reason'),
                actual_applied_throttle=current['applied_control']['throttle'],
                actual_applied_brake=current['applied_control']['brake'],
                existing_collision=bool(current['collisions'])))
    with (out/'observations.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    summaries=[]
    for mode in ['rule','base','adapter']:
        rows=[r for r in allrows if r['mode']==mode]
        waiting=[r for r in rows if not r['existing_path_points'] and r['speed_mps']>.15]
        stops=[r for r in rows if r['nominal_stop_required']]
        summaries.append(dict(mode=mode,replayed_observations=len(rows),nominal_stop_checks=len(stops),
            moving_without_route=len(waiting),moving_without_route_stop_checks=sum(r['nominal_stop_required'] for r in waiting),
            first_nominal_stop_time_s=stops[0]['time_s'] if stops else None,
            original_task_unchanged=True,actual_new_physical_executions=0))
    assert all(digest(p)==h for p,h in frozen.items())
    dump(out/'report.json',dict(tests_run=result.testsRun,tests_passed=result.wasSuccessful(),
        source_and_inputs_sha256=frozen,summaries=summaries,
        actual_replayed_existing_LiDAR_frames=len(allrows),actual_new_GPU_calls=0,
        actual_new_CARLA_episodes=0,physical_collision_improvement_proven=False,
        declared_passive_deceleration_mps2=.03,declared_reaction_s=.3,
        limitation='候选滑行模型及平坦道路高度先验尚未通过新停车标定与完整物理任务。离线发出停车建议不等于实际避免碰撞。',
        goal_complete=False))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
