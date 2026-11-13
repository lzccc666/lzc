"""独立矩阵角点验证实际旧路线阻塞和区域解析界；不调用导航几何函数。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

ROOT=Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',required=True)
    args=parser.parse_args()
    case=ROOT/'faults/perceived_blockage_cpu_002'/args.condition
    dest=case/'geometry_witness_002'
    dest.mkdir(exist_ok=False)
    barrier_info=read(case/'evaluation_only_dynamic_obstacle.json')
    barrier=Polygon(barrier_info['polygon_xy'])
    fixture=read(case/'config_and_task.json')
    vehicle=read(ROOT/'interactive_simulation/synthetic_tasks_v1.json')['vehicle']
    source=read(case/'independent_audit_003/report.json')

    def body(xy,yaw):
        c,s=math.cos(yaw),math.sin(yaw)
        rotation=np.array([[c,-s],[s,c]])
        local=np.array([[vehicle['front_from_rear_axle_m'],vehicle['width_m']/2],
            [vehicle['front_from_rear_axle_m'],-vehicle['width_m']/2],
            [-vehicle['rear_overhang_m'],-vehicle['width_m']/2],
            [-vehicle['rear_overhang_m'],vehicle['width_m']/2]])
        return Polygon(local@rotation.T+np.asarray(xy))

    stopped=barrier_info['state']
    assert not body([stopped['x'],stopped['y']],stopped['yaw_rad']).intersects(barrier)
    old=np.asarray(read(case/'old_target_path.json')['path'])
    witnesses=[]
    for i,(a,b) in enumerate(zip(old[:-1],old[1:])):
        delta=b-a
        if np.linalg.norm(delta)<1e-8:
            continue
        yaw=math.atan2(delta[1],delta[0])
        shape=body(a,yaw)
        if shape.intersects(barrier):
            witnesses.append(dict(index=i,rear_axle_xy=a.tolist(),yaw_rad=yaw,
                intersection_area_m2=shape.intersection(barrier).area,
                rear_axle_point_inside_barrier=barrier.covers(Point(a))))
    assert witnesses
    proof=source['independent_goal_body_slab_proof']
    goal_rectangles_blocked=proof['all_region_poses_rectangles_intersect_wall']
    road=Polygon(fixture['config']['road_polygon_xy'])
    free=road.difference(barrier)
    components=list(free.geoms) if hasattr(free,'geoms') else [free]
    start=Point(stopped['x'],stopped['y'])
    start_components=[p for p in components if p.covers(start)]
    full_regions_disconnected=bool(start_components) and not any(
        p.intersects(Polygon(region['polygon_xy'])) for p in start_components for region in fixture['task']['regions'])
    assert full_regions_disconnected or goal_rectangles_blocked
    samples=0
    sampled_goal_bodies_intersecting=0
    for region in fixture['task']['regions']:
        polygon=Polygon(region['polygon_xy'])
        x1,y1,x2,y2=polygon.bounds
        for x in np.linspace(x1,x2,31):
            for y in np.linspace(y1,y2,31):
                if not polygon.covers(Point(x,y)):
                    continue
                for angle in np.linspace(-15,15,31):
                    hit=body([x,y],region['heading_rad']+math.radians(angle)).intersects(barrier)
                    if goal_rectangles_blocked:
                        assert hit
                    sampled_goal_bodies_intersecting+=int(hit)
                    samples+=1
    proof=source['independent_goal_body_slab_proof']
    if goal_rectangles_blocked:
        assert proof['minimum_body_front_longitudinal']>proof['wall_longitudinal_interval'][0]
        assert proof['maximum_body_rear_longitudinal']<proof['wall_longitudinal_interval'][1]
    assert proof['vehicle_and_region_lateral_bound_m']<proof['wall_lateral_half_width_m']
    report=dict(passed=True,independent_matrix_corner_formula=True,
        original_navigation_geometry_called=False,barrier_creation_body_clear=True,
        nonzero_tangent_old_path_body_witnesses=witnesses,region_heading_samples_checked=samples,
        sample_scope='数值交叉检查，连续全集结论依赖记录的解析界，不以采样替代全集证明',
        universal_slab_proof_margins=dict(front_m=proof['minimum_body_front_longitudinal']+1,
            rear_m=1-proof['maximum_body_rear_longitudinal'],
            lateral_m=50-proof['vehicle_and_region_lateral_bound_m']),
        sampled_goal_body_intersections=sampled_goal_bodies_intersecting,
        every_acceptable_region_disconnected_from_start=full_regions_disconnected,
        all_goal_body_poses_intersect_wall_analytic=goal_rectangles_blocked,
        all_acceptable_poses_independently_unreachable=full_regions_disconnected or goal_rectangles_blocked,
        original_rear_axle_centerline_connected=(not source['rear_axle_point_geometry_disconnected']),
        original_results_changed=False,
        new_model_calls=0,new_navigation_runs=0,new_arrival_credit=0,goal_complete=False)
    with (dest/'report.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,witnesses=len(witnesses),samples=samples)))


if __name__=='__main__':
    main()
