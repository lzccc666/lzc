"""隔离重算真实屏障、旧路径、整车轨迹及实际排序投递；未暴露不是通过。"""
import argparse
from datetime import datetime, timezone
import inspect
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon

from conceptgraph.completion import realtime_navigation
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    base = OUTPUT/'faults/perceived_blockage_cpu_002'/args.condition
    dest = base/'independent_audit_003'
    dest.mkdir(exist_ok=False)
    try:
        complete = read(base/'complete.json')
        nav = base/'navigation'
        fixtures = read(base/'config_and_task.json')
        config = fixtures['config']
        initial = read(nav/'initial_state.json')
        execution, frames, events = lines(nav/'execution.jsonl'), lines(nav/'perception.jsonl'), lines(base/'events.jsonl')
        frozen = read(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json')
        geometry = frozen['vehicle']
        obstacles = [rectangle(t['center_xy'],t['yaw_rad'],t['length_m']/2,t['length_m']/2,t['width_m']) for t in config['targets']]
        barrier_file = base/'evaluation_only_dynamic_obstacle.json'
        barrier = Polygon(read(barrier_file)['polygon_xy']) if barrier_file.exists() else None
        static_body = audit_trajectory(initial, execution, geometry, Polygon(config['road_polygon_xy']), obstacles)
        body = audit_trajectory(initial, execution, geometry, Polygon(config['road_polygon_xy']),
                                obstacles+([barrier] if barrier is not None else []))
        direction = audit_execution_direction(initial, execution, config['centerline_xy'], geometry)
        dump(dest/'static_body.json', static_body)
        dump(dest/'barrier_conservative_all_time_body.json', body)
        dump(dest/'direction.json', direction)
        assert execution[-1]['state']['speed_mps'] == 0 and execution[-1]['stop_requested']
        assert complete['actual_frontend_calls'] == len(frames)
        expected = inspect.getsource(realtime_navigation.run_task).replace(
            "if final in {'ARRIVED','EXECUTION_FAILED'}:", "if final == 'EXECUTION_FAILED':").replace(
            "final = 'SAFETY_STOP'\n                    break", "final = 'SAFETY_STOP'")
        assert (base/'run_task_observation_source.py').read_text() == expected
        path_intersects = False
        center_path_intersects = False
        disconnected = False
        goal_body_impossible = False
        goal_body_proof = None

        def path_body_intersects(path):
            points = np.asarray(path, dtype=float)
            if len(points)<2:
                return False
            tangents = np.gradient(points,axis=0)
            for xy, tangent in zip(points,tangents):
                yaw = math.atan2(tangent[1],tangent[0])
                body = rectangle(xy, yaw, geometry['front_from_rear_axle_m'],
                                 geometry['rear_overhang_m'],geometry['width_m'])
                if body.intersects(barrier):
                    return True
            return False
        if barrier is not None:
            old = read(base/'old_target_path.json')
            center_path_intersects = LineString(old['path']).intersects(barrier)
            path_intersects = path_body_intersects(old['path'])
            truth = read(barrier_file)
            tangent = np.array([math.cos(truth['yaw_rad']),math.sin(truth['yaw_rad'])])
            normal = np.array([-tangent[1],tangent[0]])
            center = np.array(truth['center_xy'])
            vertices = np.asarray([p for region in fixtures['task']['regions'] for p in region['polygon_xy']])
            longitudinal = (vertices-center)@tangent
            lateral = (vertices-center)@normal
            heading_tolerance = math.radians(15)
            minimum_front = float(longitudinal.min()+geometry['front_from_rear_axle_m']*math.cos(heading_tolerance))
            maximum_rear = float(longitudinal.max()-geometry['rear_overhang_m']*math.cos(heading_tolerance))
            lateral_bound = float(np.abs(lateral).max()+math.hypot(geometry['front_from_rear_axle_m'],geometry['width_m']/2))
            aligned = all(abs(math.atan2(math.sin(region['heading_rad']-truth['yaw_rad']),
                math.cos(region['heading_rad']-truth['yaw_rad'])))<1e-10 for region in fixtures['task']['regions'])
            goal_body_impossible = bool(aligned and minimum_front > -1 and maximum_rear < 1 and lateral_bound < 50)
            goal_body_proof = dict(region_min_longitudinal=float(longitudinal.min()),
                region_max_longitudinal=float(longitudinal.max()),minimum_body_front_longitudinal=minimum_front,
                maximum_body_rear_longitudinal=maximum_rear,wall_longitudinal_interval=[-1,1],
                vehicle_and_region_lateral_bound_m=lateral_bound,wall_lateral_half_width_m=50,
                heading_tolerance_rad=heading_tolerance,region_and_wall_heading_aligned=aligned,
                all_region_poses_rectangles_intersect_wall=goal_body_impossible)
            stopped = read(barrier_file)['state']
            start = Point(stopped['x'], stopped['y'])
            goal = Point(old['goal'])
            free = Polygon(config['road_polygon_xy']).difference(barrier)
            components = list(free.geoms) if hasattr(free, 'geoms') else [free]
            disconnected = not any(p.covers(start) and p.covers(goal) for p in components)
        evidence = {}
        actual_barrier_pixels = []
        barrier_obstacle_frames = []
        after_birth_paths = []
        for frame in frames:
            assert frame['gt_sent_to_agent'] is False
            sensor = frame['sensor']
            for key in ['rgb','depth']:
                path = Path(sensor[key])
                assert digest(path) == sensor[key+'_sha256']
                evidence[str(path)] = digest(path)
            detection = nav/'detections'/f"{frame['frame_idx']:06d}.pkl.gz"
            assert digest(detection) == frame['detector_sha256']
            evidence[str(detection)] = digest(detection)
            depth = np.asarray(Image.open(sensor['depth']), dtype=float)/1000
            assert int(np.count_nonzero(depth)) == sensor['lidar_pixels']
            if barrier is not None and frame['frame_idx'] >= read(barrier_file)['frame_idx']:
                v,u = np.nonzero(depth)
                z = depth[v,u]
                xyz = np.column_stack([(u-320)*z/320,(v-180)*z/320,z])
                pose = np.asarray(sensor['camera_pose_cv'])
                world = xyz@pose[:3,:3].T+pose[:3,3]
                count = int(sum(bool(.25 < p[2] < 2.8 and barrier.distance(Point(p[:2])) <= .3) for p in world))
                actual_barrier_pixels.append(dict(frame=frame['frame_idx'], count=count))
                known = [Polygon(p) for p in frame['known_obstacle_polygons']]
                if any(p.intersects(barrier.buffer(.3)) for p in known):
                    barrier_obstacle_frames.append(frame['frame_idx'])
                after_birth_paths.append(dict(frame=frame['frame_idx'],
                    path_crosses_barrier=path_body_intersects(frame['path_world']),
                    candidate_checks=(frame['semantic_audit'] or {}).get('candidate_checks', [])))
        dump(dest/'sensor_and_detector_sha256.json', evidence)
        dump(dest/'barrier_sensor_and_path_details.json', dict(actual_barrier_pixels=actual_barrier_pixels,
            actual_obstacle_frames=barrier_obstacle_frames, paths=after_birth_paths))
        published = [e for e in events if e['kind']=='actual_lidar_obstacle_published']
        gap = [e for e in events if e['kind']=='observation_gap_begin']
        stopped_gap = [e for e in events if e['kind']=='gap_original_ttl_braking_stopped']
        gap_controls = []
        driver_start = read(nav/'driver_clock.json')['started_wall_s']
        if gap and stopped_gap:
            gap_controls = [e for e in execution if gap[0]['wall_s'] <= driver_start+e['wall_elapsed_s'] <= stopped_gap[0]['wall_s']]
        exposes = dict(physical_motion_before_fault=bool(gap and gap[0]['state']['speed_mps']>=.2),
            sensor_gap_braked_to_zero=bool(stopped_gap and gap_controls and any(x['stale_sensor_braking'] for x in gap_controls)),
            physical_barrier_created=barrier is not None,
            prior_target_path_blocked=bool(path_intersects), center_or_vehicle_goal_region_unreachable=bool(disconnected or goal_body_impossible),
            actual_LiDAR_barrier_points=any(x['count']>=2 for x in actual_barrier_pixels),
            measured_obstacle_published=bool(published and barrier_obstacle_frames))
        semantics = read(nav/'semantic_events.json')
        new_controls_after_rank = None
        ranking_valid = None
        context_rejected = None
        if args.condition == 'delayed_actual_rank':
            starts = [e for e in events if e['kind']=='actual_model_started']
            finishes = [e for e in events if e['kind']=='actual_model_finished']
            returned = [e for e in events if e['kind']=='actual_model_result_returned']
            assert len(starts) == complete['actual_CPU_Qwen_calls'] <= 1
            assert len(finishes) == complete['actual_CPU_Qwen_finished']
            model_audits = [read(p)['result'] for p in (nav/'qwen_audit').glob('*.json')]
            ranking_valid = bool(len(model_audits)==1 and model_audits[0]['output_schema_valid']
                                 and model_audits[0]['used_qwen'] and not model_audits[0]['fallback_used'])
            if starts:
                actual = read(base/'actual_model_text.json')
                assert model_audits[0]['raw_output'] == actual['raw_output']
                assert read(base/'actual_model.json')['devices'] == ['cpu']
            context_checks = [e for e in semantics['application_events'] if e['kind']=='ranking_context_checked']
            context_rejected = any(not e['applied'] and e['reason']=='candidate_or_geometry_or_query_changed' for e in context_checks)
            exposes.update(actual_CPU_rank_completed=len(finishes)==1,
                original_valid_rank_envelope=ranking_valid,
                actual_rank_delivered_after_observed_blockage=bool(returned and published and
                    returned[0]['wall_s']>=published[0]['wall_s']+1.9 and
                    any(e['kind']=='delivered' for e in semantics['events'])),
                changed_safe_context_rejected=bool(context_rejected))
            if returned:
                later = [e for e in execution if driver_start+e['wall_elapsed_s']>=returned[0]['wall_s']]
                new_controls_after_rank = len(later)
        else:
            assert complete['actual_CPU_Qwen_calls'] == 0
        late_safe = all(not e['path_crosses_barrier'] for e in after_birth_paths)
        body_passed = body['passed'] and static_body['passed']
        direction_passed = direction['passed'] is True or direction['moving_intervals']==0
        report = dict(passed=bool(all(exposes.values()) and body_passed and direction_passed and late_safe),
            updated_utc=datetime.now(timezone.utc).isoformat(), condition=args.condition,
            exposure_checks=exposes, all_required_fault_exposure=all(exposes.values()),
            old_rear_axle_centerline_intersects_barrier=bool(center_path_intersects),
            old_vehicle_path_intersects_barrier=bool(path_intersects),
            rear_axle_point_geometry_disconnected=bool(disconnected),
            all_acceptable_goal_body_poses_blocked=goal_body_impossible,
            independent_goal_body_slab_proof=goal_body_proof,
            continuous_body_including_barrier=body_passed, moving_direction=direction_passed,
            final_speed_zero=execution[-1]['state']['speed_mps']==0,
            all_completed_frames_clear_blocked_route=late_safe,
            semantic_rank_schema_valid=ranking_valid, semantic_context_rejected=context_rejected,
            actual_controls_after_rank_return=new_controls_after_rank,
            actual_controls=len(execution), actual_frontend_calls=len(frames),
            actual_CPU_Qwen_calls=complete['actual_CPU_Qwen_calls'], actual_GPU_calls=0,
            original_four_outcomes={f:complete['result'][f] for f in ['route_generated',
                'selected_object_evaluated_correct','stable_correct_lock','actual_full_perception_arrival']},
            result_is_development_combined_fault=True, formal_GPU_fault_gate=False,
            new_independent_tasks=0, actual_arrival_credit_added=0, goal_complete=False,
            source_sha256=digest(__file__), limitations=[
                '一次固定配置/任务/种子的开发组合故障；不宣称运动中突发障碍或全部动态交通安全',
                '先TTL间断停车再出现屏障；CPU TTL15秒不是正式GPU1.5秒',
                '屏障按全时段保守检查，独立逐区间整车审核覆盖其出生后的实际轨迹',
                '真实前端和真实模型分列，无GPU/人工标签/新封存或独占性能信用'])
        dump(dest/'report.json', report)
        print(json.dumps(report, ensure_ascii=False))
    except BaseException as error:
        dump(dest/'interrupted.json',dict(error=repr(error),goal_complete=False))
        raise


if __name__ == '__main__':
    main()
