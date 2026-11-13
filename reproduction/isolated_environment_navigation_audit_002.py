"""独立核验隔离源码实际执行、整车连续安全与正确身份的保守到达保持区间。"""
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import pickle
import time

import networkx as nx
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def associations(nodes, targets):
    graph = nx.Graph()
    for node in nodes:
        for target in targets:
            distance = math.dist(node['center'][:2], target['center_xy'])
            if node['category'] == target['category'] and distance <= 2.5:
                graph.add_edge(('node', node['node_id']), ('truth', target['instance_id']), weight=-distance)
    pairs = nx.max_weight_matching(graph, maxcardinality=True)
    result = {}
    for a, b in pairs:
        if a[0] != 'node':
            a, b = b, a
        result[a[1]] = b[1]
    return result


def main():
    started = time.monotonic()
    base = OUTPUT/'reproduction/isolated_cpu_executed_001'
    manifest = read(base/'manifest.json')
    root = Path(manifest['isolated_root'])
    output = root/'outputs/project_v2/thesis_completion_v1'
    case = output/'interactive_simulation/live_navigation_sessions_001'/manifest['session']
    destination = base/'independent_audit_001'
    destination.mkdir(exist_ok=False)
    try:
        result, imports = read(case/'result.json'), read(case/'import_audit.json')
        assert imports['passed'] and imports['all_conceptgraph_modules_from_isolated_root']
        assert imports['original_project_removed_from_search_path'] and not imports['CUDA_initialized']
        assert imports['Python_environment_reused'] is False
        assert imports['new_Python_environment'] and imports['clean_OS'] is False
        prefix = Path(manifest['environment_prefix']).resolve()
        assert prefix == Path(imports['environment_prefix']).resolve()
        assert Path(imports['interpreter']).resolve().is_relative_to(prefix)
        assert imports['original_site_packages_absent'] and imports['editable_finders_absent']
        assert imports['CUDA_build'] == '11.8' and imports['CPU_torch_version'] == '2.0.1+cu118'
        installed_path = Path(manifest['environment_installed_manifest'])
        assert digest(installed_path) == manifest['environment_installed_manifest_sha256']
        installed = read(installed_path)
        assert installed['installed'] and Path(installed['environment']).resolve() == prefix
        assert len(imports['verified_dependency_modules']) == 14
        for record in imports['verified_dependency_modules'].values():
            path = Path(record['path']).resolve()
            assert path.is_relative_to(prefix) and digest(path) == record['sha256']
        thread_policy_passed = imports['torch_threads'] == 1
        assert thread_policy_passed
        extension = Path(manifest['extension_path'])
        assert extension.resolve().is_relative_to(root)
        assert digest(extension) == imports['extension_sha256'] == manifest['extension_sha256']
        assert manifest['posthoc_stop_commitment_diagnostic']
        threads = lines(case/'thread_observations.jsonl')
        assert threads == imports['actual_frontend_thread_observations']
        assert len(threads) == result['actual_frontend_calls']
        for i, observation in enumerate(threads):
            assert observation['call_index'] == i
            assert observation['torch_threads_before'] == observation['torch_threads_after'] == 1
            assert not observation['CUDA_initialized']

        assert imports['clean_OS_or_new_environment']
        for name, record in imports['modules'].items():
            path = Path(record['path'])
            assert path.resolve().is_relative_to(root) and digest(path) == record['sha256'], name
        for name, expected in manifest['source_sha256'].items():
            assert digest(root/name) == digest(PROJECT/name) == expected, name
        assert manifest['original_index_sha256'] is None
        portable_source = Path(manifest['original_project'])
        assert not (portable_source/'outputs').exists() and not (portable_source/'.git').exists()
        assert manifest['original_historical_outputs_required'] is False
        frozen_checkpoint = read(OUTPUT/'source_bundle/source_checkpoint_034/checkpoint.json')
        assert digest(PROJECT/'.git/index') == frozen_checkpoint['original_index_sha256']
        for name, expected in manifest['source_sha256'].items():
            assert digest(portable_source/name) == expected, name

        task_path = output/'interactive_simulation/synthetic_tasks_v1.json'
        assert digest(task_path) == manifest['task_release_sha256']
        frozen = read(task_path)
        task = next(t for t in frozen['tasks'] if t['task_id'] == manifest['task_id'])
        config = next(c for c in frozen['configs'] if c['config_id'] == task['config_id'])
        assert task['task_id'] == 'config00-positive0' and manifest['seed'] == 17
        assert task['kind'] == 'reachable_positive'
        initial, perception, execution = read(case/'initial_state.json'), lines(case/'perception.jsonl'), lines(case/'execution.jsonl')
        assert perception[0]['sensor']['state'] == initial and initial['time_s'] == 0
        assert not any(Polygon(r['polygon_xy']).covers(Point(initial['x'], initial['y'])) for r in task['regions'])
        assert perception and execution
        files, rgb, depth, mapped, good_top, max_streak, streak, route = {}, set(), set(), [], False, 0, 0, False
        for i, frame in enumerate(perception):
            sensor = frame['sensor']
            assert frame['frame_idx'] == i and not frame['gt_sent_to_agent']
            assert abs(sensor['state']['time_s']-i*.5) < 1e-8
            assert sensor['state'] == (initial if i == 0 else execution[i*5-1]['state'])
            for key in ['rgb', 'depth']:
                path = sensor[key]
                assert digest(path) == sensor[key+'_sha256']
                files[path] = sensor[key+'_sha256']
            detector = case/'detections'/f'{i:06d}.pkl.gz'
            assert digest(detector) == frame['detector_sha256']
            files[str(detector)] = frame['detector_sha256']
            with gzip.open(detector, 'rb') as handle:
                observations = pickle.load(handle)
            assert observations['frame_provenance']['rgb_sha256'] == sensor['rgb_sha256']
            z = np.asarray(Image.open(sensor['depth']))
            assert z.shape == (360, 640) and int((z > 0).sum()) == sensor['lidar_pixels']
            matches = associations(frame['nodes'], config['targets'])
            assert {str(k):v for k, v in matches.items()} == frame['evaluation_only_matches']
            selected = matches.get(frame['selected_node_id'])
            assert selected == frame['evaluation_only_identity']
            mapped.append(selected)
            streak = streak+1 if selected in task['accepted_instance_ids'] else 0
            max_streak = max(max_streak, streak)
            candidates = frame['candidates']
            good_top |= bool(candidates and matches.get(candidates[0]['node_id']) in task['accepted_instance_ids'])
            route |= bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
            rgb.add(sensor['rgb_sha256'])
            depth.add(sensor['depth_sha256'])
        assert len(rgb) > 1 and len(depth) > 1
        assert len(perception) == result['actual_frontend_calls'] and len(execution) == result['saved_control_intervals']
        obstacles = [rectangle(t['center_xy'], t['yaw_rad'], t['length_m']/2, t['length_m']/2, t['width_m']) for t in config['targets']]
        body = audit_trajectory(initial, execution, asdict(SPEC), Polygon(config['road_polygon_xy']), obstacles)
        direction = audit_execution_direction(initial, execution, config['centerline_xy'], asdict(SPEC))
        assert body['passed'] and direction['passed'] and direction['moving_intervals'] > 0
        committed_rows = [(i, row) for i, row in enumerate(execution)
            if row['tracking'].get('reason') == 'perceived_goal_stop_commitment']
        assert committed_rows
        first_index, first_row = committed_rows[0]
        old_commit = initial if first_index == 0 else execution[first_index-1]['state']
        commitment = first_row['tracking']['commitment']
        assert commitment['committed_at_sim_s'] == old_commit['time_s']
        assert old_commit['speed_mps'] <= .15
        assert math.dist([old_commit['x'], old_commit['y']], commitment['goal_xy']) < .5
        earlier_plans = [f for f in perception if f['sim_time_s'] <= old_commit['time_s']+1e-9
            and f['planning'] and f['planning']['purpose'] == 'target_approach'
            and not f['planning']['failure'] and len(f['path_world']) >= 2]
        assert earlier_plans
        earlier = earlier_plans[-1]
        assert commitment['node_id'] == earlier['selected_node_id']
        assert commitment['goal_xy'] == earlier['planning']['goal_xy']
        assert not commitment['independent_goal_or_truth_read']
        for i, row in committed_rows:
            assert row['tracking']['commitment'] == commitment
            assert row['control']['acceleration_mps2'] < 0
            assert not row['tracking']['independent_goal_or_truth_read']
        dump(destination/'perceived_stop_commitment_proof.json', dict(
            passed=True, first_execution_index=first_index, committed_control_intervals=len(committed_rows),
            commitment=commitment, source_perception_frame=earlier['frame_idx'],
            thresholds_reused_from_existing_tracking=True, evaluation_feedback_used=False))
        certified = []
        segment_start, longest, index = None, 0., 0
        old = initial
        for row in execution:
            new = row['state']
            while index+1 < len(perception) and perception[index+1]['sim_time_s'] <= old['time_s']+1e-9:
                index += 1
            identity = mapped[index]
            dt = new['time_s']-old['time_s']
            assert abs(dt-.1) < 1e-8
            travel = new['speed_mps']*dt
            inside = False
            for region in task['regions']:
                if region['instance_id'] != identity:
                    continue
                polygon = Polygon(region['polygon_xy'])
                start = Point(old['x'], old['y'])
                # 整个位置圆弧包含于以起点为心、弧长travel为半径的圆盘。
                # 到区域边界距离大于弧长则保守证明整段中心均在独立区域内。
                position_ok = polygon.covers(start) and polygon.boundary.distance(start) >= travel+1e-9
                errors = [abs(math.atan2(math.sin(s['yaw_rad']-region['heading_rad']), math.cos(s['yaw_rad']-region['heading_rad']))) for s in [old, new]]
                turn = travel*math.tan(new['steering_rad'])/SPEC.wheelbase_m
                heading_ok = max(errors) <= math.radians(15) and abs(turn) < math.pi
                speed_ok = max(old['speed_mps'], new['speed_mps']) <= .15
                inside |= position_ok and heading_ok and speed_ok
            valid = inside and identity in task['accepted_instance_ids'] and new['time_s'] <= 90
            if valid:
                if segment_start is None:
                    segment_start = old['time_s']
                longest = max(longest, new['time_s']-segment_start)
            else:
                segment_start = None
            certified.append(dict(start_s=old['time_s'], end_s=new['time_s'],
                associated_instance=identity, current_frame=perception[index]['frame_idx'],
                whole_interval_region_heading_speed_certified=bool(valid),
                held_s=0. if segment_start is None else new['time_s']-segment_start))
            old = new
        independent_arrival = longest >= 2-1e-8 and body['passed'] and direction['passed']
        four = dict(route_generated=route, selected_object_evaluated_correct=good_top,
            stable_correct_lock=max_streak >= 3, actual_full_perception_arrival=independent_arrival)
        for key, value in four.items():
            assert result[key] == value, (key, result[key], value)
        assert result['final_state'] == execution[-1]['state']
        assert len(lines(case/'commands.jsonl')) == len(lines(case/'processed_commands.jsonl')) == 1
        assert lines(case/'processed_commands.jsonl')[0]['command'] == 'run'
        assert result['Qwen_calls'] == 0 and not result['gt_in_agent']
        dump(destination/'continuous_body.json', body)
        dump(destination/'direction.json', direction)
        dump(destination/'arrival_intervals.json', certified)
        dump(destination/'sensor_files.json', files)
        dump(destination/'report.json', dict(passed=True, updated_utc=datetime.now(timezone.utc).isoformat(),
            actual_source_files=390, imported_conceptgraph_modules=len(imports['modules']),
            source_imports_from_isolated_root=True, source_data_read_only_references_declared=True,
            clean_OS_or_new_Python_environment=True, new_Python_environment=True, clean_OS=False,
            verified_dependency_modules=14, original_site_packages_reused=False,
            shared_conda_standard_library_and_native_system_libraries_declared=True,
            environment_installed_manifest_sha256=digest(installed_path),
            actual_saved_initial_state=True,
            initial_outside_goal=True, actual_frontend_calls=len(perception), controls=len(execution),
            moving_intervals=direction['moving_intervals'], distinct_RGB=len(rgb), distinct_depth=len(depth),
            four_outcomes=four, longest_whole_interval_certified_hold_s=longest,
            continuous_body_passed=True, direction_passed=True,
            actual_torch_threads=imports['torch_threads'], requested_torch_threads=1,
            thread_resource_policy_passed=thread_policy_passed,
            successful_task_reproduction_accepted=independent_arrival and thread_policy_passed,
            prior_run_001_failure_preserved=True, posthoc_stop_commitment_diagnostic=True,
            portable_entry_actually_executed=True, source_input_has_no_outputs_or_Git=True,
            stop_commitment_intervals=len(committed_rows),
            extension_sha256=digest(extension), thread_observations_sha256=digest(case/'thread_observations.jsonl'),
            independent_association='NetworkX最大基数/最小距离；固定粗类别XY2.5m',
            independent_arrival='实际观测身份＋圆弧中心区域内界＋速度与航向全区间＋至少2秒保持',
            source_sha256=digest(__file__), input_manifest_sha256=digest(base/'manifest.json'),
            result_sha256=digest(case/'result.json'), imported_module_audit_sha256=digest(case/'import_audit.json'),
            new_independent_scene_credit=0, frozen_main_table_credit=False,
            GPU_performance_credit=False, human_labels_added=0, goal_complete=False,
            CPU_shared_wall_s=time.monotonic()-started))
        print(json.dumps(dict(passed=True, controls=len(execution), frontend=len(perception),
            independent_arrival=independent_arrival, hold_s=longest)), flush=True)
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
