"""独立复核新交互会话的源、实际传感器、控制圆弧、连续车身与方向。"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    begun = time.monotonic()
    browser = OUTPUT/'visualizations/live_navigation_browser_qa_001'
    qa = read(browser/'report.json')
    assert qa['passed']
    case = OUTPUT/'interactive_simulation/live_navigation_sessions_001'/qa['session']
    result, release = read(case/'result.json'), read(case/'release.json')
    files = read(browser/'files.json')
    assert all(digest(OUTPUT/name) == expected for name, expected in files.items())
    assert all(digest(PROJECT/name) == expected for name, expected in release['source_sha256'].items())
    task_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert digest(task_path) == release['task_release_sha256']
    tasks = read(task_path)
    task = next(t for t in tasks['tasks'] if t['task_id'] == release['task_id'])
    config = next(c for c in tasks['configs'] if c['config_id'] == task['config_id'])
    assert task['task_id'] == 'config00-positive0' and release['seed'] == 17
    initial = read(case/'initial_state.json')
    perception, execution = rows(case/'perception.jsonl'), rows(case/'execution.jsonl')
    assert initial == perception[0]['sensor']['state'] and initial['time_s'] == 0
    assert len(perception) == result['actual_frontend_calls'] == 14
    assert len(execution) == result['saved_control_intervals'] == 67
    assert result['status'] == 'USER_STOPPED' and not result['actual_full_perception_arrival']
    rgb, depth = set(), set()
    for index, frame in enumerate(perception):
        assert frame['frame_idx'] == index and not frame['gt_sent_to_agent']
        sensor = frame['sensor']
        assert digest(sensor['rgb']) == sensor['rgb_sha256']
        assert digest(sensor['depth']) == sensor['depth_sha256']
        assert digest(case/'detections'/f'{index:06d}.pkl.gz') == frame['detector_sha256']
        z = np.asarray(Image.open(sensor['depth']))
        assert z.shape == (360, 640) and int((z > 0).sum()) == sensor['lidar_pixels']
        assert abs(sensor['state']['time_s']-index*.5) < 1e-8
        expected_state = initial if index == 0 else execution[index*5-1]['state']
        assert sensor['state'] == expected_state
        rgb.add(sensor['rgb_sha256'])
        depth.add(sensor['depth_sha256'])
    assert len(rgb) == result['distinct_RGB'] == 11 and len(depth) == result['distinct_depth'] == 11
    # 本次固定正任务没有屏障；评价几何从冻结尺寸重新组成，不调用运行端setup。
    obstacles = [rectangle(t['center_xy'], t['yaw_rad'], t['length_m']/2,
        t['length_m']/2, t['width_m']) for t in config['targets']]
    continuous = audit_trajectory(initial, execution, asdict(SPEC), Polygon(config['road_polygon_xy']), obstacles)
    direction = audit_execution_direction(initial, execution, config['centerline_xy'], asdict(SPEC))
    assert continuous['passed'] and direction['passed']
    commands, applied = rows(case/'commands.jsonl'), rows(case/'processed_commands.jsonl')
    assert [r['command_id'] for r in commands] == [r['command_id'] for r in applied]
    stop = next(i for i, row in enumerate(applied) if row['command'] == 'stop')+1
    braking = [row for row in execution if row['processed_commands'] >= stop]
    assert braking and all(row['control']['acceleration_mps2'] == -SPEC.max_braking_mps2 for row in braking)
    assert braking[-1]['state']['speed_mps'] == 0 and len(braking) == 7
    assert result['final_state'] == execution[-1]['state']
    probe = read(OUTPUT/'source_bundle/live_navigation_process_probe_001.json')
    assert probe['session'] == case.name and probe['worker_confirmed_live']
    assert probe['selected_process_environment']['CUDA_VISIBLE_DEVICES'] == ''
    assert probe['live_worker_absent_from_GPU_compute_processes']
    destination = OUTPUT/'vehicle_execution/live_navigation_independent_audit_001'
    destination.mkdir(exist_ok=False)
    dump(destination/'continuous_body.json', continuous)
    dump(destination/'direction.json', direction)
    dump(destination/'report.json', dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)), session=case.name, actual_saved_initial_state=True,
        control_intervals=67, moving_intervals=direction['moving_intervals'], actual_frontend_records=14,
        command_records=10, bounded_stop_control_intervals=7, stop_zero_speed=True,
        continuous_body_passed=True, continuous_direction_passed=True, original_files_checked=len(files),
        source_files_checked=len(release['source_sha256']), distinct_RGB=11, distinct_depth=11,
        renderer_probe_checked=True, browser_report_sha256=digest(browser/'report.json'),
        process_probe_sha256=digest(OUTPUT/'source_bundle/live_navigation_process_probe_001.json'),
        independent_body_report_sha256=digest(destination/'continuous_body.json'),
        independent_direction_report_sha256=digest(destination/'direction.json'),
        actual_new_model_calls=0, new_executions=0, new_arrivals=0,
        frozen_main_table_credit=False, performance_credit=False, human_labels_added=0,
        CPU_shared_wall_s=time.monotonic()-begun, goal_complete=False))
    print(json.dumps(dict(passed=True, session=case.name, controls=67, perception=14,
        continuous_body=True, continuous_direction=True, new_arrivals=0, goal_complete=False)))


if __name__ == '__main__':
    main()
