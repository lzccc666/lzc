"""逐条件独立核对实际模型/故障事件、控制积分、连续整车与方向。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).open()]


def audit(condition, run_id='semantic_async_cpu_001'):
    base = OUTPUT/'faults'/run_id
    destination = base/condition/'independent_audit_003'
    destination.mkdir(exist_ok=False)
    try:
        root, case = base/condition, base/condition/'navigation'
        complete = read(root/'complete.json')
        initial, execution, frames = read(case/'initial_state.json'), lines(case/'execution.jsonl'), lines(case/'perception.jsonl')
        frozen = read(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json')
        config = next(c for c in frozen['configs'] if c['config_id'] == 'config00')
        task = next(t for t in frozen['tasks'] if t['task_id'] == 'config00-positive0')
        obstacles = [rectangle(t['center_xy'], t['yaw_rad'], t['length_m']/2, t['length_m']/2, t['width_m']) for t in config['targets']]
        body = audit_trajectory(initial, execution, frozen['vehicle'], Polygon(config['road_polygon_xy']), obstacles)
        direction = audit_execution_direction(initial, execution, config['centerline_xy'], frozen['vehicle'])
        assert body['passed'] and (direction['passed'] is True or direction['moving_intervals'] == 0)
        assert initial['time_s'] == 0 and complete['frontend_calls'] == len(frames)
        assert execution[-1]['state']['speed_mps'] == 0 and execution[-1]['stop_requested']
        correct, locked, streak, routes = False, False, 0, False
        sensors = {}
        for index, frame in enumerate(frames):
            assert not frame['gt_sent_to_agent'] and frame['frame_idx'] == index
            sensor = frame['sensor']
            for key in ['rgb', 'depth']:
                assert digest(sensor[key]) == sensor[key+'_sha256']
                sensors[sensor[key]] = sensor[key+'_sha256']
            z = np.array(Image.open(sensor['depth']))
            assert int((z > 0).sum()) == sensor['lidar_pixels']
            detector = case/'detections'/f'{index:06d}.pkl.gz'
            assert digest(detector) == frame['detector_sha256']
            sensors[str(detector)] = frame['detector_sha256']
            nodes = sorted(frame['nodes'], key=lambda n:n['node_id'])
            costs = np.array([[math.dist(n['center'][:2], t['center_xy']) if n['category'] == t['category'] else 1e6
                for t in config['targets']] for n in nodes])
            mapping = {}
            if nodes:
                a, b = linear_sum_assignment(costs)
                mapping = {nodes[int(x)]['node_id']:config['targets'][int(y)]['instance_id'] for x, y in zip(a, b) if costs[x, y] <= 2.5}
            selected = mapping.get(frame['selected_node_id'])
            assert selected == frame['evaluation_only_identity']
            correct |= bool(frame['candidates'] and mapping.get(frame['candidates'][0]['node_id']) in task['accepted_instance_ids'])
            streak = streak+1 if selected in task['accepted_instance_ids'] else 0
            locked |= streak >= 3
            routes |= bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
        for key, value in dict(route_generated=routes, selected_object_evaluated_correct=correct, stable_correct_lock=locked).items():
            assert complete['navigation_result'][key] == value
        model = lines(root/'model_events.jsonl')
        began = [r for r in model if r['kind'] == 'actual_model_started']
        ended = {r['identity']:r for r in model if r['kind'] == 'actual_model_finished'}
        assert len(began) == len(ended) == complete['actual_CPU_Qwen']['actual_model_started']
        assert 1 <= len(began) <= 3
        assert all(r['torch_threads'] == 1 for r in began)
        assert complete['actual_CPU_Qwen']['active_model_calls'] == 0
        assert complete['actual_CPU_Qwen']['maximum_model_concurrency'] == 1 and complete['all_backend_threads_joined']
        for record in began:
            raw = read(root/'actual_raw'/f'{record["identity"]:03d}.json')
            assert hashlib.sha256(raw['raw_output'].encode()).hexdigest() == ended[record['identity']]['raw_sha256']
            assert record['prompt_sha256'] == raw['prompt_sha256'] and record['images'] == raw['images']
        assert all(value == 1 for value in Counter(r['task'] for r in began).values())
        assert all(r['task'] in {'query_parse', 'object_attributes', 'candidate_rerank'} for r in began)
        bridge = read(case/'semantic_events.json')
        events = bridge['events']
        submitted = {r['request_id']:r for r in events if r['kind'] == 'submitted'}
        assert len(submitted) == 3
        discarded = Counter(r.get('reason') for r in events if r['kind'] == 'discarded')
        expired_queued = sum(r['kind'] == 'expired_in_queue' for r in events)
        for record in events:
            if record['kind'] == 'delivered':
                assert record['wall_s'] <= record['deadline_wall_s']
        responses = [read(p) for p in (case/'semantic_responses').glob('*.json')]
        envelopes = [read(p)['result'] for p in (case/'qwen_audit').glob('*.json')]
        faults = Counter(r.get('failure_code') for r in envelopes)
        delivered_faults = Counter(r['envelope'].get('failure_code') for r in responses)
        finished = {r['request_id']:r for r in events if r['kind'] == 'backend_completed'}
        polled = {r['request_id'] for r in events if r['kind'] in {'delivered', 'discarded'}}
        completed_not_polled = sorted(set(finished)-polled)
        for row in events:
            if row['kind'] == 'discarded' and row.get('reason') == 'expired_before_delivery':
                prior = finished[row['request_id']]
                assert prior['response_status'] == 'ready' and prior['wall_s'] <= prior['deadline_wall_s'] < row['wall_s']
        expected = dict(backend_exception='BACKEND_EXCEPTION', empty_output='EMPTY_OUTPUT',
            malformed_output='SCHEMA_VALIDATION_FAILED', backend_timeout='BACKEND_TIMEOUT')
        if condition in expected:
            assert len(envelopes) == 3 and faults[expected[condition]] == 3
            assert Counter(r['task'] for r in began) == dict(query_parse=1, object_attributes=1, candidate_rerank=1)
            assert all(not r['accepted_at_receipt'] and r['envelope']['fallback_used'] for r in responses)
            assert sum(bridge['applied_counts'].values()) == 0
        if condition == 'late_and_queue_expiry':
            assert discarded['late_expired'] > 0 and expired_queued > 0
            assert sum(bridge['applied_counts'].values()) == 0
        if condition == 'expired_before_delivery':
            assert discarded['expired_before_delivery'] > 0
        if condition == 'binding_changed':
            assert discarded['binding_changed'] > 0
        # driver.started真实单调时钟从记录中的差重算；不依赖期望控制频率。
        starts = [(r['wall_s'], ended[r['identity']]['wall_s']) for r in began]
        process_started = read(case/'driver_clock.json')['started_wall_s']
        overlapping = [sum(a <= process_started+row['wall_elapsed_s'] <= b for row in execution) for a, b in starts]
        assert sum(overlapping) > 0
        timing = np.diff([row['wall_elapsed_s'] for row in execution])
        dump(destination/'body.json', body)
        dump(destination/'direction.json', direction)
        dump(destination/'sensor_files.json', sensors)
        report = dict(passed=True, condition=condition, actual_Qwen_calls=len(began),
            actual_frontend_calls=len(frames), actual_control_intervals=len(execution),
            controls_while_each_model_generation_in_flight=overlapping,
            control_wall_interval_median_s=float(np.median(timing)),
            control_wall_interval_p95_s=float(np.percentile(timing, 95)), control_wall_interval_max_s=float(max(timing)),
            moving_intervals=direction['moving_intervals'], static_only_no_driving_credit=direction['moving_intervals'] == 0,
            actual_delegate_failure_codes=dict(faults), delivered_failure_codes=dict(delivered_faults),
            saved_delegate_envelopes=len(envelopes), delivered_envelopes=len(responses),
            backend_completed_without_main_poll_at_shutdown=completed_not_polled,
            unpolled_results_never_counted_as_discarded_or_applied=True,
            discarded_reasons=dict(discarded), expired_in_queue=expired_queued,
            original_four_outcomes={k:complete['navigation_result'][k] for k in ['route_generated',
                'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival']},
            independent_arrival_credit=0, arrival_hold_audit_separate=True,
            actual_braking_to_zero=True, single_actual_model_generation=True,
            source_sha256=digest(__file__), shared_CPU_performance_not_generalized=True,
            real_GPU_fault_gate=False, new_test_opened=False, goal_complete=False)
        dump(destination/'report.json', report)
        return report
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--run-id', default='semantic_async_cpu_001', choices=['semantic_async_cpu_001','semantic_async_cpu_002'])
    args = parser.parse_args()
    print(json.dumps(audit(args.condition, args.run_id), ensure_ascii=False))
