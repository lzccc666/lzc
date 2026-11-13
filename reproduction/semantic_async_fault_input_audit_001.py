"""独立复核故障请求的真实crop、候选安全、完整原文及有限观察窗差异。"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from PIL import Image

from conceptgraph.completion import realtime_navigation
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def forbidden_keys(value):
    if isinstance(value, dict):
        return [key for key in value if key in {'instance_id', 'accepted_instance_ids', 'regions', 'truth', 'goal_truth', 'evaluation_only_matches'}]+[
            key for item in value.values() for key in forbidden_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in forbidden_keys(item)]
    return []


def audit(run_id, condition):
    run = OUTPUT/'faults'/run_id
    case = run/condition
    target = case/'input_audit_001'
    target.mkdir(exist_ok=False)
    release = read(run/'release.json')
    assert all(digest(PROJECT/name) == expected for name, expected in release['source_sha256'].items())
    complete, imports = read(case/'complete.json'), read(case/'actual_imports.json')
    model = read(case/'actual_model.json')
    assert model['devices'] == ['cpu'] and model['parameter_count'] > 1_000_000_000
    assert not model['CUDA_initialized'] and not imports['CUDA_initialized']
    assert imports['global_torch_threads'] == 1
    if (case/'observation_policy.json').exists():
        reference = inspect.getsource(realtime_navigation.run_task)
        expected = reference.replace("if final in {'ARRIVED','EXECUTION_FAILED'}:", "if final == 'EXECUTION_FAILED':")
        expected = expected.replace("final = 'SAFETY_STOP'\n                    break", "final = 'SAFETY_STOP'")
        generated = case/'run_task_observation_source.py'
        assert generated.read_text() == expected
        policy = read(case/'observation_policy.json')
        assert digest(generated) == policy['source_sha256']
        assert not policy['controller_or_guard_changed'] and not policy['GT_control_feedback_added']
    frames = rows(case/'navigation/perception.jsonl')
    events = read(case/'navigation/semantic_events.json')['events']
    submitted_events = {event['request_id']:event for event in events if event['kind'] == 'submitted'}
    began = {row['task']:row for row in rows(case/'model_events.jsonl') if row['kind'] == 'actual_model_started'}
    returned = {read(path)['task']:read(path) for path in (case/'actual_raw').glob('*.json')}
    envelopes = {read(path)['task']:read(path)['result'] for path in (case/'navigation/qwen_audit').glob('*.json')}
    started = read(case/'navigation/driver_clock.json')['started_wall_s']
    inputs, files = [], {}
    for path in sorted((case/'submitted_requests').glob('*.json')):
        record = read(path)
        files[str(path)] = digest(path)
        payload, task = record['payload'], record['task']
        event = submitted_events[record['request_id']]
        assert event['task'] == task and event['binding'] == record['binding']
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        assert hashlib.sha256(body.encode()).hexdigest() == event['payload_sha256']
        assert not forbidden_keys(payload)
        item = dict(task=task, actual_model_called=task in began,
            request_id=record['request_id'], immutable_payload_sha256=event['payload_sha256'], forbidden_GT_keys=[])
        if task == 'query_parse':
            assert payload == dict(query_text='去红色小汽车附近')
        else:
            elapsed = event['submitted_wall_s']-started
            observed = [frame for frame in frames if frame['sensed_wall_elapsed_s'] <= elapsed <= frame['completed_wall_elapsed_s']]
            assert len(observed) == 1
            frame = observed[0]
            item['current_observation_frame'] = frame['frame_idx']
            if task == 'object_attributes':
                assert payload['frame_idx'] == frame['frame_idx']
                node = next(node for node in frame['nodes'] if node['node_id'] == payload['node_id'])
                assert node['metadata']['num_detections'] >= 3
                assert record['binding'] == {f'object_{node["node_id"]}':[node['metadata']['object_identity'], node['category']]}
                crop = Path(payload['crop_paths'][0])
                assert crop.is_relative_to(case/'navigation/actual_crops') and digest(crop) == payload['crop_sha256']
                files[str(crop)] = digest(crop)
                with Image.open(crop) as image:
                    value = image.convert('RGB')
                    pixels = dict(size=list(value.size), mode=value.mode,
                        pixels_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                if task in began:
                    assert began[task]['images'] == [pixels] == returned[task]['images']
                item['actual_crop_pixels'] = pixels
            else:
                candidates = payload['candidate_nodes']
                ids = [node['node_id'] for node in candidates]
                assert ids == frame['semantic_audit']['safety_verified_order']
                checks = {r['node_id']:r for r in frame['semantic_audit']['candidate_checks']}
                assert all(checks[i]['passed'] and checks[i]['independent_sweep_audit']['passed'] for i in ids)
                assert all(node['path_validated'] for node in candidates)
                assert all(node['category'] == next(n['category'] for n in frame['nodes'] if n['node_id'] == node['node_id']) for node in candidates)
                item['current_safety_verified_candidates'] = ids
        if task in returned:
            raw = returned[task]['raw_output']
            envelope = envelopes[task]
            assert raw and not envelope['cache']['enabled']
            if condition == 'empty_output':
                assert envelope['raw_output'] == '' and envelope['failure_code'] == 'EMPTY_OUTPUT'
            elif condition == 'malformed_output':
                assert envelope['raw_output'] == '{"schema_version":' and envelope['failure_code'] == 'SCHEMA_VALIDATION_FAILED'
            elif condition == 'backend_exception':
                assert envelope['raw_output'] is None and envelope['failure_code'] == 'BACKEND_EXCEPTION'
            elif condition == 'backend_timeout':
                assert envelope['raw_output'] is None and envelope['failure_code'] == 'BACKEND_TIMEOUT'
            else:
                assert envelope['raw_output'] == raw
            item['real_before_injection_text_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        inputs.append(item)
    assert len(inputs) == 3 and len(began) == complete['actual_CPU_Qwen']['actual_model_started']
    dump(target/'files.json', files)
    report = dict(passed=True, run_id=run_id, condition=condition, requests=inputs,
        actual_CPU_Qwen_calls=len(began), true_RGB_crop_pixels_verified=True,
        rerank_only_current_vehicle_safe_candidates=True, no_GT_keys_in_requests=True,
        wrapper_fault_after_real_model_text_verified=True, cache_disabled=True,
        bounded_runtime_source_only_declared_terminal_differences=(case/'observation_policy.json').exists(),
        actual_model_calls_added_by_audit=0, actual_navigation_runs_added_by_audit=0,
        source_sha256=digest(__file__), source_release_sha256=digest(run/'release.json'),
        original_result_sha256=digest(case/'navigation/result.json'), goal_complete=False)
    dump(target/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', choices=['semantic_async_cpu_001', 'semantic_async_cpu_002', 'semantic_async_cpu_003'], required=True)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id, args.condition), ensure_ascii=False))
