"""逻辑投递、版本、期限与任务预算边界；不加载真实模型。"""
import importlib.util
from pathlib import Path

from types import SimpleNamespace

path = Path(__file__).with_name('fixedlogical_semantic_bridge_001.py')
spec = importlib.util.spec_from_file_location('logical_bridge_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def queue(tmp_path):
    clock = module.LogicalClock()
    clock.set(0, SimpleNamespace(time_s=0.))
    calls = []

    def backend(task, payload):
        calls.append((task, payload))
        return dict(raw_output='actual injected test value')

    return module.OnePerTaskNextFrameQueue(backend, clock, tmp_path), clock, calls


def test_no_same_frame_delivery_and_next_frame_delivery(tmp_path):
    value, clock, calls = queue(tmp_path)
    value.submit('query_parse', {'query_text':'car'}, {'query_version':1}, ttl_s=2.)
    assert len(calls) == 1 and value.poll({'query_version':1}) == []
    clock.set(1, SimpleNamespace(time_s=.5))
    delivered = value.poll({'query_version':1})
    assert len(delivered) == 1 and value.poll({'query_version':1}) == []


def test_changed_identity_discards_response(tmp_path):
    value, clock, _ = queue(tmp_path)
    value.submit('object_attributes', {'node_id':1}, {'object_1':['track-1', 'car']}, ttl_s=2.)
    clock.set(1, SimpleNamespace(time_s=.5))
    assert value.poll({'object_1':['track-2', 'car']}) == []
    assert value.events[-1]['kind'] == 'discarded'


def test_deadline_does_not_revive_with_next_frame(tmp_path):
    value, clock, _ = queue(tmp_path)
    value.submit('query_parse', {}, {}, ttl_s=.1)
    clock.set(1, SimpleNamespace(time_s=.5))
    assert value.poll({}) == []


def test_only_one_request_per_task_even_after_delivery(tmp_path):
    value, clock, calls = queue(tmp_path)
    for task in ['query_parse', 'object_attributes', 'candidate_rerank']:
        value.submit(task, {}, {}, ttl_s=2.)
    clock.set(1, SimpleNamespace(time_s=.5))
    assert len(value.poll({})) == 3
    assert value.submit('candidate_rerank', {}, {}, ttl_s=2.) is None
    assert len(calls) == 3 and value.rejected_budget['candidate_rerank'] == 1


def test_submission_payload_and_binding_are_immutable(tmp_path):
    value, clock, _ = queue(tmp_path)
    payload, binding = {'items':[1]}, {'query_version':1}
    value.submit('query_parse', payload, binding, ttl_s=2.)
    payload['items'].append(2)
    binding['query_version'] = 2
    clock.set(1, SimpleNamespace(time_s=.5))
    request, _ = value.poll({'query_version':1})[0]
    assert request.payload == {'items':[1]}
