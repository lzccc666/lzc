import threading
import time

from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue


def wait_until(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_backend_block_does_not_block_controller_and_late_output_never_delivers():
    started, release = threading.Event(), threading.Event()
    def backend(task, payload):
        started.set()
        release.wait(1)
        return dict(selected_node_id=7)
    service = RealtimeSemanticQueue(backend)
    try:
        service.submit('candidate_rerank', {'ids': [7]}, {'version': 1}, ttl_s=.025)
        assert started.wait(.2)
        # 控制侧仍能推进自己的时钟与安全停止事件，不等待后台模型。
        controls = []
        for _ in range(5):
            controls.append('brake')
            assert service.poll({'version': 1}) == []
            time.sleep(.01)
        release.set()
        wait_until(lambda: any(e['kind']=='backend_completed' for e in service.snapshot_events()))
        assert service.poll({'version': 1}) == []
        assert controls == ['brake'] * 5
        assert any(e.get('reason')=='late_expired' for e in service.snapshot_events())
    finally:
        release.set()
        assert service.close()['worker_stopped']


def test_ready_response_expires_before_main_chain_poll_and_changed_binding_is_rejected():
    service = RealtimeSemanticQueue(lambda task, payload: {'raw': payload})
    try:
        service.submit('query_parse', {'query': 'car'}, {'query_version': 1}, ttl_s=.5)
        wait_until(lambda: not service.completed.empty())
        assert service.poll({'query_version': 2}) == []
        service.submit('query_parse', {'query': 'truck'}, {'query_version': 2}, ttl_s=.02)
        wait_until(lambda: not service.completed.empty())
        time.sleep(.03)
        assert service.poll({'query_version': 2}) == []
        reasons = [e.get('reason') for e in service.snapshot_events() if e['kind']=='discarded']
        assert reasons == ['binding_changed', 'expired_before_delivery']
    finally:
        service.close()


def test_waiting_request_deadline_is_not_reset_when_worker_becomes_available():
    started, release = threading.Event(), threading.Event()
    calls = []
    def backend(task, payload):
        calls.append(payload['id'])
        started.set()
        release.wait(.5)
        return {'ok': True}
    service = RealtimeSemanticQueue(backend, maximum_queued=1)
    try:
        service.submit('query_parse', {'id': 1}, {}, ttl_s=.5)
        assert started.wait(.2)
        service.submit('query_parse', {'id': 2}, {}, ttl_s=.01)
        assert service.submit('query_parse', {'id': 3}, {}, ttl_s=.5) is None
        time.sleep(.02)
        release.set()
        wait_until(lambda: any(e['kind']=='expired_in_queue' for e in service.snapshot_events()))
        assert calls == [1]
    finally:
        release.set()
        service.close()
