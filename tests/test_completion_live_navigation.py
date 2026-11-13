"""交互控制的生命周期、幂等和输入隔离；不加载真实模型。"""
import json
from pathlib import Path

import pytest

from conceptgraph.completion.live_navigation_server import SessionManager, checked_start
from conceptgraph.completion.live_navigation_worker import Commands


@pytest.fixture
def tasks():
    return dict(tasks=[dict(task_id='config00-positive0')], seeds=[17, 29, 43])


@pytest.mark.parametrize('record', [
    {'task_id':'../other', 'seed':17}, {'task_id':'config00-positive0', 'seed':True},
    {'task_id':'config00-positive0', 'seed':18},
    {'task_id':'config00-positive0', 'seed':17, 'query':'其他指令'},
    {'task_id':'config00-positive0', 'seed':17, 'accepted_instance_ids':['x']}, [],
])
def test_start_rejects_unfrozen_or_label_injection(record, tasks):
    with pytest.raises(ValueError):
        checked_start(record, tasks)


def test_valid_frozen_request(tasks):
    assert checked_start(dict(task_id='config00-positive0', seed=17), tasks)['seed'] == 17


def test_pause_cancels_pending_step_and_stop_cannot_resume():
    gate = Commands()
    gate.accept('step')
    assert gate.mode == 'PAUSED' and gate.pending_ticks == 5
    gate.accept('pause')
    assert gate.pending_ticks == 0
    gate.accept('run')
    assert gate.mode == 'RUNNING'
    gate.accept('stop')
    gate.accept('run')
    assert gate.mode == 'STOPPING' and gate.pending_ticks == 0


def test_poll_does_not_reapply_old_step(tmp_path):
    path = tmp_path/'commands.jsonl'
    path.write_text(json.dumps(dict(command='step'))+'\n')
    gate = Commands()
    assert len(gate.poll(path)) == 1
    gate.pending_ticks -= 2
    assert gate.poll(path) == [] and gate.pending_ticks == 3
    with path.open('a') as handle:
        handle.write(json.dumps(dict(command='pause'))+'\n')
    assert len(gate.poll(path)) == 1 and gate.pending_ticks == 0


@pytest.fixture
def manager(tmp_path, monkeypatch, tasks):
    task_path = tmp_path/'interactive_simulation/synthetic_tasks_v1.json'
    task_path.parent.mkdir()
    task_path.write_text(json.dumps(tasks))
    monkeypatch.setattr('conceptgraph.completion.live_navigation_server.process_ticks', lambda pid:'42')
    result = SessionManager(tmp_path)
    result.active = result.root/'test_only_session'
    result.active.mkdir()
    result.worker = dict(pid=123, start_ticks='42')
    (result.active/'worker_process.json').write_text(json.dumps(result.worker))
    (result.active/'commands.jsonl').touch()
    return result


def test_commands_are_idempotent_and_conflicting_reuse_rejected(manager):
    row = dict(run_id=manager.active.name, command='step', command_id='test-001')
    assert not manager.control(row)['duplicate']
    assert manager.control(row)['duplicate']
    assert len((manager.active/'commands.jsonl').read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        manager.control(dict(row, command='run'))


def test_restarted_service_finds_actual_worker_and_refuses_second(manager):
    recovered = SessionManager(manager.output)
    assert recovered.active == manager.active and recovered.alive()
    with pytest.raises(RuntimeError, match='已有实际导航会话'):
        recovered.start(dict(task_id='config00-positive0', seed=17))


@pytest.mark.parametrize('phase', ['FINISHED', 'FAILED'])
def test_terminal_session_cannot_accept_unapplied_control(manager, phase):
    (manager.active/'state.json').write_text(json.dumps(dict(phase=phase)))
    with pytest.raises(RuntimeError, match='会话已终止'):
        manager.control(dict(run_id=manager.active.name, command='run', command_id='test-002'))


def test_missing_process_does_not_infer_success(manager, monkeypatch):
    (manager.active/'state.json').write_text(json.dumps(dict(phase='PERCEIVING')))
    monkeypatch.setattr('conceptgraph.completion.live_navigation_server.process_ticks', lambda pid:None)
    state = manager.snapshot()
    assert not state['worker_alive'] and state['phase'] == 'WORKER_EXITED_WITHOUT_RESULT'
    assert not (manager.active/'result.json').exists()


def test_stale_control_cannot_affect_current_session(manager):
    with pytest.raises(RuntimeError):
        manager.control(dict(run_id='old_session', command='stop', command_id='test-003'))
    assert Path(manager.active/'commands.jsonl').read_text() == ''
