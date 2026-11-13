"""停车保持策略的身份、运动和无真值边界；不加载任何真实模型。"""
import importlib.util
from pathlib import Path

import pytest

from conceptgraph.completion.interactive_navigation import PerceptionVehicleAgent
from conceptgraph.navigation.vehicle_geometry import VehicleState

MODULE = Path(__file__).with_name('perceived_stop_commitment_001.py')
SPECIFICATION = importlib.util.spec_from_file_location('stop_commitment_under_test', MODULE)
MODULE_VALUE = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(MODULE_VALUE)
Agent = MODULE_VALUE.PerceivedStopCommitmentAgent


def agent(selected=5):
    value = Agent.__new__(Agent)
    value.last_successful_perceived_goal = dict(node_id=5, goal_xy=[10., 0.], frame_idx=4)
    value.stop_commitment = None
    value.selected = selected
    return value


def test_stopped_near_previous_perceived_goal_holds_without_path_or_truth():
    value = agent()
    state = VehicleState(9.8, .02, 0., .1, 0., 12.)
    acceleration, _, record = value.control(state)
    assert acceleration < 0 and record['reason'] == 'perceived_goal_stop_commitment'
    assert not record['independent_goal_or_truth_read']
    value.selected = None
    value.last_successful_perceived_goal = dict(node_id=9, goal_xy=[20., 0.])
    assert value.control(state)[2]['reason'] == 'perceived_goal_stop_commitment'
    assert value.stop_commitment['node_id'] == 5


@pytest.mark.parametrize('selected,x,speed', [(5, 9.8, .3), (5, 8., 0.), (9, 9.8, 0.), (None, 9.8, 0.)])
def test_does_not_commit_before_stopping_near_same_perceived_identity(monkeypatch, selected, x, speed):
    value = agent(selected)
    sentinel = (0., 0., dict(reason='original_control'))
    monkeypatch.setattr(PerceptionVehicleAgent, 'control', lambda self, state: sentinel)
    assert value.control(VehicleState(x, 0., 0., speed)) == sentinel
    assert value.stop_commitment is None


def test_no_successful_perceived_goal_does_not_create_commitment(monkeypatch):
    value = agent()
    value.last_successful_perceived_goal = None
    sentinel = (0., 0., dict(reason='original_control'))
    monkeypatch.setattr(PerceptionVehicleAgent, 'control', lambda self, state: sentinel)
    assert value.control(VehicleState(10., 0., 0.)) == sentinel
    assert value.stop_commitment is None
