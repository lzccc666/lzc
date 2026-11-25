import numpy as np

from conceptgraph.completion import goal_selection_ablation as module
from conceptgraph.navigation.types import ReachableGoalCandidate


def test_planner_claim_without_independent_lane_approval_is_rejected(monkeypatch):
    selector = object.__new__(module.OfflineSelector)
    selector.context = object()
    selector.start_rc = (1, 1)
    selector.heading = np.array([1., 0.])
    selector.goal_budget_s = .5
    selector.audits = {}
    monkeypatch.setattr(module, 'plan', lambda *args, **kwargs: dict(
        path=[(1, 1), (1, 2)], failure=None, expanded=2, cost_grid=1.))
    monkeypatch.setattr(module, 'audit', lambda *args: dict(
        full_lane_legal=False, independent_lane_audit=dict(illegal_transitions=[dict(index=1)])))
    goal = ReachableGoalCandidate(np.array([1., .5]), (1, 2), 2., .5, True)
    selector._plan_lane_goal(goal)
    assert not goal.path_found
    assert goal.rejection_reason == 'INDEPENDENT_LANE_AUDIT_REJECTED'
    assert selector.audits[(1, 2)]['independent_lane_audit']['illegal_transitions']
