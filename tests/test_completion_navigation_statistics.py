from conceptgraph.completion.interactive_statistics import summarize
from conceptgraph.completion.vehicle_statistics import summarize as vehicle_summary


def test_zero_motion_does_not_receive_safety_credit_and_four_outcomes_stay_separate():
    rows=[]
    for index,moved in enumerate([False,True]):
        rows.append(dict(task_id=f'task{index}',config_id='config0',seed=17,kind='reachable_positive',
            method='rule',status='WAITING' if not moved else 'PLANNING_FAILED',route_generated=moved,
            selected_object_evaluated_correct=True,stable_correct_lock=False,
            actual_full_perception_arrival=False,any_lock=False,actual_path_length_m=2. if moved else 0.,
            independent_execution_audit=dict(passed=True)))
    report=summarize(rows)['by_kind']['reachable_positive']
    assert report['actually_moving_trials']==1
    assert report['safety_among_moving']['denominator']==1
    assert report['stationary_audited_trials']==1
    assert report['four_outcomes']['route_generated']['estimate']==.5
    assert report['four_outcomes']['selected_object_evaluated_correct']['estimate']==1.
    assert report['four_outcomes']['stable_correct_lock']['estimate']==0.
    assert report['four_outcomes']['actual_full_perception_arrival']['estimate']==0.
    vehicle,_=vehicle_summary([rows[0]])
    assert vehicle['rule']['safety_rate_among_executed'] is None
    assert vehicle['rule']['executed_trials']==0
