import pytest

from conceptgraph.completion.realtime_matrix_statistics import enriched_trial, FIELDS
from conceptgraph.completion.interactive_statistics import summarize


def failed_record(tmp_path):
    return dict(task='unit-task',config_id='unit-config',kind='reachable_positive',seed=17,
        variant='new_full',status='PROCESS_FAILED_NO_RESULT',result=None,raw_case=str(tmp_path),
        returncode=1,process_timeout=False)


def test_missing_result_keeps_unknown_fields_and_stays_in_failure_denominator(tmp_path):
    row=enriched_trial(failed_record(tmp_path))
    assert all(row[key] is None for key in FIELDS)
    report=summarize([row])['by_kind']['reachable_positive']
    arrived=report['four_outcomes']['actual_full_perception_arrival']
    assert arrived['numerator']==0 and arrived['denominator']==1
    assert report['safety_among_moving']['estimate'] is None


def test_self_reported_arrival_without_successful_independent_audit_is_rejected(tmp_path):
    record=failed_record(tmp_path)
    record['result']=dict.fromkeys(FIELDS,True)
    record['result'].update(status='ARRIVED',independent_execution_audit=dict(passed=False),
        wall_duration_s=20.,gt_in_agent=False,real_clock_async=True)
    with pytest.raises(ValueError,match='independent'):
        enriched_trial(record)
