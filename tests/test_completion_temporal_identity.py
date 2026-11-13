import copy

import pytest

from conceptgraph.completion.temporal_identity_evaluation import (
    identity_counts, metric_values, prepare_sequence, return_diagnostics,
)


def frame(index, ids=('a',), gt=True, match=True):
    return dict(frame_idx=index, camera_timestamp_s=index*.5, evaluation_only_official=[
        dict(instance_token='g', category='car', center=[0, 0, 0])] if gt else [],
        predictions={'method': [dict(node_id=x, category='car', center=[0, 0, 0]) for x in ids]},
        metrics={'method': dict(matches=[dict(truth_index=0, prediction_index=0)] if gt and ids and match else [])})


def test_fragmented_track_has_no_full_identity_credit():
    records = prepare_sequence([frame(i, (str(i),)) for i in range(3)], 'method')
    result = identity_counts(records, 'all_gated')['counts']
    assert (result['IDTP'], result['IDFN'], result['IDFP']) == (1, 2, 2)
    assert metric_values(result)['IDF1'] == pytest.approx(1/3)


def test_one_persistent_track_and_single_frame_ids():
    frames = [frame(i) for i in range(3)]
    assert identity_counts(prepare_sequence(frames, 'method'), 'all_gated')['counts']['IDTP'] == 3
    for f in frames:
        f['predictions']['single_frame'] = f['predictions'].pop('method')
        f['metrics']['single_frame'] = f['metrics'].pop('method')
    records = prepare_sequence(frames, 'single_frame')
    assert identity_counts(records, 'all_gated')['counts']['IDTP'] == 1
    assert return_diagnostics(records)['counts']['adjacent_identity_changes'] == 0  # 旧相邻口径原样保留


def test_ambiguity_sensitivity_keeps_all_denominators():
    records = prepare_sequence([frame(0, ('a', 'b'))], 'method')
    all_gated = identity_counts(records, 'all_gated')['counts']
    strict = identity_counts(records, 'mutually_unique_gated')['counts']
    assert all_gated['IDTP'] == 1 and strict['IDTP'] == 0
    assert strict['GT_observations'] == 1 and strict['prediction_observations'] == 2
    assert strict['ambiguous_GT_observations'] == 1 and strict['gated_edges'] == 2


def test_global_one_to_one_cannot_credit_merged_tracks_twice():
    frames = [frame(i) for i in range(2)]
    frames[1]['evaluation_only_official'][0]['instance_token'] = 'other'
    result = identity_counts(prepare_sequence(frames, 'method'), 'all_gated')['counts']
    assert (result['IDTP'], result['IDFN'], result['IDFP']) == (1, 1, 1)


def test_empty_and_unmatched_observations():
    empty = identity_counts(prepare_sequence([frame(0, (), False)], 'method'), 'all_gated')['counts']
    assert metric_values(empty) == dict(IDR=None, IDP=None, IDF1=None)
    missing = identity_counts(prepare_sequence([frame(0, ())], 'method'), 'all_gated')['counts']
    assert missing['IDFN'] == 1 and metric_values(missing)['IDF1'] == 0


def test_official_returns_unknown_endpoint_and_shared_opportunity():
    frames = [frame(0), frame(1, (), False), frame(2, ('b',), match=False)]
    result = return_diagnostics(prepare_sequence(frames, 'method'))
    counts = result['counts']
    assert counts['official_return_opportunities'] == 1
    assert counts['official_return_unknown_id'] == 1 and counts['official_return_both_matched'] == 0
    assert counts['unreturned_visible_tail_observations'] == 1
    assert result['events'][0]['identity_status'] == 'unknown'


def test_gap_return_differs_from_adjacent_and_list_absence():
    frames = [frame(0), frame(1, (), match=False), frame(2, ('b',)), frame(3, ('b',))]
    counts = return_diagnostics(prepare_sequence(frames, 'method'))['counts']
    assert counts['method_gap_returns'] == counts['method_gap_changed_id'] == 1
    assert counts['adjacent_matched_transitions'] == 1 and counts['adjacent_identity_changes'] == 0
    assert counts['official_return_opportunities'] == 0
    frames[1]['evaluation_only_official'] = []
    events = return_diagnostics(prepare_sequence(frames, 'method'))['events']
    assert events[0]['kind'] == 'official_list_return' and events[0]['identity_status'] == 'changed'
    assert events[1]['gap_cause'] == 'GT_list_absence_included'


def test_censored_tail_and_never_matched_are_retained():
    frames = [frame(0), frame(1, (), False)]
    result = return_diagnostics(prepare_sequence(frames, 'method'))
    assert result['counts']['censored_tails'] == result['counts']['tails_GT_absent_at_scene_end'] == 1
    assert result['censored_tails'][0]['unmatched_visible_observations'] == 0
    result = return_diagnostics(prepare_sequence([frame(0, (), match=False)], 'method'))
    assert result['counts']['never_matched_instances'] == result['counts']['never_matched_GT_observations'] == 1


def test_fixed_gate_includes_boundary_but_excludes_wrong_class():
    frames = [frame(0, ('boundary', 'outside', 'truck'), match=False)]
    objects = frames[0]['predictions']['method']
    objects[0]['center'] = [3, 0, 0]
    objects[1]['center'] = [3.000001, 0, 0]
    objects[2]['category'] = 'truck'
    result = identity_counts(prepare_sequence(frames, 'method'), 'all_gated')['counts']
    assert (result['IDTP'], result['gated_edges'], result['IDFP']) == (1, 1, 2)


def test_return_gap_uses_elapsed_camera_seconds():
    frames = [frame(0), frame(1, (), False), frame(2)]
    frames[-1]['camera_timestamp_s'] = 2
    records = prepare_sequence(frames, 'method')
    event = return_diagnostics(records)['events'][0]
    assert event['gap_bucket'] == 'at_most_2s'
    frames[-1]['camera_timestamp_s'] = 2.01
    assert return_diagnostics(prepare_sequence(frames, 'method'))['events'][0]['gap_bucket'] == 'over_2s'


@pytest.mark.parametrize('bad', ['nan', 'duplicate_gt', 'duplicate_pred', 'time', 'index', 'identity', 'invalid_match'])
def test_invalid_input_is_not_silently_filtered(bad):
    frames = [frame(0), frame(1)]
    if bad == 'nan':
        frames[1]['predictions']['method'][0]['center'][0] = float('nan')
    elif bad == 'duplicate_gt':
        frames[1]['evaluation_only_official'] *= 2
    elif bad == 'duplicate_pred':
        frames[1]['predictions']['method'] *= 2
    elif bad == 'time':
        frames[1]['camera_timestamp_s'] = 0
    elif bad == 'index':
        frames[1]['frame_idx'] = 2
    elif bad == 'identity':
        frames[1]['predictions']['method'][0]['node_id'] = None
    else:
        frames[1]['metrics']['method']['matches'][0]['prediction_index'] = 4
    with pytest.raises(ValueError):
        prepare_sequence(copy.deepcopy(frames), 'method')
