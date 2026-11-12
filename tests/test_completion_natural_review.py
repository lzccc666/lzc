import json

import pytest

from conceptgraph.completion.natural_review import CAMPAIGN, validate_natural_submission
from conceptgraph.completion.stage9 import digest


@pytest.fixture
def review_fixture(tmp_path):
    root = tmp_path/'annotation'/CAMPAIGN
    root.mkdir(parents=True)
    source = tmp_path/'data_manifest/natural_distribution_audit_001/evaluation_only_natural_commands.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text('{"synthetic_unit_test_fixture": true}\n')
    rgb = tmp_path/'unit_rgb.txt'
    rgb.write_text('unit test bytes, not an actual human annotation')
    crop = root/'unit_crop.txt'
    crop.write_text('synthetic unit test crop')
    card = dict(command_token='unit-command', scene='unit-scene', camera_token='unit-camera',
        instance_token='unit-instance', rgb='unit_rgb.txt', crop='unit_crop.txt',
        source_sha256=digest(source), rgb_sha256=digest(rgb), crop_sha256=digest(crop))
    (root/'cards.json').write_text(json.dumps([card]))
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps(dict(cards_sha256=digest(root/'cards.json'))))
    record = dict(campaign=CAMPAIGN, campaign_sha256=digest(manifest), command_token=card['command_token'],
        reviewer='synthetic_unit_test', note='仅临时目录中的结构测试；不计为人工提交',
        task_scope='static_target_approach', observed_motion='static', target_visibility='visible',
        target_ambiguity='clear', observed_color='unknown',
        **{k: card[k] for k in ['source_sha256', 'rgb_sha256', 'crop_sha256']})
    return tmp_path, record, crop


def test_human_submission_validation_preserves_unknowns_and_does_not_write_labels(review_fixture):
    root, record, _ = review_fixture
    checked = validate_natural_submission(record, output=root)
    assert checked['observed_color'] == 'unknown' and checked['reachability'] == 'unknown'
    assert not checked['label_correctness_independently_verified']
    assert not (root/'annotation/human_natural_submissions').exists()
    record['observed_motion'] = 'unknown'
    with pytest.raises(ValueError, match='静止'):
        validate_natural_submission(record, output=root)


def test_changed_source_pixels_cannot_receive_a_review_of_previous_evidence(review_fixture):
    root, record, crop = review_fixture
    crop.write_text('changed unit-test pixels')
    with pytest.raises(ValueError, match='指纹'):
        validate_natural_submission(record, output=root)
