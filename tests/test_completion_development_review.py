import json

import pytest

from conceptgraph.completion.review_server import validate_submission
from conceptgraph.completion.stage9 import digest


@pytest.fixture
def review_evidence(tmp_path):
    """仅临时目录合成结构；不向项目人工记录目录写入。"""
    source = tmp_path/'frames.jsonl'
    source.write_text('{"synthetic_unit_fixture":true}\n')
    records = {}
    for run in ['corrected_dev_evidence_001', 'causal_dev_evidence_001']:
        bundle = tmp_path/'annotation'/run
        folder = bundle/'unit-scene_000'
        folder.mkdir(parents=True)
        crop = folder/'actual_crop.txt'
        crop.write_text('synthetic test crop; not a real human label')
        evidence = folder/'evidence.json'
        evidence.write_text(json.dumps(dict(scene='unit-scene', frame_idx=0,
            source_frame_file=str(source), source_frame_file_sha256=digest(source),
            predictions={'unit_method_a': [{'node_id': 'unit-node-a'}],
                'unit_method_b': [{'node_id': 'unit-node-b'}]},
            official_evaluation_only=[{'instance_token': 'unit-instance'}],
            detections=[dict(actual_clip_crop=crop.name)])))
        index = bundle/'index.json'
        index.write_text(json.dumps(dict(cases=[dict(scene='unit-scene', frame_idx=0,
            url=str(evidence.relative_to(tmp_path)))])))
        audit = bundle/'audit_001'
        audit.mkdir()
        files = audit/'files.json'
        files.write_text(json.dumps({str(path.relative_to(tmp_path)): digest(path) for path in [crop, evidence]}))
        (audit/'report.json').write_text(json.dumps(dict(passed=True, source_index_sha256=digest(index),
            files_manifest_sha256=digest(files))))
        records[run] = dict(evidence_run=run, evidence_index_sha256=digest(index),
            evidence_file_sha256=digest(evidence), scene='unit-scene', frame_idx=0,
            detection_index=0, source_sha256=digest(source), reviewer='synthetic_unit_test_only',
            mapping_method='unit_method_a', reviewed_node_ids=[],
            label='证据不足', note='临时目录结构测试，不能记为真实人工提交')
    return tmp_path, records


def test_legacy_review_without_version_still_binds_legacy_source(review_evidence):
    output, records = review_evidence
    record = records['corrected_dev_evidence_001'].copy()
    record.pop('evidence_run')
    result = validate_submission(record, output=output)
    assert result['evidence_run'] == 'corrected_dev_evidence_001'
    assert not (output/'annotation/human_submissions').exists()


def test_causal_review_preserves_unknowns_and_does_not_create_human_records(review_evidence):
    output, records = review_evidence
    result = validate_submission(records['causal_dev_evidence_001'], output=output)
    assert result['observed_color'] == result['observed_category'] == 'unknown'
    assert result['evidence_run'] == 'causal_dev_evidence_001'
    assert result['label_correctness_independently_verified'] is False
    assert not (output/'annotation/human_submissions').exists()


def test_causal_review_rejects_missing_version_index_fingerprint(review_evidence):
    output, records = review_evidence
    record = records['causal_dev_evidence_001'].copy()
    record.pop('evidence_index_sha256')
    with pytest.raises(ValueError, match='索引指纹'):
        validate_submission(record, output=output)


def test_legacy_evidence_hash_cannot_be_submitted_for_causal_bundle(review_evidence):
    output, records = review_evidence
    record = records['causal_dev_evidence_001'].copy()
    record['evidence_index_sha256'] = records['corrected_dev_evidence_001']['evidence_index_sha256']
    with pytest.raises(ValueError, match='索引指纹'):
        validate_submission(record, output=output)


@pytest.mark.parametrize('run', ['../outside', 'unregistered_evidence'])
def test_arbitrary_evidence_directory_is_not_accepted(review_evidence, run):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], evidence_run=run)
    with pytest.raises(ValueError, match='未知证据版本'):
        validate_submission(record, output=output)


def test_changed_crop_cannot_inherit_a_previous_review(review_evidence):
    output, records = review_evidence
    crop = output/'annotation/causal_dev_evidence_001/unit-scene_000/actual_crop.txt'
    crop.write_text('different synthetic pixels')
    with pytest.raises(ValueError, match='裁剪已改变'):
        validate_submission(records['causal_dev_evidence_001'], output=output)


def test_changed_prediction_evidence_cannot_inherit_previous_review(review_evidence):
    output, records = review_evidence
    evidence = output/'annotation/causal_dev_evidence_001/unit-scene_000/evidence.json'
    value = json.loads(evidence.read_text())
    value['changed_prediction'] = True
    evidence.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='证据内容已改变'):
        validate_submission(records['causal_dev_evidence_001'], output=output)


def test_causal_review_cannot_confirm_an_identity_absent_from_current_frame(review_evidence):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], instance_token='another-frame-instance')
    with pytest.raises(ValueError, match='当前帧官方评价清单'):
        validate_submission(record, output=output)
    record.update(instance_token='', label='正确关联', reviewed_node_ids=['unit-node-a'])
    with pytest.raises(ValueError, match='需要填写当前帧实例'):
        validate_submission(record, output=output)


@pytest.mark.parametrize('method', [None, 'not-in-this-evidence'])
def test_causal_review_requires_current_mapping_method(review_evidence, method):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], mapping_method=method)
    with pytest.raises(ValueError, match='建图方法'):
        validate_submission(record, output=output)


def test_node_ids_cannot_transfer_across_mapping_methods(review_evidence):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], mapping_method='unit_method_b',
        reviewed_node_ids=['unit-node-a'])
    with pytest.raises(ValueError, match='节点不在'):
        validate_submission(record, output=output)


@pytest.mark.parametrize('label', ['正确关联', '错合并', '重复节点', '身份变化'])
def test_graph_identity_claim_requires_explicit_nodes(review_evidence, label):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], label=label, instance_token='unit-instance')
    with pytest.raises(ValueError, match='明确选择'):
        validate_submission(record, output=output)


def test_explicit_method_and_nodes_are_saved_without_certifying_label(review_evidence):
    output, records = review_evidence
    record = dict(records['causal_dev_evidence_001'], label='正确关联',
        instance_token='unit-instance', reviewed_node_ids=['unit-node-a'])
    result = validate_submission(record, output=output)
    assert result['mapping_method'] == 'unit_method_a'
    assert result['reviewed_node_ids'] == ['unit-node-a']
    assert result['label_correctness_independently_verified'] is False
    assert not (output/'annotation/human_submissions').exists()
