import pytest

from conceptgraph.nuscenes.benchmark_ground_truth import accepted_targets, associate_nodes, retrieval_metrics


def test_independent_targets_use_geometry_and_allow_equal_nearest_objects():
    objects = [{"instance_token": "a", "category": "car", "center": [4, 0, 0]}, {"instance_token": "b", "category": "car", "center": [4.2, 0, 0]}, {"instance_token": "anchor", "category": "truck", "center": [4, -3, 0]}]
    assert accepted_targets(objects, "car", [0, 0], nearest=True) == ["a", "b"]
    assert accepted_targets(objects, "car", [0, 0], anchor_category="truck", relation="left_of", forward_xy=[1, 0]) == ["a", "b"]
    assert accepted_targets(objects, "car", [0, 0], anchor_category="truck", relation="right_of", forward_xy=[1, 0]) == []


def test_missing_target_counts_as_retrieval_failure_instead_of_disappearing():
    association = associate_nodes([{"node_id": 1, "category": "truck", "center": [1, 0, 0]}], [{"instance_token": "gt", "category": "car", "center": [1, 0, 0]}])
    metrics = retrieval_metrics([1], ["gt"], association)
    assert metrics["mrr"] == 0
    assert not metrics["top1"]
    assert metrics["candidate_recall"] == 0
    assert association["unmatched_instances"] == ["gt"]


def test_ambiguous_association_cannot_manufacture_a_correct_label():
    association = associate_nodes([{"node_id": 1, "category": "car", "center": [0, 0, 0]}], [{"instance_token": "a", "category": "car", "center": [1, 0, 0]}, {"instance_token": "b", "category": "car", "center": [1.1, 0, 0]}])
    assert not association["matches"]
    assert association["ambiguous_nodes"] == [1]
    assert retrieval_metrics([1], ["a"], association)["mrr"] == 0
    assert retrieval_metrics([], [], association)["correct_rejection"]
    with pytest.raises(ValueError, match="duplicate"):
        retrieval_metrics([1, 1], ["a"], association)
