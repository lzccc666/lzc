from __future__ import annotations

from conceptgraph.scenegraph.relations import SceneGraphEdge, finalize_relation_candidates


def config(**overrides):
    value = {
        "coordinate_frame": "world",
        "coordinate_frame_fallback": "world",
        "minimum_confidence": 0.0,
        "top_k_per_relation": {
            "near": 10,
            "left_of": 10,
            "right_of": 10,
            "in_front_of": 10,
            "behind": 10,
        },
        "max_out_degree": 100,
        "max_in_degree": 100,
        "max_total_edges": 100,
    }
    value.update(overrides)
    return value


def test_top_k_is_applied_separately_by_relation_type():
    edges = [
        SceneGraphEdge(0, 1, "near", confidence=0.9),
        SceneGraphEdge(0, 2, "near", confidence=0.8),
        SceneGraphEdge(0, 3, "left_of", confidence=0.7),
        SceneGraphEdge(0, 4, "left_of", confidence=0.6),
    ]
    accepted, rejected, _conflicts, stats = finalize_relation_candidates(
        edges,
        config(top_k_per_relation={"near": 1, "left_of": 1, "right_of": 1, "in_front_of": 1, "behind": 1}),
    )
    assert {(edge.source, edge.target, edge.relation) for edge in accepted} == {
        (0, 1, "near"),
        (0, 3, "left_of"),
    }
    assert len([edge for edge in rejected if edge.rejection_reason == "top_k_per_relation"]) == 2
    assert stats["after_relation_top_k_edge_count"] == 2


def test_maximum_out_degree_is_enforced_deterministically():
    edges = [
        SceneGraphEdge(0, 1, "left_of", confidence=0.9),
        SceneGraphEdge(0, 2, "in_front_of", confidence=0.8),
    ]
    accepted, rejected, _conflicts, stats = finalize_relation_candidates(
        edges,
        config(max_out_degree=1),
    )
    assert [(edge.source, edge.target) for edge in accepted] == [(0, 1)]
    assert rejected[0].rejection_reason == "max_out_degree"
    assert stats["maximum_out_degree"] == 1


def test_maximum_in_degree_is_enforced_deterministically():
    edges = [
        SceneGraphEdge(0, 3, "left_of", confidence=0.9),
        SceneGraphEdge(1, 3, "in_front_of", confidence=0.8),
    ]
    accepted, rejected, _conflicts, stats = finalize_relation_candidates(
        edges,
        config(max_in_degree=1),
    )
    assert [(edge.source, edge.target) for edge in accepted] == [(0, 3)]
    assert rejected[0].rejection_reason == "max_in_degree"
    assert stats["maximum_in_degree"] == 1


def test_maximum_total_edges_keeps_highest_confidence_edges():
    edges = [
        SceneGraphEdge(0, 1, "near", confidence=0.7),
        SceneGraphEdge(2, 3, "near", confidence=0.9),
        SceneGraphEdge(4, 5, "near", confidence=0.8),
    ]
    accepted, rejected, _conflicts, stats = finalize_relation_candidates(
        edges,
        config(max_total_edges=2),
    )
    assert {(edge.source, edge.target) for edge in accepted} == {(2, 3), (4, 5)}
    assert rejected[0].rejection_reason == "max_total_edges"
    assert stats["after_pruning_edge_count"] == 2


def test_near_counts_toward_both_endpoint_degree_limits():
    edges = [
        SceneGraphEdge(0, 1, "near", confidence=0.9),
        SceneGraphEdge(1, 2, "near", confidence=0.8),
    ]
    accepted, rejected, _conflicts, _stats = finalize_relation_candidates(
        edges,
        config(max_out_degree=1, max_in_degree=1),
    )
    assert len(accepted) == 1
    assert rejected[0].rejection_reason == "max_out_degree"
