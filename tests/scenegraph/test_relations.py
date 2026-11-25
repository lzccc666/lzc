from __future__ import annotations

import json
import pickle

import numpy as np

from conceptgraph.navigation.semantic import build_relation_index, score_relation_constraint
from conceptgraph.navigation.types import ParsedNavigationQuery, SceneGraphNode
from conceptgraph.scenegraph.relations import (
    SceneGraphEdge,
    build_relations,
    edge_relation_for_query,
    finalize_relation_candidates,
    load_scene_graph_edges,
    save_structured_edges,
)
from conceptgraph.scenegraph.visualization import edge_hover_text, filter_edges


def config(**overrides):
    value = {
        "coordinate_frame": "world",
        "coordinate_frame_fallback": "world",
        "minimum_confidence": 0.0,
        "top_k_per_relation": {
            "near": 20,
            "left_of": 20,
            "right_of": 20,
            "in_front_of": 20,
            "behind": 20,
        },
        "max_out_degree": 100,
        "max_in_degree": 100,
        "max_total_edges": 1000,
    }
    value.update(overrides)
    return value


def edge_by_relation(result, relation):
    return next(edge for edge in result.edges if edge.relation == relation)


def test_near_is_symmetric_canonical_and_deduplicated():
    result = build_relations(
        [{"id": 8, "bbox_center": [0, 0, 0]}, {"id": 2, "bbox_center": [1, 0, 0]}],
        config(),
    )
    near = edge_by_relation(result, "near")
    assert (near.source, near.target, near.directed) == (2, 8, False)
    assert edge_relation_for_query(near, 2, 8) == "near"
    assert edge_relation_for_query(near, 8, 2) == "near"


def test_left_right_share_one_canonical_fact_and_reverse_api():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [-2, 0, 0]}],
        config(),
    )
    edge = edge_by_relation(result, "left_of")
    assert (edge.source, edge.target) == (1, 0)
    assert edge_relation_for_query(edge, 1, 0) == "left_of"
    assert edge_relation_for_query(edge, 0, 1) == "right_of"


def test_front_behind_share_one_canonical_fact_and_reverse_api():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [0, 3, 0]}],
        config(),
    )
    edge = edge_by_relation(result, "in_front_of")
    assert (edge.source, edge.target) == (1, 0)
    assert edge_relation_for_query(edge, 1, 0) == "in_front_of"
    assert edge_relation_for_query(edge, 0, 1) == "behind"


def test_world_coordinate_convention_is_positive_y_front_negative_x_left():
    result = build_relations(
        [
            {"id": 0, "bbox_center": [0, 0, 0]},
            {"id": 1, "bbox_center": [0, 4, 0]},
            {"id": 2, "bbox_center": [-4, 0, 0]},
        ],
        config(),
    )
    facts = {(edge.source, edge.target, edge.relation) for edge in result.edges}
    assert (1, 0, "in_front_of") in facts
    assert (2, 0, "left_of") in facts


def test_ego_heading_uses_heading_as_forward_axis():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [4, 0, 0]}],
        config(coordinate_frame="ego_heading"),
        ego_heading=[1, 0],
    )
    edge = edge_by_relation(result, "in_front_of")
    assert (edge.source, edge.target, edge.coordinate_frame) == (1, 0, "ego_heading")


def test_diagonal_relations_preserve_both_axes_and_observation_provenance():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0], "last_seen_frame": 4,
          "metadata": {"timestamp_s": 10.0}},
         {"id": 1, "bbox_center": [4, 3, 0], "last_seen_frame": 5,
          "metadata": {"timestamp_s": 10.5}}],
        config(coordinate_frame="ego_heading", directional_mode="all_supported_axes"),
        frame_index=6, ego_heading=[1, 0],
    )
    front = edge_by_relation(result, "in_front_of")
    left = edge_by_relation(result, "left_of")
    assert (front.source, front.target) == (1, 0)
    assert (left.source, left.target) == (1, 0)
    assert front.metadata["computed_frame"] == 6
    assert front.metadata["source_observed_frame"] == 5
    assert front.metadata["target_observed_frame"] == 4
    assert front.metadata["source_timestamp_s"] == 10.5
    assert front.metadata["target_timestamp_s"] == 10.0
    assert front.metadata["forward_axis_world"] == [1.0, 0.0]


def test_axis_component_below_margin_does_not_create_direction():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [0.9, 4, 0]}],
        config(directional_mode="all_supported_axes", minimum_direction_component_m=1.0),
    )
    assert not any(edge.relation == "left_of" for edge in result.edges)
    assert any(edge.relation == "in_front_of" for edge in result.edges)


def test_local_trajectory_uses_local_tangent_as_forward_axis():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [4, 0, 0]}],
        config(coordinate_frame="local_trajectory"),
        trajectory=[[0, 0], [1, 0], [2, 0]],
        ego_position=[1, 0],
    )
    edge = edge_by_relation(result, "in_front_of")
    assert edge.coordinate_frame == "local_trajectory"


def test_degenerate_heading_returns_structured_fallback_diagnostic():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [1, 0, 0]}],
        config(coordinate_frame="ego_heading", coordinate_frame_fallback="world"),
        ego_heading=[0, 0],
    )
    diagnostic = result.diagnostics["coordinate_frame"]
    assert result.success
    assert diagnostic["actual_frame"] == "world"
    assert "degenerate_ego_heading" in diagnostic["fallback_reason"]
    assert all(edge.coordinate_frame == "world" for edge in result.edges)


def test_degenerate_trajectory_without_fallback_is_structured_failure():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [1, 0, 0]}],
        config(coordinate_frame="local_trajectory", coordinate_frame_fallback=None),
        trajectory=[[0, 0], [0, 0]],
    )
    assert not result.success
    assert result.failure_code == "COORDINATE_FRAME_UNAVAILABLE"
    assert result.edges == []


def test_self_loop_is_rejected_without_exception():
    edge = SceneGraphEdge(3, 3, "near")
    accepted, rejected, _conflicts, _pruning = finalize_relation_candidates([edge], config())
    assert accepted == []
    assert rejected[0].rejection_reason == "self_loop"


def test_duplicate_edges_merge_observations_and_time_range():
    edges = [
        SceneGraphEdge(0, 1, "near", observation_count=2, first_seen_frame=2, last_seen_frame=3),
        SceneGraphEdge(1, 0, "near", observation_count=3, first_seen_frame=1, last_seen_frame=5),
    ]
    accepted, _rejected, conflicts, _pruning = finalize_relation_candidates(edges, config())
    assert len(accepted) == 1
    assert accepted[0].observation_count == 5
    assert (accepted[0].first_seen_frame, accepted[0].last_seen_frame) == (1, 5)
    assert conflicts["duplicate_edges_merged"] == 1


def test_direction_conflict_uses_confidence_then_deterministic_tie_break():
    edges = [
        SceneGraphEdge(0, 1, "left_of", confidence=0.8, observation_count=2),
        SceneGraphEdge(1, 0, "left_of", confidence=0.9, observation_count=2),
    ]
    accepted, rejected, conflicts, _pruning = finalize_relation_candidates(edges, config())
    assert [(edge.source, edge.target) for edge in accepted] == [(1, 0)]
    assert rejected[0].rejection_reason == "direction_conflict_lower_confidence"
    assert conflicts["left_right_conflicts"] == 1


def test_front_behind_conflict_keeps_only_one_high_confidence_fact():
    edges = [
        SceneGraphEdge(0, 1, "in_front_of", confidence=0.91),
        SceneGraphEdge(0, 1, "behind", confidence=0.72),
    ]
    accepted, rejected, conflicts, _pruning = finalize_relation_candidates(edges, config())
    assert [(edge.source, edge.target, edge.relation) for edge in accepted] == [
        (0, 1, "in_front_of")
    ]
    assert rejected[0].rejection_reason == "direction_conflict_lower_confidence"
    assert conflicts["front_behind_conflicts"] == 1


def test_confidence_minimum_rejects_weak_relations_with_reason():
    result = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [7.9, 0, 0]}],
        config(
            relation_distance_thresholds_m={
                "near": 8.0,
                "directional": 8.0,
                "left_of": 8.0,
                "right_of": 8.0,
                "in_front_of": 8.0,
                "behind": 8.0,
            },
            confidence_weights={
                "distance": 1.0,
                "angle_margin": 0.0,
                "observation_count": 0.0,
                "stability": 0.0,
                "coordinate_frame": 0.0,
                "dynamic_state": 0.0,
            },
            minimum_confidence=0.1,
        ),
    )
    assert result.edges == []
    assert {edge.rejection_reason for edge in result.rejected_edges} == {"below_minimum_confidence"}


def test_empty_and_relation_free_graphs_are_successful_structured_results():
    empty = build_relations([], config(coordinate_frame="local_trajectory", coordinate_frame_fallback=None))
    isolated = build_relations(
        [{"id": 0, "bbox_center": [0, 0, 0]}, {"id": 1, "bbox_center": [100, 100, 0]}],
        config(),
    )
    assert empty.success and empty.edge_count == 0
    assert isolated.success and isolated.edge_count == 0
    assert isolated.failure_code is None


def test_legacy_pickle_and_structured_json_round_trip(tmp_path):
    legacy_path = tmp_path / "legacy.pkl"
    with legacy_path.open("wb") as handle:
        pickle.dump([(2, 1, "right_of"), (4, 3, "near")], handle)
    legacy = load_scene_graph_edges(legacy_path)
    assert legacy[0].metadata["edge_format"] == "legacy"
    assert all(edge.metadata["metadata_available"] is False for edge in legacy)

    structured_path = tmp_path / "structured.json"
    save_structured_edges(legacy, structured_path)
    restored = load_scene_graph_edges(structured_path)
    assert [edge.to_dict() for edge in restored] == [edge.to_dict() for edge in legacy]
    json.loads(structured_path.read_text(encoding="utf-8"))

    structured_pickle = tmp_path / "structured.pkl"
    save_structured_edges(restored, structured_pickle)
    assert [edge.to_dict() for edge in load_scene_graph_edges(structured_pickle)] == [
        edge.to_dict() for edge in restored
    ]


def test_visualization_filter_and_hover_use_structured_metadata():
    edges = [
        SceneGraphEdge(0, 1, "near", confidence=0.4, distance_m=2.0),
        SceneGraphEdge(
            1,
            2,
            "left_of",
            confidence=0.8,
            distance_m=3.0,
            coordinate_frame="ego_heading",
            first_seen_frame=2,
            last_seen_frame=6,
            observation_count=5,
        ),
    ]
    filtered = filter_edges(edges, relation_types=["left_of"], minimum_confidence=0.7)
    assert [(edge.source, edge.target, edge.relation) for edge in filtered] == [(1, 2, "left_of")]
    hover = edge_hover_text(filtered[0])
    for field in ("source: 1", "target: 2", "distance: 3.000 m", "confidence: 0.8000"):
        assert field in hover
    assert "coordinate_frame: ego_heading" in hover
    assert "observations: 5" in hover
    assert "valid_frames: [2, 6]" in hover


def test_output_order_and_content_are_deterministic_except_timing():
    nodes = [
        {"id": 9, "bbox_center": [0, 0, 0]},
        {"id": 2, "bbox_center": [-2, 0, 0]},
        {"id": 5, "bbox_center": [0, 3, 0]},
    ]
    first = build_relations(nodes, config()).to_dict()
    second = build_relations(list(reversed(nodes)), config()).to_dict()
    first.pop("timing")
    second.pop("timing")
    assert first == second


def navigation_node(node_id, tag, center):
    return SceneGraphNode(
        node_id=node_id,
        object_tag=tag,
        possible_tags=[tag],
        caption=tag,
        bbox_center=np.asarray(center, dtype=float),
        bbox_extent=np.ones(3, dtype=float),
    )


def relation_query(relation):
    return ParsedNavigationQuery(
        raw_text=f"car {relation} sign",
        target_text="car",
        anchor_text="sign",
        relation=relation,
    )


def test_navigation_index_preserves_directed_semantics_and_near_symmetry():
    index = build_relation_index(
        [
            SceneGraphEdge(0, 1, "left_of", confidence=0.9),
            SceneGraphEdge(0, 2, "near", directed=False, confidence=0.8),
        ]
    )
    assert [(entry["neighbor_id"], entry["relation"]) for entry in index[0]] == [
        (1, "left of"),
        (2, "near"),
    ]
    assert index[1][0]["relation"] == "right of"
    assert index[2][0]["relation"] == "near"


def test_navigation_prefers_structured_metadata_and_records_relation_chain():
    nodes_by_id = {
        0: navigation_node(0, "car", [-2, 0, 0]),
        1: navigation_node(1, "traffic sign", [0, 0, 0]),
    }
    edge = SceneGraphEdge(
        0,
        1,
        "left_of",
        confidence=0.8,
        distance_m=2.0,
        coordinate_frame="world",
        metadata={"edge_format": "structured", "metadata_available": True},
    )
    score, debug = score_relation_constraint(
        nodes_by_id[0],
        relation_query("left_of"),
        nodes_by_id,
        build_relation_index([edge]),
        anchor_text_scores={0: 0.0, 1: 1.0},
        minimum_edge_confidence=0.35,
    )
    assert score == 0.8
    assert debug["matches"][0]["edge_source"] == "structured_edge"
    assert debug["matches"][0]["relation_chain"][0]["stored_relation"] == "left_of"
    assert debug["matches"][0]["geometry_debug"] is None


def test_navigation_uses_geometry_fallback_for_legacy_or_missing_metadata():
    nodes_by_id = {
        0: navigation_node(0, "car", [-2, 0, 0]),
        1: navigation_node(1, "traffic sign", [0, 0, 0]),
    }
    legacy = SceneGraphEdge(
        0,
        1,
        "left_of",
        metadata={"edge_format": "legacy", "metadata_available": False},
    )
    score, debug = score_relation_constraint(
        nodes_by_id[0],
        relation_query("left_of"),
        nodes_by_id,
        build_relation_index([legacy]),
        anchor_text_scores={0: 0.0, 1: 1.0},
        minimum_edge_confidence=0.35,
        fallback_coordinate_frame="world",
    )
    best = debug["matches"][0]
    assert score == 1.0
    assert best["edge_source"] == "geometry_fallback"
    assert best["fallback_reason"] == ["legacy_edge_missing_metadata"]


def test_navigation_low_confidence_structured_edge_falls_back_to_geometry():
    nodes_by_id = {
        0: navigation_node(0, "car", [-2, 0, 0]),
        1: navigation_node(1, "traffic sign", [0, 0, 0]),
    }
    weak = SceneGraphEdge(
        0,
        1,
        "left_of",
        confidence=0.1,
        metadata={"edge_format": "structured", "metadata_available": True},
    )
    score, debug = score_relation_constraint(
        nodes_by_id[0],
        relation_query("left_of"),
        nodes_by_id,
        build_relation_index([weak]),
        anchor_text_scores={0: 0.0, 1: 1.0},
        minimum_edge_confidence=0.5,
    )
    assert score == 1.0
    assert debug["matches"][0]["fallback_reason"] == ["structured_edge_below_confidence"]
