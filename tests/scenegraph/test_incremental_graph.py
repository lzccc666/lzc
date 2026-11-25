from __future__ import annotations

from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.relations import build_relations


def config(**overrides):
    value = {
        "coordinate_frame": "world",
        "coordinate_frame_fallback": "world",
        "minimum_confidence": 0.0,
        "position_update_threshold_m": 0.25,
        "dynamic_edge_ttl_frames": 1,
        "static_stability_frames": 3,
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


def nodes(second_center=(2, 0, 0), dynamic=False):
    return [
        {"id": 0, "bbox_center": [0, 0, 0], "dynamic": False},
        {"id": 1, "bbox_center": list(second_center), "dynamic": dynamic},
    ]


def edge_keys(edges):
    return {(edge.source, edge.target, edge.relation) for edge in edges}


def test_static_edges_accumulate_observations_and_stability():
    graph = IncrementalSceneGraph(config())
    first = graph.update_snapshot(nodes(), frame_index=0)
    second = graph.update_snapshot(nodes(), frame_index=1)
    assert second.edge_count == first.edge_count
    assert all(edge.observation_count > 1 for edge in second.edges)
    assert all(edge.last_seen_frame == 1 for edge in second.edges)


def test_dynamic_edges_keep_valid_range_then_expire_after_ttl():
    graph = IncrementalSceneGraph(config(dynamic_edge_ttl_frames=1))
    first = graph.update_snapshot(nodes(dynamic=True), frame_index=0)
    assert first.edge_count > 0
    moved = graph.update_snapshot(nodes(second_center=(30, 0, 0), dynamic=True), frame_index=1)
    assert moved.edge_count == first.edge_count
    expired = graph.advance_frame(2)
    assert expired.edge_count == 0
    assert expired.update_statistics["expired_dynamic_edge_count"] == first.edge_count


def test_new_node_recomputes_only_its_local_pairs():
    graph = IncrementalSceneGraph(config())
    graph.update_snapshot(nodes(), frame_index=0)
    updated = graph.update_snapshot(
        nodes() + [{"id": 2, "bbox_center": [0, 3, 0]}],
        frame_index=1,
    )
    assert updated.update_statistics["affected_nodes"] == [2]
    assert updated.update_statistics["recomputed_pair_count"] == 2
    assert updated.update_statistics["reused_edge_count"] == 2


def test_subthreshold_position_change_does_not_recompute_geometry():
    graph = IncrementalSceneGraph(config(position_update_threshold_m=0.5))
    graph.update_snapshot(nodes(), frame_index=0)
    updated = graph.update_node_position(1, [2.1, 0, 0], frame_index=1)
    assert updated.update_statistics["recomputed_pair_count"] == 0
    assert updated.update_statistics["below_threshold_nodes"] == [1]


def test_position_and_dynamic_state_changes_recompute_local_neighborhood():
    graph = IncrementalSceneGraph(config())
    graph.update_snapshot(nodes(), frame_index=0)
    moved = graph.update_node_position(1, [3, 0, 0], frame_index=1)
    changed = graph.update_node_dynamic_state(1, True, frame_index=2)
    assert moved.update_statistics["affected_nodes"] == [1]
    assert moved.update_statistics["recomputed_pair_count"] == 1
    assert changed.update_statistics["state_changed_nodes"] == [1]


def test_node_invalidation_removes_incident_edges_without_exception():
    graph = IncrementalSceneGraph(config())
    graph.update_snapshot(nodes(), frame_index=0)
    result = graph.invalidate_node(1, frame_index=1)
    assert result.success
    assert result.edge_count == 0
    assert result.update_statistics["invalidated_nodes"] == [1]


def test_incremental_structure_matches_same_frame_full_rebuild():
    graph = IncrementalSceneGraph(config())
    graph.update_snapshot(nodes(), frame_index=0)
    current_nodes = nodes(second_center=(0, 4, 0))
    incremental = graph.update_snapshot(current_nodes, frame_index=1)
    full = build_relations(current_nodes, config(), frame_index=1)
    assert edge_keys(incremental.edges) == edge_keys(full.edges)


def test_empty_snapshot_and_snapshot_export_are_structured_successes():
    graph = IncrementalSceneGraph(config(coordinate_frame="local_trajectory", coordinate_frame_fallback=None))
    empty = graph.update_snapshot([], frame_index=0)
    snapshot = graph.export_snapshot()
    assert empty.success and empty.edge_count == 0
    assert snapshot.success and snapshot.update_statistics["snapshot_export"]


def test_ego_turn_recomputes_stationary_relations_without_stale_dynamic_direction():
    cfg = config(coordinate_frame="ego_heading", coordinate_frame_fallback=None)
    graph = IncrementalSceneGraph(cfg)
    stationary = nodes(second_center=(4, 0, 0), dynamic=True)
    before = graph.update_snapshot(stationary, frame_index=0, ego_heading=[1, 0])
    after = graph.update_snapshot(stationary, frame_index=1, ego_heading=[0, 1])
    full = build_relations(stationary, cfg, frame_index=1, ego_heading=[0, 1])
    assert edge_keys(before.edges) != edge_keys(after.edges)
    assert edge_keys(after.edges) == edge_keys(full.edges)
    assert after.update_statistics["coordinate_frame_changed"]
    unchanged = graph.update_snapshot(stationary, frame_index=2, ego_heading=[0, 1])
    assert unchanged.update_statistics["recomputed_pair_count"] == 0
