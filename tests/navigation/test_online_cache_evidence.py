import os
import shutil

import numpy as np
from PIL import Image

from conceptgraph.navigation.online import _object_evidence_version
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)
from tests.navigation.conftest import make_node


def test_relocated_identical_crops_hit_cache_but_changed_pixels_do_not(tmp_path):
    first = tmp_path / "first.png"
    repeat = tmp_path / "relocated.png"
    Image.new("RGB", (8, 8), "red").save(first)
    shutil.copyfile(first, repeat)
    os.utime(repeat, ns=(1, 1))
    node = make_node(1)
    metadata = {"object_tag": "car"}

    def backend(*args, **kwargs):
        return '{"schema_version":"1.0.0","summary":"red car","object_tag":"car","possible_tags":["car"],"colors":["red"],"body_type":null,"motion_state":"static","confidence":0.8}'

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True, use_qwen_object_attributes=True, cache_dir=tmp_path / "cache"
        ),
        backend=backend,
    )
    try:
        a = delegate.describe_object(
            [str(first)],
            metadata,
            object_version=_object_evidence_version(node, [str(first)]),
        )
        b = delegate.describe_object(
            [str(repeat)],
            metadata,
            object_version=_object_evidence_version(node, [str(repeat)]),
        )
        assert a["used_qwen"] and b["cache"]["hit"]
        assert delegate.backend_call_count == 1
        Image.new("RGB", (8, 8), "blue").save(repeat)
        c = delegate.describe_object(
            [str(repeat)],
            metadata,
            object_version=_object_evidence_version(node, [str(repeat)]),
        )
        assert not c["cache"]["hit"]
        assert delegate.backend_call_count == 2
    finally:
        delegate.close()


def test_sparse_depth_zbuffer_keeps_nearest_return_independent_of_order():
    points = np.array([[0, 0, 4], [0, 0, 2], [0, 0, 6], [0, 0, -1]])
    forward = nearest_depth_image(points, np.eye(3), 4, 4)
    backward = nearest_depth_image(points[::-1], np.eye(3), 4, 4)
    assert forward[0, 0] == 2
    assert (forward == backward).all()
    assert np.count_nonzero(forward) == 1


def test_explicit_camera_axis_is_identical_across_planning_conditions():
    from conceptgraph.navigation.online import OnlineSemanticNavigator
    from tests.navigation.test_lane_qwen_safety import pipeline
    from tests.navigation.test_lane_constraints import _grid

    cfg = pipeline().cfg
    navigator = OnlineSemanticNavigator(
        cfg, online_config={"pose_forward_axis": 2}, base_occupancy_grid=_grid()
    )
    try:
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        lane = navigator._normalize_vehicle_pose(pose)
        cfg.target_selection = {"mode": "joint_utility"}
        normal = navigator._normalize_vehicle_pose(pose)
        assert np.allclose(lane, normal)
        assert np.allclose(lane[3:5], [1, 0])
    finally:
        navigator.close()


def test_late_attribute_result_cannot_apply_to_changed_crop(tmp_path):
    from concurrent.futures import Future
    from conceptgraph.navigation.online import OnlineSemanticNavigator, _PendingQwenTask
    from tests.navigation.test_lane_qwen_safety import pipeline
    from tests.navigation.test_lane_constraints import _grid

    crop = tmp_path / "crop.png"
    Image.new("RGB", (8, 8), "red").save(crop)
    node = make_node(1)
    node.metadata["crop_paths"] = [str(crop)]
    version = _object_evidence_version(node, [str(crop)])
    nav = OnlineSemanticNavigator(pipeline().cfg, base_occupancy_grid=_grid())
    try:
        nav._nodes = {1: node}
        pending = _PendingQwenTask(
            task_id=1,
            task="object_attributes",
            future=Future(),
            binding={},
            submitted_frame=0,
            context={
                "node_id": 1,
                "detector_tag": node.object_tag,
                "evidence_version": version,
            },
        )
        Image.new("RGB", (8, 8), "blue").save(crop)
        applied, diagnostic = nav._accept_qwen_object_attributes(
            pending, {"used_qwen": True, "normalized_output": {"confidence": 0.9}}
        )
        assert not applied and diagnostic == "object_evidence_version_changed"
        assert not nav._qwen_node_attributes
    finally:
        nav.close()
