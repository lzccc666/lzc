import numpy as np
import pytest

from conceptgraph.completion.realtime_faults import FaultPolicy, retain_detections


def test_motion_trigger_does_not_use_frame_count_or_stationary_elapsed_time():
    p = FaultPolicy("lidar_dropout_80pct", 17)
    assert not p.begin_frame(0, [0, 0], 0, 10)
    assert not p.begin_frame(1, [0, 0], 0, 100)
    assert not p.begin_frame(2, [0.4, 0], 1, 101)
    assert not p.begin_frame(3, [0.6, 0], 0, 102)
    assert p.begin_frame(4, [0.6, 0], 1, 103)
    assert p.begin_frame(5, [0.7, 0], 1, 108.99)
    assert not p.begin_frame(6, [0.8, 0], 1, 109)
    assert p.summary()["active_frames"] == [4, 5]


@pytest.mark.parametrize(
    "condition",
    [
        "baseline",
        "sensor_blackout_4s",
        "lidar_dropout_80pct",
        "depth_noise_02m",
        "pose_noise_025m_2deg",
        "detection_dropout_30pct",
    ],
)
def test_fault_sensor_inputs_preserved_and_deterministic(condition):
    depth = np.zeros((12, 12))
    depth[2:10, 2:10] = 3.0
    pose = np.eye(4)
    before_depth = depth.copy()
    before_pose = pose.copy()
    outputs = []
    for _ in range(2):
        p = FaultPolicy(condition, 17)
        p.begin_frame(0, [0, 0], 0, 1)
        p.begin_frame(1, [1, 0], 1, 2)
        outputs.append(p.transform_sensors(depth, pose))
    assert np.array_equal(depth, before_depth) and np.array_equal(pose, before_pose)
    assert all(np.array_equal(a, b) for a, b in zip(*outputs))
    millimeters, measured = outputs[0]
    assert millimeters.dtype == np.uint16
    assert np.all(millimeters[depth == 0] == 0)
    assert np.allclose(measured[:3, :3].T @ measured[:3, :3], np.eye(3))
    if condition == "lidar_dropout_80pct":
        assert 0 < np.count_nonzero(millimeters) < np.count_nonzero(depth)
    if condition == "depth_noise_02m":
        assert not np.array_equal(millimeters / 1000, depth)
    if condition == "pose_noise_025m_2deg":
        assert not np.array_equal(measured, pose)


def test_detection_dropout_keeps_cross_field_identity_and_class_vocabulary():
    gobs = {
        k: np.arange(30)
        for k in ["xyxy", "confidence", "class_id", "mask", "image_feats", "text_feats"]
    }
    gobs.update(
        image_crops=list(range(30)),
        classes=["a", "b"],
        frame_provenance={"rgb_sha256": "actual"},
    )
    p = FaultPolicy("detection_dropout_30pct", 17)
    p.begin_frame(0, [0, 0], 0, 1)
    p.begin_frame(1, [1, 0], 1, 2)
    actual, keep = p.transform_detections(gobs)
    assert 0 < len(keep) < 30
    for key in [
        "xyxy",
        "confidence",
        "class_id",
        "mask",
        "image_feats",
        "text_feats",
        "image_crops",
    ]:
        assert np.array_equal(actual[key], keep)
    assert (
        actual["classes"] == gobs["classes"]
        and actual["frame_provenance"] == gobs["frame_provenance"]
    )
    assert len(gobs["xyxy"]) == 30


def test_no_trigger_is_unknown_exposure():
    p = FaultPolicy("sensor_blackout_4s", 17)
    p.begin_frame(0, [0, 0], 0, 1)
    assert (
        not p.summary()["actual_fault_exposed"] and p.summary()["onset_frame"] is None
    )


@pytest.mark.parametrize(
    "mask", [np.array([1, 0]), np.array([True]), np.array([[True, False]])]
)
def test_rejects_misaligned_detection_masks(mask):
    with pytest.raises(ValueError):
        retain_detections({"xyxy": np.zeros((2, 4))}, mask)


def test_rejects_misaligned_actual_detection_fields():
    with pytest.raises(ValueError):
        retain_detections(
            {"xyxy": np.zeros((2, 4)), "mask": np.zeros((1, 3, 3))},
            np.array([True, False]),
        )


def test_rejects_future_order_and_invalid_physical_inputs():
    p = FaultPolicy("baseline", 17)
    p.begin_frame(0, [0, 0], 0, 1)
    with pytest.raises(ValueError):
        p.begin_frame(0, [0, 0], 0, 2)
    with pytest.raises(ValueError):
        p.begin_frame(1, [0, 0], 0, 0.9)
    with pytest.raises(ValueError):
        p.begin_frame(1, [0, 0], -1, 2)
    with pytest.raises(ValueError):
        p.transform_sensors(np.full((2, 2), np.nan), np.eye(4))


def test_isolated_hooks_preserve_actual_files_and_restore_after_failure(
    monkeypatch, tmp_path
):
    """小图仅作I/O隔离测试，不是实际模型、人工标签或闭环实验。"""
    from types import SimpleNamespace
    from PIL import Image
    import conceptgraph.completion as completion
    import conceptgraph.scripts as scripts
    from conceptgraph.completion.stage9 import digest, dump
    from conceptgraph.completion.realtime_faults import installed_faults

    class Environment:
        def __init__(self):
            self.state = SimpleNamespace(x=0.0, y=0.0, speed_mps=0.0)

        def observe(self, dest, index):
            dest.mkdir(parents=True, exist_ok=True)
            rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
            depth = np.full((8, 8), 3000, dtype=np.uint16)
            color = dest / f"{index}.jpg"
            d = dest / f"{index}.png"
            Image.fromarray(rgb).save(color)
            Image.fromarray(depth).save(d)
            evidence = dict(
                lidar_pixels=64,
                rgb=str(color),
                rgb_sha256=digest(color),
                depth=str(d),
                depth_sha256=digest(d),
                camera_pose_cv=np.eye(4).tolist(),
            )
            dump(dest / "sensor_evidence" / f"{index:06d}.json", evidence)
            return rgb, depth / 1000, np.eye(4), color, evidence

    def detect(*_):
        return dict(
            xyxy=np.zeros((2, 4)), classes=["car"], frame_provenance={"scope": "unit"}
        ), None

    front = SimpleNamespace(
        detect_rgb_file=detect, load_detection_models=lambda cfg: cfg.device
    )
    env = SimpleNamespace(InteractiveRoadEnvironment=Environment)
    nav = SimpleNamespace(WallClockVehicle=object)
    monkeypatch.setattr(completion, "interactive_environment", env, raising=False)
    monkeypatch.setattr(completion, "realtime_navigation", nav, raising=False)
    monkeypatch.setattr(scripts, "streamlined_detections", front, raising=False)
    policy = FaultPolicy("lidar_dropout_80pct", 17)
    with pytest.raises(RuntimeError, match="injected test failure"):
        with installed_faults(policy, "cpu"):
            active = env.InteractiveRoadEnvironment()
            active.observe(tmp_path, 0)
            active.state = SimpleNamespace(x=1.0, y=0.0, speed_mps=1.0)
            rgb, depth, pose, color, evidence = active.observe(tmp_path, 1)
            assert np.array_equal(
                depth, np.asarray(Image.open(tmp_path / "depth/000001.png")) / 1000
            )
            assert digest(color) == digest(tmp_path / "unmodified_sensors/1.jpg")
            assert 0 < np.count_nonzero(depth) < 64
            assert np.all(
                np.asarray(Image.open(tmp_path / "unmodified_sensors/1.png")) == 3000
            )
            assert evidence["physical_pose_not_modified"] and np.array_equal(
                pose, np.eye(4)
            )
            assert (
                front.load_detection_models(SimpleNamespace(device="cuda:0")) == "cpu"
            )
            effective, _ = front.detect_rgb_file(None, None, color, ["car"])
            assert set(effective) == {"xyxy", "classes", "frame_provenance"}
            raise RuntimeError("injected test failure")
    assert env.InteractiveRoadEnvironment is Environment
    assert front.detect_rgb_file is detect and nav.WallClockVehicle is object
    assert (tmp_path / "fault_exposure.json").exists()
