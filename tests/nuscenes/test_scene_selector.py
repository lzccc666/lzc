from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conceptgraph.nuscenes.scene_selector import (
    MetadataValidationError,
    NuScenesSceneSelector,
    SelectorConfig,
    build_scene_scores_payload,
    build_selected_scenes_payload,
)
from conceptgraph.scripts.build_nuscenes_subset_manifest import validate_output_dir


def _record(token: str, **values: Any) -> dict[str, Any]:
    return {"token": token, **values}


def _write_fixture(
    tmp_path: Path, *, invalid_references: bool = False
) -> tuple[Path, Path]:
    metadata_root = tmp_path / "raw"
    metadata_dir = metadata_root / "v1.0-test"
    data_root = tmp_path / "local-data"
    metadata_dir.mkdir(parents=True)

    scenes = [
        _record(
            "scene-a",
            name="scene-alpha",
            description="Parking lot with static objects",
            nbr_samples=2,
            first_sample_token="sample-a1",
            last_sample_token="sample-a2",
        ),
        _record(
            "scene-b",
            name="scene-beta",
            description="Busy road",
            nbr_samples=2,
            first_sample_token="sample-b1",
            last_sample_token="sample-b2",
        ),
    ]
    samples = [
        _record(
            "sample-a1",
            scene_token="scene-a",
            timestamp=1_000_000,
            prev="",
            next="sample-a2",
        ),
        _record(
            "sample-a2",
            scene_token="scene-a",
            timestamp=1_500_000,
            prev="sample-a1",
            next="",
        ),
        _record(
            "sample-b1",
            scene_token="scene-b",
            timestamp=2_000_000,
            prev="",
            next="sample-b2",
        ),
        _record(
            "sample-b2",
            scene_token="scene-b",
            timestamp=2_500_000,
            prev="sample-b1",
            next="",
        ),
    ]
    categories = [
        _record("cat-car", name="vehicle.car"),
        _record("cat-bike", name="vehicle.bicycle"),
        _record("cat-barrier", name="movable_object.barrier"),
        _record("cat-cone", name="movable_object.trafficcone"),
        _record("cat-rack", name="static_object.bicycle_rack"),
        _record("cat-ped", name="human.pedestrian.adult"),
    ]
    attributes = [
        _record("attr-parked", name="vehicle.parked"),
        _record("attr-moving", name="vehicle.moving"),
        _record("attr-no-rider", name="cycle.without_rider"),
        _record("attr-ped-moving", name="pedestrian.moving"),
    ]
    instances = [
        _record("inst-parked", category_token="cat-car", nbr_annotations=2),
        _record("inst-bike", category_token="cat-bike", nbr_annotations=1),
        _record("inst-barrier", category_token="cat-barrier", nbr_annotations=2),
        _record("inst-cone", category_token="cat-cone", nbr_annotations=1),
        _record("inst-rack", category_token="cat-rack", nbr_annotations=1),
        _record("inst-ped", category_token="cat-ped", nbr_annotations=1),
        _record("inst-moving", category_token="cat-car", nbr_annotations=2),
    ]
    annotations = [
        _record(
            "ann-a1-car",
            sample_token="sample-a1",
            instance_token="inst-parked",
            attribute_tokens=["attr-parked"],
        ),
        _record(
            "ann-a2-car",
            sample_token="sample-a2",
            instance_token="inst-parked",
            attribute_tokens=["attr-parked"],
        ),
        _record(
            "ann-a1-bike",
            sample_token="sample-a1",
            instance_token="inst-bike",
            attribute_tokens=["attr-no-rider"],
        ),
        _record(
            "ann-a1-barrier",
            sample_token="sample-a1",
            instance_token="inst-barrier",
            attribute_tokens=[],
        ),
        _record(
            "ann-a2-barrier",
            sample_token="sample-a2",
            instance_token="inst-barrier",
            attribute_tokens=[],
        ),
        _record(
            "ann-a1-cone",
            sample_token="sample-a1",
            instance_token="inst-cone",
            attribute_tokens=[],
        ),
        _record(
            "ann-a1-rack",
            sample_token="sample-a1",
            instance_token="inst-rack",
            attribute_tokens=[],
        ),
        _record(
            "ann-a2-ped",
            sample_token="sample-a2",
            instance_token="inst-ped",
            attribute_tokens=["attr-ped-moving"],
        ),
        _record(
            "ann-b1-car",
            sample_token="sample-b1",
            instance_token="inst-moving",
            attribute_tokens=["attr-moving"],
        ),
        _record(
            "ann-b2-car",
            sample_token="sample-b2",
            instance_token="inst-moving",
            attribute_tokens=["attr-moving"],
        ),
    ]
    if invalid_references:
        annotations.extend(
            [
                _record(
                    "ann-invalid-instance",
                    sample_token="sample-a1",
                    instance_token="missing-instance",
                    attribute_tokens=[],
                ),
                _record(
                    "ann-invalid-attribute",
                    sample_token="sample-a1",
                    instance_token="inst-parked",
                    attribute_tokens=["missing-attribute"],
                ),
            ]
        )

    sensors = [
        _record("sensor-cam", channel="CAM_FRONT", modality="camera"),
        _record("sensor-lidar", channel="LIDAR_TOP", modality="lidar"),
    ]
    calibrated = [
        _record("calib-cam", sensor_token="sensor-cam"),
        _record("calib-lidar", sensor_token="sensor-lidar"),
    ]
    sample_data: list[dict[str, Any]] = []
    for sample_token in ("sample-a1", "sample-a2", "sample-b1", "sample-b2"):
        for channel, calibrated_token, suffix in (
            ("CAM_FRONT", "calib-cam", "jpg"),
            ("LIDAR_TOP", "calib-lidar", "bin"),
        ):
            short_channel = "cam" if channel == "CAM_FRONT" else "lidar"
            filename = f"samples/{channel}/{sample_token}.{suffix}"
            sample_data.append(
                _record(
                    f"sd-{sample_token}-{short_channel}",
                    sample_token=sample_token,
                    calibrated_sensor_token=calibrated_token,
                    is_key_frame=True,
                    filename=filename,
                )
            )
            if not (sample_token == "sample-b2" and channel == "LIDAR_TOP"):
                path = data_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

    tables = {
        "scene": scenes,
        "sample": samples,
        "sample_annotation": annotations,
        "instance": instances,
        "attribute": attributes,
        "category": categories,
        "calibrated_sensor": calibrated,
        "sensor": sensors,
        "sample_data": sample_data,
    }
    for name, records in tables.items():
        (metadata_dir / f"{name}.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

    converted = data_root / "scene-alpha"
    for directory, suffix in (("color", "jpg"), ("depth", "png")):
        (converted / directory).mkdir(parents=True)
        for index in range(2):
            (converted / directory / f"{index:06d}.{suffix}").write_bytes(b"fixture")
    (converted / "poses.txt").write_text("pose-1\npose-2\n", encoding="utf-8")
    return metadata_root, data_root


def _score(tmp_path: Path, *, invalid_references: bool = False):
    metadata_root, data_root = _write_fixture(
        tmp_path, invalid_references=invalid_references
    )
    return NuScenesSceneSelector(
        SelectorConfig(
            metadata_root=metadata_root,
            version="v1.0-test",
            data_roots=(data_root,),
        )
    ).score_scenes()


def test_structured_scoring_and_attribute_mapping(tmp_path: Path) -> None:
    result = _score(tmp_path)
    row = next(
        item for item in result.scene_scores if item["scene_name"] == "scene-alpha"
    )

    assert row["vehicle_parked_count"] == 2
    assert row["vehicle_parked_ratio"] == pytest.approx(2 / 3, abs=1e-6)
    assert row["cycle_without_rider_count"] == 1
    assert row["barrier_count"] == 2
    assert row["traffic_cone_count"] == 1
    assert row["bicycle_rack_count"] == 1
    assert row["static_target_count"] == 7
    assert row["static_target_instance_count"] == 5
    assert row["static_target_persistence"] == {
        "mean_frames": pytest.approx(7 / 5),
        "max_frames": 2,
        "mean_ratio": 0.7,
        "max_ratio": 1.0,
    }
    assert row["dynamic_target_count"] == 1
    assert row["cam_front_lidar_top_complete_frames"] == 2
    assert row["data_completeness"] == 1.0
    assert row["complete_for_local_conversion"] is True
    assert row["local_converted_data"]["frame_count_matches_source"] is True
    assert set(row["score_components"]) == {
        "parked_vehicle",
        "static_target_diversity",
        "static_persistence",
        "low_dynamic",
        "data_completeness",
        "sequence_length",
        "description_context",
    }


def test_selection_filters_incomplete_scenes_and_sorts_deterministically(
    tmp_path: Path,
) -> None:
    result = _score(tmp_path)
    selected = result.select(5)

    assert [row["scene_name"] for row in result.scene_scores] == [
        "scene-alpha",
        "scene-beta",
    ]
    assert [row["scene_name"] for row in selected] == ["scene-alpha"]
    assert selected[0]["selection_rank"] == 1
    beta = result.scene_scores[1]
    assert beta["complete_for_local_conversion"] is False
    assert beta["missing_sensor_file_count"] == 1
    assert (
        "requires 2 complete source pairs; found 1" == beta["local_conversion_reason"]
    )


def test_invalid_tokens_and_missing_sensor_files_are_diagnosed(tmp_path: Path) -> None:
    result = _score(tmp_path, invalid_references=True)
    report = result.integrity_report

    assert report["metadata_status"] == "issues_detected"
    assert report["invalid_reference_count"] == 2
    assert report["invalid_reference_counts"] == {
        "sample_annotation.attribute_tokens": 1,
        "sample_annotation.instance_token": 1,
    }
    assert report["sensor_file_coverage"]["missing_sensor_file_count"] == 1
    assert (
        report["sensor_file_coverage"]["missing_examples"][0]["channel"] == "LIDAR_TOP"
    )


def test_missing_required_table_has_explicit_error(tmp_path: Path) -> None:
    metadata_root, data_root = _write_fixture(tmp_path)
    (metadata_root / "v1.0-test" / "instance.json").unlink()
    selector = NuScenesSceneSelector(
        SelectorConfig(
            metadata_root=metadata_root,
            version="v1.0-test",
            data_roots=(data_root,),
        )
    )

    with pytest.raises(MetadataValidationError, match=r"instance\.json"):
        selector.score_scenes()


def test_repeated_scoring_and_payloads_are_identical(tmp_path: Path) -> None:
    metadata_root, data_root = _write_fixture(tmp_path)
    config = SelectorConfig(
        metadata_root=metadata_root,
        version="v1.0-test",
        data_roots=(data_root,),
    )
    first = NuScenesSceneSelector(config).score_scenes()
    second = NuScenesSceneSelector(config).score_scenes()

    assert first.scene_scores == second.scene_scores
    assert first.integrity_report == second.integrity_report
    assert build_scene_scores_payload(first) == build_scene_scores_payload(second)
    assert build_selected_scenes_payload(first, 2) == build_selected_scenes_payload(
        second, 2
    )


def test_num_scenes_must_be_positive(tmp_path: Path) -> None:
    result = _score(tmp_path)
    with pytest.raises(ValueError, match="greater than zero"):
        result.select(0)


def test_output_directory_cannot_escape_project_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "outputs" / "project_v2"
    monkeypatch.setattr(
        "conceptgraph.scripts.build_nuscenes_subset_manifest.PROJECT_V2_ROOT", allowed
    )
    with pytest.raises(ValueError, match="outputs/project_v2"):
        validate_output_dir(tmp_path)
    assert validate_output_dir(allowed / "subset") == (allowed / "subset").resolve()


def test_version_directory_is_preferred_over_same_named_child(tmp_path: Path) -> None:
    metadata_root, data_root = _write_fixture(tmp_path)
    metadata_dir = metadata_root / "v1.0-test"
    (metadata_dir / "v1.0-test").mkdir()

    result = NuScenesSceneSelector(
        SelectorConfig(
            metadata_root=metadata_dir,
            version="v1.0-test",
            data_roots=(data_root,),
        )
    ).score_scenes()

    assert result.metadata_dir == metadata_dir.resolve()
    assert len(result.scene_scores) == 2
