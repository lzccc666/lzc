"""Deterministic, metadata-driven selection of nuScenes parking scenes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REQUIRED_TABLES = (
    "scene",
    "sample",
    "sample_annotation",
    "instance",
    "attribute",
    "category",
    "calibrated_sensor",
    "sensor",
    "sample_data",
)
REQUIRED_CHANNELS = ("CAM_FRONT", "LIDAR_TOP")
STATIC_CATEGORIES = {
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": "bicycle_rack",
}
PARKING_CONTEXT_TERMS = ("parking", "parked", "parking lot", "park ")
SCORING_RULES: dict[str, dict[str, Any]] = {
    "parked_vehicle": {
        "max_points": 20.0,
        "rule": "65% capped parked annotation count (40) + 35% parked/vehicle ratio",
    },
    "static_target_diversity": {
        "max_points": 15.0,
        "rule": "60% capped unique static instances (10) + 40% represented static types (5)",
    },
    "static_persistence": {
        "max_points": 15.0,
        "rule": "mean static-instance observed-frame ratio",
    },
    "low_dynamic": {
        "max_points": 15.0,
        "rule": "one minus dynamic annotation ratio; zero when a scene has no annotations",
    },
    "data_completeness": {
        "max_points": 25.0,
        "rule": "available CAM_FRONT and LIDAR_TOP files / expected channel files",
    },
    "sequence_length": {
        "max_points": 5.0,
        "rule": "metadata samples capped at 40",
    },
    "description_context": {
        "max_points": 5.0,
        "rule": "parking-context term matches capped at two",
    },
}


class MetadataValidationError(RuntimeError):
    """Raised when required metadata cannot be parsed or is structurally invalid."""


@dataclass(frozen=True)
class SelectorConfig:
    metadata_root: Path
    version: str = "v1.0-trainval"
    data_roots: tuple[Path, ...] = ()
    diagnostic_example_limit: int = 50


@dataclass(frozen=True)
class SceneSelectionResult:
    metadata_dir: Path
    version: str
    data_roots: tuple[Path, ...]
    scene_scores: list[dict[str, Any]]
    integrity_report: dict[str, Any]

    def select(self, num_scenes: int) -> list[dict[str, Any]]:
        """Return only fully local-convertible scenes, in deterministic rank order."""
        if num_scenes <= 0:
            raise ValueError("num_scenes must be greater than zero")
        ready = [
            row for row in self.scene_scores if row["complete_for_local_conversion"]
        ]
        return [
            dict(row, selection_rank=index)
            for index, row in enumerate(ready[:num_scenes], 1)
        ]


@dataclass
class _Diagnostics:
    example_limit: int
    invalid_reference_counts: Counter[str] = field(default_factory=Counter)
    invalid_reference_examples: list[dict[str, str]] = field(default_factory=list)
    duplicate_token_counts: Counter[str] = field(default_factory=Counter)
    duplicate_token_examples: list[dict[str, str]] = field(default_factory=list)
    structural_issue_counts: Counter[str] = field(default_factory=Counter)
    structural_issue_examples: list[dict[str, str]] = field(default_factory=list)

    def invalid_reference(
        self,
        table: str,
        record_token: object,
        field_name: str,
        referenced_token: object,
        reason: str = "token not found",
    ) -> None:
        key = f"{table}.{field_name}"
        self.invalid_reference_counts[key] += 1
        if len(self.invalid_reference_examples) < self.example_limit:
            self.invalid_reference_examples.append(
                {
                    "table": table,
                    "record_token": str(record_token),
                    "field": field_name,
                    "referenced_token": str(referenced_token),
                    "reason": reason,
                }
            )

    def duplicate_token(self, table: str, token: object) -> None:
        self.duplicate_token_counts[table] += 1
        if len(self.duplicate_token_examples) < self.example_limit:
            self.duplicate_token_examples.append({"table": table, "token": str(token)})

    def structural_issue(self, issue: str, record_token: object, detail: str) -> None:
        self.structural_issue_counts[issue] += 1
        if len(self.structural_issue_examples) < self.example_limit:
            self.structural_issue_examples.append(
                {"issue": issue, "record_token": str(record_token), "detail": detail}
            )


@dataclass
class _SceneStats:
    sample_tokens: set[str] = field(default_factory=set)
    timestamps: list[int] = field(default_factory=list)
    annotation_count: int = 0
    vehicle_annotation_count: int = 0
    vehicle_parked_count: int = 0
    cycle_without_rider_count: int = 0
    barrier_count: int = 0
    traffic_cone_count: int = 0
    bicycle_rack_count: int = 0
    static_target_count: int = 0
    dynamic_target_count: int = 0
    static_instance_samples: dict[str, set[str]] = field(default_factory=dict)
    dynamic_instances: set[str] = field(default_factory=set)
    represented_static_types: set[str] = field(default_factory=set)
    channel_records: dict[str, dict[str, Mapping[str, Any]]] = field(
        default_factory=lambda: {channel: {} for channel in REQUIRED_CHANNELS}
    )


class NuScenesSceneSelector:
    """Score all scenes using nuScenes relational metadata and local file checks."""

    def __init__(self, config: SelectorConfig):
        self.config = config
        self.metadata_dir = resolve_metadata_dir(config.metadata_root, config.version)

    def score_scenes(self) -> SceneSelectionResult:
        table_paths = {
            name: self.metadata_dir / f"{name}.json" for name in REQUIRED_TABLES
        }
        missing_tables = [
            str(path) for path in table_paths.values() if not path.is_file()
        ]
        if missing_tables:
            formatted = "\n  - ".join(missing_tables)
            raise MetadataValidationError(
                f"Missing required nuScenes metadata tables:\n  - {formatted}"
            )

        diagnostics = _Diagnostics(self.config.diagnostic_example_limit)
        table_counts: dict[str, int] = {}

        scenes = _load_table(table_paths["scene"], "scene")
        table_counts["scene"] = len(scenes)
        scene_by_token = _index_by_token(scenes, "scene", diagnostics)
        scene_stats = {token: _SceneStats() for token in scene_by_token}

        samples = _load_table(table_paths["sample"], "sample")
        table_counts["sample"] = len(samples)
        sample_by_token = _index_by_token(samples, "sample", diagnostics)
        for sample_token, sample in sample_by_token.items():
            scene_token = sample.get("scene_token", "")
            stats = scene_stats.get(scene_token)
            if stats is None:
                diagnostics.invalid_reference(
                    "sample", sample_token, "scene_token", scene_token
                )
                continue
            stats.sample_tokens.add(sample_token)
            timestamp = sample.get("timestamp")
            if isinstance(timestamp, int):
                stats.timestamps.append(timestamp)

        self._validate_sample_links(
            samples, sample_by_token, scene_by_token, diagnostics
        )
        chain_counts = self._validate_scene_chains(scenes, sample_by_token, diagnostics)

        attributes = _load_table(table_paths["attribute"], "attribute")
        table_counts["attribute"] = len(attributes)
        attribute_by_token = _index_by_token(attributes, "attribute", diagnostics)
        categories = _load_table(table_paths["category"], "category")
        table_counts["category"] = len(categories)
        category_by_token = _index_by_token(categories, "category", diagnostics)
        instances = _load_table(table_paths["instance"], "instance")
        table_counts["instance"] = len(instances)
        instance_by_token = _index_by_token(instances, "instance", diagnostics)
        for instance_token, instance in instance_by_token.items():
            category_token = instance.get("category_token", "")
            if category_token not in category_by_token:
                diagnostics.invalid_reference(
                    "instance", instance_token, "category_token", category_token
                )

        annotations = _load_table(table_paths["sample_annotation"], "sample_annotation")
        table_counts["sample_annotation"] = len(annotations)
        self._aggregate_annotations(
            annotations,
            sample_by_token,
            scene_stats,
            instance_by_token,
            category_by_token,
            attribute_by_token,
            diagnostics,
        )
        del annotations

        sensors = _load_table(table_paths["sensor"], "sensor")
        table_counts["sensor"] = len(sensors)
        sensor_by_token = _index_by_token(sensors, "sensor", diagnostics)
        calibrated = _load_table(table_paths["calibrated_sensor"], "calibrated_sensor")
        table_counts["calibrated_sensor"] = len(calibrated)
        calibrated_by_token = _index_by_token(
            calibrated, "calibrated_sensor", diagnostics
        )
        for calibrated_token, record in calibrated_by_token.items():
            sensor_token = record.get("sensor_token", "")
            if sensor_token not in sensor_by_token:
                diagnostics.invalid_reference(
                    "calibrated_sensor", calibrated_token, "sensor_token", sensor_token
                )

        sample_data = _load_table(table_paths["sample_data"], "sample_data")
        table_counts["sample_data"] = len(sample_data)
        self._index_required_sample_data(
            sample_data,
            sample_by_token,
            scene_stats,
            calibrated_by_token,
            sensor_by_token,
            diagnostics,
        )
        del sample_data

        rows, coverage = self._build_scene_rows(scenes, scene_stats, chain_counts)
        rows.sort(key=_scene_sort_key)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank

        ready_count = sum(row["complete_for_local_conversion"] for row in rows)
        converted_checks = [
            {
                "scene_name": row["scene_name"],
                **row["local_converted_data"],
            }
            for row in rows
            if row["local_converted_data"]["scene_dir"] is not None
        ]
        converted_mismatches = sum(
            check["frame_count_matches_source"] is False for check in converted_checks
        )
        invalid_reference_count = sum(diagnostics.invalid_reference_counts.values())
        duplicate_token_count = sum(diagnostics.duplicate_token_counts.values())
        structural_issue_count = sum(diagnostics.structural_issue_counts.values())
        integrity_report = {
            "schema_version": 1,
            "metadata_dir": str(self.metadata_dir),
            "metadata_version": self.config.version,
            "data_roots": [str(path) for path in self.config.data_roots],
            "required_tables": list(REQUIRED_TABLES),
            "missing_tables": [],
            "table_record_counts": table_counts,
            "metadata_status": (
                "ok"
                if invalid_reference_count
                == duplicate_token_count
                == structural_issue_count
                == 0
                else "issues_detected"
            ),
            "invalid_reference_count": invalid_reference_count,
            "invalid_reference_counts": dict(
                sorted(diagnostics.invalid_reference_counts.items())
            ),
            "invalid_reference_examples": diagnostics.invalid_reference_examples,
            "duplicate_token_count": duplicate_token_count,
            "duplicate_token_counts": dict(
                sorted(diagnostics.duplicate_token_counts.items())
            ),
            "duplicate_token_examples": diagnostics.duplicate_token_examples,
            "structural_issue_count": structural_issue_count,
            "structural_issue_counts": dict(
                sorted(diagnostics.structural_issue_counts.items())
            ),
            "structural_issue_examples": diagnostics.structural_issue_examples,
            "sensor_file_coverage": coverage,
            "ready_for_local_conversion_scene_count": ready_count,
            "incomplete_for_local_conversion_scene_count": len(rows) - ready_count,
            "converted_scene_count": len(converted_checks),
            "converted_frame_mismatch_count": converted_mismatches,
            "converted_scene_checks": converted_checks,
        }
        return SceneSelectionResult(
            metadata_dir=self.metadata_dir,
            version=self.config.version,
            data_roots=self.config.data_roots,
            scene_scores=rows,
            integrity_report=integrity_report,
        )

    @staticmethod
    def _validate_sample_links(
        samples: Sequence[Mapping[str, Any]],
        sample_by_token: Mapping[str, Mapping[str, Any]],
        scene_by_token: Mapping[str, Mapping[str, Any]],
        diagnostics: _Diagnostics,
    ) -> None:
        for sample in samples:
            token = sample.get("token", "")
            scene_token = sample.get("scene_token", "")
            if scene_token not in scene_by_token:
                continue
            for field_name in ("prev", "next"):
                linked_token = sample.get(field_name, "")
                if not linked_token:
                    continue
                linked = sample_by_token.get(linked_token)
                if linked is None:
                    diagnostics.invalid_reference(
                        "sample", token, field_name, linked_token
                    )
                elif linked.get("scene_token") != scene_token:
                    diagnostics.invalid_reference(
                        "sample",
                        token,
                        field_name,
                        linked_token,
                        "linked sample belongs to another scene",
                    )

    @staticmethod
    def _validate_scene_chains(
        scenes: Sequence[Mapping[str, Any]],
        sample_by_token: Mapping[str, Mapping[str, Any]],
        diagnostics: _Diagnostics,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for scene in scenes:
            scene_token = scene.get("token", "")
            token = scene.get("first_sample_token", "")
            visited: set[str] = set()
            last_seen = ""
            while token:
                if token in visited:
                    diagnostics.structural_issue(
                        "sample_chain_cycle", scene_token, f"cycle at sample {token}"
                    )
                    break
                sample = sample_by_token.get(token)
                if sample is None:
                    diagnostics.invalid_reference(
                        "scene", scene_token, "first_sample_token/sample.next", token
                    )
                    break
                if sample.get("scene_token") != scene_token:
                    diagnostics.invalid_reference(
                        "scene",
                        scene_token,
                        "sample_chain",
                        token,
                        "sample chain entered another scene",
                    )
                    break
                visited.add(token)
                last_seen = token
                token = sample.get("next", "")
            counts[str(scene_token)] = len(visited)
            declared_last = scene.get("last_sample_token", "")
            if visited and declared_last and last_seen != declared_last:
                diagnostics.structural_issue(
                    "last_sample_mismatch",
                    scene_token,
                    f"declared {declared_last}, traversed {last_seen}",
                )
            declared_count = _nonnegative_int(scene.get("nbr_samples"), len(visited))
            if declared_count != len(visited):
                diagnostics.structural_issue(
                    "scene_sample_count_mismatch",
                    scene_token,
                    f"declared {declared_count}, traversed {len(visited)}",
                )
        return counts

    @staticmethod
    def _aggregate_annotations(
        annotations: Sequence[Mapping[str, Any]],
        sample_by_token: Mapping[str, Mapping[str, Any]],
        scene_stats: Mapping[str, _SceneStats],
        instance_by_token: Mapping[str, Mapping[str, Any]],
        category_by_token: Mapping[str, Mapping[str, Any]],
        attribute_by_token: Mapping[str, Mapping[str, Any]],
        diagnostics: _Diagnostics,
    ) -> None:
        for annotation in annotations:
            annotation_token = annotation.get("token", "")
            sample_token = annotation.get("sample_token", "")
            sample = sample_by_token.get(sample_token)
            if sample is None:
                diagnostics.invalid_reference(
                    "sample_annotation", annotation_token, "sample_token", sample_token
                )
                continue
            stats = scene_stats.get(sample.get("scene_token", ""))
            if stats is None:
                continue
            stats.annotation_count += 1

            instance_token = annotation.get("instance_token", "")
            instance = instance_by_token.get(instance_token)
            if instance is None:
                diagnostics.invalid_reference(
                    "sample_annotation",
                    annotation_token,
                    "instance_token",
                    instance_token,
                )
                continue
            category_token = instance.get("category_token", "")
            category = category_by_token.get(category_token)
            if category is None:
                continue
            category_name = str(category.get("name", ""))

            attribute_names: set[str] = set()
            raw_attribute_tokens = annotation.get("attribute_tokens", [])
            if not isinstance(raw_attribute_tokens, list):
                diagnostics.structural_issue(
                    "invalid_attribute_tokens_type",
                    annotation_token,
                    "attribute_tokens must be a list",
                )
                raw_attribute_tokens = []
            for attribute_token in raw_attribute_tokens:
                attribute = attribute_by_token.get(attribute_token)
                if attribute is None:
                    diagnostics.invalid_reference(
                        "sample_annotation",
                        annotation_token,
                        "attribute_tokens",
                        attribute_token,
                    )
                else:
                    attribute_names.add(str(attribute.get("name", "")))

            is_vehicle = category_name.startswith("vehicle.")
            is_parked = is_vehicle and "vehicle.parked" in attribute_names
            is_cycle_without_rider = (
                category_name in {"vehicle.bicycle", "vehicle.motorcycle"}
                and "cycle.without_rider" in attribute_names
            )
            static_type = STATIC_CATEGORIES.get(category_name)
            is_static = bool(static_type or is_parked or is_cycle_without_rider)
            is_dynamic = _is_dynamic_category(category_name) and not is_static

            if is_vehicle:
                stats.vehicle_annotation_count += 1
            if is_parked:
                stats.vehicle_parked_count += 1
                stats.represented_static_types.add("vehicle_parked")
            if is_cycle_without_rider:
                stats.cycle_without_rider_count += 1
                stats.represented_static_types.add("cycle_without_rider")
            if static_type:
                setattr(
                    stats,
                    f"{static_type}_count",
                    getattr(stats, f"{static_type}_count") + 1,
                )
                stats.represented_static_types.add(static_type)
            if is_static:
                stats.static_target_count += 1
                stats.static_instance_samples.setdefault(
                    str(instance_token), set()
                ).add(str(sample_token))
            if is_dynamic:
                stats.dynamic_target_count += 1
                stats.dynamic_instances.add(str(instance_token))

    @staticmethod
    def _index_required_sample_data(
        sample_data: Sequence[Mapping[str, Any]],
        sample_by_token: Mapping[str, Mapping[str, Any]],
        scene_stats: Mapping[str, _SceneStats],
        calibrated_by_token: Mapping[str, Mapping[str, Any]],
        sensor_by_token: Mapping[str, Mapping[str, Any]],
        diagnostics: _Diagnostics,
    ) -> None:
        seen_tokens: set[str] = set()
        for record in sample_data:
            record_token = record.get("token", "")
            if not isinstance(record_token, str) or not record_token:
                diagnostics.structural_issue(
                    "missing_record_token",
                    "",
                    "sample_data record has no non-empty token",
                )
            elif record_token in seen_tokens:
                diagnostics.duplicate_token("sample_data", record_token)
            else:
                seen_tokens.add(record_token)

            sample_token = record.get("sample_token", "")
            sample = sample_by_token.get(sample_token)
            if sample is None:
                diagnostics.invalid_reference(
                    "sample_data", record_token, "sample_token", sample_token
                )
                continue
            calibrated_token = record.get("calibrated_sensor_token", "")
            calibrated = calibrated_by_token.get(calibrated_token)
            if calibrated is None:
                diagnostics.invalid_reference(
                    "sample_data",
                    record_token,
                    "calibrated_sensor_token",
                    calibrated_token,
                )
                continue
            sensor_token = calibrated.get("sensor_token", "")
            sensor = sensor_by_token.get(sensor_token)
            if sensor is None:
                continue
            if not record.get("is_key_frame", False):
                continue
            channel = sensor.get("channel")
            if channel not in REQUIRED_CHANNELS:
                continue
            stats = scene_stats.get(sample.get("scene_token", ""))
            if stats is None:
                continue
            existing = stats.channel_records[str(channel)].get(str(sample_token))
            if existing is not None:
                diagnostics.structural_issue(
                    "duplicate_sample_channel_keyframe",
                    record_token,
                    f"sample {sample_token} has multiple {channel} keyframes",
                )
                if str(record_token) >= str(existing.get("token", "")):
                    continue
            stats.channel_records[str(channel)][str(sample_token)] = record

    def _build_scene_rows(
        self,
        scenes: Sequence[Mapping[str, Any]],
        scene_stats: Mapping[str, _SceneStats],
        chain_counts: Mapping[str, int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        missing_metadata_count = 0
        missing_file_count = 0
        missing_examples: list[dict[str, str]] = []
        available_channel_files = Counter[str]()
        metadata_channel_records = Counter[str]()
        expected_channel_files = 0

        for scene in scenes:
            scene_token = str(scene.get("token", ""))
            stats = scene_stats.get(scene_token, _SceneStats())
            scene_name = str(scene.get("name", scene_token))
            metadata_sample_count = len(stats.sample_tokens)
            nbr_samples = _nonnegative_int(
                scene.get("nbr_samples"), metadata_sample_count
            )
            expected_samples = nbr_samples if nbr_samples > 0 else metadata_sample_count
            expected_channel_files += expected_samples * len(REQUIRED_CHANNELS)

            available_by_channel = Counter[str]()
            missing_for_scene: list[dict[str, str]] = []
            complete_pair_frames = 0
            for sample_token in sorted(stats.sample_tokens):
                sample_complete = True
                for channel in REQUIRED_CHANNELS:
                    record = stats.channel_records[channel].get(sample_token)
                    if record is None:
                        missing_metadata_count += 1
                        sample_complete = False
                        issue = {
                            "scene_name": scene_name,
                            "sample_token": sample_token,
                            "channel": channel,
                            "reason": "missing keyframe record in sample_data.json",
                        }
                    else:
                        metadata_channel_records[channel] += 1
                        filename = record.get("filename", "")
                        if _valid_relative_filename(filename) and _exists_in_data_roots(
                            str(filename), self.config.data_roots
                        ):
                            available_by_channel[channel] += 1
                            available_channel_files[channel] += 1
                            continue
                        missing_file_count += 1
                        sample_complete = False
                        issue = {
                            "scene_name": scene_name,
                            "sample_token": sample_token,
                            "channel": channel,
                            "filename": str(filename),
                            "reason": (
                                "invalid or empty relative filename"
                                if not _valid_relative_filename(filename)
                                else "file not found under any data root"
                            ),
                        }
                    if len(missing_for_scene) < 10:
                        missing_for_scene.append(issue)
                    if len(missing_examples) < self.config.diagnostic_example_limit:
                        missing_examples.append(issue)
                if sample_complete:
                    complete_pair_frames += 1

            cam_metadata = len(stats.channel_records["CAM_FRONT"])
            lidar_metadata = len(stats.channel_records["LIDAR_TOP"])
            cam_available = available_by_channel["CAM_FRONT"]
            lidar_available = available_by_channel["LIDAR_TOP"]
            denominator = expected_samples * len(REQUIRED_CHANNELS)
            data_completeness = (
                min(1.0, (cam_available + lidar_available) / denominator)
                if denominator
                else 0.0
            )
            complete_pair_ratio = (
                min(1.0, complete_pair_frames / expected_samples)
                if expected_samples
                else 0.0
            )
            source_ready = bool(
                expected_samples
                and metadata_sample_count == expected_samples
                and cam_metadata == expected_samples
                and lidar_metadata == expected_samples
                and cam_available == expected_samples
                and lidar_available == expected_samples
                and complete_pair_frames == expected_samples
            )

            vehicle_parked_ratio = _safe_ratio(
                stats.vehicle_parked_count, stats.vehicle_annotation_count
            )
            dynamic_target_ratio = _safe_ratio(
                stats.dynamic_target_count, stats.annotation_count
            )
            static_instance_count = len(stats.static_instance_samples)
            persistence_values = [
                len(sample_tokens)
                for sample_tokens in stats.static_instance_samples.values()
            ]
            mean_persistence_frames = (
                sum(persistence_values) / len(persistence_values)
                if persistence_values
                else 0.0
            )
            max_persistence_frames = max(persistence_values, default=0)
            mean_persistence_ratio = (
                min(1.0, mean_persistence_frames / metadata_sample_count)
                if metadata_sample_count
                else 0.0
            )
            max_persistence_ratio = (
                min(1.0, max_persistence_frames / metadata_sample_count)
                if metadata_sample_count
                else 0.0
            )
            duration_seconds = (
                (max(stats.timestamps) - min(stats.timestamps)) / 1_000_000.0
                if len(stats.timestamps) >= 2
                else 0.0
            )

            description = str(scene.get("description", ""))
            context_matches = sum(
                description.lower().count(term) for term in PARKING_CONTEXT_TERMS
            )
            components = _score_components(
                parked_count=stats.vehicle_parked_count,
                parked_ratio=vehicle_parked_ratio,
                static_instance_count=static_instance_count,
                static_type_count=len(stats.represented_static_types),
                persistence_ratio=mean_persistence_ratio,
                dynamic_ratio=dynamic_target_ratio,
                annotation_count=stats.annotation_count,
                completeness=data_completeness,
                sample_count=metadata_sample_count,
                context_matches=context_matches,
            )
            total_score = round(sum(components.values()), 6)
            converted = self._inspect_converted_scene(
                scene_name, complete_pair_frames, expected_samples
            )
            reasons = _selection_reasons(
                source_ready=source_ready,
                parked_count=stats.vehicle_parked_count,
                parked_ratio=vehicle_parked_ratio,
                static_instance_count=static_instance_count,
                persistence_ratio=mean_persistence_ratio,
                dynamic_ratio=dynamic_target_ratio,
                annotation_count=stats.annotation_count,
                context_matches=context_matches,
            )
            rows.append(
                {
                    "scene_name": scene_name,
                    "scene_token": scene_token,
                    "description": description,
                    "nbr_samples": nbr_samples,
                    "metadata_sample_count": metadata_sample_count,
                    "sample_chain_count": chain_counts.get(scene_token, 0),
                    "trajectory_length_frames": metadata_sample_count,
                    "trajectory_duration_seconds": round(duration_seconds, 6),
                    "cam_front_metadata_frames": cam_metadata,
                    "cam_front_available_frames": cam_available,
                    "lidar_top_metadata_frames": lidar_metadata,
                    "lidar_top_available_frames": lidar_available,
                    "cam_front_lidar_top_complete_frames": complete_pair_frames,
                    "effective_sample_count": complete_pair_frames,
                    "vehicle_annotation_count": stats.vehicle_annotation_count,
                    "vehicle_parked_count": stats.vehicle_parked_count,
                    "vehicle_parked_ratio": round(vehicle_parked_ratio, 6),
                    "cycle_without_rider_count": stats.cycle_without_rider_count,
                    "barrier_count": stats.barrier_count,
                    "traffic_cone_count": stats.traffic_cone_count,
                    "bicycle_rack_count": stats.bicycle_rack_count,
                    "static_target_count": stats.static_target_count,
                    "static_target_instance_count": static_instance_count,
                    "static_target_persistence": {
                        "mean_frames": round(mean_persistence_frames, 6),
                        "max_frames": max_persistence_frames,
                        "mean_ratio": round(mean_persistence_ratio, 6),
                        "max_ratio": round(max_persistence_ratio, 6),
                    },
                    "dynamic_target_count": stats.dynamic_target_count,
                    "dynamic_target_instance_count": len(stats.dynamic_instances),
                    "dynamic_target_ratio": round(dynamic_target_ratio, 6),
                    "annotation_count": stats.annotation_count,
                    "data_completeness": round(data_completeness, 6),
                    "complete_pair_ratio": round(complete_pair_ratio, 6),
                    "missing_sensor_file_count": (
                        expected_samples * 2 - cam_available - lidar_available
                    ),
                    "missing_sensor_file_examples": missing_for_scene,
                    "score_components": components,
                    "total_score": total_score,
                    "selection_score": total_score,
                    "selection_reasons": reasons,
                    "selection_reason": "; ".join(reasons),
                    "complete_for_local_conversion": source_ready,
                    "local_conversion_reason": (
                        "all expected CAM_FRONT and LIDAR_TOP keyframe files are local"
                        if source_ready
                        else (
                            f"requires {expected_samples} complete source pairs; found "
                            f"{complete_pair_frames}"
                        )
                    ),
                    "local_converted_data": converted,
                }
            )

        coverage = {
            "expected_channel_file_count": expected_channel_files,
            "metadata_channel_record_counts": {
                channel: metadata_channel_records[channel]
                for channel in REQUIRED_CHANNELS
            },
            "available_channel_file_counts": {
                channel: available_channel_files[channel]
                for channel in REQUIRED_CHANNELS
            },
            "missing_keyframe_metadata_count": missing_metadata_count,
            "missing_sensor_file_count": missing_file_count,
            "missing_or_invalid_count": missing_metadata_count + missing_file_count,
            "missing_examples": missing_examples,
        }
        return rows, coverage

    def _inspect_converted_scene(
        self, scene_name: str, source_pair_frames: int, expected_samples: int
    ) -> dict[str, Any]:
        for root in self.config.data_roots:
            scene_dir = root / scene_name
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            if not color_dir.is_dir() and not depth_dir.is_dir():
                continue
            color_frames = _count_files(color_dir)
            depth_frames = _count_files(depth_dir)
            pose_frames = _count_nonempty_lines(scene_dir / "poses.txt")
            converted_frames = min(color_frames, depth_frames, pose_frames)
            matches = bool(
                source_pair_frames == expected_samples
                and color_frames == source_pair_frames
                and depth_frames == source_pair_frames
                and pose_frames == source_pair_frames
            )
            if matches:
                note = "converted color/depth/pose counts match complete source pairs"
            else:
                note = (
                    f"source pairs={source_pair_frames}, expected samples={expected_samples}, "
                    f"color={color_frames}, depth={depth_frames}, poses={pose_frames}"
                )
            return {
                "scene_dir": str(scene_dir),
                "color_frames": color_frames,
                "depth_frames": depth_frames,
                "pose_frames": pose_frames,
                "converted_complete_frames": converted_frames,
                "frame_count_matches_source": matches,
                "frame_count_note": note,
            }
        return {
            "scene_dir": None,
            "color_frames": 0,
            "depth_frames": 0,
            "pose_frames": 0,
            "converted_complete_frames": 0,
            "frame_count_matches_source": None,
            "frame_count_note": "no converted scene directory found under data roots",
        }


def resolve_metadata_dir(metadata_root: Path, version: str) -> Path:
    """Accept either a dataset root or the version directory itself."""
    root = Path(metadata_root).expanduser().resolve()
    version_dir = root / version
    if root.is_dir() and any(
        (root / f"{name}.json").exists() for name in REQUIRED_TABLES
    ):
        return root
    if version_dir.is_dir():
        return version_dir
    return version_dir


def build_scene_scores_payload(result: SceneSelectionResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata_dir": str(result.metadata_dir),
        "metadata_version": result.version,
        "data_roots": [str(path) for path in result.data_roots],
        "scoring_rules": SCORING_RULES,
        "scene_count": len(result.scene_scores),
        "scenes": result.scene_scores,
    }


def build_selected_scenes_payload(
    result: SceneSelectionResult, num_scenes: int
) -> dict[str, Any]:
    selected = result.select(num_scenes)
    return {
        "schema_version": 1,
        "metadata_dir": str(result.metadata_dir),
        "metadata_version": result.version,
        "requested_num_scenes": num_scenes,
        "selected_scene_count": len(selected),
        "selection_shortfall": max(0, num_scenes - len(selected)),
        "selection_policy": (
            "fully local-convertible scenes ordered by score, parked ratio, static persistence, "
            "static count, then scene name"
        ),
        "scenes": selected,
    }


def _load_table(path: Path, table_name: str) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataValidationError(
            f"Failed to read {table_name} table at {path}: {exc}"
        ) from exc
    if not isinstance(value, list) or any(
        not isinstance(record, dict) for record in value
    ):
        raise MetadataValidationError(
            f"nuScenes table {table_name} at {path} must be a JSON array of objects"
        )
    return value


def _index_by_token(
    records: Iterable[Mapping[str, Any]], table_name: str, diagnostics: _Diagnostics
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        token = record.get("token", "")
        if not isinstance(token, str) or not token:
            diagnostics.structural_issue(
                "missing_record_token",
                "",
                f"{table_name} record has no non-empty token",
            )
            continue
        if token in result:
            diagnostics.duplicate_token(table_name, token)
            continue
        result[token] = record
    return result


def _score_components(
    *,
    parked_count: int,
    parked_ratio: float,
    static_instance_count: int,
    static_type_count: int,
    persistence_ratio: float,
    dynamic_ratio: float,
    annotation_count: int,
    completeness: float,
    sample_count: int,
    context_matches: int,
) -> dict[str, float]:
    parked = 20.0 * (0.65 * min(parked_count / 40.0, 1.0) + 0.35 * parked_ratio)
    diversity = 15.0 * (
        0.6 * min(static_instance_count / 10.0, 1.0)
        + 0.4 * min(static_type_count / 5.0, 1.0)
    )
    low_dynamic = 15.0 * (1.0 - dynamic_ratio) if annotation_count else 0.0
    return {
        "parked_vehicle": round(parked, 6),
        "static_target_diversity": round(diversity, 6),
        "static_persistence": round(15.0 * persistence_ratio, 6),
        "low_dynamic": round(low_dynamic, 6),
        "data_completeness": round(25.0 * completeness, 6),
        "sequence_length": round(5.0 * min(sample_count / 40.0, 1.0), 6),
        "description_context": round(5.0 * min(context_matches / 2.0, 1.0), 6),
    }


def _selection_reasons(
    *,
    source_ready: bool,
    parked_count: int,
    parked_ratio: float,
    static_instance_count: int,
    persistence_ratio: float,
    dynamic_ratio: float,
    annotation_count: int,
    context_matches: int,
) -> list[str]:
    reasons: list[str] = []
    if source_ready:
        reasons.append("complete local CAM_FRONT/LIDAR_TOP source data")
    if parked_count >= 10 or parked_ratio >= 0.25:
        reasons.append(
            f"parked vehicle evidence ({parked_count}, ratio {parked_ratio:.3f})"
        )
    if static_instance_count >= 3:
        reasons.append(f"{static_instance_count} unique static target instances")
    if persistence_ratio >= 0.25:
        reasons.append(
            f"persistent static targets (mean ratio {persistence_ratio:.3f})"
        )
    if annotation_count and dynamic_ratio <= 0.25:
        reasons.append(f"low dynamic annotation ratio ({dynamic_ratio:.3f})")
    if context_matches:
        reasons.append("parking context in scene description")
    if not reasons:
        reasons.append("limited structured parking evidence")
    return reasons


def _scene_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    persistence = row["static_target_persistence"]["mean_ratio"]
    return (
        -int(bool(row["complete_for_local_conversion"])),
        -float(row["total_score"]),
        -float(row["vehicle_parked_ratio"]),
        -float(persistence),
        -int(row["static_target_count"]),
        str(row["scene_name"]),
    )


def _is_dynamic_category(category_name: str) -> bool:
    return bool(
        category_name.startswith("human.")
        or category_name.startswith("vehicle.")
        or category_name == "animal"
        or category_name == "movable_object.pushable_pullable"
    )


def _valid_relative_filename(filename: object) -> bool:
    if not isinstance(filename, str) or not filename:
        return False
    path = Path(filename)
    return not path.is_absolute() and ".." not in path.parts


def _exists_in_data_roots(filename: str, data_roots: Sequence[Path]) -> bool:
    return any((root / filename).is_file() for root in data_roots)


def _count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(entry.is_file() for entry in path.iterdir())


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nonnegative_int(value: object, fallback: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback
