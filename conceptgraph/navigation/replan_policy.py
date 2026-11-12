from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DEFAULT_TRIGGER_PRIORITY = (
    "path_blocked",
    "target_changed",
    "better_target_persisted",
    "target_confirmed",
    "manual_request",
    "vehicle_moved",
    "periodic_interval",
)


@dataclass(frozen=True)
class ReplanPolicyConfig:
    trigger_priority: tuple[str, ...] = DEFAULT_TRIGGER_PRIORITY
    cooldown_frames: int = 2
    cooldown_bypass_triggers: tuple[str, ...] = (
        "path_blocked",
        "target_changed",
        "better_target_persisted",
        "target_confirmed",
        "manual_request",
    )
    debounce_frames: dict[str, int] = field(
        default_factory=lambda: {
            "target_confirmed": 1,
            "target_changed": 1,
            "better_target_persisted": 1,
            "path_blocked": 1,
            "vehicle_moved": 1,
            "periodic_interval": 1,
            "manual_request": 1,
        }
    )
    vehicle_moved_distance_m: float = 3.0
    periodic_interval_frames: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any | None) -> ReplanPolicyConfig:
        if value is None:
            return cls()
        source = value if isinstance(value, Mapping) else {
            name: getattr(value, name)
            for name in cls.__dataclass_fields__
            if hasattr(value, name)
        }
        parsed = {
            name: source[name]
            for name in cls.__dataclass_fields__
            if name in source and source[name] is not None
        }
        if "trigger_priority" in parsed:
            parsed["trigger_priority"] = tuple(parsed["trigger_priority"])
        if "cooldown_bypass_triggers" in parsed:
            parsed["cooldown_bypass_triggers"] = tuple(parsed["cooldown_bypass_triggers"])
        if "debounce_frames" in parsed:
            parsed["debounce_frames"] = {
                str(key): max(1, int(item)) for key, item in dict(parsed["debounce_frames"]).items()
            }
        config = cls(**parsed)
        unknown = set(config.trigger_priority) - set(DEFAULT_TRIGGER_PRIORITY)
        if unknown:
            raise ValueError(f"unknown replan triggers: {sorted(unknown)}")
        return config


@dataclass
class ReplanDecision:
    replanned: bool
    reason: str | None
    audits: list[dict[str, Any]]
    raw_triggers: list[str]


class ReplanPolicy:
    """Evaluate replan events with stable priority, debounce, and cooldown."""

    def __init__(self, config: Mapping[str, Any] | ReplanPolicyConfig | None = None) -> None:
        self.config = config if isinstance(config, ReplanPolicyConfig) else ReplanPolicyConfig.from_mapping(config)
        self.last_plan_frame: int | None = None
        self.last_plan_vehicle_xy: np.ndarray | None = None
        self._active_counts = {trigger: 0 for trigger in DEFAULT_TRIGGER_PRIORITY}
        self._accepted_while_active: set[str] = set()
        self._planned_frames: set[int] = set()
        self._manual_requested = False

    def request_manual_replan(self) -> None:
        self._manual_requested = True

    def evaluate(
        self,
        frame_idx: int,
        vehicle_xy: np.ndarray,
        *,
        has_locked_target: bool,
        target_confirmed: bool = False,
        target_changed: bool = False,
        better_target_persisted: bool = False,
        path_blocked: bool = False,
    ) -> ReplanDecision:
        frame_idx = int(frame_idx)
        vehicle_xy = np.asarray(vehicle_xy, dtype=float)[:2]
        raw = {
            "target_confirmed": bool(target_confirmed),
            "target_changed": bool(target_changed),
            "better_target_persisted": bool(better_target_persisted),
            "path_blocked": bool(path_blocked),
            "manual_request": bool(self._manual_requested),
            "vehicle_moved": self._vehicle_moved(vehicle_xy),
            "periodic_interval": self._periodic_due(frame_idx),
        }
        raw_triggers = [trigger for trigger in self.config.trigger_priority if raw.get(trigger, False)]
        audits: list[dict[str, Any]] = []
        eligible: list[str] = []

        if frame_idx in self._planned_frames:
            for trigger in raw_triggers:
                audits.append(self._audit(trigger, False, "already_planned_this_frame"))
            self._update_active(raw)
            self._manual_requested = False
            return ReplanDecision(False, None, audits, raw_triggers)

        for trigger in self.config.trigger_priority:
            active = raw.get(trigger, False)
            if not active:
                self._active_counts[trigger] = 0
                self._accepted_while_active.discard(trigger)
                continue
            self._active_counts[trigger] += 1
            required = max(1, int(self.config.debounce_frames.get(trigger, 1)))
            if not has_locked_target:
                audits.append(self._audit(trigger, False, "no_locked_target"))
                continue
            if self._active_counts[trigger] < required:
                audits.append(self._audit(trigger, False, "debounce_pending"))
                continue
            if trigger in self._accepted_while_active:
                audits.append(self._audit(trigger, False, "debounced_active_event"))
                continue
            in_cooldown = (
                self.last_plan_frame is not None
                and frame_idx - self.last_plan_frame <= self.config.cooldown_frames
            )
            if in_cooldown and trigger not in self.config.cooldown_bypass_triggers:
                audits.append(self._audit(trigger, False, "cooldown_active"))
                continue
            eligible.append(trigger)

        selected = eligible[0] if eligible else None
        for trigger in eligible:
            if trigger == selected:
                audits.append(self._audit(trigger, True, "accepted_by_priority"))
            else:
                audits.append(self._audit(trigger, False, f"lower_priority_than:{selected}"))
        self._manual_requested = False
        if selected is None:
            return ReplanDecision(False, None, audits, raw_triggers)
        self._accepted_while_active.add(selected)
        self._planned_frames.add(frame_idx)
        return ReplanDecision(True, selected, audits, raw_triggers)

    def record_plan(self, frame_idx: int, vehicle_xy: np.ndarray) -> None:
        self.last_plan_frame = int(frame_idx)
        self.last_plan_vehicle_xy = np.asarray(vehicle_xy, dtype=float)[:2].copy()
        self._planned_frames.add(int(frame_idx))

    def _vehicle_moved(self, vehicle_xy: np.ndarray) -> bool:
        if self.last_plan_vehicle_xy is None:
            return False
        return bool(
            np.linalg.norm(vehicle_xy - self.last_plan_vehicle_xy)
            >= self.config.vehicle_moved_distance_m
        )

    def _periodic_due(self, frame_idx: int) -> bool:
        return bool(
            self.last_plan_frame is not None
            and frame_idx - self.last_plan_frame >= self.config.periodic_interval_frames
        )

    def _update_active(self, raw: Mapping[str, bool]) -> None:
        for trigger in self.config.trigger_priority:
            if raw.get(trigger, False):
                self._active_counts[trigger] += 1
            else:
                self._active_counts[trigger] = 0
                self._accepted_while_active.discard(trigger)

    @staticmethod
    def _audit(trigger: str, accepted: bool, reason: str) -> dict[str, Any]:
        return {"trigger": trigger, "accepted": bool(accepted), "reason": reason}


__all__ = [
    "DEFAULT_TRIGGER_PRIORITY",
    "ReplanDecision",
    "ReplanPolicy",
    "ReplanPolicyConfig",
]
