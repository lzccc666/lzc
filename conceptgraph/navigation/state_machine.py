from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class NavigationState(str, Enum):
    SEARCHING = "SEARCHING"
    PROVISIONAL = "PROVISIONAL"
    LOCKED = "LOCKED"
    PLANNING = "PLANNING"
    TRACKING = "TRACKING"
    REPLANNING = "REPLANNING"
    UNREACHABLE = "UNREACHABLE"


@dataclass(frozen=True)
class StateMachineConfig:
    confirmation_observations: int = 3
    confirmation_confidence: float = 0.35
    minimum_persistent_frames: int = 3
    provisional_missing_tolerance_frames: int = 0
    locked_missing_tolerance_frames: int = 1
    target_lost_after_frames: int = 4
    target_loss_state: str = NavigationState.SEARCHING.value
    better_target_min_confidence: float = 0.45
    better_target_score_margin: float = 0.08
    better_target_persistence_frames: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any | None) -> StateMachineConfig:
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
        config = cls(**parsed)
        if config.target_loss_state not in {
            NavigationState.SEARCHING.value,
            NavigationState.UNREACHABLE.value,
        }:
            raise ValueError("target_loss_state must be SEARCHING or UNREACHABLE")
        return config


@dataclass
class StateObservation:
    frame_idx: int
    state_before: str
    state_after: str
    transition_reason: str
    state_trace: list[str]
    locked_target_id: int | None
    target_confidence: float
    target_confirmed: bool = False
    target_changed: bool = False
    better_target_persisted: bool = False
    target_lost: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


class NavigationStateMachine:
    """Deterministic target confirmation and lock hysteresis."""

    def __init__(self, config: Mapping[str, Any] | StateMachineConfig | None = None) -> None:
        self.config = (
            config if isinstance(config, StateMachineConfig) else StateMachineConfig.from_mapping(config)
        )
        self.state = NavigationState.SEARCHING
        self.locked_target_id: int | None = None
        self.target_confidence = 0.0
        self._provisional_id: int | None = None
        self._provisional_observations = 0
        self._provisional_first_frame: int | None = None
        self._provisional_missing_frames = 0
        self._locked_missing_frames = 0
        self._challenger_id: int | None = None
        self._challenger_frames = 0

    def reset(self) -> None:
        self.state = NavigationState.SEARCHING
        self.locked_target_id = None
        self.target_confidence = 0.0
        self._clear_provisional()
        self._clear_challenger()
        self._locked_missing_frames = 0

    def observe(
        self,
        frame_idx: int,
        candidates: Sequence[tuple[int, float]],
        *,
        valid_target_ids: set[int] | None = None,
    ) -> StateObservation:
        before = self.state
        scores = {int(node_id): float(score) for node_id, score in candidates}
        ordered = sorted(scores.items(), key=lambda item: (-round(item[1], 12), item[0]))
        valid_ids = set(scores) if valid_target_ids is None else {int(item) for item in valid_target_ids}
        confirmed = False
        changed = False
        better_persisted = False
        target_lost = False
        reason = "state_unchanged"

        if self.locked_target_id is None:
            reason, confirmed = self._observe_unlocked(int(frame_idx), ordered, valid_ids)
        else:
            reason, changed, better_persisted, target_lost = self._observe_locked(
                ordered,
                valid_ids,
            )

        return StateObservation(
            frame_idx=int(frame_idx),
            state_before=before.value,
            state_after=self.state.value,
            transition_reason=reason,
            state_trace=[before.value] if before == self.state else [before.value, self.state.value],
            locked_target_id=self.locked_target_id,
            target_confidence=float(self.target_confidence),
            target_confirmed=confirmed,
            target_changed=changed,
            better_target_persisted=better_persisted,
            target_lost=target_lost,
            diagnostics={
                "provisional_id": self._provisional_id,
                "provisional_observations": self._provisional_observations,
                "provisional_missing_frames": self._provisional_missing_frames,
                "locked_missing_frames": self._locked_missing_frames,
                "challenger_id": self._challenger_id,
                "challenger_frames": self._challenger_frames,
            },
        )

    def begin_planning(self, *, replanning: bool) -> tuple[str, str]:
        before = self.state.value
        self.state = NavigationState.REPLANNING if replanning else NavigationState.PLANNING
        return before, self.state.value

    def planning_succeeded(self) -> tuple[str, str]:
        before = self.state.value
        self.state = NavigationState.TRACKING
        return before, self.state.value

    def planning_failed(self) -> tuple[str, str]:
        before = self.state.value
        self.state = NavigationState.UNREACHABLE
        return before, self.state.value

    def _observe_unlocked(
        self,
        frame_idx: int,
        ordered: list[tuple[int, float]],
        valid_ids: set[int],
    ) -> tuple[str, bool]:
        eligible = [
            (node_id, score)
            for node_id, score in ordered
            if node_id in valid_ids and score >= self.config.confirmation_confidence
        ]
        if not eligible:
            if (
                self._provisional_id is not None
                and self._provisional_missing_frames
                < self.config.provisional_missing_tolerance_frames
            ):
                self._provisional_missing_frames += 1
                self.state = NavigationState.PROVISIONAL
                return "provisional_temporarily_missing", False
            self._clear_provisional()
            self.state = NavigationState.SEARCHING
            self.target_confidence = 0.0
            return "no_eligible_candidate", False

        node_id, score = eligible[0]
        if node_id != self._provisional_id:
            self._provisional_id = node_id
            self._provisional_observations = 1
            self._provisional_first_frame = frame_idx
        else:
            self._provisional_observations += 1
        self._provisional_missing_frames = 0
        self.target_confidence = score
        first_frame = (
            frame_idx if self._provisional_first_frame is None else self._provisional_first_frame
        )
        persistent_frames = frame_idx - int(first_frame) + 1
        if (
            self._provisional_observations >= self.config.confirmation_observations
            and persistent_frames >= self.config.minimum_persistent_frames
        ):
            self.locked_target_id = node_id
            self.state = NavigationState.LOCKED
            self._locked_missing_frames = 0
            self._clear_challenger()
            return "candidate_confirmed", True
        self.state = NavigationState.PROVISIONAL
        return "candidate_provisional", False

    def _observe_locked(
        self,
        ordered: list[tuple[int, float]],
        valid_ids: set[int],
    ) -> tuple[str, bool, bool, bool]:
        locked_id = int(self.locked_target_id)
        scores = dict(ordered)
        if locked_id not in valid_ids or locked_id not in scores:
            self._locked_missing_frames += 1
            self._clear_challenger()
            if self._locked_missing_frames <= self.config.locked_missing_tolerance_frames:
                return "locked_target_temporarily_missing", False, False, False
            if self._locked_missing_frames < self.config.target_lost_after_frames:
                return "locked_target_missing_debounced", False, False, False
            self.locked_target_id = None
            self.target_confidence = 0.0
            self._clear_provisional()
            self.state = NavigationState(self.config.target_loss_state)
            return "locked_target_lost", False, False, True

        self._locked_missing_frames = 0
        locked_score = scores[locked_id]
        self.target_confidence = locked_score
        challenger = next(
            (
                (node_id, score)
                for node_id, score in ordered
                if node_id != locked_id
                and node_id in valid_ids
                and score >= self.config.better_target_min_confidence
                and score >= locked_score + self.config.better_target_score_margin
            ),
            None,
        )
        if challenger is None:
            self._clear_challenger()
            return "locked_target_stable", False, False, False

        challenger_id, challenger_score = challenger
        if challenger_id == self._challenger_id:
            self._challenger_frames += 1
        else:
            self._challenger_id = challenger_id
            self._challenger_frames = 1
        if self._challenger_frames < self.config.better_target_persistence_frames:
            return "better_target_provisional", False, False, False

        self.locked_target_id = challenger_id
        self.target_confidence = challenger_score
        self.state = NavigationState.LOCKED
        self._clear_challenger()
        return "better_target_persisted", True, True, False

    def _clear_provisional(self) -> None:
        self._provisional_id = None
        self._provisional_observations = 0
        self._provisional_first_frame = None
        self._provisional_missing_frames = 0

    def _clear_challenger(self) -> None:
        self._challenger_id = None
        self._challenger_frames = 0


__all__ = [
    "NavigationState",
    "NavigationStateMachine",
    "StateMachineConfig",
    "StateObservation",
]
