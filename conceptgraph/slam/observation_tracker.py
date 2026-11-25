"""已有真实三维观测上的轻量因果关联原型，不代替点云融合或车辆闭环。"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from conceptgraph.completion.association import gated_assignment


@dataclass(frozen=True)
class ObservationTrackingConfig:
    gate_m: float = 2.5
    ttl_s: float = 1.5
    prediction_horizon_s: float = 1.0
    maximum_speed_mps: float = 20.0


class CausalObservationTracker:
    def __init__(self, config: ObservationTrackingConfig | None = None):
        self.config = config or ObservationTrackingConfig()
        if any(not np.isfinite(value) or value <= 0 for value in vars(self.config).values()):
            raise ValueError("跟踪参数必须是有限正数")
        self.tracks = []
        self.last_timestamp = None
        self.next_id = 0

    def _center(self, track, timestamp):
        center = np.array(track["object"]["center"], dtype=float)
        history = track["history"]
        dt = timestamp - track["timestamp"]
        if len(history) < 2 or dt > self.config.prediction_horizon_s:
            return center
        velocities = [(history[-1][1][:2] - c[:2]) / (history[-1][0] - t)
                      for t, c in history[:-1] if history[-1][0] - t >= .25]
        if velocities:
            velocity = np.median(velocities, axis=0)
            speed = np.linalg.norm(velocity)
            if .5 < speed <= self.config.maximum_speed_mps:
                center[:2] += velocity * dt
        return center

    def update(self, observations, timestamp_s):
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("观测时间必须有限且严格递增")
        # 先校验全部输入再改变状态，失败不会消费时间或部分轨迹。
        incoming = copy.deepcopy(observations)
        for obs in incoming:
            center = np.asarray(obs["center"], dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("观测位置必须是有限三维向量")
            if not isinstance(obs["category"], str):
                raise ValueError("观测类别必须为字符串")
        tracks = [t for t in self.tracks if timestamp - t["timestamp"] <= self.config.ttl_s]
        scores = np.full((len(incoming), len(tracks)), -np.inf)
        for j, track in enumerate(tracks):
            predicted = self._center(track, timestamp)
            for i, obs in enumerate(incoming):
                if obs["category"] != track["object"]["category"]:
                    continue
                distance = np.linalg.norm((np.asarray(obs["center"]) - predicted)[:2])
                if distance <= self.config.gate_m:
                    scores[i, j] = -distance
        assignments = gated_assignment(scores)
        for i, obs in enumerate(incoming):
            if i in assignments:
                track = tracks[assignments[i]]
                track["object"] = obs
                track["timestamp"] = timestamp
                track["history"] = (track["history"] + [(timestamp, np.array(obs["center"]))])[-4:]
                track["hits"] += 1
            else:
                tracks.append(dict(node_id=f"causal-{self.next_id:08d}", object=obs,
                                   timestamp=timestamp, hits=1,
                                   history=[(timestamp, np.array(obs["center"]))]))
                self.next_id += 1
        self.tracks = tracks
        self.last_timestamp = timestamp
        output = []
        for track in tracks:
            obj = copy.deepcopy(track["object"])
            obj.update(node_id=track["node_id"], center=self._center(track, timestamp).tolist(),
                       num_detections=track["hits"], timestamp_s=track["timestamp"],
                       prediction_age_s=timestamp - track["timestamp"])
            output.append(obj)
        return output
