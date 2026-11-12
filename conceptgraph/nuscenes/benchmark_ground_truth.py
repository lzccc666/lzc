"""Independent semantic targets from official annotations, never planner scores."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

CATEGORY_TEXT = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
    "pedestrian": "行人",
}


def official_category(name):
    if name.startswith("human.pedestrian."):
        return "pedestrian"
    if name.startswith("vehicle.bus."):
        return "bus"
    return {f"vehicle.{category}": category for category in CATEGORY_TEXT}.get(name)


def accepted_targets(
    objects,
    category,
    vehicle_xy,
    *,
    nearest=False,
    anchor_category=None,
    relation=None,
    forward_xy=None,
):
    candidates = [obj for obj in objects if obj["category"] == category]
    if anchor_category:
        anchors = [obj for obj in objects if obj["category"] == anchor_category]
        forward = np.asarray(forward_xy, dtype=float)
        forward = forward / np.linalg.norm(forward)
        left = np.array([-forward[1], forward[0]])
        retained = []
        for target in candidates:
            for anchor in anchors:
                if target["instance_token"] == anchor["instance_token"]:
                    continue
                delta = np.asarray(target["center"][:2]) - anchor["center"][:2]
                component = delta @ (
                    left if relation in {"left_of", "right_of"} else forward
                )
                sign = -1 if relation in {"right_of", "behind"} else 1
                if sign * component > 1.0 and np.linalg.norm(delta) <= 15.0:
                    retained.append(target)
                    break
        candidates = retained
    if nearest and candidates:
        distances = [
            np.linalg.norm(np.asarray(obj["center"][:2]) - vehicle_xy)
            for obj in candidates
        ]
        minimum = min(distances)
        candidates = [
            obj
            for obj, distance in zip(candidates, distances)
            if distance <= minimum + 0.5
        ]
    return sorted(obj["instance_token"] for obj in candidates)


def associate_nodes(nodes, objects, *, maximum_distance_m=3.0, ambiguity_margin_m=0.5):
    """One-to-one same-category association; uncertain mappings remain explicit."""
    if not nodes or not objects:
        return {
            "matches": [],
            "unmatched_nodes": [n["node_id"] for n in nodes],
            "unmatched_instances": [o["instance_token"] for o in objects],
            "ambiguous_nodes": [],
        }
    costs = np.full((len(nodes), len(objects)), 1e9)
    for i, node in enumerate(nodes):
        for j, obj in enumerate(objects):
            if node["category"] == obj["category"]:
                costs[i, j] = np.linalg.norm(
                    np.asarray(node["center"][:2]) - obj["center"][:2]
                )
    rows, cols = linear_sum_assignment(costs)
    matches, ambiguous = [], []
    for i, j in zip(rows, cols):
        distance = costs[i, j]
        if distance > maximum_distance_m:
            continue
        alternatives = np.delete(costs[i], j)
        competing_nodes = np.delete(costs[:, j], i)
        if (
            len(alternatives) and np.min(alternatives) - distance < ambiguity_margin_m
        ) or (
            len(competing_nodes)
            and np.min(competing_nodes) - distance < ambiguity_margin_m
        ):
            ambiguous.append(nodes[i]["node_id"])
            continue
        matches.append(
            {
                "node_id": nodes[i]["node_id"],
                "instance_token": objects[j]["instance_token"],
                "distance_m": float(distance),
            }
        )
    matched_nodes = {m["node_id"] for m in matches}
    matched_instances = {m["instance_token"] for m in matches}
    return {
        "matches": matches,
        "ambiguous_nodes": ambiguous,
        "unmatched_nodes": [
            n["node_id"] for n in nodes if n["node_id"] not in matched_nodes
        ],
        "unmatched_instances": [
            o["instance_token"]
            for o in objects
            if o["instance_token"] not in matched_instances
        ],
    }


def retrieval_metrics(ranking, accepted_instances, association):
    if len(ranking) != len(set(ranking)):
        raise ValueError("duplicate ranked node ID")
    accepted = set(accepted_instances)
    mapping = {m["node_id"]: m["instance_token"] for m in association["matches"]}
    if not accepted:
        return {
            "top1": None,
            "mrr": None,
            "correct_rejection": not ranking,
            "status": "no_matching_target",
            "candidate_recall": None,
        }
    ranks = [i + 1 for i, node in enumerate(ranking) if mapping.get(node) in accepted]
    found = {mapping[node] for node in ranking if mapping.get(node) in accepted}
    return {
        "top1": bool(ranks and min(ranks) == 1),
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
        "candidate_recall": len(found) / len(accepted),
        "correct_rejection": None,
        "status": "evaluated_with_official_instances",
        "association_ambiguous_count": len(association["ambiguous_nodes"]),
    }
