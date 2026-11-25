"""Resolve explicitly versioned production jobs without hiding earlier runs."""

from __future__ import annotations

import json


def resolve_online_directory(root, query, variant, lane):
    index = root / "production_job_index.json"
    if index.exists():
        # Callers may cache the index in their own process, but always resolve
        # the declared source instead of inferring the best outcome.
        rows = json.loads(index.read_text())["jobs"]
        row = next(r for r in rows if r["query_id"] == query["query_id"] and r["variant"] == variant and r["lane"] == lane)
        return root / row["active_directory"]
    run = ("cpu_test_v1" if query["role"] == "sealed_test" else f"cpu_development_{query['scene'].replace('-', '')}_v2") if variant == "off" else f"gpu_{variant}_matrix_v1"
    return root / "experiments" / run / query["query_id"] / f"{variant}_{'lane' if lane else 'drivable'}"
