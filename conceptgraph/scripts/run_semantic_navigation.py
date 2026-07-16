from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from conceptgraph.navigation import SemanticNavigationPipeline
from conceptgraph.utils.general_utils import measure_time


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="semantic_navigation")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_params.json", "w") as f:
        import json

        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)

    pipeline = SemanticNavigationPipeline(cfg)
    plan = pipeline.run(
        query_text=cfg.query_text,
        vehicle_pose=np.asarray(cfg.vehicle_pose, dtype=float),
    )
    plan_path = pipeline.save_plan(plan, output_dir)

    print(f"Saved semantic navigation plan to: {plan_path}")
    print(f"Selected node: {plan.selected_node_id} ({plan.selected_object_tag})")
    print(f"Target point: {np.round(plan.selected_target_point, 3).tolist()}")
    print(f"Path waypoints: {len(plan.path_world)}")
    print(f"Path cost: {plan.path_cost:.3f}")


if __name__ == "__main__":
    measure_time(main)()
