# Semantic Target Navigation

This module converts a natural-language target into a reachable path using:

1. Semantic target retrieval on the concept graph
2. Reachable-goal projection onto a traversable 2D occupancy map
3. A* path planning with optional semantic bias

## Inputs

- Natural language target text, for example: `去最近停车区域附近`
- Current vehicle pose: `vehicle_pose=[x, y, z]`
- 3D concept graph artifacts:
  - `scene_graph.json`
  - `cfslam_scenegraph_edges.pkl`
  - `map/scene_map_cfslam_pruned.pkl.gz`
- Optional traversable BEV map:
  - `.npy` occupancy array
  - or `.npz` with keys `occupancy` or `bev`, and optional `origin_xy`, `resolution`

## Run

```bash
python /root/autodl-tmp/lizichong/conceptgraph/scripts/run_semantic_navigation.py \
  query_text='去最近停车区域附近' \
  vehicle_pose='[12.0, 4.0, 0.0]'
```

If no BEV map is provided, the module falls back to a conservative occupancy map built from object bounding boxes in the scene map.

## Output

The script writes:

- `semantic_navigation_plan.json`
- `config_params.json`

under:

```text
${dataset_root}/${scene_id}/exp_${exp_suffix}/navigation/
```

The plan JSON contains:

- selected semantic target node
- reachable target point
- global path in grid and world coordinates
- ranked candidate targets with semantic and relation scores
