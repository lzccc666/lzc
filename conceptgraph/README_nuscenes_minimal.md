# Minimal nuScenes Layout For ConceptGraphs

This repository now supports a minimal preprocessed nuScenes subset for the
single-front-camera mapping pipeline.

The expected directory structure is:

```text
<dataset_root>/
  scene-0061/
    color/
      000000.jpg
      000001.jpg
      ...
    depth/
      000000.png
      000001.png
      ...
    poses.txt
    intrinsics.txt
```

Notes:

- `color/` contains the ordered `CAM_FRONT` keyframe image stream.
- `depth/` contains LiDAR-projected sparse depth maps in millimeters.
- `poses.txt` stores one flattened 4x4 camera-to-world pose per line.
- `intrinsics.txt` stores a 3x3 camera intrinsic matrix.

This layout matches the same minimal format already used for the KITTI-360
adapter, so the existing detection and mapping scripts do not need path changes.

To use it with Hydra configs, replace the dataset override in the config with
`nuscenes` and edit `dataset_root` plus `scene_id`.

## Convert Raw nuScenes-mini Data

To create this minimal layout from a standard nuScenes download, run:

```bash
cd conceptgraph
python scripts/preprocess_nuscenes_to_minimal.py \
  --nuscenes-root /path/to/nuscenes \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --output-root /path/to/nuscenes_minimal \
  --stride 1 \
  --max-frames 100
```

What this script does:

- reads `CAM_FRONT` keyframe images from `samples/CAM_FRONT`
- reads `LIDAR_TOP` point clouds from `samples/LIDAR_TOP`
- reads metadata tables from `v1.0-mini/*.json`
- projects LiDAR points into the front camera image plane to create sparse depth PNGs
- writes the minimal `color/`, `depth/`, `poses.txt`, and `intrinsics.txt` output

Required downloads for the mini pipeline:

- `v1.0-mini`
- `samples`

Optional but not required for this minimal bootstrap path:

- `sweeps`
- `maps`
- `lidarseg`
- `panoptic`

Important limitations:

- this is a minimal conversion for bootstrapping the ConceptGraphs pipeline
- the produced depth is sparse LiDAR-projected depth, not dense depth
- this script uses only `CAM_FRONT` and `LIDAR_TOP`