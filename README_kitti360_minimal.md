# Minimal KITTI-360 Layout For ConceptGraphs

This repository now supports a minimal preprocessed KITTI-360 subset for the
single-front-camera mapping pipeline.

The expected directory structure is:

```text
<dataset_root>/
  2013_05_28_drive_0000_sync/
    color/
      000000.png
      000001.png
      ...
    depth/
      000000.png
      000001.png
      ...
    poses.txt
    intrinsics.txt
```

Notes:

- `color/` should contain a single ordered front-camera image stream.
- `depth/` should contain one depth map per image. PNG values are assumed to be
  stored in millimeters by default.
- `poses.txt` can be stored in either of these two formats:
  - one flattened 4x4 pose matrix per line
  - one 4x4 pose matrix over four consecutive lines
- `intrinsics.txt` is optional but recommended. It can contain either:
  - a 3x3 intrinsic matrix
  - four values: `fx fy cx cy`

This layout is intentionally simple so the existing detection and mapping
scripts can run without additional path changes.

To use it with Hydra configs, replace the dataset override in the config with
`kitti360` and edit `dataset_root` plus `scene_id`.

## Convert Raw KITTI-360 Data

To create this minimal layout from the raw KITTI-360 release, run:

```bash
cd conceptgraph
python scripts/preprocess_kitti360_to_minimal.py \
  --kitti360-root /path/to/KITTI-360 \
  --sequence 2013_05_28_drive_0000_sync \
  --output-root /path/to/kitti360_minimal \
  --cam-id 0 \
  --stride 5 \
  --max-frames 200
```

What this script does:

- reads rectified front camera images from `data_2d_raw/.../image_00/data_rect`
- reads poses from `data_poses/.../cam0_to_world.txt`
- reads Velodyne scans from `data_3d_raw/.../velodyne_points/data`
- projects LiDAR points into the rectified image plane to create sparse depth PNGs
- writes the minimal `color/`, `depth/`, `poses.txt`, and `intrinsics.txt` output

Important limitations:

- this is a minimal conversion for bootstrapping the ConceptGraphs pipeline
- the produced depth is sparse LiDAR-projected depth, not dense stereo depth
- frames without both pose and Velodyne data are skipped