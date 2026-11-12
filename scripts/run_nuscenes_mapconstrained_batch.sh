#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/lizichong
DATASET_ROOT=$ROOT/dataset/nuscenes
MAP_ROOT=$ROOT/dataset/nuscenes_map_expansion
export QWEN_VL_CKPT_PATH=$ROOT/checkpoint/Qwen2-VL-2B-Instruct
export HF_HOME=$ROOT/checkpoint/hf_cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib:${LD_LIBRARY_PATH:-}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

SCENES=(
  scene-0003
  scene-0045
  scene-0128
  scene-0133
  scene-0168
)

for SCENE in "${SCENES[@]}"; do
  echo "===== ${SCENE} ====="
  SCENE_DIR=$DATASET_ROOT/$SCENE
  EXP_DIR=$SCENE_DIR/exp_streamlined_yolo_outdoor_objects_v1
  NAV_DIR=$EXP_DIR/navigation_mapconstrained
  BEV_MAP=$EXP_DIR/navigation/bev_map.npz

  if [ ! -f "$EXP_DIR/detections/000000.pkl.gz" ]; then
    python conceptgraph/scripts/streamlined_detections.py \
      dataset_root=$DATASET_ROOT \
      scene_id=$SCENE \
      dataset_config=$ROOT/conceptgraph/nuscenes/nuscenes.yaml \
      classes_file=$ROOT/conceptgraph/vehicle_classes.txt \
      exp_suffix=streamlined_yolo_outdoor_objects_v1 \
      save_video=false
  fi

  if [ ! -f "$EXP_DIR/map/scene_map_cfslam.pkl.gz" ]; then
    python -m conceptgraph.slam.streamlined_mapping \
      dataset_root=$DATASET_ROOT \
      dataset_config=$ROOT/conceptgraph/nuscenes/nuscenes.yaml \
      scene_id=$SCENE \
      detection_folder_name=exp_streamlined_yolo_outdoor_objects_v1/detections \
      gsa_variant=streamlined_yolo \
      +exp_suffix=streamlined_yolo_outdoor_objects_v1 \
      save_suffix=outdoor_objects_tracking_gate \
      device=cuda:0 \
      save_objects_all_frames=true \
      obj_min_detections=1 \
      min_points_threshold=4 \
      dbscan_remove_noise=false \
      match_method=sim_sum \
      sim_threshold=0.35 \
      phys_bias=0.5 \
      merge_overlap_thresh=-1 \
      max_tracking_distance_m=8.0
  fi

  if [ ! -f "$EXP_DIR/cfslam_qwen_responses/0.json" ]; then
    python conceptgraph/scripts/generate_scenegraph_stub_responses.py \
      --mapfile "$EXP_DIR/map/scene_map_cfslam.pkl.gz" \
      --out-dir "$EXP_DIR/cfslam_qwen_responses"
  fi

  if [ ! -f "$EXP_DIR/cfslam_scenegraph_edges.pkl" ]; then
    python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
      --mode build-scenegraph \
      --cachedir "$EXP_DIR" \
      --mapfile "$EXP_DIR/map/scene_map_cfslam.pkl.gz" \
      --min-views-per-object 1 \
      --device cuda:0 \
      --strong-edge-max-distance-m 8.0
  fi

  if [ ! -f "$EXP_DIR/scene_graph.json" ]; then
    python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
      --mode generate-scenegraph-json \
      --cachedir "$EXP_DIR" \
      --mapfile "$EXP_DIR/map/scene_map_cfslam.pkl.gz" \
      --min-views-per-object 1 \
      --device cuda:0 \
      --strong-edge-max-distance-m 8.0
  fi

  if [ ! -f "$BEV_MAP" ]; then
    python conceptgraph/scripts/build_nuscenes_bev_map.py \
      --scene-dir "$SCENE_DIR" \
      --map-root "$MAP_ROOT" \
      --scene-name "$SCENE" \
      --output "$BEV_MAP" \
      --resolution 0.5 \
      --padding-m 20
  fi

  VEHICLE_POSE=$(python - <<PY
import numpy as np
poses=np.loadtxt("${SCENE_DIR}/poses.txt").reshape(-1,4,4)
xyz=poses[-1,:3,3]
print(f"[{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f}]")
PY
)
  VEHICLE_POSE_CSV=$(python - <<PY
import numpy as np
poses=np.loadtxt("${SCENE_DIR}/poses.txt").reshape(-1,4,4)
xyz=poses[-1,:3,3]
print(f"{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f}")
PY
)

  python conceptgraph/scripts/run_semantic_navigation.py \
    dataset_root=$DATASET_ROOT \
    scene_id=$SCENE \
    exp_suffix=streamlined_yolo_outdoor_objects_v1 \
    bev_map_path="$BEV_MAP" \
    "vehicle_pose=${VEHICLE_POSE}" \
    'query_text="去前方停车车辆附近"' \
    output_dir="$NAV_DIR"

  python conceptgraph/scripts/visualize_scenegraph.py \
    --scene-graph "$EXP_DIR/scene_graph.json" \
    --edges "$EXP_DIR/cfslam_scenegraph_edges.pkl" \
    --out-dir "$EXP_DIR/scenegraph_vis_mapconstrained" \
    --vehicle-pose "$VEHICLE_POSE_CSV" \
    --navigation-plan "$NAV_DIR/semantic_navigation_plan.json" \
    --trajectory-poses "$SCENE_DIR/poses.txt" \
    --bev-map "$BEV_MAP"
done
