#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/lizichong
DATASET_ROOT=$ROOT/dataset/nuscenes
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
  scene-0916
)

for SCENE in "${SCENES[@]}"; do
  echo "===== ${SCENE} qwen semantics ====="
  SCENE_DIR=$DATASET_ROOT/$SCENE
  BASE_EXP_DIR=$SCENE_DIR/exp_streamlined_yolo_outdoor_objects_v1
  QWEN_EXP_DIR=$SCENE_DIR/exp_qwen2vl_semantics_v1
  mkdir -p "$QWEN_EXP_DIR"

  python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
    --mode extract-node-captions \
    --cachedir "$QWEN_EXP_DIR" \
    --mapfile "$BASE_EXP_DIR/map/scene_map_cfslam.pkl.gz" \
    --device cuda:0 \
    --max-detections-per-object 5 \
    --masking-option none

  python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
    --mode refine-node-captions \
    --cachedir "$QWEN_EXP_DIR" \
    --mapfile "$BASE_EXP_DIR/map/scene_map_cfslam.pkl.gz" \
    --device cuda:0

  python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
    --mode build-scenegraph \
    --cachedir "$QWEN_EXP_DIR" \
    --mapfile "$BASE_EXP_DIR/map/scene_map_cfslam.pkl.gz" \
    --device cuda:0 \
    --min-views-per-object 1 \
    --strong-edge-max-distance-m 8.0

  python -m conceptgraph.scenegraph.build_scenegraph_cfslam \
    --mode generate-scenegraph-json \
    --cachedir "$QWEN_EXP_DIR" \
    --mapfile "$BASE_EXP_DIR/map/scene_map_cfslam.pkl.gz" \
    --device cuda:0 \
    --min-views-per-object 1 \
    --strong-edge-max-distance-m 8.0
done
