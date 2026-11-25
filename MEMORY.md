# Project Memory: ConceptGraphs + nuScenes Semantic Navigation

This file stores the implementation and experiment context for future Codex sessions.

## Basic Environment

Project root:

```text
/root/autodl-tmp/lizichong
```

Conda environment:

```bash
conda activate conceptgraph
```

Raw nuScenes path:

```text
/root/autodl-pub/nuScenes
```

Minimal nuScenes experiment root:

```text
/root/autodl-tmp/lizichong/dataset/nuscenes
```

Main experiment suffix:

```text
exp_qwen2vl_semantics_v1
```

Baseline mapping/detection suffix:

```text
exp_streamlined_yolo_outdoor_objects_v1
```

Global summary outputs:

```text
/root/autodl-tmp/lizichong/outputs/experiment_summary
```

Qwen finetune dataset:

```text
/root/autodl-tmp/lizichong/outputs/qwen_finetune_dataset
```

Qwen LoRA adapter:

```text
/root/autodl-tmp/lizichong/checkpoint/qwen_lora_semantic_delegate
```

## Completed Scenes

The following nuScenes scenes have been processed:

```text
scene-0003
scene-0045
scene-0061
scene-0103
scene-0128
scene-0133
scene-0168
scene-0553
scene-0655
scene-0757
scene-0916
```

Each final scene experiment should contain:

```text
scene_graph.json
cfslam_scenegraph_edges.pkl
cfslam_scenegraph_edge_metadata.json
scenegraph_vis/
navigation_examples/
visualization_pipeline/
evaluation/
```

## Navigation Module

New directory:

```text
conceptgraph/navigation/
```

Important files:

```text
conceptgraph/navigation/types.py
conceptgraph/navigation/io.py
conceptgraph/navigation/semantic.py
conceptgraph/navigation/planner.py
conceptgraph/navigation/pipeline.py
conceptgraph/navigation/evaluation.py
```

Implemented functions:

- Load `scene_graph.json`.
- Load `cfslam_scenegraph_edges.pkl`.
- Load `scene_map_cfslam_pruned.pkl.gz`.
- Parse Chinese semantic navigation instructions.
- Rank candidate targets by category, color, relation, and distance prior.
- Check target reachability.
- Plan paths with occupancy map + A*.
- Save raw A* path and smoothed path.
- Save navigation visualizations and evaluation metrics.

## Reachability, Occupancy, and A*

Main files:

```text
conceptgraph/navigation/planner.py
conceptgraph/navigation/pipeline.py
conceptgraph/scripts/build_nuscenes_bev_map.py
```

Current strategy:

- nuScenes BEV map provides drivable/free region.
- Scene object bboxes are overlaid as obstacles.
- Object obstacle inflation is intentionally not too conservative.
- Top-K semantic candidates are checked one by one for reachability.
- If target center is not free/reachable, planner falls back to nearby free cell.
- Raw A* path is saved together with smoothed path.

Important default thesis-friendly parameters:

```text
obstacle_profile: thesis
robot_radius_m: 0.8
object_bbox_obstacle_scale: 0.18
max_object_obstacle_radius_m: 1.2
goal_search_radius_m: 14.0
start_search_radius_m: 16.0
ignore_dynamic_obstacles: false
```

Recorded navigation fields:

```text
fallback_used
fallback_reason
fallback_distance_m
candidate_reachable_ratio
occupancy_free_ratio
planning_time_s
raw_astar_waypoints
smoothed_waypoints
```

Visualization convention:

- Light dashed path: raw A* polyline.
- Strong solid path: smoothed path.
- Mark start, selected target, candidate targets, reachable fallback target, and fallback link.

## Dynamic Object Handling

Relevant files:

```text
conceptgraph/slam/streamlined_mapping.py
conceptgraph/slam/mapping.py
conceptgraph/slam/utils.py
conceptgraph/slam/slam_classes.py
conceptgraph/scripts/build_scenegraph_from_map_stub.py
```

Design goal:

```text
Static objects are long-term fused.
Dynamic objects are short-term maintained.
```

Object motion fields:

```text
is_potential_dynamic
motion_state: static / dynamic / unknown
center_history
frame_history
velocity
is_short_term
last_seen_frame
```

Potential dynamic classes:

```text
car
truck
bus
motorcycle
bicycle
pedestrian
vehicle
van
```

Static classes:

```text
traffic sign
traffic light
road
sidewalk
building
tree
fence
pole
parking area
```

Rules:

- Vehicles are only potential dynamic, not always dynamic.
- Parked/low-speed vehicles are allowed as static long-term objects.
- Old maps without velocity default potential-dynamic vehicles to static unless observed velocity exceeds threshold.
- Dynamic objects should not accumulate long-term point clouds, to avoid point-cloud trails.
- Dynamic relation edges are marked temporal.
- Static relation edges are marked stable.

Final audit result:

```text
num_unknown_motion_nodes = 0
```

for all 11 scenes.

## Scene Graph Generation

Main script:

```text
conceptgraph/scripts/build_scenegraph_from_map_stub.py
```

Outputs:

```text
scene_graph.json
cfslam_scenegraph_edges.pkl
cfslam_scenegraph_edge_metadata.json
map/scene_map_cfslam_pruned.pkl.gz
cfslam_captions_qwen/
cfslam_captions_qwen_debug/
```

Current strong-edge parameters:

```bash
--max-edge-distance-m 8.0
--min-relation-confidence 0.45
--max-edges-per-node 4
```

Supported relation labels:

```text
near
in_front_of
behind
left_of
right_of
```

Edge metadata example:

```json
{
  "source": 0,
  "target": 1,
  "relation": "near",
  "distance_m": 3.2,
  "confidence": 0.7,
  "relation_type": "stable",
  "dynamic_relation": false,
  "source_motion_state": "static",
  "target_motion_state": "static"
}
```

Node fields include:

```text
id
object_tag
object_tag_zh
bbox_center
bbox_extent
possible_tags
caption
raw_captions
attributes.colors
motion_state
is_potential_dynamic
velocity
is_short_term
last_seen_frame
center_history
frame_history
metadata
```

## Scene Graph Visualization

Scripts:

```text
conceptgraph/scripts/visualize_scenegraph.py
conceptgraph/scripts/visualize_scenegraph_3d.py
conceptgraph/scripts/visualize_scenegraph_3d_interactive.py
```

Output directory:

```text
scenegraph_vis/
```

Output files:

```text
scenegraph_bev.png
scenegraph_network.png
scenegraph_3d.png
scenegraph_3d_interactive.html
scenegraph_summary.html
scenegraph_summary.json
```

Paper-figure conventions:

- Figure text should be Chinese when possible.
- Nodes are labeled only by id.
- Right-side table lists:
  - node id
  - English tag
  - Chinese tag
  - motion state
  - color attribute
- Different classes use fixed colors.
- Edge text is not drawn in the middle of the graph.
- Relation meanings are shown in legend:
  - solid line: near
  - dashed line: front/behind
  - dotted line: left/right
  - gray: stable relation
  - orange: dynamic/temporal relation
- BEV figure shows ego trajectory.
- Ego vehicle is shown as a visible star marker.
- 3D figure avoids exaggerated object heights or objects sinking below ground.
- Interactive HTML hover shows id, English tag, Chinese tag, motion state, colors, and caption.

## Chinese Navigation Query Generation

Main script:

```text
conceptgraph/scripts/run_nuscenes_navigation_examples.py
```

Supported base queries:

```text
前往最近的车辆附近
前往前方的车辆附近
前往小汽车附近
前往卡车附近
前往公交车附近
前往摩托车附近
前往交通标志附近
前往行人附近
前往最近的静态目标附近
```

Supported color queries:

```text
前往黑色小汽车附近
前往灰色卡车附近
前往红色车辆附近
```

Supported relation queries:

```text
前往停在灰色小汽车旁边的车辆附近
前往黑色小汽车旁边的摩托车附近
前往交通标志旁边的车辆附近
前往靠近交通标志的车辆附近
前往某颜色车辆旁边的行人附近
```

Rules:

- Generate a query only if target and anchor exist in the scene graph.
- Do not generate pedestrian queries if no pedestrian exists.
- Do not invent colors if no color attribute exists.
- Relation queries are based on scenegraph edges or geometric neighbors.
- Every retained successful case must pass `target_validation=True`.
- `navigation_examples/` contains only successful, semantically valid examples.
- Failures can be stored under evaluation files, but not in paper-display directories.

Final audit:

```text
No English category words in query_text.
All target_validation values are True.
No planning failures inside navigation_examples.
```

## Multi-Scene Experiment Results

Successful navigation examples:

```text
scene-0003: 13
scene-0045: 15
scene-0061: 19
scene-0103: 11
scene-0128: 9
scene-0133: 18
scene-0168: 12
scene-0553: 14
scene-0655: 15
scene-0757: 13
scene-0916: 16
```

Totals:

```text
successful navigation cases: 155
complex relation/color queries: 75
global failures: 0
```

Core metrics:

```text
average path length: 16.27 m
average planning time: 0.040 s
average occupancy free ratio: 0.512
average candidate reachable ratio: 0.296
average fallback distance: 2.89 m
```

Global summary files:

```text
outputs/experiment_summary/all_scene_summary.csv
outputs/experiment_summary/all_navigation_cases.csv
outputs/experiment_summary/failures.json
outputs/experiment_summary/experiment_report.md
```

Each scene's `evaluation/` contains:

```text
scene_metrics.json
navigation_cases.json
target_retrieval_metrics.json
path_planning_metrics.json
reachability_metrics.json
scenegraph_metrics.json
failures.json
summary.csv
```

## Qwen / Qwen-VL

Main files:

```text
conceptgraph/vlm/qwen_vl_chat.py
conceptgraph/vlm/qwen_semantic_delegate.py
```

Local model:

```text
/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct
```

Current design:

- Main pipeline is still rule + detection-label + geometry based for stability.
- Qwen delegate is optional.
- Qwen is disabled by default for online stages to avoid GPU/memory/runtime instability.
- If Qwen fails, fallback rule logic keeps the pipeline alive.
- Prompts, raw responses, and parsed responses can be written to `qwen_delegate_logs/`.

Potential Qwen delegate tasks:

```text
object crop caption
object_tag / possible_tags summary
color / body_type / fine-grained attributes
relation inference
navigation query parsing
candidate reranking
```

Reports:

```text
outputs/experiment_summary/qwen_usage_report.md
outputs/experiment_summary/qwen_finetune_feasibility.md
outputs/experiment_summary/qwen_lora_training_plan.md
```

## Qwen Finetune Dataset

Builder:

```text
conceptgraph/scripts/build_qwen_finetune_dataset.py
```

Output directory:

```text
outputs/qwen_finetune_dataset/
```

Files:

```text
object_caption_dataset.jsonl
relation_dataset.jsonl
navigation_query_dataset.jsonl
candidate_rerank_dataset.jsonl
dataset_stats.json
```

Sample counts:

```text
object_caption: 426
relation: 383
navigation_query: 155
candidate_rerank: 155
```

Class distribution:

```text
car: 218
truck: 66
traffic sign: 50
traffic light: 10
pedestrian: 4
bus: 40
motorcycle: 15
bicycle: 9
road: 14
```

Color distribution:

```text
gray: 322
black: 103
green: 3
red: 1
```

Relation distribution:

```text
right_of: 67
near: 110
in_front_of: 74
behind: 52
left_of: 80
```

## Qwen LoRA Finetune Experiment

Training script:

```text
scripts/train_qwen_lora_semantic_delegate.py
```

Base model:

```text
/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct
```

Adapter output:

```text
/root/autodl-tmp/lizichong/checkpoint/qwen_lora_semantic_delegate
```

Evaluation files:

```text
outputs/experiment_summary/qwen_lora_eval/comparison_metrics.json
outputs/experiment_summary/qwen_lora_eval/finetune_effect_summary.md
outputs/experiment_summary/qwen_lora_eval/per_task_metrics.csv
outputs/experiment_summary/qwen_lora_eval/baseline_predictions.json
outputs/experiment_summary/qwen_lora_eval/finetuned_predictions.json
```

Training setup:

```text
train samples: 160
eval samples: 64
max_steps: 80
batch_size: 1
grad_accum: 8
lr: 2e-4
LoRA r: 8
LoRA alpha: 16
trainable parameter ratio: about 0.0493%
training time: about 85 seconds
GPU: RTX 4090
```

Overall effect:

```text
JSON valid rate: 0.2812 -> 0.4688 (+0.1875)
field accuracy: 0.0000 -> 0.1531 (+0.1531)
exact match: 0.0000 -> 0.0000
```

Per-task effect:

```text
candidate_rerank:
  JSON valid rate 0.0000 -> 0.8889
  field accuracy 0.0000 -> 0.5556

object_caption:
  JSON valid rate 0.0526 -> 0.4211
  field accuracy 0.0000 -> 0.2000

relation:
  JSON valid rate 0.2083 -> 0.2500
  field accuracy 0.0000 -> 0.0417

navigation_query:
  JSON valid rate 1.0000 -> 0.6667
  field accuracy remains 0.0000
```

Conclusion:

- LoRA improved Qwen's structured JSON output ability for this project's semantic-delegate format.
- Main improvements are in `candidate_rerank` and `object_caption`.
- Relation reasoning remains unstable and can produce repeated fields, drifting labels, or hallucinations.
- The dataset is mostly auto-generated pseudo-labels.
- Do not replace the stable rule module with LoRA Qwen yet.
- Thesis-safe wording:
  - "Small-scale LoRA finetuning improves Qwen's structured-output usability in the proposed semantic delegation tasks."
- Avoid claiming:
  - "Finetuned Qwen already significantly improves real relation reasoning accuracy."

## Useful Commands

### Rebuild Scene Graph

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python conceptgraph/scripts/build_scenegraph_from_map_stub.py \
  --mapfile dataset/nuscenes/scene-0916/exp_streamlined_yolo_outdoor_objects_v1/map/scene_map_cfslam_pruned.pkl.gz \
  --responses-dir dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1/cfslam_qwen_responses \
  --out-dir dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1 \
  --max-edge-distance-m 8.0 \
  --min-relation-confidence 0.45 \
  --max-edges-per-node 4 \
  --write-debug-crops
```

### Generate Scene Graph Visualization

```bash
python conceptgraph/scripts/visualize_scenegraph.py \
  --scene-graph dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1/scene_graph.json \
  --edges dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1/cfslam_scenegraph_edges.pkl \
  --out-dir dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1/scenegraph_vis \
  --trajectory-poses dataset/nuscenes/scene-0916/poses.txt \
  --bev-map dataset/nuscenes/scene-0916/exp_streamlined_yolo_outdoor_objects_v1/navigation/bev_map.npz
```

### Run Navigation Examples for One Scene

```bash
python conceptgraph/scripts/run_nuscenes_navigation_examples.py \
  --scene-dir dataset/nuscenes/scene-0916 \
  --qwen-exp-dir dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1 \
  --baseline-exp-dir dataset/nuscenes/scene-0916/exp_streamlined_yolo_outdoor_objects_v1 \
  --out-dir dataset/nuscenes/scene-0916/exp_qwen2vl_semantics_v1/navigation_examples \
  --visualize
```

### Summarize Multi-Scene Experiment

```bash
python conceptgraph/scripts/run_nuscenes_experiment_suite.py \
  --minimal-root /root/autodl-tmp/lizichong/dataset/nuscenes \
  --scene-list scene-0003 scene-0045 scene-0061 scene-0103 scene-0128 scene-0133 scene-0168 scene-0553 scene-0655 scene-0757 scene-0916 \
  --num-scenes 11 \
  --exp-suffix qwen2vl_semantics_v1 \
  --baseline-suffix streamlined_yolo_outdoor_objects_v1 \
  --run-evaluation-only
```

### Build Qwen Finetune Dataset

```bash
python conceptgraph/scripts/build_qwen_finetune_dataset.py \
  --minimal-root /root/autodl-tmp/lizichong/dataset/nuscenes \
  --exp-suffix qwen2vl_semantics_v1 \
  --scene-list scene-0003 scene-0045 scene-0061 scene-0103 scene-0128 scene-0133 scene-0168 scene-0553 scene-0655 scene-0757 scene-0916
```

### Qwen LoRA Finetune

```bash
python scripts/train_qwen_lora_semantic_delegate.py \
  --max-train-samples 160 \
  --max-eval-samples 64 \
  --max-steps 80 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 2e-4 \
  --output-dir checkpoint/qwen_lora_semantic_delegate \
  --report-dir outputs/experiment_summary/qwen_lora_eval
```

## Thesis Writing Suggestions

Good scenes for main text:

```text
scene-0061
scene-0133
scene-0916
```

Reason:

- Moderate node count.
- Clear navigation paths.
- Richer relation/color query examples.
- Good BEV scenegraph visual effect.

Good scene for appendix:

```text
scene-0655
```

Reason:

- More nodes and higher complexity.
- Useful to show large-scene capacity, but may be too crowded for main text.

Safe claims:

- The system completes object-level 3D mapping, scene graph generation, and semantic navigation planning on multiple nuScenes scenes.
- Reachability filtering and fallback-to-nearby-free-cell reduce planning failures caused by map/out-of-drivable-area targets.
- Dynamic object handling uses potential dynamic class plus velocity/motion-state, rather than treating all vehicles as dynamic.
- Qwen LoRA finetuning is an exploratory semantic delegation experiment; it improves structured output usability, but relation reasoning still needs rules or cleaned labels.

Avoid overclaiming:

- Do not claim real online autonomous driving.
- Do not claim finetuned Qwen has reliable real-world relation reasoning.
- Do not claim paths fully satisfy vehicle kinematics. Current planner is BEV + A* + smoothing, not a full vehicle dynamics planner.

## Known Limitations

1. nuScenes map boundaries and object projection can cause targets to fall outside free/drivable regions, so fallback is often needed.
2. A* paths are geometric paths. Smoothing improves display quality but does not produce full control trajectories.
3. Color attributes are mostly from image statistics or pseudo labels; gray/black dominate and red/rare colors are underrepresented.
4. Pedestrian samples are very few. Do not overstate pedestrian navigation performance.
5. Qwen LoRA data is auto-generated pseudo-labeled data, suitable for feasibility experiments but not strong semantic ground truth.
6. Relation edges are still primarily geometric strong associations. Qwen relation inference remains unstable.
7. Full mapping + Qwen is more suitable for offline batch processing. Online navigation should use existing scenegraph + rule retrieval.

## Future Work Priority

1. Manually inspect and clean Qwen finetune samples, especially relation and navigation query samples.
2. Design stricter relation schema to prevent invalid/repeated Qwen fields.
3. Add optional LoRA adapter loading to `QwenSemanticDelegate`.
4. Run ablations:
   - without reachability filter
   - without relation score
   - without dynamic handling
   - without path smoothing
   - without Qwen rerank
5. Select best figures for thesis main text and appendix.
6. If path realism becomes important, add vehicle kinematic constraints or Dubins/Reeds-Shepp smoothing.

## Persistent Change Log

Use this section as a running log whenever the project is modified in future sessions. Each entry should record:

```text
date:
files changed:
what changed:
why:
validation:
outputs:
paper relevance:
remaining risks:
```

### 2026-07-17: Added Project Memory

Files changed:

```text
MEMORY.md
```

What changed:

- Created this persistent memory file.
- Recorded project paths, scene list, modules, experiment outputs, Qwen LoRA results, known limitations, and future work priorities.
- Added this change-log area for future modifications.

Why:

- Future Codex sessions need a compact but complete context entry point.
- The project now spans mapping, scenegraph generation, navigation, Qwen delegation, experiments, and thesis figures; relying on chat history alone is fragile.

Validation:

- File created successfully in project root.

Outputs:

```text
/root/autodl-tmp/lizichong/MEMORY.md
```

Paper relevance:

- This memory file can help reconstruct method evolution, experimental design decisions, and limitations for thesis chapters.

Remaining risks:

- This file must be updated manually after future changes; otherwise it will drift from the actual code.

## Design Rationale Notes

Use this section to preserve not just what changed, but why the project uses the current design. These notes are useful when writing thesis method sections.

### Why Use Object-Level Scene Graphs Instead of Only 2D Detection

- Object-level maps make navigation targets persistent across frames.
- 2D detections are frame-local and disappear when the target leaves the camera view.
- Scene graphs support relational navigation queries, such as "go to the vehicle near the traffic sign".
- The object-level graph is more interpretable for thesis visualization than raw neural features or dense point clouds.

Paper wording:

```text
Compared with frame-level object detection, the object-level scene graph maintains persistent object instances and explicit spatial relations, enabling semantic target retrieval and relation-aware navigation in a map-level representation.
```

### Why Use Detection Before Segmentation

- YOLO/YOLO-World provides object-level bounding boxes and category priors.
- MobileSAM refines the object mask inside each detection region.
- This combination reduces the search area for segmentation and gives each mask a semantic label.
- Direct segmentation without detection would make instance identity and object class assignment harder.

Paper wording:

```text
The detection stage provides semantic object proposals, while the segmentation stage refines object boundaries for 3D projection. This two-stage design balances semantic discrimination and geometric precision.
```

### Why Keep CLIP/OpenCLIP Features

- CLIP image/text features are still useful for object matching and semantic similarity.
- Qwen/Qwen-VL is used as a language/vision-language semantic delegate, not as a replacement for all visual embeddings.
- CLIP features are lightweight and suitable for repeated similarity computation during mapping and retrieval.
- Qwen is heavier and better suited for offline captioning, relation explanation, or structured semantic outputs.

Paper wording:

```text
CLIP features provide compact visual-semantic embeddings for object association and retrieval, while the VLM module is used for higher-level semantic delegation such as caption summarization and structured query interpretation.
```

### Why Treat Vehicles as Potential Dynamic Objects

- In driving scenes, vehicles can be moving or parked.
- Treating every vehicle as dynamic would remove useful parked cars from the static navigation graph.
- Treating every vehicle as static would cause moving-object point-cloud trails and unstable graph nodes.
- The implemented compromise uses potential dynamic categories plus velocity/motion-state checks.

Paper wording:

```text
Vehicle categories are modeled as potentially dynamic rather than inherently dynamic. This allows parked vehicles to be fused as stable landmarks while preventing fast-moving objects from contaminating long-term geometric maps.
```

### Why Use Reachability Filtering Before Final Target Selection

- The most semantically similar object may lie outside the drivable map or inside an obstacle.
- Directly selecting top-1 semantic target can produce invalid navigation goals.
- Top-K reachability checking allows the system to choose the best semantic target that can actually be reached.
- Fallback to nearby free cells handles small projection/map-boundary errors.

Paper wording:

```text
To bridge semantic retrieval and physical navigation feasibility, candidate targets are filtered by occupancy-map reachability before final goal selection. When the semantic target center is not directly reachable, the planner searches for the nearest reachable free cell around the target.
```

### Why Keep Rule-Based Logic Even After Qwen LoRA

- Qwen LoRA improved structured JSON output, but relation reasoning remains unstable.
- Rule-based geometry is deterministic and easier to validate.
- The thesis system should prioritize correctness and reproducibility over black-box semantic guesses.
- Qwen is best described as an optional semantic delegate or feasibility exploration.

Paper wording:

```text
Although LoRA finetuning improves the VLM's structured-output compliance, deterministic geometric and semantic rules are retained for core navigation decisions to ensure reproducibility and avoid unstable relation hallucinations.
```

## Thesis Reference Bank

This section stores reusable thesis text, tables, and figure guidance.

### Method Contributions

Possible thesis contribution wording:

```text
1. An object-level semantic mapping pipeline is built for nuScenes scenes by combining object detection, instance segmentation, LiDAR-projected depth, and cross-frame object fusion.
2. A dynamic-object-aware fusion strategy is introduced to distinguish long-term static landmarks from short-term potentially dynamic objects.
3. A relation-aware scene graph is constructed from object-level maps, preserving object category, position, motion state, color attributes, and stable/temporal spatial relations.
4. A semantic navigation pipeline is implemented that combines Chinese instruction parsing, scene-graph target retrieval, reachability filtering, occupancy-map planning, and A* path smoothing.
5. A Qwen/Qwen2-VL semantic delegation and LoRA finetuning experiment is conducted to evaluate the feasibility of structured VLM outputs for object semantics, relation reasoning, and candidate reranking.
```

### Experiment Summary Table Values

Current multi-scene experiment:

```text
number of scenes: 11
successful navigation cases: 155
complex relation/color queries: 75
global failures: 0
average path length: 16.27 m
average planning time: 0.040 s
average occupancy free ratio: 0.512
average candidate reachable ratio: 0.296
average fallback distance: 2.89 m
```

Per-scene success count:

```text
scene-0003: 13
scene-0045: 15
scene-0061: 19
scene-0103: 11
scene-0128: 9
scene-0133: 18
scene-0168: 12
scene-0553: 14
scene-0655: 15
scene-0757: 13
scene-0916: 16
```

### Qwen LoRA Table Values

Overall:

```text
JSON valid rate: 0.2812 -> 0.4688 (+0.1875)
field accuracy: 0.0000 -> 0.1531 (+0.1531)
exact match: 0.0000 -> 0.0000
```

Per task:

```text
candidate_rerank:
  JSON valid rate 0.0000 -> 0.8889
  field accuracy 0.0000 -> 0.5556

object_caption:
  JSON valid rate 0.0526 -> 0.4211
  field accuracy 0.0000 -> 0.2000

relation:
  JSON valid rate 0.2083 -> 0.2500
  field accuracy 0.0000 -> 0.0417

navigation_query:
  JSON valid rate 1.0000 -> 0.6667
  field accuracy remains 0.0000
```

Suggested conclusion:

```text
Small-scale LoRA finetuning improves Qwen's structured-output compliance in the proposed semantic delegation tasks, especially for candidate reranking and object semantic formatting. However, relation reasoning remains unstable due to pseudo-label noise and limited manually verified supervision; therefore, rule-based geometric relation reasoning is retained in the final navigation pipeline.
```

### Recommended Figures for Thesis

Main text candidates:

```text
scene-0061
scene-0133
scene-0916
```

Use these for:

- clean BEV scene graph
- relation-aware semantic navigation example
- A* raw path vs smoothed path
- target reachability/fallback visualization

Appendix candidate:

```text
scene-0655
```

Use for:

- large-scene graph complexity
- dense object map visualization

### Claims to Avoid

Do not claim:

- The system is a complete real-time autonomous driving stack.
- Qwen LoRA already provides reliable real-world relation reasoning.
- The A* smoothed path is a full vehicle-kinodynamic trajectory.
- nuScenes pedestrian navigation results are strong; pedestrian samples are too few.
- Automatically generated color/relation labels are equivalent to human ground truth.

Safer wording:

```text
The proposed system demonstrates an offline semantic mapping and navigation planning pipeline on nuScenes scenes. It emphasizes interpretable object-level representation, reachability-aware target selection, and reproducible semantic navigation experiments rather than full closed-loop autonomous driving.
```
