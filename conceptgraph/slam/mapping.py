import torch
import torch.nn.functional as F
import numpy as np

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accuracte_batch, 
    compute_3d_giou_accurate_batch,
)
from conceptgraph.slam.utils import (
    merge_obj2_into_obj1, 
    compute_overlap_matrix_2set,
    ensure_motion_metadata,
    estimate_observation_velocity,
    initialize_motion_metadata,
    should_use_short_term_update,
    update_dynamic_object_short_term,
)

def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return sims

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    if getattr(cfg, "association_policy", "legacy") == "one_to_one":
        return _merge_one_to_one(cfg, detection_list, objects, agg_sim)
    if not bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        return _merge_detections_to_objects_legacy(cfg, detection_list, objects, agg_sim)

    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        initialize_motion_metadata(cfg, detection_list[i])
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            det_classes = set(map(int, np.atleast_1d(detection_list[i]["class_id"]).tolist()))
            best_j = None
            best_score = float("-inf")
            for j in torch.argsort(agg_sim[i], descending=True).tolist():
                obj_classes = set(map(int, np.atleast_1d(objects[j]["class_id"]).tolist()))
                if det_classes.isdisjoint(obj_classes):
                    continue
                det_center = _bbox_center_from_object(detection_list[i]["bbox"])
                obj_center = _bbox_center_from_object(objects[j]["bbox"])
                if float(np.linalg.norm(det_center - obj_center)) > float(getattr(cfg, "max_tracking_distance_m", 8.0)):
                    continue
                score = float(agg_sim[i, j])
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                objects.append(detection_list[i])
                continue
            j = best_j
            matched_det = detection_list[i]
            matched_obj = objects[j]
            ensure_motion_metadata(cfg, matched_obj)
            velocity, current_center, current_frame = estimate_observation_velocity(cfg, matched_obj, matched_det)
            if should_use_short_term_update(cfg, matched_obj, matched_det, velocity):
                merged_obj = update_dynamic_object_short_term(
                    cfg,
                    matched_obj,
                    matched_det,
                    velocity=velocity,
                    current_center=current_center,
                    current_frame=current_frame,
                )
            else:
                merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
    
    return objects


def _merge_one_to_one(cfg, detection_list, objects, agg_sim):
    """Freeze all associations before mutating geometry; each track gets one observation."""
    from conceptgraph.completion.association import gated_assignment

    scores = agg_sim.detach().cpu().numpy().copy()
    for i, det in enumerate(detection_list):
        initialize_motion_metadata(cfg, det)
        for j, obj in enumerate(objects):
            det_classes = set(np.atleast_1d(det["class_id"]).tolist())
            obj_classes = set(np.atleast_1d(obj["class_id"]).tolist())
            distance = np.linalg.norm(_bbox_center_from_object(det["bbox"])
                                      - _bbox_center_from_object(obj["bbox"]))
            if (det_classes.isdisjoint(obj_classes)
                    or distance > float(getattr(cfg, "max_tracking_distance_m", 8.0))):
                scores[i, j] = -np.inf
    assignments = gated_assignment(scores)
    for i, det in enumerate(detection_list):
        if i not in assignments:
            objects.append(det)
            continue
        j = assignments[i]
        obj = objects[j]
        velocity, center, frame_idx = estimate_observation_velocity(cfg, obj, det)
        if should_use_short_term_update(cfg, obj, det, velocity):
            merged = update_dynamic_object_short_term(cfg, obj, det, velocity, center, frame_idx)
        else:
            merged = merge_obj2_into_obj1(cfg, obj, det, run_dbscan=False)
        objects[j] = merged
    return objects


def _merge_detections_to_objects_legacy(
    cfg,
    detection_list: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
) -> MapObjectList:
    for i in range(agg_sim.shape[0]):
        if agg_sim[i].max() == float("-inf"):
            objects.append(detection_list[i])
        else:
            det_classes = set(map(int, np.atleast_1d(detection_list[i]["class_id"]).tolist()))
            best_j = None
            best_score = float("-inf")
            for j in torch.argsort(agg_sim[i], descending=True).tolist():
                obj_classes = set(map(int, np.atleast_1d(objects[j]["class_id"]).tolist()))
                if det_classes.isdisjoint(obj_classes):
                    continue
                det_center = _bbox_center_from_object(detection_list[i]["bbox"])
                obj_center = _bbox_center_from_object(objects[j]["bbox"])
                if float(np.linalg.norm(det_center - obj_center)) > float(getattr(cfg, "max_tracking_distance_m", 8.0)):
                    continue
                score = float(agg_sim[i, j])
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                objects.append(detection_list[i])
                continue
            objects[best_j] = merge_obj2_into_obj1(cfg, objects[best_j], detection_list[i], run_dbscan=False)
    return objects


def _bbox_center_from_object(bbox) -> np.ndarray:
    if hasattr(bbox, "center"):
        return np.asarray(bbox.center, dtype=float)
    if hasattr(bbox, "get_center"):
        return np.asarray(bbox.get_center(), dtype=float)
    if isinstance(bbox, np.ndarray):
        return np.asarray(bbox, dtype=float).mean(axis=0)
    return np.asarray(bbox["center"], dtype=float)
