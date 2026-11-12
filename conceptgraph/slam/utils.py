from collections import Counter
import copy
import json
import cv2

import numpy as np
from omegaconf import DictConfig
import open3d as o3d
import torch

import torch.nn.functional as F

import faiss

from conceptgraph.utils.general_utils import to_tensor, to_numpy
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList

from conceptgraph.utils.ious import compute_3d_iou, compute_3d_iou_accuracte_batch, mask_subtract_contained, compute_iou_batch
from conceptgraph.dataset.datasets_common import from_intrinsics_matrix


DEFAULT_DYNAMIC_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
}

DEFAULT_STATIC_CLASSES = {
    "traffic sign",
    "traffic light",
    "road",
    "sidewalk",
    "building",
    "tree",
    "fence",
    "pole",
}

MOTION_STATE_KEYS = {
    "timestamp_history_s",
    "timestamp_s",
    "stable_object_id",
    "is_potential_dynamic",
    "motion_state",
    "center_history",
    "frame_history",
    "velocity",
    "is_short_term",
    "last_seen_frame",
}

def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors

def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    
    # get the classes, should be saved when making the dataset
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes  = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # create the class colors, or load them if they exist
    class_colors  = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors

def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.asarray(mask, dtype=bool) & np.isfinite(depth_array) & (depth_array > 0)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)
    
    if points.shape[0] == 0:
        import pdb
        pdb.set_trace()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def process_pcd(pcd, cfg, run_dbscan=True):
    
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
        
    if cfg.dbscan_remove_noise and run_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
        
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()


def _cfg_list(cfg, key: str, default: set[str]) -> set[str]:
    values = getattr(cfg, key, None)
    if values is None:
        return set(default)
    return {normalize_class_name(item) for item in values}


def normalize_class_name(value) -> str:
    return str(value).strip().lower().replace("_", " ")


def get_object_class_names(obj: dict) -> set[str]:
    raw_names = obj.get("class_name", obj.get("object_tag", []))
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    names = {normalize_class_name(name) for name in raw_names if str(name).strip()}
    return names


def get_object_center(obj: dict) -> np.ndarray:
    bbox = obj.get("bbox")
    if bbox is not None:
        if hasattr(bbox, "center"):
            return np.asarray(bbox.center, dtype=float)
        if hasattr(bbox, "get_center"):
            return np.asarray(bbox.get_center(), dtype=float)
        if isinstance(bbox, dict) and "center" in bbox:
            return np.asarray(bbox["center"], dtype=float)
    if obj.get("center_history"):
        return np.asarray(obj["center_history"][-1], dtype=float)
    bbox_np = obj.get("bbox_np")
    if bbox_np is not None:
        points = np.asarray(bbox_np, dtype=float)
        return (points.min(axis=0) + points.max(axis=0)) / 2.0
    raise KeyError("Object has neither bbox nor center_history")


def get_object_frame_idx(obj: dict, fallback: int = 0) -> int:
    if "last_seen_frame" in obj:
        return int(obj["last_seen_frame"])
    image_idx = obj.get("image_idx", [])
    if image_idx:
        return int(np.max(np.asarray(image_idx, dtype=int)))
    return int(fallback)


def is_potential_dynamic_class(cfg, class_names: set[str]) -> bool:
    dynamic_classes = _cfg_list(cfg, "dynamic_classes", DEFAULT_DYNAMIC_CLASSES)
    return bool(class_names & dynamic_classes)


def is_static_class(cfg, class_names: set[str]) -> bool:
    static_classes = _cfg_list(cfg, "static_classes", DEFAULT_STATIC_CLASSES)
    return bool(class_names & static_classes)


def initialize_motion_metadata(cfg, obj: dict, frame_idx: int | None = None) -> dict:
    """Attach motion metadata to a new detection or old deserialized object."""
    if not bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        return obj
    class_names = get_object_class_names(obj)
    is_potential_dynamic = is_potential_dynamic_class(cfg, class_names)
    center = get_object_center(obj).astype(float).tolist()
    if frame_idx is None:
        frame_idx = get_object_frame_idx(obj)

    obj.setdefault("is_potential_dynamic", bool(is_potential_dynamic))
    # 兼容旧配置；新协议中潜在动态类别的首帧不含静止证据。
    initial_unknown = bool(getattr(cfg, "initial_motion_unknown", False)) and is_potential_dynamic
    obj.setdefault("motion_state", "unknown" if initial_unknown else "static")
    obj.setdefault("center_history", [center])
    obj.setdefault("frame_history", [int(frame_idx)])
    obj.setdefault("velocity", 0.0)
    obj.setdefault("is_short_term", False)
    obj.setdefault("last_seen_frame", int(frame_idx))
    trim_motion_history(cfg, obj)
    return obj


def ensure_motion_metadata(cfg, obj: dict) -> dict:
    return initialize_motion_metadata(cfg, obj, frame_idx=get_object_frame_idx(obj))


def trim_motion_history(cfg, obj: dict) -> None:
    max_size = int(getattr(cfg, "center_history_size", 8))
    if max_size <= 0:
        return
    for key in ("center_history", "frame_history", "timestamp_history_s"):
        if key in obj and isinstance(obj[key], list) and len(obj[key]) > max_size:
            obj[key] = obj[key][-max_size:]


def estimate_observation_velocity(cfg, obj: dict, det: dict) -> tuple[float, np.ndarray, int]:
    ensure_motion_metadata(cfg, obj)
    initialize_motion_metadata(cfg, det)
    current_center = get_object_center(det)
    previous_center = np.asarray(obj.get("center_history", [get_object_center(obj)])[-1], dtype=float)
    current_frame = get_object_frame_idx(det, get_object_frame_idx(obj) + 1)
    previous_frame = get_object_frame_idx(obj)
    frame_dt = float(getattr(cfg, "frame_dt", 1.0))
    delta_frames = max(1, int(current_frame) - int(previous_frame))
    delta_t = frame_dt * delta_frames
    if obj.get("timestamp_s") is not None and det.get("timestamp_s") is not None:
        delta_t = float(det["timestamp_s"]) - float(obj["timestamp_s"])
        if delta_t == 0 and current_frame == previous_frame:
            # Legacy many-to-one may fuse two detections from the same image.
            # There is no second time interval: preserve the prior estimate.
            det["motion_evidence_duplicate_timestamp"] = True
            return float(obj.get("velocity", 0.0)), current_center, current_frame
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("object observations require strictly increasing timestamps")
    velocity = float(np.linalg.norm(current_center - previous_center) / delta_t)
    if getattr(cfg, "motion_estimator", "legacy") == "robust_xy":
        from conceptgraph.completion.motion import robust_xy_speed
        history=obj.get("timestamp_history_s", [])
        centers=obj.get("center_history", [])
        if len(history)==len(centers) and len(history)>0 and det.get("timestamp_s") is not None:
            velocity=robust_xy_speed([*centers,current_center], [*history,float(det["timestamp_s"])])
        else:
            velocity=float(np.linalg.norm((current_center-previous_center)[:2])/delta_t)
    return velocity, current_center, current_frame


def update_motion_metadata_after_match(cfg, obj: dict, det: dict, velocity: float, current_center: np.ndarray, current_frame: int) -> dict:
    if not bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        return obj
    if det.get("motion_evidence_duplicate_timestamp", False):
        obj["motion_evidence_duplicate_timestamp"] = True
        return obj
    class_names = get_object_class_names(obj) | get_object_class_names(det)
    is_potential_dynamic = is_potential_dynamic_class(cfg, class_names)
    is_static = is_static_class(cfg, class_names)
    velocity_threshold = float(getattr(cfg, "velocity_threshold", 0.5))

    obj["is_potential_dynamic"] = bool(is_potential_dynamic)
    obj["velocity"] = float(velocity)
    obj["last_seen_frame"] = int(current_frame)
    if det.get("timestamp_s") is not None:
        obj["timestamp_s"] = float(det["timestamp_s"])
        obj.setdefault("timestamp_history_s", []).append(float(det["timestamp_s"]))
    obj.setdefault("center_history", [])
    obj.setdefault("frame_history", [])
    obj["center_history"].append(np.asarray(current_center, dtype=float).tolist())
    obj["frame_history"].append(int(current_frame))
    trim_motion_history(cfg, obj)

    if is_static:
        obj["motion_state"] = "static"
        obj["is_short_term"] = False
    elif is_potential_dynamic and velocity > velocity_threshold:
        obj["motion_state"] = "dynamic"
        obj["is_short_term"] = True
    elif is_potential_dynamic:
        minimum_observations = int(getattr(cfg, "static_confirmation_observations", 1))
        obj["motion_state"] = "static" if len(obj["frame_history"]) >= minimum_observations else "unknown"
        obj["is_short_term"] = False
    else:
        obj["motion_state"] = obj.get("motion_state", "unknown")
        obj["is_short_term"] = bool(obj.get("is_short_term", False))
    return obj


def should_use_short_term_update(cfg, obj: dict, det: dict, velocity: float) -> bool:
    if not bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        return False
    class_names = get_object_class_names(obj) | get_object_class_names(det)
    if is_static_class(cfg, class_names):
        return False
    return is_potential_dynamic_class(cfg, class_names) and velocity > float(getattr(cfg, "velocity_threshold", 0.5))


def update_dynamic_object_short_term(cfg, obj: dict, det: dict, velocity: float, current_center: np.ndarray, current_frame: int) -> dict:
    """Update identity, semantics and current geometry without accumulating old/new point clouds."""
    n_obj = int(obj.get("num_detections", 1))
    n_det = int(det.get("num_detections", 1))
    for key, value in det.items():
        if key in {"pcd", "bbox", "clip_ft", "text_ft", "inst_color"}:
            continue
        if key in MOTION_STATE_KEYS:
            continue
        if isinstance(value, list):
            obj.setdefault(key, [])
            obj[key] = obj[key] + value
        elif isinstance(value, (int, float, np.integer, np.floating)) and key == "num_detections":
            obj[key] = n_obj + n_det
        elif key not in obj:
            obj[key] = copy.deepcopy(value)

    obj["pcd"] = copy.deepcopy(det["pcd"])
    obj["bbox"] = copy.deepcopy(det["bbox"])
    obj["bbox"].color = [1, 0.6, 0]

    obj["clip_ft"] = (to_tensor(obj["clip_ft"]) * n_obj + to_tensor(det["clip_ft"]) * n_det) / max(n_obj + n_det, 1)
    obj["clip_ft"] = F.normalize(obj["clip_ft"], dim=0)
    obj["text_ft"] = (to_tensor(obj["text_ft"]) * n_obj + to_tensor(det["text_ft"]) * n_det) / max(n_obj + n_det, 1)
    obj["text_ft"] = F.normalize(obj["text_ft"], dim=0)
    update_motion_metadata_after_match(cfg, obj, det, velocity, current_center, current_frame)
    obj["motion_state"] = "dynamic"
    obj["is_short_term"] = True
    return obj


def is_long_term_mergeable(cfg, obj: dict) -> bool:
    if not bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        return True
    ensure_motion_metadata(cfg, obj)
    return obj.get("motion_state") != "dynamic" and not bool(obj.get("is_short_term", False))

def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in list(obj1.keys()):
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
            if k in MOTION_STATE_KEYS:
                continue
            if k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            elif k not in obj2:
                continue
            elif isinstance(obj1[k], list):
                obj1[k] += obj2[k]
            elif isinstance(obj1[k], bool):
                obj1[k] = bool(obj1[k])
            elif isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif isinstance(obj1[k], float):
                obj1[k] = float(obj1[k])
            elif isinstance(obj1[k], str):
                obj1[k] = obj1[k]
            elif isinstance(obj1[k], dict):
                merged = copy.deepcopy(obj1[k])
                if isinstance(obj2[k], dict):
                    merged.update(obj2[k])
                obj1[k] = merged
            else:
                obj1[k] = obj1[k]
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)

    if bool(getattr(cfg, "enable_dynamic_object_handling", True)):
        velocity, current_center, current_frame = estimate_observation_velocity(cfg, obj1, obj2)
        update_motion_metadata_after_match(cfg, obj1, obj2, velocity, current_center, current_frame)
    
    return obj1

def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, nearest_indices = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < cfg.downsample_voxel_size ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, nearest_indices = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < cfg.downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        if not is_long_term_mergeable(cfg, objects[i]) or not is_long_term_mergeable(cfg, objects[j]):
            continue
        visual_sim = F.cosine_similarity(
            to_tensor(objects[i]['clip_ft']),
            to_tensor(objects[j]['clip_ft']),
            dim=0
        )
        text_sim = F.cosine_similarity(
            to_tensor(objects[i]['text_ft']),
            to_tensor(objects[j]['text_ft']),
            dim=0
        )
        if ratio > cfg.merge_overlap_thresh:
            if visual_sim > cfg.merge_visual_sim_thresh and \
                text_sim > cfg.merge_text_sim_thresh:
                if kept_objects[j]:
                    # Then merge object i into object j
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects

def denoise_objects(cfg, objects: MapObjectList):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def filter_objects(cfg, objects: MapObjectList):
    # Remove the object that has very few points or viewed too few times
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects

def merge_objects(cfg, objects: MapObjectList):
    if cfg.merge_overlap_thresh > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))
    
    return objects

def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES = ["wall", "floor", "ceiling"],
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs
    
    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        
        # SKip masks that are too small
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        
        # Skip the BG classes
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue
        
        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                # print(f"Skipping {class_name} with area {bbox_area} > {cfg.max_bbox_area_ratio} * {image_area}")
                continue
            
        # Skip masks with low confidence
        if gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < cfg.mask_conf_threshold:
                continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k in {"classes", "frame_provenance"}: # 帧级字段
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs

def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs

def gobs_to_detection_list(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        
        from conceptgraph.completion.depth import depth_support
        robust_depth = getattr(cfg, "depth_policy", "legacy") == "largest_supported_cluster"
        supported_mask, depth_evidence = depth_support(depth_array, mask, robust=robust_depth)
        # 保持原配置的深度范围，只在显式启用的新方法使用质量筛选。
        if robust_depth:
            mask = supported_mask
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None
        )
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        global_object_pcd = process_pcd(global_object_pcd, cfg)
        
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            'depth_evidence': [depth_evidence],
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }
        initialize_motion_metadata(cfg, detected_object, frame_idx=idx)
        
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list

def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    '''
    Transform the detection list by the given transform
    
    Args:
        detection_list: DetectionList
        transform: 4x4 torch.Tensor
        
    Returns:
        transformed_detection_list: DetectionList
    '''
    transform = to_numpy(transform)
    
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    
    for i in range(len(detection_list)):
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)
    
    return detection_list
