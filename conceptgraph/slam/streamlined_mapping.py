'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import gzip
import pickle

# Related third party imports
from PIL import Image
import imageio
import numpy as np
import open3d as o3d
import torch
from tqdm import trange

import hydra
import omegaconf
from omegaconf import DictConfig

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer
from conceptgraph.utils.ious import (
    compute_2d_box_contained_batch
)
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

BG_CLASSES = ["wall", "floor", "ceiling"]

# Disable torch gradient computation
torch.set_grad_enabled(False)


@dataclass(frozen=True)
class MappingFrame:
    frame_idx: int
    image_rgb: np.ndarray
    depth_array: np.ndarray
    cam_K: np.ndarray
    camera_pose: np.ndarray
    detections: dict
    color_path: str
    timestamp_s: float | None = None


@dataclass
class MappingUpdate:
    frame_idx: int
    objects: MapObjectList
    bg_objects: dict | None
    changed_object_ids: list[int]
    skipped: bool
    foreground_detection_count: int
    object_count: int


class StreamlinedMapper:
    """Reusable single-frame object fusion core used by the legacy CLI and stage 5."""

    def __init__(self, cfg, classes, class_colors=None):
        self.cfg = cfg
        self.classes = list(classes)
        self.class_colors = class_colors
        self.objects = MapObjectList(device=cfg.device)
        self.bg_objects = None if cfg.skip_bg else {name: None for name in BG_CLASSES}
        self.last_observations = []
        self._next_object_id = 0
        self._last_timestamp_s = None

    def update(self, frame: MappingFrame) -> MappingUpdate:
        if frame.timestamp_s is not None:
            if (not np.isfinite(frame.timestamp_s) or (self._last_timestamp_s is not None
                    and frame.timestamp_s <= self._last_timestamp_s)):
                raise ValueError("mapping requires strictly increasing finite timestamps")
            self._last_timestamp_s = frame.timestamp_s
        if getattr(self.cfg, "association_policy", "legacy") == "one_to_one":
            ttl = float(getattr(self.cfg, "dynamic_ttl_s", 2.0))
            self.objects = MapObjectList([
                obj for obj in self.objects
                if not (frame.timestamp_s is not None and obj.get("is_short_term", False)
                        and obj.get("timestamp_s") is not None
                        and frame.timestamp_s - obj["timestamp_s"] > ttl)
            ])
        before = [_mapping_object_signature(obj) for obj in self.objects]
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg=self.cfg,
            image=frame.image_rgb,
            depth_array=frame.depth_array,
            cam_K=frame.cam_K,
            idx=int(frame.frame_idx),
            gobs=frame.detections,
            trans_pose=frame.camera_pose,
            class_names=self.classes,
            BG_CLASSES=BG_CLASSES,
            color_path=frame.color_path,
        )
        for det in fg_detection_list:
            if frame.timestamp_s is not None:
                det["timestamp_s"] = float(frame.timestamp_s)
                det["timestamp_history_s"] = [float(frame.timestamp_s)]
            if getattr(self.cfg, "association_policy", "legacy") == "one_to_one":
                det["stable_object_id"] = f"track-{self._next_object_id:08d}"
                self._next_object_id += 1
        self.last_observations = copy.deepcopy(fg_detection_list)

        if len(bg_detection_list) > 0 and self.bg_objects is not None:
            for detected_object in bg_detection_list:
                class_name = detected_object["class_name"][0]
                if self.bg_objects[class_name] is None:
                    self.bg_objects[class_name] = detected_object
                else:
                    self.bg_objects[class_name] = merge_obj2_into_obj1(
                        self.cfg,
                        self.bg_objects[class_name],
                        detected_object,
                        run_dbscan=False,
                    )

        if len(fg_detection_list) == 0:
            return MappingUpdate(
                frame_idx=int(frame.frame_idx),
                objects=self.objects,
                bg_objects=self.bg_objects,
                changed_object_ids=[],
                skipped=True,
                foreground_detection_count=0,
                object_count=len(self.objects),
            )

        contain_numbers = None
        if self.cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch("xyxy", 0)
            contain_numbers = compute_2d_box_contained_batch(
                xyxy,
                self.cfg.contain_area_thresh,
            )
            for index in range(len(fg_detection_list)):
                fg_detection_list[index]["contain_number"] = [contain_numbers[index]]

        if len(self.objects) == 0:
            for detected_object in fg_detection_list:
                self.objects.append(detected_object)
            skipped = True
        else:
            spatial_sim = compute_spatial_similarities(
                self.cfg,
                fg_detection_list,
                self.objects,
            )
            visual_sim = compute_visual_similarities(
                self.cfg,
                fg_detection_list,
                self.objects,
            )
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
            if self.cfg.use_contain_number:
                contain_numbers_objects = torch.Tensor(
                    [obj["contain_number"][0] for obj in self.objects]
                )
                detection_contained = (contain_numbers > 0).unsqueeze(1)
                object_contained = (contain_numbers_objects > 0).unsqueeze(0)
                agg_sim[detection_contained ^ object_contained] -= self.cfg.contain_mismatch_penalty
            agg_sim[agg_sim < self.cfg.sim_threshold] = float("-inf")
            self.objects = merge_detections_to_objects(
                self.cfg,
                fg_detection_list,
                self.objects,
                agg_sim,
            )
            frame_number = int(frame.frame_idx) + 1
            if self.cfg.denoise_interval > 0 and frame_number % self.cfg.denoise_interval == 0:
                self.objects = denoise_objects(self.cfg, self.objects)
            if self.cfg.filter_interval > 0 and frame_number % self.cfg.filter_interval == 0:
                self.objects = filter_objects(self.cfg, self.objects)
            if self.cfg.merge_interval > 0 and frame_number % self.cfg.merge_interval == 0:
                self.objects = merge_objects(self.cfg, self.objects)
            skipped = False

        after = [_mapping_object_signature(obj) for obj in self.objects]
        changed = [
            index
            for index in range(max(len(before), len(after)))
            if index >= len(before) or index >= len(after) or before[index] != after[index]
        ]
        return MappingUpdate(
            frame_idx=int(frame.frame_idx),
            objects=self.objects,
            bg_objects=self.bg_objects,
            changed_object_ids=changed,
            skipped=skipped,
            foreground_detection_count=len(fg_detection_list),
            object_count=len(self.objects),
        )

    def effective_objects(self) -> MapObjectList:
        return MapObjectList(
            [obj for obj in self.objects if obj["num_detections"] >= self.cfg.obj_min_detections]
        )


def _mapping_object_signature(obj) -> tuple:
    bbox = obj.get("bbox")
    center = np.asarray(bbox.get_center(), dtype=float) if bbox is not None else np.zeros(3)
    return (
        tuple(np.round(center, 6).tolist()),
        int(obj.get("num_detections", 0)),
        str(obj.get("motion_state", "unknown")),
        bool(obj.get("is_short_term", False)),
    )

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()


def save_scene_map_for_downstream(cfg, objects: MapObjectList, bg_objects, classes, class_colors) -> Path:
    """Write the map path consumed by scenegraph/navigation stages."""
    exp_suffix = getattr(cfg, "exp_suffix", None)
    if exp_suffix is None:
        exp_suffix = getattr(cfg, "save_suffix", "streamlined")
    exp_dir = cfg.dataset_root / cfg.scene_id / f"exp_{exp_suffix}"
    map_dir = exp_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_dir / "scene_map_cfslam.pkl.gz"
    result = {
        "objects": objects.to_serializable(),
        "bg_objects": None if bg_objects is None else bg_objects.to_serializable(),
        "cfg": cfg,
        "class_names": classes,
        "class_colors": class_colors,
    }
    with gzip.open(map_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved downstream scene map to {map_path}")
    return map_path
    
def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg
    
@hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def main(cfg : DictConfig):
    cfg = process_cfg(cfg)
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()
    
    classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)

    mapper = StreamlinedMapper(cfg, classes, class_colors)
    objects = mapper.objects
    bg_objects = mapper.bg_objects
        
    # For visualization
    if cfg.vis_render:
        view_param = o3d.io.read_pinhole_camera_parameters(cfg.render_camera_path)
            
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
        
    if cfg.save_objects_all_frames:
        save_all_folder = cfg.dataset_root \
            / cfg.scene_id / "objects_all_frames" / f"{cfg.gsa_variant}_{cfg.save_suffix}"
        os.makedirs(save_all_folder, exist_ok=True)

    for idx in trange(len(dataset)):
        # get color image
        color_path = dataset.color_paths[idx]
        image_original_pil = Image.open(color_path)

        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        # image_rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        # Get the RGB image
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"
        
        # Get the depth image
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        # Get the intrinsics matrix
        cam_K = intrinsics.cpu().numpy()[:3, :3]
        
        # load grounded SAM detections
        gobs = None # stands for grounded SAM observations

        color_path = Path(color_path)
        detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
        detections_path = detections_path.with_suffix(".pkl.gz")
        color_path = str(color_path)
        detections_path = str(detections_path)
        
        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)

        
        # depth_image = Image.open(depth_path)
        # depth_array = np.array(depth_image) / dataset.png_depth_scale
        # depth_tensor = torch.from_numpy(depth_array).float().to(cfg['device']).T

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[idx]
        unt_pose = unt_pose.cpu().numpy()
        
        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
            
        # if idx == 71:
        #     fg_detection_list, bg_detection_list = gobs_to_detection_list(
        #         cfg = cfg,
        #         image = image_rgb,
        #         depth_array = depth_array,
        #         cam_K = cam_K,
        #         idx = idx,
        #         gobs = gobs,
        #         trans_pose = adjusted_pose,
        #         class_names = classes,
        #         BG_CLASSES = BG_CLASSES,
        #         color_path = color_path,
        #     )
        #     for det in fg_detection_list:
        #         o3d.visualization.draw_geometries([det['pcd']])
                
        #     exit()
        # else:
        #     continue
        
        mapping_update = mapper.update(
            MappingFrame(
                frame_idx=idx,
                image_rgb=image_rgb,
                depth_array=depth_array,
                cam_K=cam_K,
                camera_pose=adjusted_pose,
                detections=gobs,
                color_path=color_path,
            )
        )
        objects = mapping_update.objects
        bg_objects = mapping_update.bg_objects
        if mapping_update.skipped:
            continue
            
        if cfg.save_objects_all_frames:
            save_all_path = save_all_folder / f"{idx:06d}.pkl.gz"
            objects_to_save = MapObjectList([
                _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])
            
            objects_to_save = prepare_objects_save_vis(objects_to_save)
            
            if not cfg.skip_bg:
                bg_objects_to_save = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
                bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
            else:
                bg_objects_to_save = None
            
            result = {
                "camera_pose": adjusted_pose,
                "objects": objects_to_save,
                "bg_objects": bg_objects_to_save,
            }
            with gzip.open(save_all_path, 'wb') as f:
                pickle.dump(result, f)
        
        if cfg.vis_render:
            objects_vis = MapObjectList([
                copy.deepcopy(_) for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])
            
            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(class_colors)
            
            rendered_image, vis = obj_renderer.step(
                image = image_original_pil,
                gt_pose = adjusted_pose,
                new_objects = objects_vis,
                paint_new_objects=False,
                return_vis_handle = cfg.debug_render,
            )

            if cfg.debug_render:
                vis.run()
                del vis
            
            # Convert to uint8
            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)
                frames.append(rendered_image)
            
        # print(
        #     f"Finished image {idx} of {len(dataset)}", 
        #     f"Now we have {len(objects)} objects.",
        #     f"Effective objects {len([_ for _ in objects if _['num_detections'] >= cfg.obj_min_detections])}"
        # )
        
    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects)
        
    objects = denoise_objects(cfg, objects)
    
    # Save the full point cloud before post-processing
    if cfg.save_pcd:
        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
            'class_names': classes,
            'class_colors': class_colors,
        }

        pcd_save_path = cfg.dataset_root / \
            cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        # make the directory if it doesn't exist
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        pcd_save_path = str(pcd_save_path)
        
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud to {pcd_save_path}")
    
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    
    # Save again the full point cloud after the post-processing
    if cfg.save_pcd:
        results['objects'] = objects.to_serializable()
        pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud after post-processing to {pcd_save_path}")

    save_scene_map_for_downstream(cfg, objects, bg_objects, classes, class_colors)
        
    if cfg.save_objects_all_frames:
        save_meta_path = save_all_folder / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': classes,
                'class_colors': class_colors,
            }, f)
        
    if cfg.vis_render:
        # Still render a frame after the post-processing
        objects_vis = MapObjectList([
            _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
        ])

        if cfg.class_agnostic:
            objects_vis.color_by_instance()
        else:
            objects_vis.color_by_most_common_classes(class_colors)
        
        rendered_image, vis = obj_renderer.step(
            image = image_original_pil,
            gt_pose = adjusted_pose,
            new_objects = objects_vis,
            paint_new_objects=False,
            return_vis_handle = False,
        )
        
        # Convert to uint8
        rendered_image = (rendered_image * 255).astype(np.uint8)
        frames.append(rendered_image)
        
        # Save frames as a mp4 video
        frames = np.stack(frames)
        video_save_path = (
            cfg.dataset_root
            / cfg.scene_id
            / ("objects_mapping-%s-%s.mp4" % (cfg.gsa_variant, cfg.save_suffix))
        )
        imageio.mimwrite(video_save_path, frames, fps=10)
        print("Save video to %s" % video_save_path)
        
if __name__ == "__main__":
    main()
