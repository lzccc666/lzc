"""
Build a scene graph from the segment-based map and captions from a VLM.
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap
from conceptgraph.utils.general_utils import prjson

import cv2
import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging
from rich.console import Console

# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()


def _load_vehicle_classes() -> list[str]:
    classes_path = Path(__file__).resolve().parents[1] / "vehicle_classes.txt"
    if not classes_path.exists():
        return ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "traffic sign", "traffic light", "road", "sidewalk", "building", "tree", "fence", "pole"]
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prefer_conda_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(lib_dir)] + [item for item in current.split(":") if item and item != str(lib_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
    libstdcxx = lib_dir / "libstdc++.so.6"
    if libstdcxx.exists():
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)


_prefer_conda_runtime_libs()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass
class ProgramArgs:
    mode: Literal[
        "extract-node-captions",
        "refine-node-captions",
        "build-scenegraph",
        "generate-scenegraph-json",
        "annotate-scenegraph",
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Device to use
    device: str = "cuda:0"

    # Local VLM checkpoint used for captioning and caption refinement
    qwen_vl_ckpt: str = os.getenv(
        "QWEN_VL_CKPT_PATH",
        str(Path(__file__).resolve().parents[2] / "checkpoint" / "Qwen-VL-Chat-Int4"),
    )

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "none"

    # Strong outdoor scenegraph edges: keep only close object pairs by default.
    strong_edge_max_distance_m: float = 8.0
    strong_edge_relations: List[str] = None

def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")



def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """ Draw a red outline around the object i nan image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # Optionally, add padding around the object by dilating the drawn contours
    kernel = np.ones((5, 5), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    
    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        raise ValueError("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)
    
    return image_crop, mask_crop

def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """ This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""
    
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()



def extract_node_captions(args):
    from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # Scene map is in CFSLAM format
    # keys: 'image_idx', 'mask_idx', 'color_path', 'class_id', 'num_detections',
    # 'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'clip_ft',
    # 'text_ft', 'pcd_np', 'bbox_np', 'pcd_color_np'

    # Imports to help with feature extraction
    # from extract_mask_level_features import (
    #     crop_bbox_from_img,
    #     get_model_and_preprocessor,
    #     preprocess_and_encode_pil_image,
    # )

    # rich console for pretty printing
    console = Console()

    qwen_model_path = args.qwen_vl_ckpt or os.getenv("QWEN_VL_CKPT_PATH")
    chat = QwenVLChat(qwen_model_path, args.device)
    print("Qwen-VL chat initialized...")
    query = "Describe the central object in the image."
    # query = "Describe the object in the image that is outlined in red."

    # Directories to save features and captions
    savedir_captions = Path(args.cachedir) / "cfslam_captions_qwen"
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_qwen_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []
    existing_caption_path = Path(args.cachedir) / "cfslam_qwen_captions.json"
    if existing_caption_path.exists():
        with open(existing_caption_path, "r", encoding="utf-8") as f:
            caption_dict_list = json.load(f)
    caption_index = {int(item["id"]): item for item in caption_dict_list if isinstance(item, dict) and "id" in item}

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
        if idx_obj in caption_index and caption_index[idx_obj].get("captions"):
            continue
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        captions = []
        low_confidences = []
        
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []  # New list for masks
        if len(idx_most_conf) < 1:
            continue 
        idx_most_conf = idx_most_conf[:args.max_detections_per_object]

        for idx_det in tqdm(idx_most_conf):
            # image = Image.open(correct_path).convert("RGB")
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            class_id = obj["class_id"][idx_det]
            # Retrieve and crop mask
            mask = obj["mask"][idx_det]
            
            padding = 10
            x1, y1, x2, y2 = xyxy
            # image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop  # No modification

            _w, _h = image_crop.size
            if _w * _h < 70 * 70:
                # captions.append("small object")
                print("small object. Skipping LLaVA captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            # image_tensor = chat.image_processor.preprocess(image_crop, return_tensors="pt")["pixel_values"][0]
            chat.reset()
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat(query=query, images=[image_crop_modified])
            console.print("[bold green]Qwen-VL:[/bold green] " + outputs)
            captions.append(outputs)
        
            # print(f"Line 274, obj['mask'][idx_det].shape: {obj['mask'][idx_det].shape}")
            # print(f"Line 276, image.size: {image.size}")
            
            # For the LLava debug folder
            conf_value = conf[idx_det]
            image_list.append(image_crop)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask

        record = {
            "id": idx_obj,
            "captions": captions,
            "low_confidences": low_confidences,
        }
        caption_index[idx_obj] = record
        caption_dict_list = [caption_index[key] for key in sorted(caption_index)]

        # Save per-object captions for inspection
        with open(savedir_captions / f"{idx_obj}.json", "w", encoding="utf-8") as f:
            json.dump(caption_dict_list[-1], f, indent=2, ensure_ascii=False)

        # Again for the VLM debug folder
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)

    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    for item in caption_dict_list:
        item["captions"] = [caption.replace("The central object in the image is ", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "cfslam_qwen_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def parse_or_fallback_object_response(response_text: str, captions: list[str]) -> dict:
    valid_tags = _load_vehicle_classes()
    text = str(response_text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and {"summary", "possible_tags", "object_tag"} <= set(parsed):
            parsed["possible_tags"] = valid_tags
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict) and {"summary", "possible_tags", "object_tag"} <= set(parsed):
                parsed["possible_tags"] = valid_tags
                return parsed
        except json.JSONDecodeError:
            pass

    joined = " ".join(captions).lower()
    candidate_tags = {
        "motorcycle": ["motorcycle", "motor bike", "bike", "摩托车"],
        "bicycle": ["bicycle", "bike", "自行车"],
        "pedestrian": ["pedestrian", "person", "human", "行人", "人"],
        "traffic sign": ["traffic sign", "road sign", "bus stop sign", "sign", "交通标志"],
        "traffic light": ["traffic light", "signal light", "红绿灯"],
        "truck": ["truck", "lorry", "货车", "卡车"],
        "bus": ["bus", "公交", "巴士"],
        "car": ["car", "sedan", "suv", "van", "auto", "vehicle", "汽车", "车辆"],
        "road": ["road", "street", "道路"],
        "sidewalk": ["sidewalk", "人行道"],
        "building": ["building", "house", "建筑"],
        "tree": ["tree", "树"],
        "fence": ["fence", "围栏"],
        "pole": ["pole", "杆"],
    }
    best_tag = "invalid"
    best_count = 0
    for tag, words in candidate_tags.items():
        count = sum(joined.count(word) for word in words)
        if count > best_count:
            best_count = count
            best_tag = tag
    if best_count == 0:
        best_tag = "invalid"
    return {
        "summary": f"{best_tag} observed in nuScenes crops",
        "possible_tags": valid_tags,
        "object_tag": best_tag,
    }


def refine_node_captions(args):
    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.scenegraph.GPTPrompt import GPTPrompt
    from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

    # Load the captions for each segment
    caption_file = Path(args.cachedir) / "cfslam_qwen_captions.json"
    captions = None
    with open(caption_file, "r") as f:
        captions = json.load(f)
    # loaddir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    # captions = []
    # for idx_obj in range(len(os.listdir(loaddir_captions))):
    #     with open(loaddir_captions / f"{idx_obj}.pkl", "rb") as f:
    #         captions.append(pkl.load(f))

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # load the prompt
    gpt_messages = GPTPrompt().get_json()

    TIMEOUT = 25  # Timeout in seconds

    responses_savedir = Path(args.cachedir) / "cfslam_qwen_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    chat = QwenVLChat(args.qwen_vl_ckpt, args.device)
    print("Qwen-VL chat initialized for caption refinement...")

    responses = []
    unsucessful_responses = 0

    # loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]) == 0:
            continue
        
        # Prepare the object prompt 
        _dict = {}
        _caption = captions[_i]
        _bbox = scene_map[_i]["bbox"]
        # _bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(scene_map[_i]["bbox"]))
        _dict["id"] = _caption["id"]
        # _dict["bbox_extent"] = np.round(_bbox.extent, 1).tolist()
        # _dict["bbox_center"] = np.round(_bbox.center, 1).tolist()
        _dict["captions"] = _caption["captions"]
        # _dict["low_confidences"] = _caption["low_confidences"]
        # Convert to printable string
        
        # Make and format the full prompt
        preds = json.dumps(_dict, indent=0)

        start_time = time.time()
        prompt = (
            gpt_messages[0]["content"]
            + "\n\nReturn only valid JSON with keys summary, possible_tags, object_tag. "
            + "Do not include markdown, tables, or explanation.\n\n"
            + preds
        )
        try:
            response_text = chat(query=prompt, images=None)
        except Exception as exc:
            print(f"Qwen-VL failed: {exc}")
            _dict["response"] = "FAIL"
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            continue

        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            continue
        
        # count unsucessful responses
        if "invalid" in response_text.strip("\n"):
            unsucessful_responses += 1
            
        # print output
        prjson([{"role": "user", "content": preds}])
        print(response_text)
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        _dict["response"] = parse_or_fallback_object_response(response_text, _caption["captions"])
        
        # save the response
        responses.append(json.dumps(_dict))
        save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
        # responses.append(chat_completion["choices"][0]["message"]["content"].strip("\n"))

    # tags = []
    # for response in responses:
    #     try:
    #         parsed = json.loads(response)
    #         tags.append(parsed["object_tag"])
    #     except:
    #         tags.append("FAIL")

    # Save the responses to a text file
    # with open(Path(args.cachedir) / "gpt-3.5-turbo_responses.txt", "w") as f:
    #     for response in responses:
    #         f.write(response + "\n")
    with open(Path(args.cachedir) / "cfslam_qwen_responses.pkl", "wb") as f:
        pkl.dump(responses, f)


def extract_object_tag_from_json_str(json_str):
    start_str_found = False
    is_object_tag = False
    object_tag_complete = False
    object_tag = ""
    r = json_str.strip().split()
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # Searching for open parenthesis of JSON
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # Start string found. Now skip everything until the object_tag field
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # object_tag field found. Read it
        if is_object_tag and not object_tag_complete:
            if _r == '"':
                continue
            else:
                if _r.strip() in [",", "}"]:
                    break
                object_tag += f" {_r}"
                continue
    return object_tag


def build_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix
    from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

    chat = QwenVLChat(args.qwen_vl_ckpt, args.device)
    print("Qwen-VL chat initialized for scenegraph relations...")

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    response_dir = Path(args.cachedir) / "cfslam_qwen_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in range(len(scene_map)):
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                if isinstance(_d["response"], str):
                    _d["response"] = json.loads(_d["response"])
                elif not isinstance(_d["response"], dict):
                    raise TypeError("Unsupported response type")
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'Qwen json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            except TypeError:
                _d["response"] = {
                    'summary': f'Qwen json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            if not isinstance(_d["response"], dict) or "object_tag" not in _d["response"]:
                _d["response"] = {
                    'summary': f'Qwen json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # # Load the responses from the json file
    # responses = None
    # # Load json file into a list
    # with open(Path(args.cachedir) / "cfslam_qwen_responses.pkl", "rb") as f:
    #     responses = pkl.load(f)
    # object_tags = []
    # for response in responses:
    #     # _tag = extract_object_tag_from_json_str(response)
    #     # _tag = _tag.lower().replace('"', "").strip()
    #     _d = json.loads(json.loads(response)["responses"])
    #     object_tags.append(_d["object_tag"])

    # Remove segments that correspond to "invalid" tags
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set(indices_to_remove)
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)
    # combine with also_indices_to_remove and sort the list
    indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    # Filtering responses based on segment_ids_to_retain
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]

    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp['response']['object_tag'] for resp in responses]


    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    del pruned_scene_map
    # del pruned_object_tags
    gc.collect()
    num_segments = len(scene_map)

    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]
        # scene_map[i]["object_tag"] = object_tags[i]

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 25  # timeout in seconds

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx2],
                        },
                    }
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    # Default prompt
                    DEFAULT_PROMPT = """
                    The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
                    string (and nothing else), with two keys: "object_relation", and "reason".

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    1. bbox_extent: the 3D bounding box extents of the object
                    2. bbox_center: the 3D bounding box center of the object
                    3. object_tag: an extremely brief description of the object

                    Produce an "object_relation" field that best describes the outdoor spatial relationship between the two objects.
                    The field must be one of the following (verbatim):
                    1. "near"
                    2. "left_of"
                    3. "right_of"
                    4. "in_front_of"
                    5. "behind"
                    6. "none of these"

                    Before producing the "object_relation" field, produce a "reason" field that explains why
                    the chosen "object_relation" field is the best.
                    """

                    start_time = time.time()
                    response_text = chat(query=DEFAULT_PROMPT + "\n\n" + input_json_str, images=None)
                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    if elapsed_time > TIMEOUT:
                        print("Timed out exceeded!")
                        output_dict["object_relation"] = _fallback_object_relation(input_dict)
                        output_dict["reason"] = "fallback_after_timeout"
                    else:
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(response_text)
                            # If the output is a valid JSON, then add it to the output dictionary
                            relation = str(chat_output_json.get("object_relation", "none of these")).strip().lower()
                            if relation == "fail":
                                relation = _fallback_object_relation(input_dict)
                                reason = "fallback_after_parse_failure"
                            else:
                                reason = chat_output_json.get("reason", "")
                            output_dict["object_relation"] = relation
                            output_dict["reason"] = reason
                        except:
                            output_dict["object_relation"] = _fallback_object_relation(input_dict)
                            output_dict["reason"] = "fallback_after_parse_failure"
                    relations.append(output_dict)

                    # print(chat_completion["choices"][0]["message"]["content"])

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            relation = str(relations[_idx]["object_relation"]).strip().lower()
            if _is_strong_scenegraph_edge(
                relation,
                scene_map[segmentidx1],
                scene_map[segmentidx2],
                args.strong_edge_max_distance_m,
                args.strong_edge_relations,
            ):
                scenegraph_edges.append((segmentidx1, segmentidx2, relation))
            _idx += 1
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)
    edge_metadata = []
    for src, dst, relation in scenegraph_edges:
        dynamic = _is_dynamic_scenegraph_node(scene_map[src]) or _is_dynamic_scenegraph_node(scene_map[dst])
        edge_metadata.append(
            {
                "source": int(src),
                "target": int(dst),
                "relation": str(relation),
                "relation_type": "temporal" if dynamic else "stable",
                "dynamic_relation": bool(dynamic),
            }
        )
    with open(Path(args.cachedir) / "cfslam_scenegraph_edge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(edge_metadata, f, indent=2, ensure_ascii=False)


def generate_scenegraph_json(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    

    # Generate the JSON file summarizing the scene, if it doesn't exist already
    # or if the --recopmute_scenegraph_json flag is set
    scene_desc = []
    print("Generating scene graph JSON file...")
    response_dir = Path(args.cachedir) / "cfslam_qwen_responses"

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        loaded_data = pkl.load(f)
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        else:
            scene_map.load_serializable(loaded_data)
    print(f"Loaded scene map with {len(scene_map)} objects")

    for i, segment in enumerate(scene_map):
        bbox_center = _object_last_observation_bbox_center(segment)
        if bbox_center is None:
            bbox_center = np.round(segment['bbox'].center, 1).tolist()
        caption_dict = segment["caption_dict"]
        response_file = response_dir / f"{caption_dict['id']}.json"
        if response_file.exists():
            try:
                with open(response_file, "r", encoding="utf-8") as f:
                    response_wrapper = json.load(f)
                if isinstance(response_wrapper, dict) and "response" in response_wrapper:
                    caption_dict = dict(caption_dict)
                    caption_dict["response"] = response_wrapper["response"]
            except Exception:
                pass
        raw_captions = caption_dict.get("captions", [])
        response = caption_dict.get("response", {})
        response_summary = response.get("summary", "")
        response_possible_tags = response.get("possible_tags", [])
        response_object_tag = response.get("object_tag", "")
        attributes = _extract_caption_attributes(raw_captions, response_summary, response_object_tag)
        display_caption = _build_display_caption(response_object_tag, raw_captions, attributes, response_summary)
        _d = {
            "id": caption_dict["id"],
            "bbox_extent": np.round(segment['bbox'].extent, 1).tolist(),
            "bbox_center": bbox_center,
            "possible_tags": response_possible_tags,
            "object_tag": response_object_tag,
            "caption": display_caption,
            "raw_captions": raw_captions,
            "attributes": attributes,
            "motion_state": str(segment.get("motion_state", "unknown")),
            "is_potential_dynamic": bool(segment.get("is_potential_dynamic", False)),
            "velocity": float(segment.get("velocity", 0.0) or 0.0),
            "is_short_term": bool(segment.get("is_short_term", False)),
            "last_seen_frame": int(segment.get("last_seen_frame", -1)),
            "center_history": _json_safe_history(segment.get("center_history", [])),
            "frame_history": [int(v) for v in segment.get("frame_history", [])],
            "metadata": {
                "motion_state": str(segment.get("motion_state", "unknown")),
                "is_potential_dynamic": bool(segment.get("is_potential_dynamic", False)),
                "velocity": float(segment.get("velocity", 0.0) or 0.0),
                "is_short_term": bool(segment.get("is_short_term", False)),
                "last_seen_frame": int(segment.get("last_seen_frame", -1)),
            },
        }
        scene_desc.append(_d)
    with open(Path(args.cachedir) / "scene_graph.json", "w") as f:
        json.dump(scene_desc, f, indent=4)


def _extract_caption_attributes(raw_captions: list[str], summary: str, object_tag: str) -> dict:
    text = " ".join([summary, object_tag] + [str(item) for item in raw_captions]).lower()
    colors = ["white", "black", "blue", "red", "gray", "grey", "silver", "green", "yellow", "orange"]
    brands = [
        "honda", "toyota", "nissan", "bmw", "ford", "chevrolet", "mazda", "mitsubishi", "subaru", "volvo",
        "mini", "rogue", "qashqai", "prius", "fit", "mirage", "orlando"
    ]
    body_types = ["suv", "sedan", "hatchback", "van", "truck", "bus", "motorcycle", "bike", "car"]
    found_colors = [c for c in colors if c in text]
    found_brands = [b for b in brands if b in text]
    found_body_types = [t for t in body_types if t in text]
    return {
        "colors": sorted(set(found_colors)),
        "brands": sorted(set(found_brands)),
        "body_types": sorted(set(found_body_types)),
        "raw_text": text,
    }


def _build_display_caption(object_tag: str, raw_captions: list[str], attributes: dict, fallback_summary: str) -> str:
    colors = attributes.get("colors", []) if isinstance(attributes, dict) else []
    brands = attributes.get("brands", []) if isinstance(attributes, dict) else []
    body_types = attributes.get("body_types", []) if isinstance(attributes, dict) else []
    parts = []
    if colors:
        parts.append("/".join(colors[:2]))
    if brands:
        parts.append("/".join(brands[:2]))
    body_without_tag = [item for item in body_types if item != object_tag]
    if body_without_tag:
        parts.append("/".join(body_without_tag[:2]))
    if object_tag:
        parts.append(str(object_tag))
    if parts:
        return " ".join(parts)

    for caption in raw_captions:
        text = str(caption).strip()
        if text and "observed in nuscenes" not in text.lower():
            return text[:120]
    return str(fallback_summary or object_tag or "object")


def _fallback_object_relation(input_dict: dict) -> str:
    tag1 = str(input_dict["object1"]["object_tag"]).lower()
    tag2 = str(input_dict["object2"]["object_tag"]).lower()
    c1 = np.asarray(input_dict["object1"]["bbox_center"], dtype=float)
    c2 = np.asarray(input_dict["object2"]["bbox_center"], dtype=float)
    delta = c1 - c2
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    if np.hypot(dx, dy) < 5.0:
        return "near"
    if abs(dx) >= abs(dy):
        return "in_front_of" if dx > 0 else "behind"
    return "left_of" if dy > 0 else "right_of"


def _json_safe_history(history) -> list[list[float]]:
    safe = []
    for item in history or []:
        try:
            arr = np.asarray(item, dtype=float).reshape(-1)
            safe.append([float(v) for v in arr[:3]])
        except Exception:
            continue
    return safe


def _is_dynamic_scenegraph_node(obj: dict) -> bool:
    return str(obj.get("motion_state", "unknown")).lower() == "dynamic" or bool(obj.get("is_short_term", False))


def _object_last_observation_bbox_center(segment) -> list[float] | None:
    try:
        image_indices = segment.get("image_idx", [])
        xyxy_list = segment.get("xyxy", [])
        if not image_indices or not xyxy_list:
            return None
        last_idx = int(np.argmax(np.asarray(image_indices, dtype=int)))
        xyxy = np.asarray(xyxy_list[last_idx], dtype=float)
        center_xy = [(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0]
        bbox = segment["bbox"]
        center = np.asarray(bbox.center, dtype=float)
        center[0] = center[0]
        center[1] = center[1]
        return np.round(center, 1).tolist()
    except Exception:
        return None


def _is_strong_scenegraph_edge(
    relation: str,
    obj1: dict,
    obj2: dict,
    max_distance_m: float,
    allowed_relations: List[str] | None,
) -> bool:
    relation = str(relation).strip().lower()
    if relation in {"none of these", "fail", "invalid", ""}:
        return False

    allowed = set(allowed_relations or ["near"])
    if relation not in allowed:
        return False

    c1 = np.asarray(obj1["bbox"].center, dtype=float)
    c2 = np.asarray(obj2["bbox"].center, dtype=float)
    planar_distance = float(np.linalg.norm(c1[:2] - c2[:2]))
    return planar_distance <= float(max_distance_m)


def display_images(image_list):
    num_images = len(image_list)
    cols = 2  # Number of columns for the subplots (you can change this as needed)
    rows = (num_images + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = image_list[i]
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def annotate_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        loaded_data = pkl.load(f)
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        else:
            scene_map.load_serializable(loaded_data)

    annot_inds = None
    if args.annot_inds is not None:
        annot_inds = args.annot_inds
    # If annot_inds is not None, we also need to load the annotation json file and only
    # annotate the objects that are specified in the annot_inds list
    annots = []
    if annot_inds is not None:
        annots = json.load(open(Path(args.cachedir) / "annotated_scenegraph.json", "r"))

    if annot_inds is None:
        annot_inds = list(range(len(scene_map)))

    for idx in annot_inds:
        print(f"Object {idx} out of {len(annot_inds)}...")
        obj = scene_map[idx]

        prev_annot = None
        if len(annots) >= idx + 1:
            prev_annot = annots[idx]

        annot = {}
        annot["id"] = idx

        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        print(obj.keys())

        imgs = []

        for idx_det in idx_most_conf:
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            mask_crop = crop_image_pil(Image.fromarray(mask), x1, y1, x2, y2, padding=padding)
            mask_crop = np.array(mask_crop)[..., None]
            mask_crop[mask_crop == 0] = 0.05
            image_crop = np.array(image_crop) * mask_crop
            imgs.append(image_crop)
            if len(imgs) >= 5:
                break
            # if idx_det >= 5:
            #     break

        # Display the images
        display_images(imgs)
        plt.close("all")

        # Ask the user to annotate the object
        if prev_annot is not None:
            print("Previous annotation:")
            print(prev_annot)
        annot["object_tags"] = input("Enter object tags (comma-separated): ")
        annot["colors"] = input("Enter colors (comma-separated): ")
        annot["materials"] = input("Enter materials (comma-separated): ")

        if prev_annot is not None:
            annots[idx] = annot
        else:
            annots.append(annot)

        go_on = input("Continue? (y/n): ")
        if go_on == "n":
            break

    # Save the annotations
    with open(Path(args.cachedir) / "annotated_scenegraph.json", "w") as f:
        json.dump(annots, f, indent=4)


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    
    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        build_scenegraph(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    elif args.mode == "annotate-scenegraph":
        annotate_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
