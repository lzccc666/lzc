"""只读回查L1退休见证的真实相机/检测/CLIP/原LiDAR点，不重跑建图。"""
import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.project_paths import PROJECT_ROOT
from multiview_identity_repair_20261123_prepare_validation import digest, dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out-id",required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    events=[]
    for scene in manifest["scenes"]:
        files=list((args.run/scene["scene"]).glob("*predictions*.jsonl.gz"))
        assert len(files)==1
        with gzip.open(files[0],"rt") as f:
            rows=[json.loads(t) for t in f]
        for group,row in zip(scene["groups"],rows):
            audit=row["audit"]["L1_lifecycle"]
            for retired in audit["retired_states"]:
                assert retired["last_measured_timestamp_s"]<=group["reference_timestamp_s"]
                witnesses=[w for w in audit["registry"]["witnesses"] if w.get("accepted") and retired["logical_id"] in (w.get("track_a"),w.get("track_b"))]
                assert witnesses,"退休缺可回查的当前组见证"
                witness=witnesses[0]
                assert witness["reason"]=="current_shared_real_support"
                recovered=[]
                for key in ("source_key_a","source_key_b"):
                    camera,timestamp,index=witness[key]
                    view=group["views"][camera]
                    assert view["camera_timestamp_s"]==timestamp and 0<=group["reference_timestamp_s"]-timestamp<=.075
                    with gzip.open(view["detections"],"rb") as f:
                        gobs=pickle.load(f)
                    assert 0<=index<len(gobs["xyxy"])
                    mask=gobs["mask"][index]
                    projection=np.load(view["projection_evidence"])
                    selected=mask[projection["pixel_y"],projection["pixel_x"]]
                    ids=projection["point_index"][selected]
                    points=np.fromfile(view["lidar_raw"],np.float32).reshape(-1,5)[ids,:3]
                    pose=np.asarray(view["lidar_world_pose"])
                    world=points@pose[:3,:3].T+pose[:3,3]
                    recovered.append(dict(camera=camera,timestamp=timestamp,index=index,world=world,ids=set(map(int,ids)),
                        feature=np.asarray(gobs["image_feats"][index],float),category=gobs["classes"][gobs["class_id"][index]],
                        lidar_token=view["lidar_token"],RGB_sha256=gobs["frame_provenance"]["rgb_sha256"]))
                a,b=recovered
                assert a["camera"]!=b["camera"] and a["category"]==b["category"]
                cosine=float(a["feature"]@b["feature"]/(np.linalg.norm(a["feature"])*np.linalg.norm(b["feature"])))
                assert abs(cosine-witness["feature_cosine"])<1e-5 and cosine>=.65
                assert min(len(a["world"]),len(b["world"]))>=2
                small,large=sorted([a["world"],b["world"]],key=len)
                close=int((cKDTree(large).query(small)[0]<=.2).sum())
                assert close>=2
                identical=len(a["ids"] & b["ids"]) if a["lidar_token"]==b["lidar_token"] else None
                events.append(dict(scene=scene["scene"],frame_idx=group["frame_idx"],retired=retired,witness=witness,
                    original_mask_point_counts=[len(a["world"]),len(b["world"])],source_original_point_ID_intersection=identical,
                    original_mask_world_shared_neighbours_0p2m=close,recomputed_current_CLIP_cosine=cosine,
                    source_keys_match_actual_frame=True,no_future_sensor=True,
                    limitation="Raw-mask original-point support independently exists; producer's selected-cluster witness counts are retained, not relabelled as exact raw-mask counts. Shared support is not human identity ground truth."))
    frozen=PROJECT_ROOT/"outputs/results/nuscenes/validation/iterations/mvrepair_lifecycle_20261123_candidate_L1_v1/mvrepair_lifecycle_20261123_release.json"
    release=json.loads(frozen.read_text())
    assert all(digest(p)==r["sha256"] and digest(r["snapshot"])==r["sha256"] for p,r in release["source_sha256"].items())
    result=dict(passed=True,retirements=len(events),events=events,frozen_sources_verified=len(release["source_sha256"]),
        GT_read=False,new_mapping_calls=0,new_model_calls=0,source_sha256=digest(__file__),manifest_sha256=digest(args.manifest),
        frozen_release_sha256=digest(frozen))
    dump(PROJECT_ROOT/"outputs/reports/multiview_identity_repair_20261123/validation"/(args.out_id+"_L1_causal_witness.json"),result)
    print(json.dumps(dict(passed=True,retirements=len(events),frozen_sources_verified=result["frozen_sources_verified"],GT_read=False)),flush=True)


if __name__=="__main__":
    main()
