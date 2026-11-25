"""确认集已看后，只读分解L1三条duplicate机会；不改预测、不删节点。"""
import gzip
import json
from pathlib import Path
import pickle

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.project_paths import PROJECT_ROOT
from multiview_identity_repair_20261123_independent_scores import in_view, score_frame
from multiview_identity_repair_20261123_prepare_validation import digest, dump


def main():
    folder=PROJECT_ROOT/"dataset/processed/multiview_identity_repair_20261123_validation_v1"
    manifest=json.loads((folder/"manifest.json").read_text())
    truth={(r["scene"],r["frame_idx"]):r["objects"] for r in map(json.loads,(folder/"multiview_identity_repair_20261123_evaluation_ground_truth.jsonl").read_text().splitlines())}
    run=PROJECT_ROOT/"outputs/results/nuscenes/validation/iterations/mvrepair_L1_crosslog_validation_20261123_v1"
    cases=[]
    for scene in manifest["scenes"]:
        with gzip.open(run/scene["scene"]/"mvrepair_lifecycle_20261123_predictions.jsonl.gz","rt") as f:
            rows=list(map(json.loads,f))

        def source(pred,frame_idx):
            for i in range(frame_idx,-1,-1):
                audit=rows[i]["audit"]["L1_lifecycle"]
                raw=audit["output_representatives"].get(pred["node_id"],pred["node_id"])
                ids={pred["node_id"],raw}
                keys=[]
                for w in audit["registry"]["witnesses"]:
                    for suffix in ("a","b"):
                        if w.get("track_"+suffix) in ids and w.get("source_key_"+suffix):
                            key=w["source_key_"+suffix]
                            if key not in keys:keys.append(key)
                if keys:
                    assert len(keys)==1,"多来源需另行核验，不猜一个来源"
                    camera,timestamp,index=keys[0]
                    view=scene["groups"][i]["views"][camera]
                    assert timestamp==view["camera_timestamp_s"]
                    with gzip.open(view["detections"],"rb") as f:gobs=pickle.load(f)
                    assert gobs["classes"][gobs["class_id"][index]]==pred["category"]
                    mask=gobs["mask"][index]
                    evidence=np.load(view["projection_evidence"])
                    selected=mask[evidence["pixel_y"],evidence["pixel_x"]]
                    full_points=int(selected.sum())
                    quantiles=(pred.get("depth_evidence") or [{}])[-1].get("depth_quantiles_m")
                    if quantiles:
                        depth=imageio.imread(view["depth"])[evidence["pixel_y"],evidence["pixel_x"]]/1000.
                        selected &= (depth>=quantiles[0]-1e-6)&(depth<=quantiles[-1]+1e-6)
                    ids=evidence["point_index"][selected]
                    points=np.fromfile(view["lidar_raw"],np.float32).reshape(-1,5)[ids,:3]
                    pose=np.asarray(view["lidar_world_pose"])
                    world=points@pose[:3,:3].T+pose[:3,3]
                    return dict(frame_idx=i,camera=camera,camera_timestamp_s=timestamp,detection_index=index,
                        rgb=view["rgb"],depth=view["depth"],projection_evidence=view["projection_evidence"],lidar_raw=view["lidar_raw"],
                        lidar_token=view["lidar_token"],confidence=float(gobs["confidence"][index]),category=pred["category"],
                        RGB_sha256=gobs["frame_provenance"]["rgb_sha256"],full_mask_measured_points=full_points,
                        selected_depth_interval_points=len(ids),xyxy=gobs["xyxy"][index].tolist(),
                        producer_reported_selected_pixels=(pred.get("depth_evidence") or [{}])[-1].get("retained_pixels"),
                        source_key=keys[0],feature=np.asarray(gobs["image_feats"][index],float),mask=mask,point_ids=set(map(int,ids)),world=world)
            return None

        for group,row in zip(scene["groups"],rows):
            pred=[p for p in row["predictions"]["L1_lifecycle"] if any(in_view(p["center"],v) for v in group["views"].values())]
            gt=truth[scene["scene"],group["frame_idx"]]
            _,pairs,gate=score_frame(pred,gt)
            used={p for _,p in pairs}
            for j,item in enumerate(pred):
                if j in used or not gate[:,j].any():continue
                nearby={g for g in range(len(gt)) if gate[g,j]}
                partners=[(g,p) for g,p in pairs if g in nearby]
                assert len(partners)==1,"有多个近邻GT时保留歧义，不强指一个对象"
                gi,pi=partners[0];matched=pred[pi]
                a,b=source(item,group["frame_idx"]),source(matched,group["frame_idx"])
                assert a is not None and b is not None
                cosine=float(a["feature"]@b["feature"]/(np.linalg.norm(a["feature"])*np.linalg.norm(b["feature"])))
                same_frame_camera=a["frame_idx"]==b["frame_idx"] and a["camera"]==b["camera"]
                mask_iou=float((a["mask"]&b["mask"]).sum()/(a["mask"]|b["mask"]).sum()) if same_frame_camera else None
                common=len(a["point_ids"]&b["point_ids"]) if a["lidar_token"]==b["lidar_token"] else None
                small,large=sorted([a["world"],b["world"]],key=len)
                close=int((cKDTree(large).query(small)[0]<=.2).sum()) if len(small) else 0
                ws=[w for w in row["audit"]["L1_lifecycle"]["registry"]["witnesses"] if {w.get("track_a"),w.get("track_b")}=={item["node_id"],matched["node_id"]}]
                registry=row["audit"]["L1_lifecycle"]["registry"]
                delta=np.asarray(item["center"])-matched["center"]
                def clean(x):return {k:v for k,v in x.items() if k not in ("feature","mask","point_ids","world")}
                cases.append(dict(scene=scene["scene"],frame_idx=group["frame_idx"],automatic_duplicate_GT_instance=gt[gi]["instance_token"],
                    GT_category=gt[gi]["category"],duplicate_prediction=item,matched_prediction=matched,
                    duplicate_source=clean(a),matched_source=clean(b),same_current_camera=same_frame_camera,
                    current_or_old=["current" if x.get("prediction_age_s",0)==0 else "past_state" for x in (item,matched)],
                    actual_CLIP_cosine=cosine,raw_mask_IoU_same_image=mask_iou,
                    center_delta_duplicate_minus_matched_m=delta.tolist(),center_XY_distance_m=float(np.linalg.norm(delta[:2])),
                    center_3D_distance_m=float(np.linalg.norm(delta)),
                    same_scan_selected_original_point_ID_intersection=common,selected_world_points_close_0p2m=close,
                    selected_point_smaller_denominator=len(small),source_capture_gap_s=abs(a["camera_timestamp_s"]-b["camera_timestamp_s"]),
                    pair_witnesses=ws,canonical_ids=[registry["canonical_by_track"].get(x["node_id"],x["node_id"]) for x in (item,matched)],
                    direct_alias_exists=registry["aliases"].get(item["node_id"])==matched["node_id"] or registry["aliases"].get(matched["node_id"])==item["node_id"],
                    diagnosis="same-camera duplicate detections cannot enter cross-camera-only witness" if same_frame_camera else "past-state and current observation lack simultaneous witness; source gap exceeds 75ms gate" if abs(a["camera_timestamp_s"]-b["camera_timestamp_s"])>.075 else "requires additional source examination",
                    human_identity_verified=False,not_an_instruction_to_delete=True))
    assert len(cases)==3
    out=dict(passed=True,scope="confirmation data now inspected; subsequent method adjustment/evaluation is debug regression",duplicate_object_frame_opportunities=3,
        unique_scene_count=len({c["scene"] for c in cases}),cases=cases,new_frontend_model_mapping_calls=0,predictions_modified=False,
        limitation="3m nearest class-compatible unmatched output is an automatic duplicate diagnostic, not human identity ground truth. Depth interval source reconstruction is read-only; no online alias or deletion is produced.",source_sha256=digest(__file__))
    report=PROJECT_ROOT/"outputs/reports/multiview_identity_repair_20261123/validation"
    dump(report/"crosslog_L1_remaining_duplicate_diagnosis.json",out)
    lines=["# L1跨log剩余重复：只读来源诊断","","该确认集现已看过，后续据此修改方法再跑须称验证调试/回归。没有改预测、删节点或重跑模型。3条是对象帧重复机会，不是3个独立物体的人审真值。",""]
    for c in cases:
        lines += [f"- {c['scene']} frame{c['frame_idx']}，{c['GT_category']}：{c['duplicate_prediction']['node_id']} 对 {c['matched_prediction']['node_id']}。{c['diagnosis']}。状态{c['current_or_old']}；CLIP={c['actual_CLIP_cosine']:.4f}，同图mask IoU={c['raw_mask_IoU_same_image']}，同扫描点ID交集={c['same_scan_selected_original_point_ID_intersection']}，0.2m近邻支持={c['selected_world_points_close_0p2m']}/{c['selected_point_smaller_denominator']}；曝光差={c['source_capture_gap_s']:.3f}s，直接alias={c['direct_alias_exists']}。"]
    lines += ["","同相机前端重复与跨相机身份统一是不同入口，不能仅放松跨相机见证或全局缩短TTL。旧状态与当前状态也不能只凭邻近、同类、外观相似强删；需要可审计的跨时身份/真实运动证据。上述材料用于算法代理定位缺口，不是自动修复许可或真实身份真值。"]
    (report/"L1跨log剩余重复来源.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(passed=True,cases=3,diagnoses=[c["diagnosis"] for c in cases])),flush=True)


if __name__=="__main__":main()
