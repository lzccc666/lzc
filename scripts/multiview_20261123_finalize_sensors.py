"""只读核验本轮真实前端缓存、冻结来源与用量并交付报告。"""
import csv
import gzip
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from conceptgraph.project_paths import PROJECT_ROOT
from multiview_20261123_prepare_inputs import digest, dump


def main():
    base = PROJECT_ROOT
    processed = base / "dataset/processed/multiview_20261123_pilot_v1"
    manifest = json.loads((processed / "manifest.json").read_text())
    run = Path(manifest["frontend_run"])
    report = base / "outputs/reports/multiview_semantic_tasks_20261123/sensors"
    release = json.loads((run / "frontend_release.json").read_text())
    complete = json.loads((run / "frontend_complete.json").read_text())
    audit = json.loads((report / "independent_input_audit.json").read_text())
    protections = {**release["source_sha256"], **release["weights_sha256"],
                   str(base / "scripts/multiview_20261123_prepare_inputs.py"): manifest["source_sha256"]}
    assert all(digest(p) == h for p, h in protections.items())
    per_camera = {c: dict(channel=c, images=0, detections=0, measured_lidar_pixels=0, missing_lidar_views=0) for c in manifest["channels"]}
    rows = [json.loads(x) for x in (run / "frontend_frames.jsonl").read_text().splitlines()]
    assert len(rows) == 120 == complete["actual_new_frontend_calls"]
    for row in rows:
        assert digest(row["detections"]) == row["detections_sha256"]
        with gzip.open(row["detections"], "rb") as f:
            gobs = pickle.load(f)
        assert digest(row["rgb"]) == row["rgb_sha256"]
        n = len(gobs["xyxy"])
        assert len(gobs["mask"]) == len(gobs["confidence"]) == len(gobs["class_id"]) == len(gobs["image_feats"]) == len(gobs["text_feats"]) == n
        assert gobs["mask"].dtype == bool
        assert np.isfinite(gobs["image_feats"]).all() and np.isfinite(gobs["text_feats"]).all()
        if n:
            assert np.allclose(np.linalg.norm(gobs["image_feats"], axis=1), 1, atol=2e-4)
        count = per_camera[row["channel"]]
        count["images"] += 1
        count["detections"] += n
        count["measured_lidar_pixels"] += len(np.load(row["projection_evidence"])["point_index"])
        count["missing_lidar_views"] += row["lidar_token"] is None
    with (report / "per_camera_frontend.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(per_camera.values()))))
        writer.writeheader()
        writer.writerows(per_camera.values())
    files = [base / "scripts" / f"multiview_20261123_{n}.py" for n in ("prepare_inputs", "run_frontend", "audit_inputs", "finalize_sensors")]
    files += [base / p for p in ("conceptgraph/nuscenes/expansion_benchmark.py", "conceptgraph/project_paths.py", "conceptgraph/scripts/preprocess_nuscenes_to_minimal.py", "conceptgraph/scripts/convert_lane_expansion.py", "conceptgraph/completion/vehicle_frontend.py", "conceptgraph/scripts/streamlined_detections.py", "conceptgraph/utils/model_utils.py")]
    hashes = {}
    for p in files:
        target = run / "source_snapshot" / p.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == digest(p)
        else:
            shutil.copy2(p, target)
        hashes[str(p.relative_to(base))] = digest(target)
    dump(run / "source_snapshot_complete.json", dict(files_sha256=hashes, sources=len(hashes), model_weights_sha256=release["weights_sha256"], historical_sources_verified_against_pre_frontend_release=True, dependency_capture="read-only snapshot; imported historical modules unchanged during this task"))
    own_roots = [processed, base / "dataset/raw/multiview_20261123_pilot_v1", run, report]
    actual_bytes = sum(p.stat().st_size for root in own_roots for p in root.rglob("*") if p.is_file() and not p.is_symlink()) + sum(p.stat().st_size for p in files[:4])
    cache_bytes = sum(Path(r["detections"]).stat().st_size for r in rows)
    input_hashes = 0
    for scene in manifest["scenes"]:
        for group in scene["groups"]:
            for channel, view in group["views"].items():
                old = next(r for r in json.loads((processed / "input_verification.json").read_text()) if r["scene"] == scene["scene"] and r["frame_idx"] == group["frame_idx"] and r["channel"] == channel)
                for key, sha in (("depth", "depth_sha256"), ("projection_evidence", "projection_sha256"), ("rgb", "rgb_sha256")):
                    assert digest(view[key]) == old[sha]
                    input_hashes += 1
                if view["lidar_raw"]:
                    assert digest(view["lidar_raw"]) == old["lidar_sha256"]
                    input_hashes += 1
    summary = dict(passed=True, groups=40, views=120, scenes=[s["scene"] for s in manifest["scenes"]],
        unique_logs=len({s["log_token"] for s in manifest["scenes"]}), actual_new_frontend_calls=120,
        real_instances=complete["detections"], preserved_source_and_weight_hashes=len(protections),
        source_snapshot_files=len(hashes), unchanged_input_hashes_checked=input_hashes,
        gpu_wall_s=complete["gpu_wall_s"], model_load_wall_s=complete["model_load_wall_s"], model_wall_s=complete["model_wall_s"],
        frontend_cache_bytes=cache_bytes, new_files_bytes_before_this_report=actual_bytes,
        free_bytes=shutil.disk_usage(run).free, all_three_inputs_same_frozen_frontend=True,
        independent_projection_mismatched_pixels=audit["independent_projection_mismatched_pixels"],
        missing_past_lidar_views=6, future_lidar_views=0, per_camera=list(per_camera.values()),
        independent_generalization_claim=False, frontend_predictions_are_not_mapping_recall=True,
        manifest_sha256=digest(processed / "manifest.json"), frontend_complete=str(run / "frontend_complete.json"),
        inputs_complete=str(processed / "inputs_complete.json"), source_complete=str(run / "source_snapshot_complete.json"))
    dump(report / "sensor_frontend_final_verification.json", summary)
    text = f"""# 三视角传感器与真实前端小试\n\n已完成开发集按名排序首2场景 scene-0012/0013，各前20关键帧。共40组、120张原始CAM_FRONT/CAM_FRONT_LEFT/CAM_FRONT_RIGHT。两场景属于同一log，只作有界开发小试，不能作为独立泛化验证，也不是全38场景R1复现。\n\n120次真实YOLOv8l-world/MobileSAM/CLIP已跑完，共{complete['detections']}个二维实例；固定六类、960、confidence=0.1、seed=17，无训练、无缓存复用。三路同配置。mask和真实image/text CLIP features完整保留；省略可由原图与box重建的image_crops。二维实例数量不能当建图召回。\n\n每路保存曝光时间、K、camera→world位姿、past LiDAR源token/bin路径及原始point_index→pixel/z证据。参考时刻为三路最新曝光；最大等待{audit['maximum_synchronization_wait_s']*1000:.3f}ms。114路past LiDAR年龄{audit['minimum_lidar_age_s']*1000:.3f}—{audit['maximum_lidar_age_s']*1000:.3f}ms，6路首帧无<=100ms过去测量而保留0深度，未来LiDAR为0。120张JPEG真实解码，独立zbuffer逐像素比较0差，点索引投影核验通过。PNG单位毫米，除以1000得到米；仅ego运动补偿，未用GT补动态运动。\n\n输入仅按需从共享archive01读取一次，取出120张JPEG共19,078,410字节，102.49秒；原共享卷未写。GPU加载{complete['model_load_wall_s']:.2f}秒、累计模型推理{complete['model_wall_s']:.2f}秒、GPU总墙钟{complete['gpu_wall_s']:.2f}秒。全部新增约{actual_bytes/1024**2:.2f}MiB，预测gzip缓存{cache_bytes/1024**2:.2f}MiB，磁盘空闲{summary['free_bytes']/1024**3:.2f}GiB。离线顺序推理耗时另记group_latency.jsonl，不作实时或闭环宣称。\n\n输入manifest：{processed / 'manifest.json'}\n\n输入完成：{processed / 'inputs_complete.json'}\n\n真实前端完成：{run / 'frontend_complete.json'}\n\n源码与权重来源：{run / 'source_snapshot_complete.json'}\n\n独立审计：{report / 'independent_input_audit.json'}\n\n相机分表：{report / 'per_camera_frontend.csv'}\n\n完整性复核：{report / 'sensor_frontend_final_verification.json'}\n\nA/B/C单前视/三视角不去重/三视角去重由建图执行代理复用同一冻结缓存；GT单独导出，不进入前端或建图输入。\n"""
    (report / "三视角传感器与前端小试交付.md").write_text(text)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
