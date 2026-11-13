"""用完整P2直接投影与独立欧拉矩阵核验90帧；不导入待核验几何实现。"""

from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import time
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_manifest/kitti_tracking_center90_002"
DEST = DATA / "independent_audit_001"
DEST.mkdir(exist_ok=False)
START = time.monotonic()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    with p.open("x") as h:
        json.dump(x, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write("\n")


def mat(v):
    m = np.eye(4)
    a = np.asarray(v)
    m[: a.shape[0], : a.shape[1]] = a
    return m


try:
    comp = json.loads((DATA / "complete.json").read_text())
    agent = json.loads((DATA / "agent_manifest.json").read_text())
    assert sha(DATA / "agent_manifest.json") == comp["agent_manifest_sha256"]
    assert sha(DATA / "evaluation_only/truth.json") == comp["truth_sha256"]
    old = json.loads(
        (ROOT / "data_manifest/kitti_tracking_center90_001/selection.json").read_text()
    )
    new = json.loads((DATA / "selection.json").read_text())
    assert new["selected"] == old["selected"]
    assert len(agent["frames"]) == 90
    oxts = zipfile.ZipFile(DATA / "official_small_archives/data_tracking_oxts.zip")
    poses = {}
    checks = []
    counter = Counter()
    allhash = {}
    for seq in new["selected"]:
        name = next(
            n
            for n in oxts.namelist()
            if n.startswith("training/") and n.endswith("/" + seq + ".txt")
        )
        rows = np.loadtxt(oxts.read(name).decode().splitlines())
        scale = math.cos(rows[0, 0] * math.pi / 180)
        world = []
        for raw in rows:
            lat, lon, alt, roll, pitch, yaw = raw[:6]
            x = scale * 6378137 * lon * math.pi / 180
            y = scale * 6378137 * math.log(math.tan(math.pi / 4 + lat * math.pi / 360))
            sr, cr, sp, cp, sy, cy = (
                math.sin(roll),
                math.cos(roll),
                math.sin(pitch),
                math.cos(pitch),
                math.sin(yaw),
                math.cos(yaw),
            )
            rotation = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            pose = mat(rotation)
            pose[:3, 3] = [x, y, alt]
            world.append(pose)
        poses[seq] = [np.linalg.solve(world[0], p) for p in world]
    for row in agent["frames"]:
        assert time.monotonic() - START < 600
        for p, h in row["input_sha256"].items():
            assert sha(p) == h
            allhash[p] = h
        assert (
            row["nominal_timestamp_s"] == row["original_frame_id"] * 0.1
            and row["sensor_timestamp_measured"] is False
        )
        seq = row["sequence"]
        idx = row["original_frame_id"]
        width, height = row["image_wh"]
        values = {}
        for line in Path(row["calib"]).read_text().splitlines():
            key, rest = line.split(maxsplit=1)
            values[key.rstrip(":")] = np.fromstring(rest, sep=" ")
        p2 = values["P2"].reshape(3, 4)
        rect = mat(values["R_rect"].reshape(3, 3))
        velo = mat(values["Tr_velo_cam"].reshape(3, 4))
        imu = mat(values["Tr_imu_velo"].reshape(3, 4))
        cloud = np.fromfile(row["lidar"], dtype="<f4").reshape(-1, 4).astype(float)
        cloud[:, 3] = 1
        ref = (rect @ velo @ cloud.T).T
        projected = ref @ p2.T
        valid = (
            np.isfinite(projected).all(1)
            & (projected[:, 2] > 0)
            & (projected[:, 2] <= 65.535)
        )
        px = np.zeros((len(cloud), 2), dtype=int)
        px[valid] = np.floor(projected[valid, :2] / projected[valid, 2, None]).astype(
            int
        )
        valid &= (
            (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
        )
        nearest = np.full(width * height, np.inf)
        np.minimum.at(nearest, px[valid, 1] * width + px[valid, 0], projected[valid, 2])
        occupied = np.isfinite(nearest)
        expected = np.zeros(len(nearest), dtype=np.uint16)
        expected[occupied] = np.rint(nearest[occupied] * 1000).astype(np.uint16)
        depth = np.asarray(Image.open(row["depth"]))
        assert np.array_equal(expected.reshape(height, width), depth)
        shift = np.eye(4)
        shift[:3, 3] = np.linalg.solve(p2[:, :3], p2[:, 3])
        expected_pose = poses[seq][idx] @ np.linalg.inv(shift @ rect @ velo @ imu)
        residual = float(np.max(np.abs(expected_pose - np.asarray(row["pose"]))))
        assert residual < 1e-6
        assert np.max(np.abs(np.asarray(row["intrinsic"]) - p2[:, :3])) < 1e-10
        counter["frames"] += 1
        counter["occupied_depth_pixels"] += int(occupied.sum())
        checks.append(
            dict(
                sequence=seq,
                original_frame_id=idx,
                depth_pixels=int(occupied.sum()),
                world_pose_max_absolute_difference=residual,
                depth_exact_mm=True,
            )
        )
    truth = json.loads((DATA / "evaluation_only/truth.json").read_text())
    for f in truth["frames"]:
        row = next(
            r
            for r in agent["frames"]
            if r["sequence"] == f["sequence"]
            and r["original_frame_id"] == f["original_frame_id"]
        )
        pose = np.asarray(row["pose"])
        p = {}
        for line in Path(row["calib"]).read_text().splitlines():
            key, rest = line.split(maxsplit=1)
            p[key.rstrip(":")] = np.fromstring(rest, sep=" ")
        shift = np.linalg.solve(
            p["P2"].reshape(3, 4)[:, :3], p["P2"].reshape(3, 4)[:, 3]
        )
        for obj in f["objects"]:
            assert (
                obj["evaluation_only"] is True
                and obj["instance_token"] == f"{f['sequence']}:{obj['track_id']}"
            )
            c = np.array(obj["bottom_center_camera0"])
            c[1] -= obj["dimensions_hwl"][0] / 2
            assert np.allclose(c, obj["center_camera0"]) and np.allclose(
                c + shift, obj["center_camera2"]
            )
            assert np.allclose(
                pose[:3, :3] @ (c + shift) + pose[:3, 3], obj["center"], atol=1e-7
            )
            counter["official_label_records"] += 1
    save(DEST / "checks.json", checks)
    save(
        DEST / "report.json",
        dict(
            passed=True,
            counts=dict(counter),
            sequences=3,
            input_sha256=allhash,
            complete_sha256=sha(DATA / "complete.json"),
            audit_source_sha256=sha(__file__),
            selection_same_as_failed_001=True,
            independent_full_P2_depth_exact=True,
            independent_world_pose=True,
            nominal_time_only=True,
            frontends_run=0,
            navigation_arrivals=0,
            goal_complete=False,
            wall_s=time.monotonic() - START,
        ),
    )
    print(json.dumps(dict(passed=True, counts=dict(counter))))
except BaseException as exc:
    save(DEST / "interrupted.json", dict(error=repr(exc), goal_complete=False))
    raise
