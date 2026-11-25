"""读取已有感知图，不读官方答案；构建真实逐帧浏览页及深度显示图。"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from export_sealed_public_078 import LABELS
from sealed_resource_guard_074 import ResourceGuard

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent
SOURCE = ROOT/'main_experiments/sealed_public_058'
OUT = ROOT/'visualizations/evidence_browser_082'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    assets = Path(__file__).with_name('viewer_assets_082')
    release = json.loads((SOURCE/'release.json').read_text())
    sources = {str(Path(__file__)): sha(__file__), str(SOURCE/'release.json'): sha(SOURCE/'release.json')}
    write(OUT/'protocol.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        purpose='逐帧真实证据浏览，不读取或修改封存测试答案',
        scenes=30, frames=1196, mapping_conditions=10,
        depth_preview='原始深度毫米转米；有效深度0至65m线性映射到白至深灰，缺测为黑；不补点、不插值、不改变原图',
        geometry='保存的三维中心及世界轴对齐尺寸；不是完整点云、目标真值或车辆轨迹',
        raw_mapping_files_reused=True, original_sensor_images_copied=False,
        CPU_budget_s=600, maximum_new_output_MiB=64,
        actual_new_GPU_calls=0, actual_new_CARLA_calls=0, goal_complete=False))
    begun = time.monotonic()
    guard = ResourceGuard(ROOT, OUT)
    manifest = dict(old_output_prefix=str(OLD), scenes=[],
        conditions=[dict(id=c, label=LABELS[c]) for c in release['conditions']],
        frames=1196, mapping_conditions=10,
        raw_images_are_original=True, arrival_credit=False)
    with (SOURCE/'agent_frames.jsonl').open() as f:
        inputs = list(map(json.loads, f))
    previews = {}
    for number, row in enumerate(inputs):
        if number % 20 == 0:
            guard(begun, 600)
        path = Path(row['depth'])
        expected = row['input_sha256'][str(path)]
        assert sha(path) == expected
        sources[str(path)] = expected
        with Image.open(path) as image:
            depth = np.asarray(image, dtype=np.float32)/row['depth_scale']
        valid = np.isfinite(depth) & (depth > 0)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        preview[valid] = np.rint(32+223*(1-np.clip(depth[valid]/65., 0, 1))).astype(np.uint8)
        output = OUT/'depth_preview'/row['scene']/f"{row['frame_idx']:06d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(output, compress_level=6)
        previews.setdefault(row['scene'], []).append(dict(frame_idx=row['frame_idx'],
            valid_depth_pixels=int(valid.sum()), depth_available=row['depth_available'],
            missing_lidar_reason=row.get('missing_lidar_reason'),
            path=str(output.relative_to(OUT)), sha256=sha(output)))
    for selected in release['scenes']:
        scene = selected['scene']
        mappings = {}
        for condition in release['conditions']:
            folder = SOURCE/'mapping'/condition/scene
            report_path = folder/'report.json'
            report = json.loads(report_path.read_text())
            path = folder/'frames.jsonl.gz'
            assert sha(path) == report['frames_sha256']
            sources[str(path)] = report['frames_sha256']
            sources[str(report_path)] = sha(report_path)
            with gzip.open(path, 'rt') as f:
                frames = list(map(json.loads, f))
            assert len(frames) == selected['frames']
            assert all(frame['gt_in_inference'] is False for frame in frames)
            mappings[condition] = '/'+str(path.relative_to(OLD))
        assert len(previews[scene]) == selected['frames']
        manifest['scenes'].append(dict(id=scene, frames=selected['frames'],
            mapping_urls=mappings, previews=previews[scene]))
    for name in ['index.html', 'viewer.js']:
        sources[str(assets/name)] = sha(assets/name)
        shutil.copyfile(assets/name, OUT/name)
    write(OUT/'manifest.json', manifest)
    assert all(sha(p) == h for p, h in sources.items())
    guard(begun, 600, force=True)
    with (OUT/'使用说明.md').open('x') as f:
        f.write('# 真实逐帧浏览入口\n\n沿用本机12465转发到服务器8877，打开 http://localhost:12465/public_experiment_completion_20261115/visualizations/evidence_browser_082/index.html 。需要HTTP访问，直接双击HTML不能读取压缩帧数据。\n\n')
        f.write('选择场景、方案和帧号，即可看真实RGB、检测与SAM掩膜叠加、稀疏深度显示图、保存的三维对象框及关系。拖动三维区域旋转、滚轮缩放；框由三维中心和世界轴对齐尺寸构成，不是完整点云。关系显示最多200条，完整边保留在原始JSON中。\n\n')
        f.write('深度只为显示做统一灰度映射：越近越亮，无测量为黑，超过65m显示为深灰。不补点、不填洞、不修改原始深度。缺少过去100ms内LiDAR的首帧保留并明确标注；空图不等于整段道路无目标。\n\n')
        f.write('本页不读取官方答案或计算正确率，离线对象图不是车辆执行。KITTI入口和CARLA10回合证据在页面顶部单独链接；不把三个数据来源的完成状态混为一谈。所有原始证据只读复用，没有新GPU调用或新CARLA回合。\n')
    write(OUT/'receipt.json', dict(UTC=datetime.now(timezone.utc).isoformat(), scenes=30,
        frames=1196, mapping_files=300, depth_previews=len(inputs),
        sources_sha256=sources, manifest_sha256=sha(OUT/'manifest.json'),
        source_files_unchanged=True, official_ground_truth_opened=False,
        elapsed_s=time.monotonic()-begun, browser_QA_pending=True,
        actual_new_GPU_calls=0, actual_new_CARLA_calls=0, new_arrival_credit=0, goal_complete=False))
    print('30 scenes / 1196 frame inputs / 300 saved mapping files indexed; browser QA pending', flush=True)


if __name__ == '__main__':
    main()
