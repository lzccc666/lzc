"""被动重放固定L4，保存时序碎片的真实点/关联分数；不读取GT。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from conceptgraph.slam.mvrepair_20261123_combined import CanonicalPhysicalFrontMapper
from scripts.multiview_20261123_run_graph import load_frame


ROOT = PROJECT_ROOT
P = 'mvrepair_20261123_'


def safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class PassiveMapper(CanonicalPhysicalFrontMapper):
    def refine_association_scores(self, frame, detections, spatial_sim, visual_sim, scores):
        self.pair_scores = dict(frame_idx=frame.frame_idx, detections=[dict(
            raw_id=o['stable_object_id'], category=o['class_name'][-1], center=o['bbox'].get_center(),
            points=np.asarray(o['pcd'].points), source_key=self.measurement_sources[o['mask_idx'][-1]]['_observation_key']) for o in detections],
            objects=[dict(raw_id=o['stable_object_id'], category=o['class_name'][-1],
                center=o['bbox'].get_center(), num_detections=o['num_detections'], timestamp_s=o['timestamp_s']) for o in self.objects],
            spatial=spatial_sim.cpu().numpy(), appearance=visual_sim.cpu().numpy(),
            score=scores.cpu().numpy())
        return super().refine_association_scores(frame, detections, spatial_sim, visual_sim, scores)


def main():
    manifest_path = ROOT/'dataset/processed/multiview_identity_repair_20261123_validation_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads((ROOT/'outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1/multiview_20261123_release.json').read_text())
    run = experiment_directory('nuscenes', 'validation', P+'temporal_diagnosis_v2', create=True)
    shutil.copy2(__file__, run/Path(__file__).name)
    mapper = PassiveMapper(OmegaConf.create(baseline['config']), CLASSES)
    scene = next(s for s in manifest['scenes'] if s['scene']=='scene-0035')
    reference = ROOT/'outputs/results/nuscenes/validation/iterations/mvrepair_L4_crosslog_regression_20261123_v1/scene-0035'/ (P+'predictions.jsonl.gz')
    old = list(map(json.loads, gzip.open(reference, 'rt')))
    records = []
    for group in scene['groups']:
        if group['frame_idx'] > 11:
            break
        np.random.seed(17+group['frame_idx'])
        frames = [(c, load_frame(group, c)) for c in group['views']]
        output = mapper.update_group(frames, {c:group['views'][c]['camera_timestamp_s'] for c, _ in frames})
        assert output == old[group['frame_idx']]['predictions']['L4_three']
        if group['frame_idx'] < 8:
            continue
        states = [*mapper.position_layer.tracks, *mapper.position_layer.dense_states.values()]
        records.append(copy.deepcopy(dict(frame_idx=group['frame_idx'], timestamp_s=group['reference_timestamp_s'],
            association=copy.deepcopy(mapper.pair_scores), states=[dict(raw_id=s['node_id'],
                canonical=mapper.lifecycle.resolve(mapper.position_layer.aliases.get(s['node_id'], s['node_id'])),
                category=s['category'], center=s['center'], last_measured_center=s['last_measured_center'],
                timestamp=s['timestamp'], velocity=s.get('velocity'), variance=s['variance'],
                hits=s['hits'], real_points=s.get('_last_world_points'),
                source_keys=s.get('_observation_keys'), feature=s['feature']) for s in states], predictions=output)))
    # 真实来源图像，只画现有检测框，不生成场景内容。
    panels = []
    for frame_idx, detection_idx in [(10, 5), (11, 0)]:
        view = scene['groups'][frame_idx]['views']['CAM_FRONT_LEFT']
        frame = load_frame(scene['groups'][frame_idx], 'CAM_FRONT_LEFT')
        image = Image.open(view['rgb']).convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(frame.detections['xyxy'][detection_idx]), outline='red', width=4)
        draw.text((20, 20), f'Actual CAM_FRONT_LEFT frame {frame_idx}; detection {detection_idx}', fill='red')
        panels.append(image.resize((800, 450)))
    combined = Image.new('RGB', (1600, 450))
    for i, panel in enumerate(panels):
        combined.paste(panel, (i*800, 0))
    combined.save(run/(P+'actual_truck_sources.jpg'))
    release = dict(UTC=datetime.now(timezone.utc).isoformat(), GT_read=False, exact_L4_prefix_groups=12,
        scope='one scene first12groups passive diagnostic; not new complete performance result',
        budget='15min CPU/32MiB within parent120min768MiB',
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/(P+'release.json')).write_text(json.dumps(release, ensure_ascii=False, indent=2))
    with gzip.open(run/(P+'temporal_evidence.json.gz'), 'wt') as out:
        json.dump(records, out, default=safe, allow_nan=True)
    print(run)


if __name__ == '__main__':
    main()
