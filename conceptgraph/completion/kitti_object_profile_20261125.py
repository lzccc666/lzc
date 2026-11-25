"""Timing-only diagnostic of the frozen Object frontend; no new method."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from conceptgraph.completion import kitti_object_all_20261125 as impl


def main():
    import torch
    import cv2
    from conceptgraph.scripts import streamlined_detections as frontend
    original_rows = impl.rows; original_write = impl.write
    profile = []; current = {}; original_load = impl.load_frame
    impl.RUN_ID = 'kitti_object_all_20261125_profile_v2'
    impl.TEMP = impl.ROOT / 'outputs/reports/kitti_object_all_20261125_profile_tmp'
    def row_input(path):
        for index, row in enumerate(original_rows(path)):
            if Path(path).name == 'method_frames.jsonl' and index >= 10:
                break
            yield row
    def write(path, value):
        p = Path(path)
        if p.name in ['gpu_started_v1.json', 'gpu_released_v1.json']:
            p = p.with_name(p.name.replace('_v1', '_profile_v2'))
        return original_write(p, value)
    def load(archives, row):
        current.clear(); current['frame_id'] = row['frame_id']
        started = time.monotonic(); result = original_load(archives, row)
        current['load_and_projection_s'] = time.monotonic() - started
        return result
    def timed(fn, name):
        def invoke(*args, **kwargs):
            started = time.monotonic(); result = fn(*args, **kwargs)
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            current[name] = current.get(name, 0.) + time.monotonic() - started
            return result
        return invoke
    original_loader = frontend.load_detection_models
    def loader(cfg):
        result = original_loader(cfg)
        for model, name in zip(result[:2], ['YOLO_predict_s', 'SAM_predict_s']):
            model.predict = timed(model.predict, name)
        return result
    frontend.load_detection_models = loader
    frontend.compute_clip_features_batched = timed(frontend.compute_clip_features_batched, 'CLIP_features_s')
    original_detector = frontend.detect_rgb_file
    def detector(*args, **kwargs):
        started = time.monotonic(); result = original_detector(*args, **kwargs)
        current['frontend_total_s'] = time.monotonic() - started
        current['torch_threads'] = torch.get_num_threads(); current['cv2_threads'] = cv2.getNumThreads()
        return result
    frontend.detect_rgb_file = detector
    original_observations = impl.old.observations
    def observations(*args, **kwargs):
        started = time.monotonic(); result = original_observations(*args, **kwargs)
        current['numeric_depth_postprocess_s'] = time.monotonic() - started
        profile.append(dict(current)); print(json.dumps(current), flush=True)
        with (impl.REPORT / 'phase_profile_v2.jsonl').open('a') as stream:
            stream.write(json.dumps(current) + '\n')
        return result
    impl.old.observations = observations; impl.load_frame = load
    impl.rows = row_input; impl.write = write
    original_write(impl.DATA / 'timing_diagnostic_amendment_v2.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='diagnostic10frames_only_not_full_results', changes='timers and explicit CUDA synchronization only',
        source_sha256={str(Path(__file__)): impl.sha(__file__),
            str(impl.ROOT / 'scripts/kitti_object_profile_20261125.py'): impl.sha(impl.ROOT / 'scripts/kitti_object_profile_20261125.py')},
        thresholds_or_weights_or_prompts_changed=False, original_prefix_retained=True))
    impl.infer()


if __name__ == '__main__':
    main()
