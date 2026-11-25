"""Same frozen Object method, explicit CUDA barriers and auditable phase timing."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

from conceptgraph.completion import kitti_object_all_20261125 as impl


class PrefixMismatch(BaseException):
    pass


def register():
    paths = [Path(__file__), impl.ROOT / 'scripts/kitti_object_runtime_20261125_v3.py']
    amendment = impl.DATA / 'runtime_execution_amendment_v3.json'
    hashes = {str(p): impl.sha(p) for p in paths}
    if amendment.exists():
        if impl.read(amendment)['runtime_source_sha256'] != hashes:
            raise PrefixMismatch('runtime_source_changed')
    else:
        impl.write(amendment, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            runtime_source_sha256=hashes,
            changes='same frontend calls with explicit CUDA synchronization after YOLO/SAM/CLIP; phase timers and prefix equality guard',
            no_change_to_model_weights_or_thresholds_or_seed_or_inputs_or_metrics=True,
            prior79_prefix_and10_diagnostic_retained=True,
            diagnostic10_all_numeric_predictions_and_maskhash_exact=True,
            prefix_required=79, total_frames=7481, configuration_count=1,
            GPU_budget_includes_prior_v1_and10_profile=True,
            underlying_latency_cause='synchronization-sensitive execution observed; no stronger hardware root-cause claim'))
    impl.RUN_ID = 'kitti_object_all_20261125_v3'
    impl.TEMP = impl.ROOT / 'outputs/reports/kitti_object_all_20261125_runtime_tmp'


def infer():
    import numpy as np
    import torch
    from conceptgraph.scripts import streamlined_detections as frontend
    register()
    old_dest = impl.experiment_directory('kitti', 'validation', 'kitti_object_all_20261125_v1')
    prefix = {r['frame_id']: r for r in impl.rows(old_dest / 'inference_frames.jsonl.gz')}
    assert len(prefix) == 79
    current = {}; matched = []
    prior_gpu_s = sum(impl.read(impl.REPORT / name)['GPU_wall_s']
        for name in ['gpu_released_v1.json', 'gpu_released_profile_v2.json'])
    original_guard = impl.guard; original_write = impl.write; original_load = impl.load_frame
    def guard(started, allowance=6000):
        return original_guard(started, allowance=6000 - prior_gpu_s if allowance == 6000 else allowance)
    def write(path, value):
        p = Path(path)
        if p.name in ['gpu_started_v1.json', 'gpu_released_v1.json']:
            p = p.with_name(p.name.replace('_v1', '_v3'))
            value = dict(value, prior_GPU_wall_s=prior_gpu_s, total_GPU_budget_s=6000)
        return original_write(p, value)
    def load(archives, row):
        current.clear(); current['frame_id'] = row['frame_id']
        started = time.monotonic(); result = original_load(archives, row)
        current['load_and_projection_s'] = time.monotonic() - started
        current['input_sha256'] = {key: hashlib.sha256(content).hexdigest() for key, content in result[0].items()}
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
        return result
    frontend.detect_rgb_file = detector
    original_observations = impl.old.observations
    def observations(*args, **kwargs):
        started = time.monotonic(); result = original_observations(*args, **kwargs)
        current['numeric_depth_postprocess_s'] = time.monotonic() - started
        for prediction, mask in zip(result, args[0]['mask']):
            prediction['packed_mask_sha256'] = hashlib.sha256(np.packbits(mask.astype(bool)).tobytes()).hexdigest()
            prediction['mask_shape'] = list(mask.shape)
        name = current['frame_id']
        if name in prefix:
            original = prefix[name]
            if result != original['predictions'] or current['input_sha256'] != original['input_member_sha256']:
                original_write(impl.REPORT / 'prefix_mismatch_v3.json', dict(frame_id=name, exact=False))
                raise PrefixMismatch('original79_prefix_numeric_or_input_mismatch:' + name)
            matched.append(name)
            if len(matched) == len(prefix):
                original_write(impl.REPORT / 'prefix_equivalence_v3.json', dict(passed=True,
                    compared=len(matched), inputs_and_all_prediction_numbers_and_maskhashes_exact=True,
                    original_prefix_sha256=impl.sha(old_dest / 'inference_frames.jsonl.gz'),
                    no_parameter_or_label_changes=True))
        phase = dict(current); phase.pop('input_sha256')
        with (impl.REPORT / 'phase_timing_v3.jsonl').open('a') as stream:
            stream.write(json.dumps(phase) + '\n')
        return result
    impl.guard = guard; impl.write = write; impl.load_frame = load; impl.old.observations = observations
    impl.infer()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['infer', 'evaluate'])
    action = parser.parse_args().action
    if action == 'infer':
        infer()
    else:
        register()
        if not impl.read(impl.REPORT / 'prefix_equivalence_v3.json')['passed']:
            raise PrefixMismatch('prefix_equivalence_required')
        impl.evaluate()


if __name__ == '__main__':
    main()
