"""只重放已冻结 R1 缓存并补录定位/确认/输出过滤证据，逐字段核验原预测。"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer, RefinedEvidenceMapper
from conceptgraph.slam.evidence_mapper import confirmed_observation
from conceptgraph.slam.streamlined_mapping import MappingFrame


def state_record(state):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in state.items() if k != 'feature'}


class DiagnosticLayer(DenseProtectedPositionLayer):
    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        self.before_sparse = [state_record(t) for t in self.tracks]
        result = super().update(sparse, dense_predictions, dense_measurements, timestamp)
        self.before_fov = copy.deepcopy(result)
        return result


class DiagnosticR1(RefinedEvidenceMapper):
    def __init__(self, cfg):
        super().__init__(cfg, CLASSES, condition='R1')
        self.position_layer = DiagnosticLayer()
        self.sparse_details = []

    def extract_detections(self, frame):
        self.sparse_details = []
        return super().extract_detections(frame)

    def sparse_callback(self, frame):
        original = super().sparse_callback(frame)

        def save(depth, mask, gobs, index, evidence):
            before = len(self.sparse_measurements)
            original(depth, mask, gobs, index, evidence)
            if len(self.sparse_measurements) > before:
                measurement = self.sparse_measurements[-1]
                self.sparse_details.append(dict(state_record(measurement),
                    filtered_mask_index=int(index), raw_index=int(gobs['_audit_indices'][index]),
                    depth_evidence=copy.deepcopy(evidence)))
        return save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-budget-s', type=float, default=1800)
    args = parser.parse_args()
    start = time.monotonic()
    original = PROJECT / 'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
    release = json.loads((original / 'release.json').read_text())
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/slam/dense_refinement.py',
        PROJECT / 'conceptgraph/slam/sparse_observations.py', PROJECT / 'conceptgraph/slam/quality_mapper.py']
    for path in sources:
        target = dest / 'source_snapshot' / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump(dest / 'release.json', dict(mode='R1_frozen_internal_diagnostic_replay', original_run=str(original),
        original_release_sha256=digest(original / 'release.json'), sources_sha256={str(p): digest(p) for p in sources},
        scenes=release['scenes'], seed=release['seed'], new_candidate=False, parameter_changes=False,
        new_frontend_calls=0, main_threshold_m=3., diagnostic_oriented_box_margin_m=.5,
        max_new_bytes=2 * 2**30, minimum_free_bytes=20 * 2**30, wall_budget_s=args.wall_budget_s,
        GT_in_inference=False, test_accessed=False))
    frames = 0
    for scene in release['scenes']:
        name = scene['scene']
        mapper = DiagnosticR1(OmegaConf.create(release['config']))
        path = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
        target = dest / name
        target.mkdir()
        with path.open() as source, gzip.open(original / name / 'frames.jsonl.gz', 'rt') as reference, gzip.open(target / 'states.jsonl.gz', 'wt') as out:
            aa, bb = list(map(json.loads, source)), list(map(json.loads, reference))
            if len(aa) != len(bb):
                raise AssertionError('帧数变化')
            for row, expected in zip(aa, bb):
                if time.monotonic() - start > args.wall_budget_s or shutil.disk_usage(dest).free < 20 * 2**30:
                    raise RuntimeError('达到预算或磁盘保护线；保留真实前缀')
                idx = row['frame_idx']
                p = Path(expected['input']['detections'])
                if digest(p) != expected['input']['detections_sha256']:
                    raise ValueError('缓存改变')
                with gzip.open(p, 'rb') as stream:
                    gobs = pickle.load(stream)
                before = {o['stable_object_id'] for o in mapper.objects}
                frame = MappingFrame(idx, np.asarray(Image.open(row['rgb']).convert('RGB')),
                    np.asarray(Image.open(row['depth']), dtype=float) / 1000, np.asarray(row['intrinsic']),
                    np.asarray(row['camera_pose']), gobs, row['rgb'], timestamp_s=row['camera_timestamp_s'])
                np.random.seed(release['seed'] + idx)
                mapper.update(frame)
                raw = predict(mapper.objects, CLASSES)
                confirmed = predict(mapper.confirmed_objects(), CLASSES)
                output = mapper.position_predictions(confirmed, frame)
                if json.loads(json.dumps(output, allow_nan=False)) != expected['predictions']['R1']:
                    raise AssertionError(f'R1预测改变 {name}/{idx}')
                if mapper.frame_audit != expected['audit']:
                    raise AssertionError(f'R1审计改变 {name}/{idx}')
                observations = []
                geometry = {a['filtered_mask_index']: a for a in mapper.frame_audit if a['stage'] == 'geometry'}
                for obj in mapper.last_observations:
                    mask_idx = int(obj['mask_idx'][-1])
                    observations.append(dict(category=obj['class_name'][-1], filtered_mask_index=mask_idx,
                        raw_index=geometry[mask_idx]['detection_index'], center=obj['bbox'].get_center().tolist(),
                        points=np.asarray(obj['pcd'].points).tolist(), confidence=float(obj['conf'][-1])))
                associations, confirmations = [], []
                for obj in mapper.objects:
                    node_id = obj['stable_object_id']
                    if obj['image_idx'][-1] == idx:
                        associations.append(dict(node_id=node_id, filtered_mask_index=int(obj['mask_idx'][-1]),
                            raw_index=geometry[int(obj['mask_idx'][-1])]['detection_index'],
                            reason='matched_existing' if node_id in before else 'created',
                            current_history_center=np.asarray(obj['center_history'][-1]).tolist()))
                    confirmations.append(dict(node_id=node_id, confirmed=confirmed_observation(obj,
                        strong_confidence=float(mapper.cfg.strong_confidence),
                        minimum_observations=int(mapper.cfg.weak_confirmation_observations)),
                        distinct_frames=len(set(obj['image_idx'])), observations=int(obj['num_detections'])))
                record = dict(frame_idx=idx, predictions_exact=True, audits_exact=True,
                    observations=observations, sparse_measurements=mapper.sparse_details,
                    dense_internal=raw, dense_confirmed=confirmed,
                    position_outputs_before_fov=mapper.position_layer.before_fov,
                    sparse_tracks_before=mapper.position_layer.before_sparse,
                    sparse_tracks=[state_record(t) for t in mapper.position_layer.tracks],
                    dense_states={k: state_record(v) for k, v in mapper.position_layer.dense_states.items()},
                    aliases=mapper.position_layer.aliases, associations=associations, confirmations=confirmations,
                    expired_dense_node_ids=sorted(before - {o['stable_object_id'] for o in mapper.objects}))
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                frames += 1
        print(name, frames, 'predictions/audits exact', flush=True)
    if frames != 1508:
        raise AssertionError('范围不完整')
    dump(dest / 'report.json', dict(complete=True, scenes=38, frames=frames, every_prediction_and_audit_exact=True,
        parameter_changes=False, new_frontend_calls=0, wall_s=time.monotonic() - start,
        output_sha256={str(p.relative_to(dest)): digest(p) for p in dest.glob('scene-*/*.gz')}))


if __name__ == '__main__':
    main()
