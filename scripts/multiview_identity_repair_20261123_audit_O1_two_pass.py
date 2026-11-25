"""独立复现O1碰撞帧两次原地输入过滤/包含mask改写，仅读缓存。"""
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


def subtract(boxes, mask):
    areas = np.prod(boxes[:, 2:]-boxes[:, :2], 1)
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = np.prod(np.maximum(rb-lt, 0), 2)
    ratio = inter/areas[:, None]
    aa, bb = np.nonzero((ratio < .7) & (ratio.T > .8))
    out = mask.copy()
    for a, b in zip(aa, bb):
        out[a] &= ~out[b]
    return out


def main():
    root = PROJECT_ROOT
    output = experiment_directory('nuscenes', 'validation', 'mvrepair_O1_two_pass_input_audit_20261123_v1', create=True)
    records = []
    for split, folder in [('pilot', 'multiview_20261123_pilot_v1'), ('regression', 'multiview_identity_repair_20261123_validation_v1')]:
        manifest = json.loads((root/'dataset/processed'/folder/'manifest.json').read_text())
        run = root/'outputs/results/nuscenes/validation/iterations'/('mvrepair_lifecycle_20261123_O1_ownership_'+split+'_v1')
        for scene in manifest['scenes']:
            file = run/scene['scene']/'mvrepair_lifecycle_20261123_predictions.jsonl.gz'
            for group, row in zip(scene['groups'], map(json.loads, gzip.open(file, 'rt'))):
                own = row['audit']['O1_ownership_three']['unique_support']
                if not own['collision_points']:
                    continue
                for camera, view in group['views'].items():
                    with gzip.open(view['detections'], 'rb') as stream:
                        det = pickle.load(stream)
                    boxes, mask = np.asarray(det['xyxy']), np.asarray(det['mask'], bool)
                    assert mask.shape[1:] == (900, 1600)
                    area = np.prod(boxes[:, 2:]-boxes[:, :2], 1)
                    keep = np.flatnonzero((mask.sum((1, 2)) >= 25) & (area <= 900*1600) & (det['confidence'] >= .1))
                    first = subtract(boxes[keep], mask[keep])
                    keep_second = np.flatnonzero(first.sum((1, 2)) >= 25)
                    second = subtract(boxes[keep][keep_second], first[keep_second])
                    changed = int(np.count_nonzero(first[keep_second] != second))
                    original_keys, second_keys = keep[keep_second].tolist(), keep_second.tolist()
                    saved_input = [e for e in row['audit']['O1_ownership_three']['mapping'] if e.get('stage') == 'input_filter' and e.get('camera') == camera]
                    assert len(saved_input) == len(keep)
                    assert [e['detection_index'] for e in saved_input] == list(range(len(keep)))
                    expected_final_keys = [e['detection_index'] for e in saved_input if e['reason'] == 'retained']
                    assert expected_final_keys == second_keys
                    records.append(dict(split=split, scene=scene['scene'], frame_idx=group['frame_idx'], camera=camera,
                        original_detections=len(mask), first_retained_detections=len(keep), second_retained_detections=len(keep_second),
                        first_original_keys=keep.tolist(), second_saved_keys=second_keys, corresponding_original_keys=original_keys,
                        saved_key_original_index_mismatches=sum(a != b for a, b in zip(original_keys, second_keys)),
                        second_containment_changed_pixels=changed,
                        first_mask_areas=first.sum((1, 2)).tolist(), second_mask_areas=second.sum((1, 2)).tolist()))
    bad = [r for r in records if r['saved_key_original_index_mismatches'] or r['second_containment_changed_pixels'] or r['second_retained_detections'] != r['first_retained_detections']]
    result = dict(passed=True, collision_camera_views=len(records), changed_camera_views=len(bad),
        total_saved_key_original_index_mismatches=sum(r['saved_key_original_index_mismatches'] for r in records),
        conclusion='Frozen O1 reuses mutable frame.detections. Actual second-pass input filtering/key/mask differences are listed; this is an implementation control audit, not a conclusion about ownership principle.',
        records=records, affected=bad, method_or_model_runs=0, predictions_changed=False)
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    report = root/'outputs/reports/multiview_identity_repair_20261123/validation/O1_two_pass_input_audit.json'
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(collision_camera_views=len(records), affected=bad, total_index_mismatches=result['total_saved_key_original_index_mismatches'])))


if __name__ == '__main__':
    main()
