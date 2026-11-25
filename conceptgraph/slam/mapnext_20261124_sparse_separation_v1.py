"""S1: preserve confirmed sparse compact instances with independent image evidence.

The original proximity suppression is retained unless *every* nearby map output
has a current same-camera mask/box that is disjoint from the sparse instance.
"""
import numpy as np

from conceptgraph.slam.mapnext_20261124_mapper import AuditedNativeMapper
from conceptgraph.slam.mapnext_20261124_depth_history_v1 import UniqueHistoryMapper
from conceptgraph.slam.multiview_20261123_fusion import in_view

EXTRA_SOURCES = ['conceptgraph/slam/mapnext_20261124_mapper.py',
                 'conceptgraph/slam/mapnext_20261124_depth_history_v1.py']


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-intersection
    return float(intersection/union) if union else 0.


class IndependentSparseMixin:
    def settings_record(self):
        record = super().settings_record()
        record['mapnext_sparse_independence'] = dict(version='S1',
            categories=['pedestrian','bicycle','motorcycle'], original_confirmation_and_TTL=True,
            current_observation_only=True, every_blocker_same_camera_current=True,
            maximum_mask_overlap_smaller=.1, maximum_box_IoU=.1,
            centers_separation_m=.5, no_hit_sum_or_new_points=True)
        return record

    def position_predictions(self, predictions, frame):
        output = super().position_predictions(predictions, frame)
        layer, decisions = self.position_layer, []
        states = [*layer.tracks, *layer.dense_states.values()]
        roots = {p['node_id'] for p in output}
        def root_of(state):
            return self.lifecycle.resolve(layer.aliases.get(state['node_id'],state['node_id']))
        for track in layer.tracks:
            root = root_of(track)
            settings = layer.settings
            frames_needed = (settings.minimum_frames if track['confidence'] >= settings.strong_confidence
                             else settings.weak_minimum_frames)
            if (root in roots or track['category'] not in ['pedestrian','bicycle','motorcycle']
                    or track['hits'] < frames_needed or track['total_points'] < settings.minimum_total_points
                    or track['timestamp'] != frame.timestamp_s
                    or frame.timestamp_s-track['timestamp'] > settings.maximum_prediction_age_s
                    or not any(in_view(track['center'], v) for v in self.group_views)):
                continue
            blockers = [o for o in output if o['category']==track['category'] and
                np.linalg.norm((np.asarray(o['center'])-track['center'])[:2]) <= settings.association_distance_m]
            if not blockers:
                continue
            key = track.get('_observation_keys', [None])[-1]
            if key is None or key[0] not in self.current_views:
                continue
            view = self.current_views[key[0]]
            mask = view['masks'][key[2]]
            current_frame = next(f for c,f in self.group_frames if c==key[0])
            box = current_frame.detections['xyxy'][key[2]]
            checks = []
            for blocker in blockers:
                current = [s for s in states if root_of(s)==blocker['node_id'] and s['timestamp']==frame.timestamp_s]
                witnesses = []
                for state in current:
                    other = state.get('_observation_keys', [None])[-1]
                    if other is None or other[0] != key[0] or other[1] != key[1]:
                        continue
                    other_mask = view['masks'][other[2]]
                    denominator = min(int(mask.sum()),int(other_mask.sum()))
                    overlap = float((mask & other_mask).sum()/denominator) if denominator else 1.
                    iou = box_iou(box,current_frame.detections['xyxy'][other[2]])
                    separation = float(np.linalg.norm((track['last_measured_center']-state['last_measured_center'])[:2]))
                    witnesses.append(dict(source_key=other,mask_overlap_smaller=overlap,box_IoU=iou,
                        separation_m=separation,independent=overlap<.1 and iou<.1 and separation>.5))
                checks.append(dict(blocker=blocker['node_id'],witnesses=witnesses,
                    independent=bool(witnesses) and all(w['independent'] for w in witnesses)))
            accepted=all(c['independent'] for c in checks)
            decisions.append(dict(canonical=root,source_key=key,accepted=accepted,blocker_checks=checks,
                actual_hits=track['hits'],actual_points=track['total_points']))
            if not accepted:
                continue
            output.append(dict(node_id=root,category=track['category'],center=track['center'].tolist(),
                num_detections=track['hits'],timestamp_s=track['timestamp'],prediction_age_s=0.,
                position_covariance_diag=track['variance'].tolist(),position_source='confirmed_sparse_independent_mask',
                measured_points_total=track['total_points'],shape_available=False,
                state_last_measurement_timestamp_s=track['timestamp'],render_timestamp_s=frame.timestamp_s,
                state_measurement_age_s=0.,mapnext_actual_source_key=key))
            roots.add(root)
        self.mapnext_sparse_decisions=decisions
        return output

    def update_group(self, frames, capture_times, projections=None):
        result = super().update_group(frames,capture_times,projections)
        self.audit['mapnext_sparse_independence']=self.mapnext_sparse_decisions
        return result


class IndependentSparseMapper(IndependentSparseMixin,AuditedNativeMapper):
    pass


class UniqueHistoryIndependentMapper(IndependentSparseMixin,UniqueHistoryMapper):
    pass
