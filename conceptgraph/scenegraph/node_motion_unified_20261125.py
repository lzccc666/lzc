"""因果测量旁路与统一节点视图；不改变原关联、几何、身份或默认配置。"""
from __future__ import annotations

import copy
import math
from itertools import combinations
import numpy as np

from conceptgraph.completion.association import gated_assignment
from conceptgraph.completion.motion import robust_xy_speed
from conceptgraph.navigation.mf_position_20261125_v2 import POLICY
from conceptgraph.slam.dense_refinement import RefinedEvidenceMapper
from conceptgraph.slam.mapnext_20261124_sparse_separation_v2 import IndependentSparseMapperV2
from conceptgraph.slam.sparse_observations import measured_points
import conceptgraph.slam.sparse_observations as sparse_module

SCHEMA_VERSION = 'node_motion_unified_20261125_v1'
FRAME_SOURCE = 'CAM_FRONT_camera_pose_and_optical_axis_projected_world_XY_proxy_not_rear_axle_heading'


def _key(camera, timestamp, index):
    return (str(camera), float(timestamp), int(index))


class _EvidenceLayerMixin:
    """在原位置层调用前后记录；原数值路径通过super执行一次。"""
    def _nm_record(self, state, observation, timestamp):
        owner = self._nm_owner
        if '_nm_uid' not in state:
            state['_nm_uid'] = owner._nm_serial
            owner._nm_serial += 1
        raw = observation.get('_nm_raw')
        if raw is None:
            return
        row = copy.deepcopy(raw)
        row['assignment'] = copy.deepcopy(observation.get('_nm_assignment', {}))
        row['source_closed'] = bool(row['assignment'].get('unambiguous', False))
        state.setdefault('_nm_trace', []).append(row)
        state['_nm_trace'] = state['_nm_trace'][-POLICY['history_length']:]
        owner._nm_events.append(dict(kind='state_raw_update', state_uid=state['_nm_uid'],
            raw_state_id=state['node_id'], observation_key=row['key'],
            source_closed=row['source_closed'], assignment=row['assignment']))

    def _new_state(self, observation, timestamp, node_id):
        state = super()._new_state(observation, timestamp, node_id)
        self._nm_record(state, observation, timestamp)
        return state

    def _update_state(self, state, observation, timestamp):
        super()._update_state(state, observation, timestamp)
        self._nm_record(state, observation, timestamp)

    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        owner = self._nm_owner
        before = {s.get('_nm_uid'): s['node_id'] for s in [*self.tracks, *self.dense_states.values()]}
        aliases = dict(self.aliases)
        for node_id, observation in dense_measurements.items():
            raw, assignment = owner._nm_dense_source(node_id)
            observation['_nm_raw'] = raw
            observation['_nm_assignment'] = assignment
        for observation in sparse:
            key = observation.get('_observation_key', observation.get('_nm_source_key'))
            observation['_nm_raw'] = owner._nm_raw.get(tuple(key)) if key else None
        original_assignment = sparse_module.gated_assignment
        calls = []

        def watched_assignment(scores, **kwargs):
            result = original_assignment(scores, **kwargs)
            index = len(calls)
            if index == 0:
                rows = sparse
                for i, observation in enumerate(rows):
                    observation['_nm_prior_dense_feasible'] = int(np.isfinite(scores[i]).sum())
                    observation['_nm_prior_dense_assigned'] = i in result
            elif index == 1:
                rows = [o for i, o in enumerate(sparse) if i not in calls[0]]
            else:
                raise AssertionError('unexpected_sparse_assignment_stage')
            for i, observation in enumerate(rows):
                row_count = int(np.isfinite(scores[i]).sum())
                j = result.get(i)
                column_count = int(np.isfinite(scores[:, j]).sum()) if j is not None else 0
                unresolved = bool(index == 1 and observation.get('_nm_prior_dense_feasible', 0)
                                  and not observation.get('_nm_prior_dense_assigned', False))
                closed = ((row_count == 1 and column_count == 1) if j is not None else row_count == 0)
                observation['_nm_assignment'] = dict(stage='sparse_to_dense' if index == 0 else 'sparse_to_sparse_or_new',
                    row_feasible=row_count, column_feasible=column_count, assigned=j is not None,
                    unresolved_prior_dense_competition=unresolved, unambiguous=bool(closed and not unresolved))
            calls.append(result)
            return result

        sparse_module.gated_assignment = watched_assignment
        try:
            result = super().update(sparse, dense_predictions, dense_measurements, timestamp)
        finally:
            sparse_module.gated_assignment = original_assignment
        assert len(calls) == 2, len(calls)
        after = {s.get('_nm_uid'): s['node_id'] for s in [*self.tracks, *self.dense_states.values()]}
        for uid, node_id in before.items():
            if uid not in after:
                owner._nm_events.append(dict(kind='state_no_longer_active', state_uid=uid,
                    raw_state_id=node_id, history_merged=False))
        for key, value in self.aliases.items():
            if aliases.get(key) != value:
                owner._nm_events.append(dict(kind='original_alias_handoff', original_id=key,
                    alias_id=value, history_merged=False))
        return result


class _CausalEvidenceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nm_serial = 0
        self._nm_raw = {}
        self._nm_dense_assignment = {}
        self._nm_events = []
        layer = self.position_layer
        layer.__class__ = type('NodeMotionEvidenceLayer', (_EvidenceLayerMixin, type(layer)), {})
        layer._nm_owner = self

    def begin_evidence(self, scene, frame_idx):
        self._nm_scene, self._nm_frame_idx = scene, int(frame_idx)
        self._nm_raw, self._nm_dense_assignment, self._nm_events = {}, {}, []
        self._nm_started_empty = not self.objects

    def observation_policy(self, frame):
        original = super().observation_policy(frame)
        camera = next((c for c, f in getattr(self, 'group_frames', []) if f is frame), 'CAM_FRONT')
        capture = float(getattr(self, 'capture_times', {}).get(camera, frame.timestamp_s))
        def select(depth, mask, gobs, index):
            selected, evidence = original(depth, mask, gobs, index)
            points = measured_points(depth, selected, frame.cam_K, frame.camera_pose)
            raw_index = int(gobs.get('_audit_indices', np.arange(len(gobs['mask'])))[index])
            source_key = _key(camera, capture, raw_index)
            self._nm_raw[source_key] = dict(key=f'{self._nm_scene}:{self._nm_frame_idx}:{camera}:{capture:.6f}:{raw_index}',
                source_key=list(source_key), scene=self._nm_scene, frame_idx=self._nm_frame_idx,
                timestamp_s=capture, render_timestamp_s=float(frame.timestamp_s), category=gobs['classes'][gobs['class_id'][index]],
                filtered_detection_index=int(index), original_detection_index=raw_index,
                raw_center_world_m=np.median(points, axis=0).tolist() if len(points) else None,
                current_points=len(points), depth_evidence=copy.deepcopy(evidence),
                source='current_projected_lidar', filtered_position_used_as_measurement=False)
            return selected, evidence
        return select

    def sparse_callback(self, frame):
        original = super().sparse_callback(frame)
        camera = next((c for c, f in getattr(self, 'group_frames', []) if f is frame), 'CAM_FRONT')
        capture = float(getattr(self, 'capture_times', {}).get(camera, frame.timestamp_s))
        def save(depth, mask, gobs, index, evidence):
            count = len(self.sparse_measurements)
            original(depth, mask, gobs, index, evidence)
            if len(self.sparse_measurements) > count:
                raw_index = int(gobs.get('_audit_indices', np.arange(len(gobs['mask'])))[index])
                self.sparse_measurements[-1]['_nm_source_key'] = list(_key(camera, capture, raw_index))
        return save

    def merge_associated_detections(self, detections, scores):
        values = scores.detach().cpu().numpy().copy()
        for i, det in enumerate(detections):
            for j, obj in enumerate(self.objects):
                if (set(np.atleast_1d(det['class_id'])).isdisjoint(set(np.atleast_1d(obj['class_id'])))
                        or np.linalg.norm(np.asarray(det['bbox'].get_center())-obj['bbox'].get_center()) > self.cfg.max_tracking_distance_m):
                    values[i, j] = -np.inf
        assigned = gated_assignment(values)
        for i, det in enumerate(detections):
            row = int(np.isfinite(values[i]).sum());j = assigned.get(i)
            column = int(np.isfinite(values[:, j]).sum()) if j is not None else 0
            self._nm_dense_assignment[int(det['mask_idx'][-1])] = dict(stage='original_dense_one_to_one',
                row_feasible=row, column_feasible=column, assigned=j is not None,
                unambiguous=bool(row == 1 and column == 1) if j is not None else row == 0)
        return super().merge_associated_detections(detections, scores)

    def _nm_dense_source(self, node_id):
        obj = next((o for o in self.objects if o['stable_object_id'] == node_id), None)
        if obj is None:
            return None, dict(unambiguous=False, reason='no_original_dense_object')
        index = int(obj['mask_idx'][-1])
        source = getattr(self, 'measurement_sources', {}).get(index)
        if source:
            raw = self._nm_raw.get(tuple(source['_observation_key']))
        else:
            raw = next((r for r in self._nm_raw.values() if r['filtered_detection_index'] == index), None)
        assignment = self._nm_dense_assignment.get(index)
        if assignment is None:
            assignment = dict(stage='first_dense_batch', row_feasible=0,
                unambiguous=bool(self._nm_started_empty), reason='no_preexisting_track' if self._nm_started_empty else 'missing_association_evidence')
        return raw, assignment

    def evidence_states(self):
        layer = self.position_layer
        return [dict(node_id=s['node_id'], state_uid=s.get('_nm_uid'), category=s['category'],
                     timestamp_s=s['timestamp'], variance=np.asarray(s['variance']).tolist(),
                     raw_trace=copy.deepcopy(s.get('_nm_trace', [])))
                for s in [*layer.tracks, *layer.dense_states.values()]]

    def resolve_evidence_id(self, node_id):
        logical = self.position_layer.aliases.get(node_id, node_id)
        return self.lifecycle.resolve(logical) if hasattr(self, 'lifecycle') else logical


class UnifiedR1Mapper(_CausalEvidenceMixin, RefinedEvidenceMapper):
    pass


class UnifiedS2Mapper(_CausalEvidenceMixin, IndependentSparseMapperV2):
    pass


def motion_evidence(trace, variance, render_timestamp):
    trace = trace[-POLICY['history_length']:];reasons=[]
    times = [r['timestamp_s'] for r in trace]
    groups = [(r['scene'], r['frame_idx']) for r in trace]
    if len(set(groups)) < 3 or len(set(groups)) != len(trace):reasons.append('insufficient_independent_groups')
    if not times or any(b <= a for a, b in zip(times, times[1:])) or any(t > render_timestamp for t in times):reasons.append('noncausal_or_unordered_measurements')
    span = times[-1]-times[0] if times else 0.
    if span < 1.-1e-8:reasons.append('span_below_one_second')
    latest = trace[-1] if trace else {};depth=latest.get('depth_evidence', {})
    current = bool(trace) and abs(latest['render_timestamp_s']-render_timestamp) < 1e-6
    if not current:reasons.append('no_current_update')
    if (depth.get('source') != 'current_projected_lidar' or depth.get('units') != 'm'
            or latest.get('current_points', 0) < 1 or depth.get('support_fraction', 0) < .4):reasons.append('invalid_current_measurement_support')
    if not trace or any(not r.get('source_closed', False) for r in trace):reasons.append('unresolved_identity_or_competition')
    var = np.asarray(variance if variance is not None else [np.nan]*3, dtype=float)
    if not np.isfinite(var).all() or np.any(var <= 0) or max(var[:2]) > 4.:reasons.append('heuristic_uncertainty_too_large_or_missing')
    speed, heading = None, None
    if len(times) >= 2 and all(b > a for a, b in zip(times, times[1:])) and all(r.get('raw_center_world_m') is not None for r in trace):
        centers = np.asarray([r['raw_center_world_m'] for r in trace])
        speed = robust_xy_speed(centers, times)
        pairs = [(centers[j,:2]-centers[i,:2])/(times[j]-times[i]) for i,j in combinations(range(len(times)),2) if times[j]-times[i] >= .75]
        vector = np.median(pairs,axis=0) if pairs else (centers[-1,:2]-centers[-2,:2])/(times[-1]-times[-2])
        if speed > .5:heading = float(math.atan2(vector[1], vector[0]))
    if speed is None:reasons.append('missing_raw_speed')
    state = 'unknown' if reasons else ('static' if speed <= .5 else 'dynamic')
    return dict(state=state, speed_mps=speed if not reasons else None, measured_raw_speed_mps=speed,
                motion_heading_world_rad=heading if state == 'dynamic' else None, reasons=reasons,
                span_s=span, current_update=current, independent_groups=len(set(groups)),
                first_measurement_s=times[0] if times else None,last_measurement_s=times[-1] if times else None,
                source_closed=bool(trace) and all(r.get('source_closed',False) for r in trace))


def unified_nodes(predictions, states, resolve, render_timestamp):
    by_id={}
    for state in states:by_id.setdefault(resolve(state['node_id']),[]).append(state)
    nodes=[]
    for index, obj in enumerate(predictions):
        matches=by_id.get(obj['node_id'],[])
        # Do not concatenate histories across original IDs or retired/canonical members.
        current=[s for s in matches if abs(s['timestamp_s']-render_timestamp)<1e-6]
        source=max(current or matches,key=lambda s:(s['timestamp_s'],-s.get('state_uid',0))) if matches else None
        trace=source['raw_trace'] if source else []
        variance=obj.get('position_covariance_diag',source['variance'] if source else None)
        motion=motion_evidence(trace,variance,render_timestamp)
        if len(current)>1:
            motion.update(state='unknown',speed_mps=None,motion_heading_world_rad=None)
            motion['reasons'].append('multiple_current_canonical_states_not_merged')
        latest=trace[-1] if trace else {};depth=latest.get('depth_evidence',{})
        extent=obj.get('extent');shape=bool(extent is not None and np.isfinite(extent).all() and np.all(np.asarray(extent)>0))
        node=dict(original_index=index,node_id=obj['node_id'],category=obj['category'],center_world_m=obj['center'],
            extent_world_aabb_m=extent if shape else None,shape_available=shape,
            motion_state=motion['state'],motion_speed_xy_mps=motion['speed_mps'],motion_reference_frame='world_xy',
            motion_heading_world_rad=motion['motion_heading_world_rad'],body_heading_world_rad=None,body_heading_source='unknown',
            motion_time={k:motion[k] for k in ['first_measurement_s','last_measurement_s','span_s','current_update']},
            measurement=dict(source=latest.get('source','unknown'),current_points=latest.get('current_points',0) if motion['current_update'] else 0,
                current_support_fraction=depth.get('support_fraction') if motion['current_update'] else None,
                raw_center_world_m=latest.get('raw_center_world_m'),observation_keys=[r['key'] for r in trace],
                independent_groups=motion['independent_groups'],source_closed=motion['source_closed'],unknown_reasons=motion['reasons'],
                state_uid=source.get('state_uid') if source else None,raw_state_id=source['node_id'] if source else None,
                original_dense_or_sparse='sparse_position' if obj.get('shape_available') is False else 'dense_object',
                measured_raw_speed_mps=motion['measured_raw_speed_mps']),
            uncertainty=dict(diag_m2=variance,kind='uncalibrated_heuristic_not_probability'),
            semantic_position_candidate=motion['state']=='static',vehicle_safety_authorized=False,
            legacy_motion_state=obj.get('motion_state','unknown'),legacy_velocity_mps=obj.get('velocity_mps'),
            motion_history_cross_id_merged=False)
        node['motion_time'].update(render_timestamp_s=render_timestamp,age_s=None if not trace else render_timestamp-latest['timestamp_s'])
        nodes.append(node)
    return nodes


def task_predictions(predictions,nodes):
    """统一任务视图：保留几何／原索引；三态与真实当前深度由旁路提供。"""
    out=copy.deepcopy(predictions)
    for obj,node in zip(out,nodes):
        obj['motion_state']=node['motion_state']
        obj['shape_available']=node['shape_available']
        obj['position_confirmation']=dict(semantic_position_candidate=node['semantic_position_candidate'],
            vehicle_safety_authorized=False,unknown_reasons=node['measurement']['unknown_reasons'])
    return out
