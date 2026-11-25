"""对已冻结 R1 输出做逐真值支持诊断；诊断真值从不进入 mapper。"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from conceptgraph.completion.mapping_recall_diagnosis import box_iou, projected_box


BOX_IOU = .25


def xy_error(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b))[:2]))


def inside_official_box(points, obj, margin_m=0.):
    """几何支持代理；时差/标注误差下不是激光点的实例真值。"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    w, x, y, z = obj['rotation_wxyz']
    rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    local = (points - np.asarray(obj['center'])) @ rotation
    width, length, height = obj['size_wlh']
    return np.all(np.abs(local) <= np.asarray([length, width, height]) / 2 + margin_m, axis=1)


def state_evidence(row, states, truth):
    """把同帧真实内部状态连至支持框；所有近邻/框支持都保留代理属性。"""
    result = dict(row)
    ids = {c['raw_index'] for c in row['candidates']}
    dense = [o for o in states['observations'] if o['raw_index'] in ids and o['category'] == row['category']]
    sparse = [o for o in states['sparse_measurements'] if o['raw_index'] in ids and o['category'] == row['category']]
    observations = []
    for kind, entries in [('dense', dense), ('sparse', sparse)]:
        for observation in entries:
            entry = dict(kind=kind, raw_index=observation['raw_index'], center=observation['center'],
                error_xy_m=xy_error(observation['center'], row['center']),
                center_inside_official_box_margin05=bool(inside_official_box([observation['center']], truth, .5)[0]))
            if kind == 'dense':
                points = np.asarray(observation['points']).reshape(-1, 3)
                entry.update(points=len(points),
                    points_inside_official_box=int(inside_official_box(points, truth, 0.).sum()),
                    points_inside_official_box_margin05=int(inside_official_box(points, truth, .5).sum()))
            else:
                entry['points'] = observation['points']
            observations.append(entry)
    correct = [o for o in observations if o['error_xy_m'] <= 3]
    correct_dense = [o for o in correct if o['kind'] == 'dense']
    correct_sparse = [o for o in correct if o['kind'] == 'sparse']

    def near(entries):
        return [dict(node_id=o['node_id'], center=o['center'], error_xy_m=xy_error(o['center'], row['center']))
            for o in entries if o['category'] == row['category'] and xy_error(o['center'], row['center']) <= 3]

    internal = near(states['dense_internal'])
    confirmed = near(states['dense_confirmed'])
    prefov = near(states['position_outputs_before_fov'])
    sparse_near = near(states['sparse_tracks'])
    linked = []
    for observation in correct_dense:
        for assoc in states['associations']:
            if assoc['raw_index'] != observation['raw_index']:
                continue
            node = assoc['node_id']
            check = next(c for c in states['confirmations'] if c['node_id'] == node)
            raw = next(p for p in states['dense_internal'] if p['node_id'] == node)
            alias = states['aliases'].get(node, node)
            refined = next((p for p in states['position_outputs_before_fov'] if p['node_id'] == alias), None)
            linked.append(dict(assoc, confirmed=check['confirmed'],
                internal_center_error_xy_m=xy_error(raw['center'], row['center']),
                current_history_error_xy_m=xy_error(assoc['current_history_center'], row['center']),
                refined_center_error_xy_m=xy_error(refined['center'], row['center']) if refined else None))
    sparse_links = []
    for observation in correct_sparse:
        for kind, entries in [('sparse_track', states['sparse_tracks']), ('dense_state', states['dense_states'].values())]:
            for state in entries:
                if state['category'] != row['category'] or np.max(np.abs(np.asarray(state['last_measured_center']) - observation['center'])) > 1e-8:
                    continue
                needed = 2 if state['confidence'] >= .2 else 3
                sparse_links.append(dict(kind=kind, node_id=state['node_id'], hits=state['hits'],
                    total_points=state['total_points'], frames_needed=needed,
                    confirmed_support=bool(state['hits'] >= needed and state['total_points'] >= 4),
                    center_error_xy_m=xy_error(state['center'], row['center'])))

    if row['matched']:
        reason = 'matched_R1'
    elif row['final_within_3m']:
        reason = 'output_3m_support_competition'
    elif prefov:
        reason = 'position_output_3m_removed_by_camera_fov'
    elif correct_dense:
        if linked and all(not link['confirmed'] for link in linked):
            reason = 'accurate_dense_observation_unconfirmed'
        elif linked and all(link['refined_center_error_xy_m'] is None or link['refined_center_error_xy_m'] > 3 for link in linked):
            reason = 'accurate_dense_observation_map_position_outside_3m'
        else:
            reason = 'accurate_dense_observation_other_output_loss'
    elif correct_sparse:
        if sparse_links and all(link['kind'] == 'sparse_track' and not link['confirmed_support'] for link in sparse_links):
            reason = 'accurate_sparse_observation_insufficient_confirmation'
        elif sparse_links and all(link['center_error_xy_m'] > 3 for link in sparse_links):
            reason = 'accurate_sparse_observation_filter_position_outside_3m'
        elif sparse_links:
            reason = 'accurate_sparse_observation_suppressed_or_not_output'
        else:
            reason = 'accurate_sparse_observation_not_linked_to_saved_current_state'
    elif not row['iou025_detection_count']:
        reason = 'no_iou025_same_class_2d_support'
    elif not row['input_retained_count']:
        reason = 'overlap_exists_input_filter_rejected'
    elif row['max_valid_mask_depth_points'] == 0:
        reason = 'retained_mask_no_current_valid_depth'
    elif row['max_selected_mask_depth_points'] == 0:
        reason = 'depth_selection_no_retained_points'
    elif observations:
        reason = 'current_' + ('dense' if dense else 'sparse') + '_measurement_center_outside_3m'
    else:
        reason = 'geometry_rejected_or_no_saved_measurement'
    result.update(reason=reason, evidence_bin='no_iou025_support' if not row['iou025_detection_count'] else 'overlap_depth_' + row['depth_bin'],
        current_measurements=observations, current_measurement_accurate=bool(correct),
        current_sparse_accurate=bool(correct_sparse), nearby_dense_internal=internal,
        nearby_dense_confirmed=confirmed, nearby_pre_fov=prefov, nearby_sparse_states=sparse_near,
        accurate_dense_links=linked, accurate_sparse_links=sparse_links)
    return result


def frame_evidence(source, frozen, detections, *, log_token, first_seen):
    """复用真实 input_filter/geometry 审计，避免重新解释 R1 深度门限。"""
    if source['evaluation_only_official'] != frozen['evaluation_only_official']:
        raise ValueError('GT 范围或顺序变化')
    pose, intrinsic = np.asarray(source['camera_pose']), np.asarray(source['intrinsic'])
    filters = {a['detection_index']: a for a in frozen['audit'] if a['stage'] == 'input_filter'}
    geometry = {a['detection_index']: a for a in frozen['audit'] if a['stage'] == 'geometry'}
    recovered = {a['truth_index']: a for a in frozen['metrics']['R1']['matches']}
    preds = frozen['predictions']['R1']
    rows = []
    for ti, obj in enumerate(frozen['evaluation_only_official']):
        center = np.asarray(obj['center'])
        camera = pose[:3, :3].T @ (center - pose[:3, 3])
        uvz = intrinsic @ camera
        uv = uvz[:2] / uvz[2]
        box = projected_box(obj, pose, intrinsic)
        candidates = []
        all_same = []
        for j, cid in enumerate(detections['class_id']):
            if detections['classes'][int(cid)] != obj['category']:
                continue
            raw_box = np.asarray(detections['xyxy'][j], dtype=float)
            iou = box_iou(box, raw_box)
            all_same.append(dict(index=j, box_iou=iou,
                center_inside=bool(raw_box[0] <= uv[0] <= raw_box[2] and raw_box[1] <= uv[1] <= raw_box[3])))
            if iou < BOX_IOU:
                continue
            g = geometry.get(j)
            candidate = dict(raw_index=j, box_iou=iou, xyxy=raw_box.tolist(),
                confidence=float(detections['confidence'][j]),
                filter_reason=filters.get(j, {}).get('reason', 'not_recorded'), geometry=g)
            if g is not None and 'observation_center' in g:
                candidate['current_center_error_xy_m'] = xy_error(g['observation_center'], center)
            candidates.append(candidate)
        retained = [c for c in candidates if c['filter_reason'] == 'retained']
        valid = max((c['geometry']['depth_evidence']['valid_pixels'] for c in retained if c['geometry']), default=0)
        points = max((c['geometry']['depth_evidence']['retained_pixels'] for c in retained if c['geometry']), default=0)
        accurate = [c for c in candidates if c.get('current_center_error_xy_m', float('inf')) <= 3]
        same_pred = [(i, p, xy_error(p['center'], center)) for i, p in enumerate(preds) if p['category'] == obj['category']]
        within = [dict(prediction_index=i, node_id=p['node_id'], error_xy_m=e) for i, p, e in same_pred if e <= 3]
        match = recovered.get(ti)
        if match:
            reason = 'matched_R1'
        elif within:
            reason = 'output_within_3m_one_to_one_competition'
        elif not candidates:
            reason = 'no_iou025_same_class_2d_support'
        elif not retained:
            reason = 'overlap_exists_input_filter_rejected'
        elif valid == 0:
            reason = 'retained_mask_no_current_valid_depth'
        elif points == 0:
            reason = 'depth_selection_no_retained_points'
        elif points < 5:
            reason = 'sparse_1_to_4_points_requires_state_audit'
        elif not any(c['geometry'] and c['geometry']['reason'] == 'accepted' for c in retained):
            reason = 'dense_points_geometry_rejected'
        elif not accurate:
            reason = 'current_dense_center_outside_3m'
        else:
            reason = 'accurate_current_dense_but_final_output_missing'
        instance = obj['instance_token']
        first = instance not in first_seen
        first_seen.add(instance)
        dist = xy_error(center, pose[:3, 3])
        rows.append(dict(scene=frozen['scene'], log_token=log_token, frame_idx=frozen['frame_idx'],
            truth_index=ti, instance_token=instance, annotation_token=obj['annotation_token'],
            category=obj['category'], center=obj['center'], size_wlh=obj['size_wlh'],
            visibility_token=obj['visibility_token'], official_sample_lidar_pts=obj['num_lidar_pts'],
            distance_xy_m=dist, range_group='0_20m' if dist < 20 else '20_40m' if dist < 40 else '40m_plus',
            scene_first_frame=frozen['frame_idx'] == 0, first_instance_occurrence=first,
            gt_camera_z_m=float(camera[2]), gt_projected_center_uv=uv.tolist(), gt_projected_box=box.tolist() if box is not None else None,
            denominator_geometry_valid=bool(1 < camera[2] < 65 and 0 <= uv[0] < 1600 and 0 <= uv[1] < 900),
            projected_box_area=float(np.prod(box[2:] - box[:2])) if box is not None else 0.,
            projected_box_touches_border=bool(box is not None and (box[0] == 0 or box[1] == 0 or box[2] == 1600 or box[3] == 900)),
            matched=match is not None, saved_match=match, preliminary_reason=reason,
            same_class_detection_count=len(all_same), iou025_detection_count=len(candidates),
            any_overlap_same_class=any(c['box_iou'] > 0 for c in all_same),
            projected_center_in_same_class_box=any(c['center_inside'] for c in all_same),
            best_same_class_iou=max((c['box_iou'] for c in all_same), default=0),
            input_retained_count=len(retained), max_valid_mask_depth_points=valid,
            max_selected_mask_depth_points=points, depth_bin='0' if points == 0 else '1_to_4' if points < 5 else '5_plus',
            current_dense_accurate=bool(accurate), candidates=candidates,
            final_same_class_nearest_m=min((x[2] for x in same_pred), default=None),
            final_within_3m=within,
            source_records=dict(frozen_R1=frozen.get('_path'), raw_frame=source.get('_path'), **frozen['input'])))
    return rows


def summarize(rows):
    counts, groups, cross = Counter(), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        counts['truth'] += 1
        counts['matched'] += row['matched']
        counts[row.get('reason', row['preliminary_reason'])] += 1
        outcome = 'TP' if row['matched'] else 'FN'
        for label in ['same_class_detection_count', 'iou025_detection_count', 'input_retained_count']:
            cross[label + ':' + str(row[label] > 0)][outcome] += 1
        for label in ['any_overlap_same_class', 'projected_center_in_same_class_box', 'current_dense_accurate']:
            cross[label + ':' + str(row[label])][outcome] += 1
        for label in ['depth_bin', 'preliminary_reason']:
            cross[label + ':' + str(row[label])][outcome] += 1
        for key in ['category', 'range_group', 'visibility_token', 'scene_first_frame', 'first_instance_occurrence', 'log_token', 'scene']:
            value = groups[key + ':' + str(row[key])]
            value['truth'] += 1
            value[outcome] += 1
            value[row.get('reason', row['preliminary_reason'])] += 1
    return dict(counts=dict(counts), groups={k: dict(v) for k, v in groups.items()}, cross={k: dict(v) for k, v in cross.items()})
