"""补充评价：首次/曾经、确定选错/无法关联、几何进入/身份条件进入分别统计。"""
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from diagnose_carla_ten_bottlenecks_043 import sha

R = Path(__file__).resolve().parents[1]


def objects_to_truth(nodes, targets):
    if not nodes or not targets:
        return {}
    costs = np.array([[math.dist(n['center'][:2], t['center_xy'])
        if n['category'] == t['category'] else 1.e6 for t in targets] for n in nodes])
    ri, ci = linear_sum_assignment(costs)
    return {nodes[i]['node_id']: str(targets[j]['instance_id'])
            for i, j in zip(ri, ci) if costs[i, j] <= 2.5}


def main():
    out = R / 'carla_official/identity_geometry_audit_046'
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(p):
        inputs[str(p)] = sha(p)
        return json.loads(p.read_text())

    summary = []
    frames = []
    for number in range(1, 11):
        job = read(R / f'json/expansion_jobs_011_primary/job_{number:03d}.json')
        d = Path(job['episode_directory'])
        old = read(d / 'independent_expansion_audit_009/report.json')
        truth = read(d / 'evaluation_only/truth.json')
        T = truth['arrival_truth']
        accepted = set(map(str, T['accepted_instance_ids']))
        targets = [dict(instance_id=truth['target_actor_id'], category=truth['category'],
                        center_xy=truth['center_xy'])] + truth['other_controlled_actors']
        regions = [Polygon(r['polygon_xy']) for r in T['regions']]
        pp, ep = d / 'perception.jsonl.gz', d / 'execution.jsonl.gz'
        inputs[str(pp)] = sha(pp)
        inputs[str(ep)] = sha(ep)
        with gzip.open(pp, 'rt') as f:
            observations = [json.loads(line) for line in f]
        mappings = {p['frame_idx']: objects_to_truth(p['nodes'], targets) for p in observations}
        c = Counter()
        first = None
        raw_first = None
        for p in observations:
            m = mappings[p['frame_idx']]
            node = p['candidates'][0]['node_id'] if p['candidates'] else None
            raw = p['semantic_audit'].get('proposed_node_id')
            if raw is not None and raw_first is None:
                raw_first = dict(frame_idx=p['frame_idx'], node_id=raw,
                    match=m.get(raw), correct=m.get(raw) in accepted)
            if node is not None and first is None:
                first = dict(frame_idx=p['frame_idx'], node_id=node,
                    match=m.get(node), correct=m.get(node) in accepted)
            if node is not None and m.get(node) in accepted:
                c['ever_authorized_correct_proposal'] += 1
            if p['lock']['state_after'] == 'LOCKED' and p['selected_node_id'] is not None:
                known = m.get(p['selected_node_id'])
                if known in accepted:
                    c['verified_correct_lock_frames'] += 1
                elif known is None:
                    c['unassociated_lock_frames'] += 1
                else:
                    c['verified_other_object_lock_frames'] += 1
        assert bool(c['ever_authorized_correct_proposal']) == old['correct_first_proposal']
        assert bool(c['verified_other_object_lock_frames'] or c['unassociated_lock_frames']) == old['wrong_lock']
        cursor = 0
        geometric = Counter()
        min_inside_speed = math.inf
        max_geometric_hold = 0.
        held = None
        first_unassociated = None
        with gzip.open(ep, 'rt') as f:
            for line in f:
                e = json.loads(line)
                while cursor + 1 < len(observations) and observations[cursor + 1]['sensor_frame'] <= e['frame']:
                    cursor += 1
                p = observations[cursor]
                selected = p['selected_node_id']
                matched = mappings[p['frame_idx']].get(selected)
                state = e['state']
                xy = Point(state['x'], state['y'])
                inside = any(region.covers(xy) for region in regions)
                heading = any(region.covers(xy) and abs(math.atan2(
                    math.sin(state['yaw_rad'] - r['heading_rad']),
                    math.cos(state['yaw_rad'] - r['heading_rad']))) <= T['max_heading_error_rad']
                    for region, r in zip(regions, T['regions']))
                stopped = state['speed_mps'] <= T['max_speed_mps']
                if inside:
                    geometric['inside_ticks'] += 1
                    min_inside_speed = min(min_inside_speed, state['speed_mps'])
                    if matched not in accepted:
                        geometric['inside_without_confirmed_identity_ticks'] += 1
                    if stopped:
                        geometric['inside_and_stopped_ticks'] += 1
                    if not stopped:
                        geometric['inside_and_moving_ticks'] += 1
                if inside and stopped and heading:
                    held = state['time_s'] if held is None else held
                    max_geometric_hold = max(max_geometric_hold, state['time_s'] - held)
                else:
                    held = None
                if selected is not None and matched is None and first_unassociated is None:
                    node = next((n for n in p['nodes'] if n['node_id'] == selected), None)
                    first_unassociated = dict(tick=e['tick'], node_id=selected,
                        stable_identity=node['metadata'].get('object_identity') if node else None,
                        predicted_category=node['category'] if node else None,
                        distance_to_target_center_m=math.dist(node['center'][:2], truth['center_xy']) if node else None)
                if number == 6:
                    frames.append(dict(tick=e['tick'], selected_node_id=selected,
                        matched_truth_identity=matched, correct_identity=matched in accepted,
                        geometrically_inside_region=inside, geometry_heading_ok=heading,
                        stopped=stopped, original_inside_flag=e['evaluation']['inside_independent_region']))
        summary.append(dict(job=number, episode=job['episode'], task_kind=old['task_kind'],
            first_authorized_proposal_correct=first['correct'] if first else None,
            ever_authorized_proposal_correct=bool(c['ever_authorized_correct_proposal']),
            raw_semantic_first_proposal_correct=raw_first['correct'] if raw_first else None,
            raw_semantic_first_proposal=raw_first, first_authorized_proposal=first,
            **{k: c[k] for k in ['verified_correct_lock_frames',
                'verified_other_object_lock_frames', 'unassociated_lock_frames']},
            physical_geometric_region_ticks=geometric['inside_ticks'],
            geometric_region_without_confirmed_identity_ticks=geometric['inside_without_confirmed_identity_ticks'],
            geometric_region_stopped_ticks=geometric['inside_and_stopped_ticks'],
            min_speed_geometrically_inside_mps=min_inside_speed if math.isfinite(min_inside_speed) else None,
            geometric_stop_heading_max_hold_s=max_geometric_hold,
            first_unassociated_lock=first_unassociated,
            original_strict_safe_arrival=old['strict_safe_arrival'],
            original_wrong_lock_including_unassociated=old['wrong_lock']))
    fields = [k for k in summary[0] if k not in ['raw_semantic_first_proposal',
        'first_authorized_proposal', 'first_unassociated_lock']]
    with (out / 'episodes.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(summary)
    with (out / 'job06_geometry_identity.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(frames[0]))
        w.writeheader()
        w.writerows(frames)
    report = dict(episodes=summary,
        original_metrics_unchanged=True,
        definitions=dict(old_correct_first_proposal='曾经至少一次授权后的首位候选正确，并非第一次提议',
            old_wrong_lock='确定关联到其他对象，或不能关联到任何已登记真值对象',
            old_inside_independent_region='当前所选对象身份匹配的区域内；不是无条件的几何进入',
            new_unassociated_lock='2.5m同类一对一关联门限内没有真值匹配；未知，不授予正确或确定错误信用',
            geometric_hold='仅位置、方向和速度；不满足身份与完整安全条件时仍不授予到达'),
        inference_receives_GT=False, actual_new_CARLA_calls=0, actual_new_GPU_calls=0,
        inputs_sha256=inputs, source_sha256=sha(__file__), goal_complete=False)
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps([{k: r[k] for k in fields} for r in summary], ensure_ascii=False))


if __name__ == '__main__':
    main()
