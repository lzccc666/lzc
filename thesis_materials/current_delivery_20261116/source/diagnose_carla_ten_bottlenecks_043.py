"""只读分析首批10个官方回合；真值仅在本诊断器隔离读取，不进入智能体。"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

R = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stream(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def path_key(path):
    return hashlib.sha256(json.dumps(path, separators=(',', ':')).encode()).hexdigest()


def main():
    out = R / 'carla_official/ten_bottlenecks_043'
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(p):
        inputs[str(p)] = sha(p)
        return json.loads(p.read_text())

    summaries = []
    trace = []
    projections = []
    for number in range(1, 11):
        job = read(R / f'json/expansion_jobs_011_primary/job_{number:03d}.json')
        assert job['exit_code'] == 0 and all(a['exit_code'] == 0 for a in job['audits'])
        d = Path(job['episode_directory'])
        audit = read(d / 'independent_expansion_audit_009/report.json')
        truth = read(d / 'evaluation_only/truth.json')
        read(d / 'release.json')
        execution_path, perception_path = d / 'execution.jsonl.gz', d / 'perception.jsonl.gz'
        inputs[str(execution_path)] = sha(execution_path)
        inputs[str(perception_path)] = sha(perception_path)
        observations = list(stream(perception_path))
        plan_passes = 0
        path_revisions = 0
        same_identity_refresh = 0
        last_id = None
        last_path = None
        for p in observations:
            current = path_key(p['path_world'])
            if last_path is not None and current != last_path:
                path_revisions += 1
                same_identity_refresh += int(last_id == p['selected_node_id']
                    and p['selected_node_id'] is not None and bool(p['path_world']))
            check = p['semantic_audit'].get('plan_check') or {}
            plan_passes += bool(check.get('passed'))
            last_path, last_id = current, p['selected_node_id']
            if number == 7:
                anchor = truth['other_controlled_actors'][0]
                # 隔离上限诊断：车辆外形水平轮廓的0—4m高保守包围柱。
                # 不是精确3D box；全部包围柱在视野外足以证明参照车不在前相机中。
                column = np.array([[x, y, z, 1.] for x, y in anchor['polygon_xy']
                                   for z in (0., 4.)])
                cam = (np.linalg.inv(p['camera_pose_cv']) @ column.T).T[:, :3]
                visible = cam[:, 2] > .1
                uv = cam[visible, :2] / cam[visible, 2, None] * 320 + [320., 180.]
                all_left = bool(len(uv) == len(column) and np.max(uv[:, 0]) < 0)
                projections.append(dict(frame_idx=p['frame_idx'],
                    u_min=float(np.min(uv[:, 0])), u_max=float(np.max(uv[:, 0])),
                    entire_conservative_column_left_of_front_camera=all_left,
                    graph_has_truck=any(n['category'] == 'truck' for n in p['nodes']),
                    search_suppressed=p.get('pending_semantics_search_suppressed', False)))
        cursor = 0
        checks = Counter()
        reasons = Counter()
        ticks = 0
        final = None
        for e in stream(execution_path):
            while cursor + 1 < len(observations) and observations[cursor + 1]['sensor_frame'] <= e['frame']:
                cursor += 1
            p = observations[cursor]
            for k, v in e['evaluation'].items():
                if isinstance(v, bool):
                    checks[k] += v
            g = e['tracking']['always_on_motion_guard']
            if g['stop_required']:
                for kind in ('route_following_forecast', 'current_steering_coasting_forecast'):
                    reasons[(kind, g[kind].get('reason'))] += 1
            ticks += 1
            final = e
            if number in (2, 5, 6, 10):
                trace.append(dict(job=number, tick=e['tick'], time_s=e['state']['time_s'],
                    selected=p['selected_node_id'], plan_passed=p['semantic_audit']['plan_check'].get('passed'),
                    plan_failure=p['semantic_audit']['plan_check'].get('failure'),
                    active_path_sha256=path_key(p['path_world']), active_path_points=len(p['path_world']),
                    speed_mps=e['state']['speed_mps'], motion_guard_stop=g['stop_required'],
                    cross_track_error_m=e['tracking'].get('cross_track_error_m'),
                    next_throttle=e['next_control']['throttle'], next_brake=e['next_control']['brake'],
                    inside_independent_region=e['evaluation']['inside_independent_region'],
                    heading_ok=e['evaluation']['heading_ok'], stopped=e['evaluation']['stopped']))
        assert ticks == audit['physical_ticks']
        summaries.append(dict(job=number, episode=job['episode'], task_kind=audit['task_kind'],
            status=audit['status'], strict_safe_arrival=audit['strict_safe_arrival'],
            physical_ticks=ticks, evaluation_condition_ticks=dict(checks),
            plan_checks_passed=plan_passes, observed_active_path_revisions=path_revisions,
            active_path_refreshes_without_identity_change=same_identity_refresh,
            guard_reasons=[dict(forecast=k[0], reason=k[1], ticks=v) for k, v in reasons.items()],
            final_pose=final['state'], final_evaluation=final['evaluation']))
    for name, rows in [('timeline.csv', trace), ('reference_projection.csv', projections)]:
        with (out / name).open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report = dict(episodes=summaries,
        reference_projection=dict(frames=len(projections),
            whole_column_outside_front_camera_frames=sum(p['entire_conservative_column_left_of_front_camera'] for p in projections),
            graph_has_reference_frames=sum(p['graph_has_truck'] for p in projections),
            max_projected_u_px=max(p['u_max'] for p in projections),
            interpretation='这是相机视野覆盖缺失证据，不是检测模型在可见卡车上漏检的证据；未使用真值填充推理图'),
        code_contract_issue='原执行端只有plan_target!=selected时接收prepared_plans；当前身份不变时可能丢弃本帧已经重新检查的计划。测试修复不能自动获得实际到达信用。',
        causal_claim=False, actual_new_CARLA_calls=0, actual_new_GPU_calls=0,
        goal_complete=False, inputs_sha256=inputs, source_sha256=sha(__file__))
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(reference_projection=report['reference_projection'],
        positive=[{k: e[k] for k in ['job', 'plan_checks_passed',
            'active_path_refreshes_without_identity_change', 'evaluation_condition_ticks']}
            for e in summaries if e['task_kind'] == 'positive']), ensure_ascii=False))


if __name__ == '__main__':
    main()
