"""固定38开发场景的HD有向连接任务；不运行规划器或按成功选分支。"""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import time

import numpy as np
from shapely.geometry import LineString

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def polyline_slice(points, start_s, end_s):
    points = np.asarray(points)[:, :2]
    line = LineString(points)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    a, b = line.interpolate(start_s), line.interpolate(end_s)
    return [[a.x, a.y]]+points[(distances > start_s)&(distances < end_s)].tolist()+[[b.x, b.y]]


def main():
    started = time.monotonic()
    source = OUTPUT/'ablations/causal_maps_full_001'
    parent_release = json.loads((source/'release.json').read_text())
    root = OUTPUT/'vehicle_execution/HDmap_connection_tasks_001'
    root.mkdir(exist_ok=False)
    contexts = {}
    for path in (source/'contexts').glob('*/context.json'):
        data = json.loads(path.read_text())
        if data['scene'] not in contexts or data['frame_idx'] < contexts[data['scene']][1]['frame_idx']:
            contexts[data['scene']] = (path, data)
    assert set(contexts) == set(parent_release['scene_logs']) and len(contexts) == 38
    protocol = OUTPUT/'task_protocol/HDmap_vehicle_connection_001.md'
    dump(root/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), scenes=38,
        source_module_sha256=digest(__file__), protocol_sha256=digest(protocol),
        parent_release_sha256=digest(source/'release.json'), map_sha256=parent_release['map_sha256'],
        all_context_sha256={str(path): digest(path) for path, _ in contexts.values()},
        wall_budget_s=300, storage_budget_bytes=15000000, method_run=False,
        independent_test_opened=False, goal_complete=False))
    rows = []
    for scene, (path, context) in sorted(contexts.items()):
        assert time.monotonic()-started < 300
        pose = np.asarray(context['camera_pose'])
        xy, yaw = pose[:2, 3], math.atan2(pose[1, 2], pose[0, 2])
        geometries = context['lane_geometries']
        eligible = []
        for token, lane in geometries.items():
            points = np.asarray(lane['centerline'])
            if lane['lane_type'] != 'lane' or len(points) < 2:
                continue
            index = int(np.argmin(np.linalg.norm(points[:, :2]-xy, axis=1)))
            distance = float(np.linalg.norm(points[index, :2]-xy))
            if distance <= 10 and math.cos(points[index, 2]-yaw) >= .5:
                eligible.append((distance, token))
        record = dict(task_id=scene+'-HD-connection', scene=scene, frame_idx=context['frame_idx'],
            log_token=parent_release['scene_logs'][scene], context_path=str(path), context_sha256=digest(path),
            camera_xy=xy.tolist(), camera_yaw_rad=yaw, semantic_arrival_credit=0,
            artificial_rear_axle_initialization=True, current_static_predicted_obstacles=True)
        if not eligible:
            record['status'] = 'unknown_no_aligned_nearby_entry_lane'
        else:
            distance, entry_token = min(eligible)
            record.update(entry_lane_token=entry_token, entry_distance_from_current_camera_m=distance)
            connectors = sorted(t for t in context['transitions'].get(entry_token, [])
                                if t in geometries and geometries[t]['lane_type'] == 'lane_connector')
            if not connectors:
                record['status'] = 'unknown_no_declared_outgoing_connector'
            else:
                connector_token = connectors[0]
                exits = sorted(t for t in context['transitions'].get(connector_token, [])
                               if t in geometries and geometries[t]['lane_type'] == 'lane')
                record['connector_token'] = connector_token
                if not exits:
                    record['status'] = 'unknown_connector_has_no_declared_exit_lane'
                else:
                    exit_token = exits[0]
                    chain = [geometries[t] for t in [entry_token, connector_token, exit_token]]
                    first = LineString(np.asarray(chain[0]['centerline'])[:, :2])
                    last = LineString(np.asarray(chain[-1]['centerline'])[:, :2])
                    record.update(exit_lane_token=exit_token, lane_tokens=[g['token'] for g in chain])
                    if min(first.length, last.length) < 6:
                        record['status'] = 'unknown_entry_or_exit_lane_too_short'
                    else:
                        start_s, end_s = max(2., first.length-8), min(10., last.length-4)
                        start = first.interpolate(start_s)
                        goal = last.interpolate(end_s)
                        h0 = np.array(first.interpolate(start_s+.1).coords[0])-np.array(first.interpolate(start_s-.1).coords[0])
                        h1 = np.array(last.interpolate(end_s+.1).coords[0])-np.array(last.interpolate(end_s-.1).coords[0])
                        reference = polyline_slice(chain[0]['centerline'], start_s, first.length)
                        reference += np.asarray(chain[1]['centerline'])[:, :2].tolist()
                        reference += polyline_slice(chain[-1]['centerline'], 0., end_s)
                        # 只删除参考线连续重复点；不修补地图多边形、连接或原车道几何。
                        reference = [p for i, p in enumerate(reference) if not i or math.dist(p, reference[i-1]) > 1e-10]
                        record.update(start_xy=[start.x, start.y], start_yaw_rad=math.atan2(h0[1], h0[0]),
                            goal_xy=[goal.x, goal.y], goal_yaw_rad=math.atan2(h1[1], h1[0]),
                            artificial_start_distance_to_recorded_camera_m=math.dist([start.x, start.y], xy),
                            reference_centerline_xy=reference, reference_length_m=LineString(reference).length)
                        record['status'] = 'ready_given_map_route' if record['artificial_start_distance_to_recorded_camera_m'] <= 30 else 'unknown_artificial_start_exceeds_30m_bound'
        rows.append(record)
    dump(root/'tasks.json', rows)
    dump(root/'manifest.json', dict(tasks=38, logs=len({r['log_token'] for r in rows}),
        status_counts=dict(Counter(r['status'] for r in rows)), tasks_sha256=digest(root/'tasks.json'),
        seeds=[17, 29, 43], methods=['hybrid_astar', 'point_astar'], expected_trials=228,
        method_inference_executed=False, semantic_targets_evaluated=False, semantic_arrival_credit=0,
        independent_test_opened=False, source_protocol_sha256=digest(protocol), goal_complete=False))
    print((root/'manifest.json').read_text())


if __name__ == '__main__':
    main()
