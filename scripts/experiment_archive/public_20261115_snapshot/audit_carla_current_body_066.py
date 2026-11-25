"""用保存的当前观测逐帧独立拆分初始车身不合法原因；不运行仿真或规划器。"""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'carla_official/current_body_causes_066'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rows(path):
    with gzip.open(path, 'rt') as f:
        yield from map(json.loads, f)


def main():
    OUT.mkdir(exist_ok=False)
    begun = time.monotonic()
    inputs = {str(Path(__file__)): sha(__file__)}
    geometry = Path('/root/autodl-tmp/lizichong/conceptgraph/navigation/vehicle_geometry.py')
    inputs[str(geometry)] = sha(geometry)
    episodes = []
    for number in [1, 2, 5, 6, 10]:
        jobfile = ROOT / f'json/expansion_jobs_011_primary/job_{number:03d}.json'
        job = json.loads(jobfile.read_text())
        folder = Path(job['episode_directory'])
        for p in [jobfile, folder/'release.json', folder/'perception.jsonl.gz', folder/'execution.jsonl.gz']:
            inputs[str(p)] = sha(p)
        episodes.append((number, folder))
    write(OUT/'protocol.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=inputs, episode_numbers=[e[0] for e in episodes],
        selection='第1个成功作为参照，加全部4个正确选中/规划但未到达的已运行正回合',
        algorithm='独立GEOS当前车身与已声明道路/当前障碍相交；不调用规划器，不读取目标身份真值',
        frozen_padding_m=.05, nominal_body_padding_m=0.,
        frame_alignment='perception.sensor_frame与execution.frame完全相等，不插值、不读取未来',
        CPU_budget_s=240, output_budget_bytes=16*1024**2,
        actual_new_CARLA_calls=0, actual_new_GPU_calls=0,
        arrival_or_counterfactual_credit=0, causal_claim=False, goal_complete=False))
    summaries = []
    with (OUT/'per_frame.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = None
        for number, folder in episodes:
            release = json.loads((folder/'release.json').read_text())
            road = Polygon(release['prior']['road_polygon_xy'])
            reference = LineString(release['prior']['centerline_xy'])
            road_check = prep(road)
            spec = VehicleSpec(**release['vehicle_spec'])
            execution = iter(rows(folder/'execution.jsonl.gz'))
            current = next(execution)
            counts, causes, event_samples = Counter(), Counter(), []
            first_invalid = None
            for observation in rows(folder/'perception.jsonl.gz'):
                assert time.monotonic()-begun < 240
                while current['frame'] < observation['sensor_frame']:
                    current = next(execution)
                assert current['frame'] == observation['sensor_frame']
                state = VehicleState(**current['state'])
                nominal = footprint_polygon(state, spec, 0.)
                body = footprint_polygon(state, spec, .05)
                polygons = [Polygon(p) for p in observation['known_obstacle_polygons']]
                assert all(p.is_valid for p in polygons)
                outside = not road_check.covers(body)
                nominal_outside = not road_check.covers(nominal)
                intersecting = [p for p in polygons if p.intersects(body)]
                nominal_intersections = [p for p in polygons if p.intersects(nominal)]
                cause = ('road_and_current_obstacle' if outside and intersecting else
                         'road_only' if outside else 'current_obstacle_only' if intersecting else 'current_body_legal')
                check = observation['semantic_audit'].get('plan_check') or {}
                reported = check.get('failure') == 'INVALID_INITIAL_BODY'
                progress = reference.project(Point(state.x, state.y))
                before = reference.interpolate(max(0, progress-.2))
                after = reference.interpolate(min(reference.length, progress+.2))
                lane_yaw = math.atan2(after.y-before.y, after.x-before.x)
                yaw_error = math.atan2(math.sin(state.yaw_rad-lane_yaw), math.cos(state.yaw_rad-lane_yaw))
                row = dict(job=number, frame_idx=observation['frame_idx'], sensor_frame=current['frame'],
                    sim_time_s=state.time_s, planner_failure=check.get('failure'),
                    reported_invalid_initial_body=reported, current_body_cause=cause,
                    padded_body_outside_road=outside, nominal_body_outside_road=nominal_outside,
                    padding_only_road_violation=outside and not nominal_outside,
                    padded_obstacle_intersections=len(intersecting),
                    nominal_obstacle_intersections=len(nominal_intersections),
                    maximum_obstacle_overlap_area_m2=max([body.intersection(p).area for p in intersecting] or [0.]),
                    outside_road_area_m2=body.difference(road).area,
                    boundary_distance_m=body.distance(road.boundary),
                    lateral_centerline_distance_m=reference.distance(Point(state.x,state.y)),
                    heading_error_to_lane_deg=math.degrees(yaw_error), speed_mps=state.speed_mps,
                    motion_guard_stopped=current['tracking']['always_on_motion_guard']['stop_required'],
                    selected_node=observation['selected_node_id'], active_path_points=len(observation['path_world']))
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                counts['observed_frames'] += 1
                counts['reported_invalid_initial_body_frames'] += reported
                counts['current_body_illegal_frames'] += cause != 'current_body_legal'
                counts['reported_invalid_reproduced_frames'] += reported and cause != 'current_body_legal'
                counts['reported_invalid_not_reproduced_frames'] += reported and cause == 'current_body_legal'
                counts['nominal_body_outside_road_frames'] += nominal_outside
                counts['padding_only_road_violation_frames'] += outside and not nominal_outside
                counts['nominal_body_intersects_current_obstacles_frames'] += bool(nominal_intersections)
                if reported:
                    causes[cause] += 1
                    if first_invalid is None:
                        first_invalid = row
                if len(event_samples) < 4 and cause != 'current_body_legal':
                    event_samples.append(row)
            summary = dict(job=number, episode=folder.parts[-4], counts=dict(counts),
                reported_initial_failure_causes=dict(causes), first_reported_invalid=first_invalid,
                first_illegal_frames=event_samples)
            summaries.append(summary)
            print(json.dumps(dict(job=number, counts=dict(counts), causes=dict(causes))), flush=True)
    unchanged = all(sha(p) == h for p, h in inputs.items())
    assert unchanged
    write(OUT/'report.json', dict(UTC=datetime.now(timezone.utc).isoformat(), episodes=summaries,
        observed_frames=sum(s['counts']['observed_frames'] for s in summaries),
        reported_invalid_frames=sum(s['counts']['reported_invalid_initial_body_frames'] for s in summaries),
        reproduced_invalid_frames=sum(s['counts']['reported_invalid_reproduced_frames'] for s in summaries),
        inputs_unchanged=unchanged, source_protocol_sha256=sha(OUT/'protocol.json'),
        per_frame_sha256=sha(OUT/'per_frame.csv'), elapsed_s=time.monotonic()-begun,
        actual_new_CARLA_calls=0, actual_new_GPU_calls=0, new_arrival_credit=0,
        conclusions_scope='只拆分当前观测下的几何拒绝原因，不能证明障碍真实存在、唯一根因或修复后轨迹',
        no_GT_into_agent=True, causal_claim=False, goal_complete=False))
    assert sum(p.stat().st_size for p in OUT.iterdir()) < 16*1024**2


if __name__ == '__main__':
    main()
