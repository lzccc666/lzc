"""Freeze official-instance follow tasks and independently audit recorded witnesses.

This is a geometry/time benchmark preparation tool, not a navigation method.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import math
from pathlib import Path
import shutil
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import unary_union

from conceptgraph.completion.navbench_stage1_20261124 import (
    ROOT, canonical, content_sha, dump, jsonl, now, read, rows, sha, source_ref,
)
from conceptgraph.completion.reference_regions import load_reference_prior, object_polygon
from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, angle_difference, footprint_polygon, swept_body_polygon,
)
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import experiment_directory

RUN_ID = 'navsemantic_tasks_20261124_v3'
DEST = ROOT / 'dataset/processed' / RUN_ID
REPORT = ROOT / 'outputs/reports' / RUN_ID
META = ROOT / 'dataset/raw/nuscenes_trainval/v1.0-trainval'
SENSOR = ROOT / 'dataset/processed/mv136_20261123_sixscene_v1/manifest.json'
PROTECTED = ROOT / 'dataset/processed/navbench_stage1_20261124_v1'
SCENES = ['scene-0012', 'scene-0013', 'scene-0019', 'scene-0035', 'scene-0042', 'scene-0047']
POLICY = dict(seed=17, scenes=SCENES, issue_frames=[0, 10], horizon_s=3.,
    output_mode='timed_xy_yaw_trajectory', reference_step_s=.1,
    target_categories=['vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid'],
    visible_front_channels=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    minimum_visibility_token=2, minimum_initial_forward_distance_m=5.,
    maximum_initial_forward_distance_m=60., minimum_longitudinal_body_gap_m=5.,
    maximum_longitudinal_body_gap_m=60., maximum_heading_difference_deg=30.,
    maximum_lane_hops=8, maximum_lane_search_m=100., maximum_lane_join_gap_m=.75,
    minimum_ego_forward_progress_m=.5,
    vehicle={key:value for key,value in asdict(VehicleSpec()).items() if key in
        ['wheelbase_m','front_from_rear_axle_m','rear_overhang_m','width_m']},
    engineering_reference='Official ego_pose origin is declared rear axle in this engineering model; xy and yaw retained, z projected out. No actual nuScenes axle measurement is claimed.',
    initial_state='Known official LIDAR_TOP keyframe ego_pose at issue time; no from-rest assumption or forced stop. No controls/acceleration/bicycle-execution claim.',
    geometry='Whole engineering vehicle on raw official drivable with holes; same directed lane route and all-category official obstacle endpoint convex swept envelopes in every subinterval.',
    dynamics_scope='Evaluation-only official 2Hz boxes; vertices interpolate linearly between anchors. Unmatched endpoint boxes conservatively occupy the full anchor interval. No unobserved-real-traffic guarantee.',
    witness_role='Recorded ego future is only an attempted existence witness after checking this contract, never automatically a correct answer or a unique scoring trajectory.',
    admission='All 12 fixed candidates retained. Only geometry/time independently passed candidates may enter a NEW geometry-time conditional feasible table, never the old main table or historical 81.',
    source_roles='Current labels/geometry evaluation-only. Method inputs contain instruction, current/past sensor references and known initial pose, not official target identity, future sequence, annotations or witness.',
    budget=dict(max_elapsed_s=1800, max_generated_bytes=64*1024**2, GPU=0, model_calls=0, planner_calls=0))


def transform(row):
    w, x, y, z = np.asarray(row['rotation'], float)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = row['translation']
    return result


def yaw(row):
    matrix = transform(row)
    return math.atan2(matrix[1, 0], matrix[0, 0])


def lane_at(xy, heading, prior):
    candidates = []
    for lane in prior['lanes']:
        polygon = Polygon(lane['polygon'])
        if not polygon.covers(Point(xy)):
            continue
        line = LineString(np.asarray(lane['centerline'])[:, :2])
        arc = line.project(Point(xy))
        a, b = line.interpolate(max(0., arc-.25)), line.interpolate(min(line.length, arc+.25))
        tangent = math.atan2(b.y-a.y, b.x-a.x)
        if abs(angle_difference(heading, tangent)) <= math.radians(POLICY['maximum_heading_difference_deg']):
            candidates.append((line.distance(Point(xy)), lane['token'], lane))
    return min(candidates, key=lambda v: v[:2])[2] if candidates else None


def directed_routes(start_lane, prior, start_xy):
    by_token = {l['token']: l for l in prior['lanes']}
    queue = deque([[start_lane['token']]])
    routes = {}
    while queue:
        route = queue.popleft()
        token = route[-1]
        if token in routes:
            continue
        arrays = [np.asarray(by_token[t]['centerline'])[:, :2] for t in route]
        line = LineString(np.concatenate(arrays))
        routes[token] = dict(tokens=route, line=line,
            polygon=unary_union([Polygon(by_token[t]['polygon']) for t in route]))
        if len(route)-1 >= POLICY['maximum_lane_hops'] or line.length-line.project(Point(start_xy)) > POLICY['maximum_lane_search_m']:
            continue
        for next_token in sorted(by_token[token]['outgoing']):
            if next_token in by_token and next_token not in route:
                if math.dist(by_token[token]['end_pose'][:2], by_token[next_token]['start_pose'][:2]) <= POLICY['maximum_lane_join_gap_m']:
                    queue.append(route+[next_token])
    return routes


def visible(obj, group):
    for channel in POLICY['visible_front_channels']:
        view = group['views'][channel]
        pose = np.asarray(view['camera_pose'])
        camera_xyz = pose[:3, :3].T @ (np.asarray(obj['translation'])-pose[:3, 3])
        if camera_xyz[2] <= 1:
            continue
        uvz = np.asarray(view['intrinsic']) @ camera_xyz
        u, v = uvz[:2]/uvz[2]
        if 0 <= u < view['width'] and 0 <= v < view['height']:
            return True
    return False


def choose_target(objects, ego_xy, ego_yaw, group, prior, categories):
    lane = lane_at(ego_xy, ego_yaw, prior)
    if lane is None:
        return None, None, 'initial_rear_axle_not_on_heading_compatible_directed_lane'
    routes = directed_routes(lane, prior, ego_xy)
    candidates = []
    for obj in objects:
        if categories[obj['instance_token']] not in POLICY['target_categories']:
            continue
        if int(obj['visibility_token']) < POLICY['minimum_visibility_token'] or not visible(obj, group):
            continue
        target_lane = lane_at(obj['translation'][:2], yaw(obj), prior)
        if target_lane is None or target_lane['token'] not in routes:
            continue
        route = routes[target_lane['token']]
        distance = route['line'].project(Point(obj['translation'][:2]))-route['line'].project(Point(ego_xy))
        if POLICY['minimum_initial_forward_distance_m'] <= distance <= POLICY['maximum_initial_forward_distance_m']:
            candidates.append((distance, obj['instance_token'], obj, route))
    if not candidates:
        return None, None, 'no_visible_official_forward_vehicle_on_same_directed_lane_route'
    _, _, target, route = min(candidates, key=lambda t:t[:2])
    # Following continues through declared directed successor lanes; the initial
    # target's lane end is not a semantic endpoint or a stopping goal.
    prefix=route['tokens']
    route=dict(route,continuations=[r for r in routes.values() if r['tokens'][:len(prefix)]==prefix])
    return target, route, None


def route_measure(route, ego_xy, target_xy):
    common=[]
    for candidate in route['continuations']:
        if candidate['polygon'].covers(Point(ego_xy)) and candidate['polygon'].covers(Point(target_xy)):
            ego_arc=candidate['line'].project(Point(ego_xy))
            target_arc=candidate['line'].project(Point(target_xy))
            common.append((candidate['line'].distance(Point(ego_xy))+candidate['line'].distance(Point(target_xy)),
                candidate['tokens'],target_arc-ego_arc))
    if not common:return False,None
    return True,min(common,key=lambda r:r[:2])[2]


def verify_witness(anchor_frames, initial, target, route, prior, spec):
    failures = []; checks = []; states = []; full_future = []
    origin_t = anchor_frames[0]['timestamp_s']
    for frame in anchor_frames:
        objects = {o['instance_token']:o for o in frame['objects']}
        if target['instance_token'] not in objects:
            failures.append('target_identity_missing_in_reference_window')
        full_future.append(dict(time_s=frame['timestamp_s']-origin_t, sample_token=frame['sample_token'],
            lidar_token=frame['lidar']['token'], official_ego_pose=frame['ego_pose'], objects=frame['objects']))
    if failures:
        return dict(passed=False, failures=sorted(set(failures)), checks=checks), states, full_future
    start = anchor_frames[0]
    assert np.allclose(initial, transform(start['ego_pose']))
    initial_state=VehicleState(float(initial[0,3]),float(initial[1,3]),yaw(start['ego_pose']))
    initial_body=footprint_polygon(initial_state,spec)
    initial_objects={o['instance_token']:o for o in start['objects']}
    initial_gap=(route['line'].project(Point(target['translation'][:2]))
        -route['line'].project(Point(initial[:2,3]))-spec.front_from_rear_axle_m-target['size'][1]/2)
    initial_checks=dict(time_s=0.,road_sweep=bool(prior['drivable'].covers(initial_body)),
        obstacle_sweep=all(not initial_body.intersects(object_polygon(o)) for o in initial_objects.values()),
        same_directed_route=bool(route['polygon'].covers(Point(initial[:2,3]))),heading_compatible=True,
        longitudinal_body_gap_m=float(initial_gap),
        gap_accepted=POLICY['minimum_longitudinal_body_gap_m']<=initial_gap<=POLICY['maximum_longitudinal_body_gap_m'])
    checks.append(initial_checks)
    for key in ['road_sweep','obstacle_sweep','same_directed_route','gap_accepted']:
        if not initial_checks[key]:failures.append('initial_'+key)
    for left, right in zip(anchor_frames, anchor_frames[1:]):
        begin, end = left['timestamp_s']-origin_t, min(right['timestamp_s']-origin_t, POLICY['horizon_s'])
        if begin >= POLICY['horizon_s']:
            break
        dt = right['timestamp_s']-left['timestamp_s']
        if not 0 < dt <= .6:
            failures.append('irregular_or_missing_official_keyframe_interval')
        a, b = left['ego_pose'], right['ego_pose']
        y0, y1 = yaw(a), yaw(b)
        grid = np.linspace(begin, end, max(1, math.ceil((end-begin)/POLICY['reference_step_s']))+1)
        local = []
        for t in grid:
            alpha = (t-begin)/dt
            xy = (1-alpha)*np.asarray(a['translation'][:2])+alpha*np.asarray(b['translation'][:2])
            heading = y0+alpha*angle_difference(y1,y0)
            state = VehicleState(float(xy[0]), float(xy[1]), heading, time_s=float(t))
            local.append(state)
            if not states or abs(t-states[-1]['time_s']) > 1e-8:
                states.append(asdict(state))
        lo = {o['instance_token']:o for o in left['objects']}
        ro = {o['instance_token']:o for o in right['objects']}
        obstacle_envelopes = [unary_union([object_polygon(z[tok]) for z in [lo,ro] if tok in z]).convex_hull
                              for tok in sorted(lo.keys() | ro.keys())]
        for before, after in zip(local, local[1:]):
            sweep = swept_body_polygon(before, after, spec)
            body = footprint_polygon(after, spec)
            road_ok = bool(prior['drivable'].covers(sweep))
            obstacle_ok = all(not sweep.intersects(p) for p in obstacle_envelopes)
            alpha = (after.time_s-begin)/dt
            target_xy = (1-alpha)*np.asarray(lo[target['instance_token']]['translation'][:2])+alpha*np.asarray(ro[target['instance_token']]['translation'][:2])
            target_heading = yaw(lo[target['instance_token']])+alpha*angle_difference(yaw(ro[target['instance_token']]),yaw(lo[target['instance_token']]))
            same_route,center_distance=route_measure(route,[after.x,after.y],target_xy)
            target_length = max(lo[target['instance_token']]['size'][1],ro[target['instance_token']]['size'][1])
            gap = center_distance-spec.front_from_rear_axle_m-target_length/2 if same_route else None
            heading_ok = bool(abs(angle_difference(after.yaw_rad,target_heading)) <= math.radians(POLICY['maximum_heading_difference_deg']))
            gap_ok = bool(gap is not None and POLICY['minimum_longitudinal_body_gap_m'] <= gap <= POLICY['maximum_longitudinal_body_gap_m'])
            checks.append(dict(time_s=after.time_s, road_sweep=road_ok, obstacle_sweep=obstacle_ok,
                same_directed_route=same_route, heading_compatible=heading_ok,
                longitudinal_body_gap_m=float(gap) if gap is not None else None, gap_accepted=gap_ok))
            for key, passed in [('road_sweep',road_ok),('obstacle_sweep',obstacle_ok),
                                ('same_directed_route',same_route),('heading',heading_ok),('gap',gap_ok)]:
                if not passed:
                    failures.append(key)
    if not states or states[-1]['time_s'] < POLICY['horizon_s']-1e-8:
        failures.append('insufficient_reference_horizon')
    elif route['line'].project(Point(states[-1]['x'],states[-1]['y']))-route['line'].project(Point(states[0]['x'],states[0]['y'])) < POLICY['minimum_ego_forward_progress_m']:
        failures.append('no_forward_follow_progress')
    return dict(passed=not failures, failures=sorted(set(failures)), checks=checks,
        scope='Geometry/time witness under declared box interpolation, engineering rear-axle and drivable-map model; no bicycle controls or real execution certification.'), states, full_future


def main():
    begun = time.monotonic()
    assert shutil.disk_usage(ROOT).free > 20*1024**3
    protected = {str(p):sha(p) for p in PROTECTED.rglob('*') if p.is_file()}
    run = experiment_directory('nuscenes','validation',RUN_ID,create=True)
    DEST.mkdir(exist_ok=False); REPORT.mkdir(exist_ok=False)
    dump(DEST/'preregistration.json',dict(created_utc=now(),policy=POLICY,
        source_manifest=source_ref(SENSOR), all_fixed_candidates_retained=True,
        old_registry=source_ref(PROTECTED/'task_registry.jsonl'),old_denominator=source_ref(PROTECTED/'denominator_freeze.json')))
    manifests = {s['scene']:s for s in read(SENSOR)['scenes']}
    selected_groups = {g['sample_token']:(name,g) for name in SCENES for g in manifests[name]['groups']}
    cache=ROOT/'dataset/processed/navsemantic_tasks_20261124_v2/evaluation_only/official_selected_frames.jsonl'
    if cache.exists():
        cache_manifest=read(cache.parent.parent/'manifest.json')
        assert sha(cache)==cache_manifest['files']['evaluation_only/official_selected_frames.jsonl']
        samples={r['sample_token']:r for r in rows(cache)}
        assert set(samples)==set(selected_groups)
    else:
        lidar = {r['sample_token']:r for r in iter_records(META/'sample_data.json')
                 if r['sample_token'] in selected_groups and r['is_key_frame'] and '/LIDAR_TOP/' in r['filename']}
        pose_ids = {r['ego_pose_token'] for r in lidar.values()}
        poses = {r['token']:r for r in iter_records(META/'ego_pose.json') if r['token'] in pose_ids}
        samples = {tok:dict(scene=name,frame_idx=g['frame_idx'],sample_token=tok,
            timestamp_s=lidar[tok]['timestamp']/1e6,lidar=lidar[tok],ego_pose=poses[lidar[tok]['ego_pose_token']],objects=[])
            for tok,(name,g) in selected_groups.items()}
    categories = {r['token']:r['name'] for r in iter_records(META/'category.json')}
    instances = {r['token']:categories[r['category_token']] for r in iter_records(META/'instance.json')}
    if not cache.exists():
        for obj in iter_records(META/'sample_annotation.json'):
            if obj['sample_token'] in samples:
                samples[obj['sample_token']]['objects'].append(obj)
    log_locations = {r['token']:r['location'] for r in iter_records(META/'log.json')}
    priors = {}; candidates=[]; inputs=[]; summary_rows=[]
    spec=VehicleSpec()
    for name in SCENES:
        manifest=manifests[name]; groups={g['frame_idx']:g for g in manifest['groups']}
        location=log_locations[manifest['log_token']]
        if location not in priors:
            priors[location]=load_reference_prior(location)
        prior=priors[location]
        for issue_idx in POLICY['issue_frames']:
            group=groups[issue_idx]; first=samples[group['sample_token']]
            initial=transform(first['ego_pose']); initial_yaw=yaw(first['ego_pose'])
            assert all(v['camera_timestamp_s']<=first['timestamp_s'] for c,v in group['views'].items() if c in POLICY['visible_front_channels'])
            task_id=f'{name}:f{issue_idx}:follow-nearest-same-route-v1'
            contract=dict(**POLICY,task_id=task_id,scene=name,issue_time_s=first['timestamp_s'],
                issue_sample_token=group['sample_token'],initial_rear_axle_world_pose=initial.tolist(),
                target_definition='Nearest visible forward official vehicle on a heading-compatible lane and directed successor route, selected at issue time only; no color or human-verification claim.',
                instruction='跟随前方同车道最近的车辆。')
            target,route,reason=choose_target(first['objects'],initial[:2,3],initial_yaw,group,prior,instances)
            record=dict(task_id=task_id,scene=name,log_token=manifest['log_token'],issue_frame_idx=issue_idx,
                action='follow_vehicle',contract=contract,contract_sha256=content_sha(contract),
                label_source='official_instance_and_declared_independent_directed_lane_geometry',
                human_verified=False,feasibility_status='pending',main_table_eligible=False,
                new_geometry_time_conditional_eligible=False,reason=reason,method_scores_used=False,
                source_images=str(ROOT/'dataset/processed/navbench_stage1_candidates_20261124_supplement_v1'/f'{name}.md'))
            if target:
                window=[]
                for frame_idx in range(issue_idx,len(groups)):
                    item=samples[groups[frame_idx]['sample_token']]
                    window.append(item)
                    if item['timestamp_s']-first['timestamp_s']>=POLICY['horizon_s']:
                        break
                audit,states,future=verify_witness(window,initial,target,route,prior,spec)
                witness=dict(task_id=task_id,contract_sha256=record['contract_sha256'],
                    chosen_initial_official_instance=target['instance_token'],initial_official_annotation=target,
                    route_tokens=route['tokens'],route_line_xy=list(route['line'].coords),
                    allowed_successor_routes=[dict(tokens=r['tokens'],line_xy=list(r['line'].coords),
                        polygon_geojson=mapping(r['polygon'])) for r in route['continuations']],
                    route_polygon_geojson=mapping(route['polygon']),reference_states=states,
                    future_official_anchors=future,independent_audit=audit,
                    method_input_access=False,reference_is_unique_correct_answer=False)
                witness_path=DEST/'evaluation_only'/f'{name}_f{issue_idx}_follow_reference.json'
                dump(witness_path,witness)
                record.update(target_instance_token=target['instance_token'],witness=source_ref(witness_path),
                    reason='independent_geometry_time_recorded_witness_passed' if audit['passed'] else 'attempted_recorded_witness_failed; not proven infeasible',
                    feasibility_status='geometry_time_feasible' if audit['passed'] else 'pending',
                    new_geometry_time_conditional_eligible=audit['passed'],reference_failure_reasons=audit['failures'])
            candidates.append(record)
            # No target annotation, route choice, future ego or future official objects reach this file.
            past=[]
            for frame_idx in range(issue_idx+1):
                g=groups[frame_idx]
                past.append(dict(frame_idx=frame_idx,views={c:dict(rgb=v['rgb'],depth=v['depth'],
                    intrinsic=v['intrinsic'],camera_pose=v['camera_pose'],camera_timestamp_s=v['camera_timestamp_s'])
                    for c,v in g['views'].items() if c in POLICY['visible_front_channels']}))
            inputs.append(dict(task_id=task_id,scene=name,issue_frame_idx=issue_idx,
                issue_time_s=first['timestamp_s'],instruction=contract['instruction'],
                initial_rear_axle_world_pose=initial.tolist(),initial_yaw_rad=initial_yaw,
                task_scope='3s geometry/time following; no from-rest or parking-hold requirement',
                past_and_current_sensor_references=past,gt_in_inference=False))
            summary_rows.append(dict(task_id=task_id,status=record['feasibility_status'],reason=record['reason'],
                failures=record.get('reference_failure_reasons',[])))
    jsonl(DEST/'evaluation_only/task_candidates.jsonl',candidates)
    jsonl(DEST/'method_inputs/follow_tasks.jsonl',inputs)
    jsonl(DEST/'evaluation_only/official_selected_frames.jsonl',samples.values())
    prerequisites=dict(pull_over=dict(status='pending',available='Official drivable/lane boundary geometry and real RGB',
        missing='Independent explicitly designated curbside stopping region and side/boundary meaning. Drivable alone does not grant parking permission.',
        human_question='如要做靠边，您期望 scene0047 f19 左右哪一侧、哪块可见区域？无法确认可保留待标。'),
        park_nearest=dict(status='pending',available='Real three-front RGB, official map carpark areas if present',
        missing='Individual-space geometry, complete acceptable space set, visible occupancy and frozen nearest definition; no extra data available.',
        human_question='六场景固定画面里，您是否能指出一个确实有单独边界的车位？看不到就保持待标，不把停车场区域当车位。'),
        follow_vehicle=dict(status='official_candidates_created',count=len(candidates),future_for_evaluation_only=True))
    dump(DEST/'three_task_readiness.json',prerequisites)
    source_paths=[META/x for x in ['sample_data.json','ego_pose.json','sample_annotation.json','category.json','instance.json','log.json']]+[SENSOR]
    if cache.exists():source_paths.append(cache)
    for prior in priors.values():source_paths.append(Path(prior['path']))
    protected_ok=all(sha(p)==digest for p,digest in protected.items());assert protected_ok
    size=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    assert size<POLICY['budget']['max_generated_bytes'] and time.monotonic()-begun<1800
    source_files=[Path(__file__),ROOT/'scripts/navsemantic_tasks_20261124_prepare.py']
    for source in source_files:
        target=run/'source_snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    summary=dict(created_utc=now(),seed=17,policy_sha256=content_sha(POLICY),
        revision='v2 first complete negative results retained. v3 fixes route truncation at initial target lane by allowing declared map successor routes; no threshold, candidate or semantic contract change.',
        fixed_follow_candidates=12,status_counts=dict(Counter(t['feasibility_status'] for t in candidates)),
        geometry_time_feasible_ids=[t['task_id'] for t in candidates if t['new_geometry_time_conditional_eligible']],
        historical_2015_81_unchanged=True,old_main_table_unchanged=True,new_control_or_real_execution_certificates=0,
        source_sha256={str(p):sha(p) for p in source_files},input_sha256={str(p):sha(p) for p in source_paths},
        bytes_before_manifest=size,elapsed_s=round(time.monotonic()-begun,3),rows=summary_rows,
        files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file()},
        model_calls=0,GPU_calls=0,planner_calls=0,method_evaluated=False)
    dump(DEST/'manifest.json',summary);dump(run/'preparation_summary.json',summary)
    report=['# 三类真实语义任务最小准备包','',
        '本包只做官方标注与几何/时序参考准备，尚未评被测导航方法；旧2015/81及主表完全不变。',
        '跟车契约事前固定3秒，只检查几何/时序、同向同道、5–60m纵向车身净距、整车道路和连续扫掠。'
        '官方ego原点明确声明为本工程后轴；初始为真实移动状态，无强制从静止或停车保持。'
        '未声称控制积分、真实交通安全或闭环执行。参考不是唯一正确轨迹。','',
        '|固定候选|状态|原因|','|---|---|---|']
    report += [f"|{r['task_id']}|{r['status']}|{', '.join(r['failures']) or r['reason']}|" for r in summary_rows]
    report += ['',f"状态统计：{summary['status_counts']}。所有12题与失败记录保留；参考未通过只表示pending。",'',
        '靠边：现有道路几何不等于指定停车区域，仍待绑定；找车位：没有独立单车位集合/占用，不造标签。',
        '无需用户先判断路线可解。最多两个具体图像问题保存在three_task_readiness.json，属于任务域选择，不阻止跟车工程工作。','',
        f'[六场景真实候选浏览]({ROOT/"dataset/processed/navbench_stage1_candidates_20261124_supplement_v1/index.md"})',
        f'[逐条评端任务]({DEST/"evaluation_only/task_candidates.jsonl"})；[方法侧输入]({DEST/"method_inputs/follow_tasks.jsonl"})。',
        f"生成{size:,}字节，CPU耗时{summary['elapsed_s']}秒，0GPU/模型/规划调用。"]
    (REPORT/'三类任务最小真实包.md').write_text('\n'.join(report)+'\n')
    print(canonical({k:v for k,v in summary.items() if k not in ['files','input_sha256','source_sha256']}))


if __name__=='__main__':main()
