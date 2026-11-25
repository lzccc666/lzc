"""冻结、运行与独立评价分进程入口，历史回归不选参数。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np

from conceptgraph.navigation.navstage2_20261124 import POLICY, MethodMap, run_task
from conceptgraph.navigation.vehicle_geometry import VehicleSpec
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory

ROOT = PROJECT_ROOT
PACKAGE = ROOT/'dataset/processed/navbench_stage1_20261124_v1'
INPUT = ROOT/'dataset/processed/navstage2_20261124_v2'
RUN_ID = 'navstage2_20261124_v2'
BASE = ROOT/'outputs/project_v2/thesis_completion_v1'
SEALED = BASE/'public_experiment_completion_20261115/main_experiments/sealed_public_058'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def space_guard():
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('disk_below_20GiB')


def verify_frozen(manifest):
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in manifest[group].items():
            if sha(path) != digest:
                raise ValueError(f'frozen_file_changed:{path}')


def prepare():
    space_guard(); INPUT.mkdir(exist_ok=False)
    frozen = read(PACKAGE/'denominator_freeze.json')
    ids = frozen['conditional_historical_reference_ids']
    assert len(ids) == 81 and len(set(ids)) == 81
    candidates = {r['task_id']: r for r in rows(PACKAGE/'method_inputs/candidate_tasks.jsonl')}
    registry = {r['task_id']: r for r in rows(PACKAGE/'task_registry.jsonl')}
    logs = {r['token']: r['location'] for r in read(ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval/log.json')}
    agents = {(r['scene'], r['frame_idx']): r for r in rows(SEALED/'agent_frames.jsonl')}
    source_inputs = {str(PACKAGE/p): sha(PACKAGE/p) for p in ['manifest.json','protocol_v2.json',
        'denominator_freeze.json','method_inputs/candidate_tasks.jsonl','task_registry.jsonl',
        'effective_amendment_v3.json','source_binding_overlay_v1.jsonl']}
    source_inputs[str(SEALED/'agent_frames.jsonl')] = sha(SEALED/'agent_frames.jsonl')
    graphs = {}; exported = []
    for task_id in ids:
        task = candidates[task_id]; path = Path(task['historical_prediction_path'])
        source_inputs[str(path)] = sha(path)
        if task['scene'] not in graphs:
            with gzip.open(path,'rt') as f:
                graphs[task['scene']] = {r['frame_idx']: r for r in map(json.loads,f)}
        frame = graphs[task['scene']][task['issue_frame_idx']]
        agent = agents[task['scene'], task['issue_frame_idx']]
        assert frame['gt_in_inference'] is False and agent['gt_in_agent'] is False and not agent['future_lidar_used']
        assert frame['camera_timestamp_s'] == agent['camera_timestamp_s']
        for field in ['camera_pose','intrinsic','rgb','depth']:
            assert frame[field] == agent[field]
        original = Path(task['initial_input_manifest']); source_inputs[str(original)] = sha(original)
        assert sha(original) == agent['input_sha256'][str(original)]
        location = logs[registry[task_id]['log_token']]
        map_path = ROOT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
        source_inputs[str(map_path)] = sha(map_path)
        # 只导出方法需要字段；registry的参考答案/GT contract hash不进入推理包。
        exported.append(dict(task_id=task_id, instruction=task['instruction'], scene=task['scene'],
            issue_frame_idx=task['issue_frame_idx'], map_location=location,
            rear_axle_reference_world_pose=agent['rear_axle_reference_world_pose'],
            rear_axle_yaw_rad=agent['rear_axle_yaw_rad'], initial_speed_mps=0., initial_steering_rad=0.,
            sample_token=agent['sample_token'], camera_token=agent['camera_token'],
            prediction_frame={k:frame[k] for k in ['scene','frame_idx','camera_timestamp_s','camera_pose',
                'intrinsic','objects','edges','gt_in_inference']}))
    path = INPUT/'method_inputs.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in exported))
    source_inputs[str(path)] = sha(path)
    src = sorted(set([Path(__file__),ROOT/'scripts/navstage2_20261124.py',
        ROOT/'conceptgraph/project_paths.py'] + list((ROOT/'conceptgraph/navigation').glob('*.py')) +
        [ROOT/'conceptgraph/completion'/n for n in ['reference_regions.py','selection_ablation.py',
        'navigation_diagnostics.py','evidence.py','topology_vehicle.py','execution_direction_audit.py',
        'reference_reachability_audit.py','reference_region_audit.py']] +
        [ROOT/'conceptgraph/scenegraph/relations.py',ROOT/'conceptgraph/scripts/build_lane_constrained_bev.py']))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=17,
        task_ids=ids, denominator=81, scope=POLICY['scope'], policy=POLICY,
        vehicle=asdict(VehicleSpec()), source_sha256={str(p):sha(p) for p in src}, input_sha256=source_inputs,
        wall_budget_s=2700, storage_budget_bytes=256*1024**2, GPU_calls=0,
        previous_used_test=True, parameter_selection_allowed=False, configurations=1,
        technical_revision='v1保留原样：官方中心线退化段触发原检查器ValueError；v2只将无法认证的候选记失败并继续，未修改几何或规划参数',
        reference_trajectory_inference_access=False, oracle_target_geometry_inference_access=False,
        empty_graphs=sum(not r['prediction_frame']['objects'] for r in exported),
        new_three_tasks_not_executed=True,
        evaluation='语义对象正确、完整GT道路/车身/2-8m近物/车辆/末端条件、有限区域成员资格分报。参考每对象最多4个小框只是充分可行子域，域外不等于附近任务语义失败；有限区域评价允许原可接受对象集合的全部已冻结区域，禁止只用选中见证的实例/区域。参考轨迹仅证明存在，不要求逐点模仿。',
        evaluation_only_sha256={str(p):sha(p) for p in [
            PACKAGE/'evaluation_only/conditional_reference_labels.jsonl',
            PACKAGE/'evaluation_only/certificates.jsonl',
            BASE/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl',
            BASE/'annotation/reference_regions_reserved_full_003/targets.jsonl']})
    dump(INPUT/'manifest.json', manifest)
    print(json.dumps(dict(prepared=str(INPUT),denominator=81,empty=manifest['empty_graphs'],manifest_sha256=sha(INPUT/'manifest.json'))))


def run():
    space_guard(); manifest = read(INPUT/'manifest.json'); verify_frozen(manifest)
    dest = experiment_directory('nuscenes','test',RUN_ID,create=True)
    shutil.copy2(INPUT/'manifest.json',dest/'manifest.json')
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as tar:
        for path in manifest['source_sha256']:
            tar.add(path, arcname=str(Path(path).relative_to(ROOT)))
    tasks = rows(INPUT/'method_inputs.jsonl'); assert [r['task_id'] for r in tasks] == manifest['task_ids']
    maps = {}; begun = time.monotonic(); planning_used = 0.; counts = Counter()
    with (dest/'method_predictions.jsonl').open('x') as output:
        for index, task in enumerate(tasks):
            space_guard()
            if task['map_location'] not in maps:
                maps[task['map_location']] = MethodMap(task['map_location'])
            if planning_used >= POLICY['total_planning_budget_s']:
                result = dict(task_id=task['task_id'],scene=task['scene'],frame_idx=task['issue_frame_idx'],
                    status='TOTAL_BUDGET_EXHAUSTED',selected_node_id=None,selected_prediction_index=None,
                    selected=None,semantic_candidates=[],planning_attempts=[],predicted_rows=[],duration_s=0.,
                    actual_arrival_credit=False,gt_in_inference=False)
            else:
                result = run_task(task, maps[task['map_location']])
                planning_used += result['duration_s']
            output.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); output.flush()
            counts[result['status']] += 1
            if (index+1)%10 == 0:
                print(json.dumps(dict(completed=index+1,total=81,counts=dict(counts),wall_s=time.monotonic()-begun)),flush=True)
    assert sum(counts.values()) == 81
    verify_frozen(manifest)
    dump(dest/'run_summary.json',dict(denominator=81,counts=dict(counts),wall_s=time.monotonic()-begun,
        planning_wall_s=planning_used,GPU_calls=0,actual_arrival_credit=False,
        predictions_sha256=sha(dest/'method_predictions.jsonl'), source_and_inputs_unchanged=True,
        output_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())))
    print(str(dest))


def evaluate():
    # 此命令在方法进程结束后执行；GT只在这个评价进程加载。
    from shapely.geometry import Point, Polygon, shape
    from shapely.ops import unary_union
    from conceptgraph.completion.reference_reachability_audit import (
        audit_body_rows, audit_control_rows, audit_directed_rows, audit_terminal_rows, angular_error)
    from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes
    from conceptgraph.scripts.build_lane_constrained_bev import _sample_arcline
    dest = experiment_directory('nuscenes','test',RUN_ID); manifest = read(dest/'manifest.json'); verify_frozen(manifest)
    for path,digest in manifest['evaluation_only_sha256'].items():
        assert sha(path)==digest
    predictions = rows(dest/'method_predictions.jsonl'); inputs = {r['task_id']:r for r in rows(INPUT/'method_inputs.jsonl')}
    labels = {r['task_id']:r for r in rows(PACKAGE/'evaluation_only/conditional_reference_labels.jsonl')}
    certificates = {r['certificate_id']:r for r in rows(PACKAGE/'evaluation_only/certificates.jsonl')}
    region_targets={(r['scene'],r['frame_idx'],r['instance_token']):r for r in rows(BASE/'annotation/reference_regions_reserved_full_003/targets.jsonl')}
    official = {(r['scene'],r['frame_idx']):r for r in rows(BASE/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl')}
    maps = {}; evaluations=[]; counts=Counter(); by_scene=defaultdict(Counter)
    for pred in predictions:
        task=inputs[pred['task_id']]; label=labels[pred['task_id']]; truth=official[task['scene'],task['issue_frame_idx']]['objects']
        mapping=associate_nodes(task['prediction_frame']['objects'],truth,maximum_distance_m=3.,ambiguity_margin_m=.5)
        identities={r['node_id']:r['instance_token'] for r in mapping['matches']}
        selected_identity=identities.get(pred['selected_node_id'])
        accepted=label['source_query']['accepted_instance_tokens']; correct=selected_identity in accepted
        evaluation=dict(task_id=pred['task_id'],scene=pred['scene'],frame_idx=pred['frame_idx'],
            status=pred['status'],candidate_count=len(pred['semantic_candidates']),
            candidate_target_covered=any(identities.get(c['node_id']) in accepted for c in pred['semantic_candidates']),
            selected_instance=selected_identity, selected_correct=correct,
            predicted_route=bool(pred['predicted_rows']), official_body_safe=False,
            control_valid=False,directed_route_valid=False,broad_terminal_geometry=False,
            reference_region_membership=False,frozen_region_contract_success=False,
            broad_semantic_geometry_success=False,actual_arrival_credit=False,
            eval_errors=[])
        if pred['predicted_rows']:
            certs=[certificates[c] for c in label['certificate_ids']]
            context=read(certs[0]['artifacts']['static_context']['path'])
            road,lane_union=shape(context['drivable_geojson']),shape(context['lane_union_geojson'])
            obstacles=unary_union([shape(o['polygon_geojson']) for o in context['objects']])
            states=pred['predicted_rows']; vehicle=manifest['vehicle']
            control=audit_control_rows(states,vehicle,np.asarray(task['rear_axle_reference_world_pose']))
            body=audit_body_rows(states,vehicle,road,lane_union,obstacles)
            location=task['map_location']
            if location not in maps:
                data=read(ROOT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json')
                lane_records={}
                localnodes={n['token']:(n['x'],n['y']) for n in data['node']}; polygons={p['token']:p for p in data['polygon']}
                for kind in ['lane','lane_connector']:
                    for r in data[kind]:
                        paths=data['arcline_path_3'].get(r['token'],[])
                        if paths:
                            polygon=polygons[r['polygon_token']]
                            lane_records[r['token']]=dict(polygon=Polygon([localnodes[t] for t in polygon['exterior_node_tokens']],
                                [[localnodes[t] for t in h['node_tokens']] for h in polygon.get('holes',[])]),
                                samples=np.concatenate([_sample_arcline(p,.25) for p in paths]))
                maps[location]=(lane_records,data['connectivity'])
            lanes,conn=maps[location]
            direction=audit_directed_rows(states,pred['selected_lane_tokens'],lanes,conn,np.pi/3)
            hold=states[-21:]; hold_ok=len(hold)==21 and hold[-1]['state']['time_s']-hold[0]['state']['time_s']>=2.-1e-8
            terminal=False
            for obj in context['objects']:
                if obj['instance_token'] not in accepted:
                    continue
                target=shape(obj['polygon_geojson'])
                valid=hold_ok
                for row in hold:
                    s=row['state']; point=Point(s['x'],s['y']); dist=target.distance(point)
                    yaw_ok=False
                    for token in pred['selected_lane_tokens']:
                        lane=lanes[token]
                        if lane['polygon'].covers(point):
                            samples=lane['samples']; k=int(np.argmin(np.linalg.norm(samples[:,:2]-[s['x'],s['y']],axis=1)))
                            yaw_ok |= abs(angular_error(s['yaw_rad'],float(samples[k,2])))<=np.pi/36
                    valid &= (2.<=dist<=8. and s['speed_mps']<=.02 and s['time_s']<=90. and yaw_ok)
                terminal |= bool(valid)
            allowed_regions=[region for instance in accepted
                for region in region_targets.get((pred['scene'],pred['frame_idx'],instance),{}).get('regions',[])]
            membership=any(not audit_terminal_rows(states,region,dict(hold_steps=21,minimum_hold_duration_s=2.,maximum_reference_time_s=90.)) for region in allowed_regions)
            evaluation.update(control_valid=not control,official_body_safe=not body,directed_route_valid=not direction,
                broad_terminal_geometry=terminal,reference_region_membership=membership,
                frozen_region_contract_success=bool(correct and not control and not body and not direction and membership),
                broad_semantic_geometry_success=bool(correct and not control and not body and not direction and terminal),
                eval_errors=sorted(set(control+body+direction)))
        evaluations.append(evaluation)
        for key in ['candidate_target_covered','selected_correct','predicted_route','official_body_safe','control_valid',
                    'directed_route_valid','broad_terminal_geometry','reference_region_membership',
                    'frozen_region_contract_success','broad_semantic_geometry_success']:
            counts[key]+=int(evaluation[key]);by_scene[pred['scene']][key]+=int(evaluation[key])
        by_scene[pred['scene']]['denominator']+=1
    assert [r['task_id'] for r in predictions]==manifest['task_ids']
    (dest/'evaluation_only').mkdir(exist_ok=False)
    (dest/'evaluation_only/queries.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluations))
    dump(dest/'evaluation_only/summary.json',dict(denominator=81,counts=dict(counts),by_scene={k:dict(v) for k,v in by_scene.items()},
        note='有限参考小框仅单列契约成员关系；broad是原语言附近/2-8m完整几何诊断，不代表新停车/跟车任务或实际到达',
        predictions_sha256=sha(dest/'method_predictions.jsonl'),previous_used_test_regression=True,
        evaluation_ground_truth_in_method=False,actual_arrival_credit=False))
    print(json.dumps(dict(denominator=81,counts=dict(counts)),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'evaluate':evaluate}[args.action]()
