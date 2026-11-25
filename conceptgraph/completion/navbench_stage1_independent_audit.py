"""Independent, CPU-only source/data preservation audit for Navbench stage 1."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'outputs/reports/navbench_stage1_20261124/independent'
MIN_FREE=20*1024**3


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def dump(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')


def now():return datetime.now(timezone.utc).isoformat()


def protected_paths():
    items=['AGENTS.md','MEMORY.md','PAPER_MEMORY.md','项目导航.md','conceptgraph/project_paths.py',
      'paper/当前版本.json','paper/当前版_v4/sources/论文正文.md',
      'conceptgraph/completion/navigation_diagnostics.py','conceptgraph/completion/selection_map_transfer.py',
      'conceptgraph/completion/goal_selection_ablation.py','conceptgraph/completion/reference_reachability.py',
      'conceptgraph/completion/reference_reachability_audit.py','conceptgraph/completion/reference_reachability_finalize.py',
      'outputs/project_v2/thesis_completion_v1/ablations/causal_goals_full_001/queries.jsonl',
      'outputs/project_v2/thesis_completion_v1/ablations/causal_goals_full_001/report.json',
      'outputs/results/nuscenes/validation/iterations/paper_nav_20261123_first_104746/same_task_chain_records.jsonl',
      'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/queries.jsonl',
      'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/report.json']
    paths=[ROOT/item for item in items]
    for pattern in ['conceptgraph/navigation/*.py','conceptgraph/hydra_configs/*.yaml',
      'conceptgraph/slam/*mapper.py','outputs/reports/paper_campaign_20261123/navigation/*.json',
      'outputs/reports/paper_campaign_20261123/navigation/*.csv',
      'outputs/project_v2/thesis_completion_v1/annotation/reference_reachability_reserved_full_002/audit_001/*']:
        paths.extend(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(set(paths))


def baseline():
    start=time.perf_counter();free=shutil.disk_usage(ROOT).free
    if free<MIN_FREE:raise RuntimeError('below 20GiB disk floor')
    REPORT.mkdir(parents=True,exist_ok=True)
    items=[];missing=[]
    for path in protected_paths():
        if not path.is_file():missing.append(str(path));continue
        items.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path),is_symlink=path.is_symlink(),resolved_path=str(path.resolve())))
    q=ROOT/'outputs/project_v2/thesis_completion_v1/ablations/causal_goals_full_001/queries.jsonl'
    queries=[json.loads(line) for line in q.open() if line.strip()]
    groups={}
    for name in sorted({row['method'] for row in queries}):
        subset=[row for row in queries if row['method']==name]
        groups[name]=dict(records=len(subset),unique_tasks=len({(row['scene'],row['frame_idx'],row['query_index']) for row in subset}),
          positive=sum(bool(row['positive']) for row in subset),negative=sum(not row['positive'] for row in subset))
    chains=ROOT/'outputs/results/nuscenes/validation/iterations/paper_nav_20261123_first_104746/same_task_chain_records.jsonl'
    joined=[json.loads(line) for line in chains.open() if line.strip()]
    chain_groups={}
    for name in sorted({row['method'] for row in joined}):
        subset=[row for row in joined if row['method']==name]
        chain_groups[name]={key:sum(bool(row.get(key)) for row in subset) for key in [
          'positive','correct_selected','correct_and_predicted_approach_conditions','correct_approach_point_road_route',
          'correct_approach_point_road_static_body_route','negative_false_selection','route_generated']}
        chain_groups[name]['records']=len(subset)
    data=dict(created_utc=now(),audit_scope='CPU-only, independent accounting/schema/source review; no tested-method planning, inference, training or implementation edits',
      budget=dict(wall_seconds=5400,new_artifact_bytes=512*1024**2,minimum_free_bytes=MIN_FREE,GPU_calls=0,model_calls=0),
      initial_disk_free_bytes=free,protected=items,missing=missing,total_hashed_bytes=sum(p['bytes'] for p in items),
      full_base_weight_hashing=False,history_recomputed=dict(original_queries=groups,same_task_chain=chain_groups),
      baseline_wall_s=time.perf_counter()-start,audit_source_sha256=digest(Path(__file__)))
    dump(REPORT/'initial_protection_and_budget.json',data)
    print(json.dumps(dict(report=str(REPORT/'initial_protection_and_budget.json'),files=len(items),hashed_bytes=data['total_hashed_bytes'],history=data['history_recomputed']),ensure_ascii=False))


def verify_protection():
    baseline=json.loads((REPORT/'initial_protection_and_budget.json').read_text())
    results=[dict(path=p['path'],unchanged=Path(p['path']).is_file() and digest(p['path'])==p['sha256']) for p in baseline['protected']]
    summary=dict(created_utc=now(),passed=all(x['unchanged'] for x in results),checked=len(results),results=results,
                 disk_free_bytes=shutil.disk_usage(ROOT).free,model_calls=0,GPU_calls=0)
    return summary


def historical_reference():
    """Recompute witness/query membership and controls; inherit audited geometry honestly."""
    started=time.perf_counter()
    source=ROOT/'outputs/project_v2/thesis_completion_v1/annotation/reference_reachability_reserved_full_002'
    release=json.loads((source/'release.json').read_text())
    audited=json.loads((source/'audit_001/report.json').read_text())
    checks=json.loads((source/'audit_001/witness_checks.json').read_text())
    by_key={(r['scene'],r['frame_idx'],r['instance_token']):r for r in checks}
    hashes=[]
    for group in ['source_sha256','input_sha256']:
        for path,expected in release[group].items():hashes.append(dict(path=path,passed=digest(path)==expected))
    errors=[];targets=[];state_count=0;accepted=set()
    vehicle=release['vehicle'];policy=release['policy']
    clip=lambda x,a,b:max(a,min(b,x))
    wrap=lambda a:(a+math.pi)%(2*math.pi)-math.pi
    for line in (source/'targets.jsonl').open():
        row=json.loads(line)
        if not row.get('reference'):continue
        key=row['scene'],row['frame_idx'],row['instance_token'];witness_path=source/row['reference']['path']
        h=digest(witness_path);previous=by_key.get(key);fail=[]
        if h!=row['reference']['sha256'] or previous is None or h!=previous['witness_sha256']:fail.append('witness_hash_or_old_audit_link')
        if previous is None or not previous['passed'] or previous['failures']:fail.append('old_independent_geometry_not_passed')
        witness=json.loads(witness_path.read_text());states=witness['rows'];state_count+=len(states)
        initial_path=Path(witness['known_initial_pose']['path'])
        if digest(initial_path)!=witness['known_initial_pose']['sha256']:fail.append('initial_pose_source_hash')
        initial=json.loads(initial_path.read_text());pose=initial['rear_axle_reference_world_pose']
        values=[pose[0][3],pose[1][3],math.atan2(pose[1][0],pose[0][0]),0.,0.,0.]
        names=['x','y','yaw_rad','speed_mps','steering_rad','time_s']
        if any(abs(states[0]['state'][name]-value)>1e-8 for name,value in zip(names,values)):fail.append('initial_state_mismatch')
        for first,second in zip(states,states[1:]):
            a=first['state'];b=second['state'];control=second['control'];dt=vehicle['dt_s']
            acceleration=clip(control['acceleration_mps2'],-vehicle['max_braking_mps2'],vehicle['max_acceleration_mps2'])
            speed=clip(a['speed_mps']+acceleration*dt,0,vehicle['max_speed_mps'])
            command=clip(control['steering_command_rad'],-vehicle['max_steering_rad'],vehicle['max_steering_rad'])
            steering=a['steering_rad']+clip(command-a['steering_rad'],-vehicle['max_steering_rate_radps']*dt,vehicle['max_steering_rate_radps']*dt)
            distance=(a['speed_mps']+speed)/2*dt;angle=distance*math.tan((a['steering_rad']+steering)/2)/vehicle['wheelbase_m']
            chord=distance*(math.sin(angle/2)/(angle/2) if abs(angle)>1e-10 else 1.)
            expected=[a['x']+chord*math.cos(a['yaw_rad']+angle/2),a['y']+chord*math.sin(a['yaw_rad']+angle/2),wrap(a['yaw_rad']+angle),speed,steering,a['time_s']+dt]
            if not all(math.isfinite(b[n]) and abs(b[n]-v)<=1e-6 for n,v in zip(names,expected)):fail.append('control_replay_mismatch')
        tail=states[-policy['hold_steps']:]
        if tail[-1]['state']['time_s']-tail[0]['state']['time_s']<policy['minimum_hold_duration_s']-1e-8:fail.append('inadequate_historical_hold')
        if states[-1]['state']['time_s']>policy['maximum_reference_time_s']+1e-8:fail.append('exceeded_historical_horizon')
        if any(r['state']['speed_mps']>.02 or abs(wrap(r['state']['yaw_rad']-witness['region']['heading_rad']))>witness['region']['max_heading_error_rad'] for r in tail):fail.append('terminal_speed_heading')
        if witness.get('evaluated_method_called') is not False or witness.get('actual_vehicle_execution') is not False:fail.append('method_or_execution_flag')
        if fail:errors.append(dict(key=key,failures=sorted(set(fail))))
        else:accepted.add(key)
        targets.append(dict(scene=key[0],frame_idx=key[1],instance_token=key[2],path=str(witness_path),sha256=h,states=len(states),passed=not fail,failures=sorted(set(fail))))
    query_rows=[json.loads(line) for line in (source/'audit_001/queries.jsonl').open()]
    certified=0;unknown=0
    for row in query_rows:
        expected=sorted(t for t in row['accepted_instance_tokens'] if (row['scene'],row['frame_idx'],t) in accepted)
        if expected!=sorted(row['certified_instance_tokens']):errors.append(dict(query_id=row['query_id'],failure='certified_target_membership'))
        if bool(expected)!=bool(row['reachable_label']):errors.append(dict(query_id=row['query_id'],failure='reachable_label'))
        if row['inaccessible_label'] is not None:errors.append(dict(query_id=row['query_id'],failure='unknown_promoted_to_infeasible'))
        certified+=bool(expected);unknown+=not bool(expected)
    output=dict(created_utc=now(),passed=not errors and all(r['passed'] for r in hashes),queries=len(query_rows),
       conditional_feasible=certified,pending=unknown,infeasible_proven=0,witnesses=len(targets),control_states_recomputed=state_count,
       original_geometry_audit_inherited=True,geometry_recomputed_this_run=False,
       geometry_scope='Existing independent full road/lane/obstacle/continuous-sweep audits are inherited only for unchanged witness hashes; this run freshly checks hashes, controls, terminal speed/heading/time and query membership.',
       applicable_action='historical_object_approach_only',three_new_semantic_actions_certified=0,
       hashes=hashes,witnesses_detail=targets,failures=errors,elapsed_s=time.perf_counter()-started,GPU_calls=0,model_calls=0,planner_calls=0)
    dump(REPORT/'historical_reference_recheck_v1.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['hashes','witnesses_detail']},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['baseline','protection','reference']);args=parser.parse_args()
    if args.mode=='baseline':baseline()
    elif args.mode=='reference':historical_reference()
    else:print(json.dumps(verify_protection(),ensure_ascii=False))


if __name__=='__main__':main()
