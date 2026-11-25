"""Read-only compatibility audit of the existing 30-scene causal predicted graphs."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import time

from conceptgraph.completion.navbench_stage1_independent_audit import ROOT,REPORT,digest,dump,now


def main():
    started=time.perf_counter()
    base=ROOT/'outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/main_experiments/sealed_public_058'
    graphs=base/'mapping/one_to_one_depth_quality'
    release=json.loads((base/'release.json').read_text())
    inputs=[json.loads(line) for line in (base/'agent_frames.jsonl').open()]
    queries=[json.loads(line) for line in (base/'query_inputs.jsonl').open()]
    references=[json.loads(line) for line in (ROOT/'outputs/project_v2/thesis_completion_v1/annotation/reference_reachability_reserved_full_002/audit_001/queries.jsonl').open()]
    by={(row['scene'],row['frame_idx']):row for row in inputs}
    q_by={row['query_id']:row for row in queries}
    failures=[];frame_index={};source_files=[];total_objects=0;early_flag=0;missing_depth=0
    if digest(base/'agent_frames.jsonl')!=release['agent_frames_sha256']:failures.append('agent_frames_hash')
    if digest(base/'query_inputs.jsonl')!=release['query_inputs_sha256']:failures.append('query_inputs_hash')
    for path in sorted(graphs.glob('scene-*/frames.jsonl.gz')):
        scene=path.parent.name;report=json.loads((path.parent/'report.json').read_text());h=digest(path)
        if h!=report['frames_sha256']:failures.append(scene+':graph_hash')
        source_files.append(dict(path=str(path),sha256=h,bytes=path.stat().st_size))
        previous=-1e99;count=0
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);key=row['scene'],row['frame_idx'];agent=by[key]
                if key in frame_index:failures.append(str(key)+':duplicate')
                if row['camera_timestamp_s']<previous:failures.append(str(key)+':future_order')
                previous=row['camera_timestamp_s'];count+=1
                for name in ['camera_timestamp_s','camera_pose','intrinsic','rgb','depth']:
                    if row[name]!=agent[name]:failures.append(str(key)+':'+name+'_mismatch')
                metadata_path=next(Path(p) for p in agent['input_sha256'] if '/inputs/' in p)
                if digest(metadata_path)!=agent['input_sha256'][str(metadata_path)]:failures.append(str(key)+':input_metadata_hash')
                metadata=json.loads(metadata_path.read_text())
                if metadata['sample_token']!=agent['sample_token'] or metadata['rear_axle_reference_world_pose']!=agent['rear_axle_reference_world_pose']:failures.append(str(key)+':sample_or_rear_axle')
                if row['gt_in_inference'] is not False or agent['gt_in_agent'] is not False:failures.append(str(key)+':GT_flag')
                if any(k.startswith('evaluation_only') or k in ['acceptable_instances','accepted_instance_tokens','official_objects'] for k in row):failures.append(str(key)+':evaluation_label_in_graph')
                if agent['future_lidar_used'] is not False:failures.append(str(key)+':future_LiDAR')
                if agent['lidar_timestamp_s'] is not None:
                    age=agent['camera_timestamp_s']-agent['lidar_timestamp_s']
                    if not 0<=age<=.100001:failures.append(str(key)+':past_LiDAR_age')
                else:missing_depth+=1
                if metadata.get('inference_executed') is False:early_flag+=1
                for obj in row['objects']:
                    if obj.get('timestamp_s') is not None and obj['timestamp_s']>row['camera_timestamp_s']+1e-8:failures.append(str(key)+':future_node')
                    if obj.get('last_seen_frame',row['frame_idx'])>row['frame_idx']:failures.append(str(key)+':future_node_frame')
                total_objects+=len(row['objects']);frame_index[key]=dict(graph_path=str(path),sample_token=metadata['sample_token'],camera_timestamp_s=row['camera_timestamp_s'],objects=len(row['objects']))
        if count!=report['frames']:failures.append(scene+':count')
    reference_count=0;certified_count=0;certified_empty_graph=0
    for row in references:
        query=q_by.get(row['query_id']);key=row['scene'],row['frame_idx']
        if query is None or key not in frame_index:failures.append(row['query_id']+':missing_query_or_frame');continue
        if (query['scene'],query['frame_idx'])!=key or query['timestamp_s']!=frame_index[key]['camera_timestamp_s']:failures.append(row['query_id']+':reference_time_or_frame')
        reference_count+=1
        if row['reachability']=='conditionally_reachable_static_from_rest':
            certified_count+=1;certified_empty_graph+=frame_index[key]['objects']==0
    source=ROOT/'outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/source/sealed_public_benchmark_058.py'
    source_frozen=digest(source)==release['frozen_files_sha256'][str(source)]
    if not source_frozen:failures.append('frozen_generator_source_hash')
    report=dict(created_utc=now(),passed=not failures,scenes=len(source_files),frames=len(frame_index),predicted_object_frames=total_objects,
      source_files=source_files,source_generator=dict(path=str(source),sha256=digest(source),matches_frozen_release=source_frozen,
         reviewed_mapping_lines='345-417: mapper receives RGB/depth/intrinsic/pose/current detections/timestamp only; official labels enter separate evaluate path'),
      reference_queries_compatible=reference_count,conditional_feasible_reference_queries=certified_count,
      certified_queries_with_empty_graph=certified_empty_graph,initial_conversion_inference_executed_false=early_flag,
      missing_depth_frames_preserved=missing_depth,input_metadata_hashes_checked=len(frame_index),
      sample_pose_time_checked=True,past_only_inputs_checked=True,no_GT_flag_and_payload_checked=True,
      raw_RGB_depth_rehashed_this_audit=False,model_weights_rehashed_this_audit=False,
      graph_version='historical_sealed_public_058_one_to_one_depth_quality; not R1/T3',
      conclusion='Existing causal predicted graphs are available and compatible with all historical reference query times/initial metadata. The earlier input-conversion inference_executed=false is stale preparation metadata, not evidence that graphs are missing. Eligible for a new frozen method-input export, not proof of method success.',
      split='already_used_test_regression',failures=failures,elapsed_s=time.perf_counter()-started,model_calls=0,GPU_calls=0,planner_calls=0)
    dump(REPORT/'existing30_predicted_graph_compatibility_v1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'},ensure_ascii=False))


if __name__=='__main__':main()
