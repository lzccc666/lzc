"""Frozen, CPU-only replay of 1115/front and 1200/front3 historical tasks.

Prediction and evaluation are separate commands.  The ranker never receives
official instances, task answers, visibility labels or object motion truth.
This is a seen-development regression, not new language generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.project_paths import experiment_directory

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/processed/nu_task_replay_20261125_v1'
REPORT = ROOT / 'outputs/reports/nu_task_replay_20261125'
RUN_ID = 'nu_task_replay_20261125_v1'
METHODS = ('single_frame', 'original_fusion', 'R1', 'S2_1', 'S2_3', 'S2_6')
OLD_QUERY = ROOT / 'outputs/project_v2/thesis_completion_v1/main_experiments/causal_retrieval_full_001/queries.jsonl'
OLD_MAP = ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/causal_mapping_full_001'
ALIGN = ROOT / 'outputs/project_v2/thesis_completion_v1/data_manifest/development_causal_inputs_full_001/alignment_manifest.json'
S2 = ROOT / 'dataset/processed/relation_nav_unified_20261125_v1'
POLICY = dict(seed=17, configuration_count=1, parameter_search=False,
    split='seen_validation_debug_regression', inference_uses_GT=False,
    scopes={'legacy1115': ['CAM_FRONT'], 'unified1200': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']},
    near_distance_m={'legacy1115': 8., 'unified1200': 6.},
    candidate_visibility='predicted 3D center projected into current RGB; 1<camera_z<65m and 0<=u<W,0<=v<H; union of protocol cameras',
    candidate_motion='unified verified three-state only; unknown never restored from legacy',
    candidate_evidence='raw semantic_position_candidate; dense also retains historical depth gate',
    association='same-category maximum assignment <=3m world XY, unchanged historical evaluator',
    clock='exact source sample_token plus exact CAM_FRONT capture timestamp; common render clock is supplied max6, recorded separately',
    negative_labels='original front/front3 visibility scopes only, no side-view extension labels',
    vehicle_safety_authorized=False, trajectory_generated=False, GPU_calls=0, Qwen_calls=0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def jsonl(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for value in values: f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def guard():
    if shutil.disk_usage(DATA.resolve()).free < 20 * 1024**3:
        raise OSError('actual_output_volume_below_20GiB')


def verify(*, method_only=False):
    manifest = json.loads((DATA/'manifest.json').read_text())
    for section in ('source_sha256', 'frozen_inputs_sha256'):
        for p, expected in manifest[section].items():
            if method_only and section == 'frozen_inputs_sha256' and Path(p) != DATA/'method_tasks.jsonl':
                continue
            if sha(p) != expected: raise AssertionError(f'frozen_source_changed:{p}')
    return manifest


def prepare():
    if DATA.exists(): raise FileExistsError(DATA)
    DATA.mkdir(parents=True); (DATA/'evaluation_only').mkdir()
    guard()
    binding = {}
    for p in json.loads(ALIGN.read_text())['plans']:
        c = p['camera']
        binding[c['scene'], c['frame_idx']] = dict(sample_token=c['sample_token'],
            camera_timestamp_s=c['timestamp']/1e6, source_camera_token=c['token'],
            log=c['log_token'])
    labels_by_key = {}
    for q in rows(OLD_QUERY):
        key = (q['scene'], q['frame_idx'], q['query_index'])
        value = {k:q[k] for k in ('query_text','positive','acceptable_instances')}
        if key in labels_by_key and labels_by_key[key] != value:
            raise AssertionError('old_method_answers_disagree')
        labels_by_key[key] = value
    assert len(labels_by_key) == 1115
    inputs, labels, frame_truth = [], [], []
    needed_frames = set()
    for (scene, idx, qi), value in sorted(labels_by_key.items()):
        tid = f'legacy1115:{scene}:{idx}:{qi}'
        inputs.append(dict(task_id=tid, protocol='legacy1115', scene=scene, original_frame_idx=idx,
            query_index=qi, query_text=value['query_text'], reference_frame='ego_heading', **binding[scene,idx]))
        labels.append(dict(task_id=tid, label_status='scorable', positive=value['positive'],
            acceptable_instances=value['acceptable_instances']))
        needed_frames.add((scene,idx))
    assert sum(x['positive'] for x in labels) == 733
    input_paths = [OLD_QUERY, ALIGN, S2/'method_tasks.jsonl', S2/'evaluation_only/task_labels.jsonl', S2/'evaluation_only/frames.jsonl']
    for scene in sorted({k[0] for k in needed_frames}):
        path = OLD_MAP/scene/'frames.jsonl'; input_paths.append(path)
        for frame in rows(path):
            key = scene, frame['frame_idx']
            if key not in needed_frames: continue
            if abs(frame['camera_timestamp_s'] - binding[key]['camera_timestamp_s']) > 1e-6:
                raise AssertionError('legacy_clock_binding_mismatch')
            frame_truth.append(dict(protocol='legacy1115',scene=scene,original_frame_idx=key[1],
                sample_token=binding[key]['sample_token'], objects=frame['evaluation_only_official']))
    s2frames = {(r['scene'],r['frame_idx']):r for r in rows(S2/'method_frames.jsonl') if r['cohort']=='S2'}
    s2manifest_path = ROOT/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json'
    input_paths.extend([S2/'method_frames.jsonl',s2manifest_path])
    group_binding = {}
    for scene in json.loads(s2manifest_path.read_text())['scenes']:
        for group in scene['groups']:
            group_binding[scene['scene'],group['frame_idx']] = dict(sample_token=group['sample_token'],
                camera_timestamp_s=group['views']['CAM_FRONT']['camera_timestamp_s'], log=scene['log_token'])
    tasks = list(rows(S2/'method_tasks.jsonl'))
    for t in tasks:
        b = group_binding[t['scene'],t['frame_idx']]
        inputs.append(dict(task_id='unified1200:'+t['task_id'],protocol='unified1200',scene=t['scene'],
            original_frame_idx=t['frame_idx'],query_text=t['query_text'],reference_frame=t['reference_frame'],**b))
    for t in rows(S2/'evaluation_only/task_labels.jsonl'):
        labels.append({**t, 'task_id':'unified1200:'+t['task_id']})
    for t in rows(S2/'evaluation_only/frames.jsonl'):
        if t['cohort'] == 'S2':
            frame_truth.append(dict(protocol='unified1200',scene=t['scene'],original_frame_idx=t['frame_idx'],
                sample_token=group_binding[t['scene'],t['frame_idx']]['sample_token'],objects=t['truth']))
    assert len(inputs)==2315 and len({r['task_id'] for r in inputs})==2315
    assert len(labels)==2315 and sum(r['label_status']!='scorable' for r in labels)==60
    assert len(frame_truth) == len({(r['protocol'],r['scene'],r['original_frame_idx']) for r in inputs})
    jsonl(DATA/'method_tasks.jsonl',inputs)
    jsonl(DATA/'evaluation_only/task_labels.jsonl',labels)
    jsonl(DATA/'evaluation_only/frame_truth.jsonl',frame_truth)
    sources = [Path(__file__),ROOT/'scripts/nu_task_replay_20261125.py',
        ROOT/'conceptgraph/completion/relation_nav_runtime_20261125.py',
        ROOT/'conceptgraph/navigation/relation_nav_unified_20261125.py',
        ROOT/'conceptgraph/navigation/semantic.py',ROOT/'conceptgraph/scenegraph/relations.py',
        ROOT/'conceptgraph/scenegraph/relation_nav_unified_20261125.py',
        ROOT/'conceptgraph/completion/evidence.py',ROOT/'conceptgraph/completion/quality.py']
    input_paths += [DATA/'method_tasks.jsonl',DATA/'evaluation_only/task_labels.jsonl',DATA/'evaluation_only/frame_truth.jsonl']
    write(DATA/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,
        task_counts={'legacy1115':dict(all=1115,positive=733,negative=382,pending=0),
                     'unified1200':dict(all=1200,positive=141,negative=999,pending=60)},
        scene_count=len({r['scene'] for r in inputs}),frame_tasks=len(frame_truth),
        source_sha256={str(p):sha(p) for p in sources},
        frozen_inputs_sha256={str(p):sha(p) for p in input_paths},
        new_prediction_streams='per-scene input receipt at consumption after producer closes file; not yet present at preregistration',
        selection_metrics_separate_from_mapping=True))
    print(json.dumps(dict(tasks=2315,frame_tasks=len(frame_truth))))


def visible(center, view):
    pose=np.asarray(view['camera_pose'],dtype=float)
    xyz=pose[:3,:3].T@(np.asarray(center,dtype=float)-pose[:3,3])
    if not 1.<xyz[2]<65.:return False
    intrinsic=view.get('intrinsic',view.get('cam_K'))
    uv=np.asarray(intrinsic,dtype=float)@xyz
    return bool(0<=uv[0]/uv[2]<view['width'] and 0<=uv[1]/uv[2]<view['height'])


def select(protocol, query, records, predictions, pose, idx, stamp):
    from conceptgraph.completion.relation_nav_runtime_20261125 import adapt_nodes
    from conceptgraph.navigation.relation_nav_unified_20261125 import select_task
    from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
    from conceptgraph.scenegraph.relations import build_relations, RelationNode
    from conceptgraph.scenegraph.relation_nav_unified_20261125 import POLICY as REL_POLICY
    safe=[]
    for record in records:
        record=copy.deepcopy(record)
        index=record['original_index']
        if record['node_id']!=predictions[index]['node_id']:raise AssertionError('node_original_index_mismatch')
        record['legacy_prediction']=predictions[index]
        safe.append(record)
    nodes,allowed,indices=adapt_nodes(safe,{})
    forward=pose[:2,2]; forward=forward/np.linalg.norm(forward)
    if protocol=='legacy1115':
        edges=build_relations(nodes,dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),frame_index=idx,ego_heading=forward).edges
        parsed=parse_navigation_query(query,8.);parsed.vehicle_heading_xy=forward
        ranked=rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
            relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,evidence_policy=None)
        ranked=[r for r in ranked if allowed[r.node_id]]
        result=dict(status='TARGET_SELECTED' if ranked else 'NO_CONFIRMED_STATIC_TARGET',
            selected_id=ranked[0].node_id if ranked else None,candidate_ids=[r.node_id for r in ranked],
            GT_in_method=False,Qwen_calls=0,vehicle_safety_authorized=False,trajectory_generated=False)
    else:
        relation_input={i:RelationNode(i,n.bbox_center,dynamic=n.motion_state=='dynamic',
            observation_count=n.metadata.get('num_detections',0),last_seen_frame=idx,
            metadata=dict(timestamp_s=stamp,source_node_id=n.metadata['frozen_object_id'],motion_state=n.motion_state)) for i,n in nodes.items()}
        edges=build_relations(relation_input,REL_POLICY,frame_index=idx,ego_heading=forward).edges
        result=select_task(query,nodes,allowed,edges,heading=forward,origin=pose[:2,3])
    selected=result['selected_id']
    result.update(selected_original_index=indices.get(selected),
        selected_source_node_id=nodes[selected].metadata['frozen_object_id'] if selected is not None else None,
        eligible_original_indices=[indices[i] for i in nodes if allowed[i]],
        visible_original_indices=[r['original_index'] for r in records],
        original_index_by_local={str(i):indices[i] for i in nodes})
    return result


def infer(scene, frame_file):
    verify(method_only=True);guard()
    path=Path(frame_file).resolve()
    if not path.exists():raise FileNotFoundError(path)
    complete_path=path.parent/'complete.json'
    complete=json.loads(complete_path.read_text())
    before=dict(sha256=sha(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    if complete['frames_sha256'] != before['sha256']:
        raise AssertionError('producer_complete_sha_mismatch')
    tasks=defaultdict(list)
    for task in rows(DATA/'method_tasks.jsonl'):
        if task['scene']==scene:tasks[task['sample_token']].append(task)
    if not tasks:raise ValueError('scene_has_no_frozen_tasks')
    dest=experiment_directory('nuscenes','validation',RUN_ID)
    output=dest/scene/'method_predictions.jsonl'
    if output.exists():raise FileExistsError(output)
    by_sample={};counts=Counter();results=[];started=time.monotonic()
    for frame in rows(path):
        token=frame['sample_token']
        if token not in tasks:continue
        if token in by_sample:raise AssertionError('duplicate_sample_in_new_stream')
        if frame['scene']!=scene:raise AssertionError('scene_binding_mismatch')
        front=frame['views']['CAM_FRONT'];pose=np.asarray(front['camera_pose'],dtype=float)
        for task in tasks[token]:
            if abs(task['camera_timestamp_s']-front['camera_timestamp_s'])>1e-6:
                raise AssertionError('source_camera_timestamp_mismatch')
            for method in METHODS:
                incoming=frame['methods'][method]
                predictions=incoming['predictions'];records=incoming['nodes']
                if len(predictions)!=len(records):raise AssertionError('node_prediction_cardinality')
                scope=POLICY['scopes'][task['protocol']]
                filtered=[r for r in records if any(visible(r['center_world_m'],frame['views'][c]) for c in scope)]
                result=select(task['protocol'],task['query_text'],filtered,predictions,pose,frame['frame_idx'],frame['render_timestamp_s'])
                results.append(dict(**task,method=method,new_frame_idx=frame['frame_idx'],
                    render_timestamp_s=frame['render_timestamp_s'],front_capture_timestamp_s=front['camera_timestamp_s'],
                    visible_scope=scope,inference=result,predictions_for_evaluation=[
                        dict(original_index=r['original_index'],node_id=r['node_id'],category=r['category'],center=r['center_world_m']) for r in filtered]))
                counts[task['protocol']]+=1
        by_sample[token]=frame['frame_idx']
    if set(by_sample)!=set(tasks):raise AssertionError(f'missing_frozen_samples:{set(tasks)-set(by_sample)}')
    if sha(path)!=before['sha256'] or path.stat().st_size!=before['size']:raise AssertionError('source_changed_during_inference')
    jsonl(output,results)
    write(dest/scene/'method_receipt.json',dict(scene=scene,path=str(path),input_receipt=before,
        producer_complete_path=str(complete_path),producer_complete_sha256=sha(complete_path),
        note='prediction stream frozen on consumption, source tasks and method policy preregistered',
        source_manifest_sha256=sha(DATA/'manifest.json'),method_output_sha256=sha(output),
        counts=dict(counts),sample_binding=by_sample,elapsed_s=time.monotonic()-started,
        GT_read_in_command=False,GPU_calls=0))
    print(json.dumps(dict(scene=scene,counts=dict(counts))))


def evaluate(scene):
    from conceptgraph.completion.quality import object_frame_metrics
    verify();guard()
    dest=experiment_directory('nuscenes','validation',RUN_ID)
    path=dest/scene/'method_predictions.jsonl'
    receipt=json.loads((dest/scene/'method_receipt.json').read_text())
    if sha(path)!=receipt['method_output_sha256']:raise AssertionError('method_output_changed')
    labels={r['task_id']:r for r in rows(DATA/'evaluation_only/task_labels.jsonl')}
    truths={(r['protocol'],r['scene'],r['original_frame_idx']):r['objects'] for r in rows(DATA/'evaluation_only/frame_truth.jsonl')}
    metrics=defaultdict(Counter);by_log=defaultdict(Counter);output=[];binding_cache={}
    for row in rows(path):
        protocol=row['protocol'];method=row['method'];key=(protocol,scene,row['original_frame_idx'])
        binding_key=protocol,method,row['sample_token']
        predictions=row['predictions_for_evaluation']
        if binding_key not in binding_cache:
            matches=object_frame_metrics(predictions,truths[key],threshold_m=3.)['matches']
            binding_cache[binding_key]={predictions[m['prediction_index']]['original_index']:truths[key][m['truth_index']]['instance_token'] for m in matches}
        answer=labels[row['task_id']];scorable=answer['label_status']=='scorable';positive=answer['positive']
        selected=row['inference']['selected_original_index'];identity=binding_cache[binding_key].get(selected)
        correct=bool(identity in answer['acceptable_instances']) if scorable and positive else None
        covered=bool(set(binding_cache[binding_key].values())&set(answer['acceptable_instances'])) if scorable and positive else None
        count=dict(tasks=1,positive=int(scorable and bool(positive)),negative=int(scorable and not positive),
            pending=int(not scorable),selected=int(selected is not None),rejected=int(selected is None),
            positive_selected_correct=int(bool(correct)),positive_target_in_visible_map=int(bool(covered)),
            negative_false_selection=int(scorable and not positive and selected is not None),
            pending_selected=int(not scorable and selected is not None))
        metrics[protocol,method].update(count);by_log[protocol,method,row['log']].update(count)
        output.append(dict(task_id=row['task_id'],protocol=protocol,method=method,scene=scene,
            original_frame_idx=row['original_frame_idx'],sample_token=row['sample_token'],
            selected_original_index=selected,selected_identity=identity,label=answer,
            selected_correct=correct,target_in_visible_map=covered))
    jsonl(dest/scene/'evaluation.jsonl',output)
    write(dest/scene/'evaluation_summary.json',dict(scene=scene,
        methods={p:{m:dict(metrics[p,m]) for m in METHODS} for p in ('legacy1115','unified1200')},
        by_log=[dict(protocol=p,method=m,log=l,counts=dict(v)) for (p,m,l),v in by_log.items()],
        method_receipt_sha256=sha(dest/scene/'method_receipt.json'),
        evaluation_sha256=sha(dest/scene/'evaluation.jsonl'),
        scope='fixed historical task universe, not side-view label extension',
        region_correctness_evaluated=False,trajectory_success_evaluated=False))
    print(json.dumps(dict(scene=scene,rows=len(output))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','infer','evaluate'))
    parser.add_argument('--scene');parser.add_argument('--frame-file');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='infer':infer(args.scene,args.frame_file)
    else:evaluate(args.scene)


if __name__=='__main__':main()
