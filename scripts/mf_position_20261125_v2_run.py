"""38场景因果观测导出与固定多帧位置候选完整验证；不运行GPU或车辆规划。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,gzip,json,pickle,shutil,time,subprocess
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.stage9 import predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.completion.selection_map_transfer import select_frozen_map
from conceptgraph.navigation.mf_position_20261125_v2 import LedgerR1Mapper,POLICY,select_position_candidates
from conceptgraph.navigation.navstage2_repair_diagnosis_20261124 import ROOT,BASE,sha,read,lines,dump
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.streamlined_mapping import MappingFrame

DATA=ROOT/'dataset/processed/mf_position_20261125_v2'
R1=ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
TRANSFER=ROOT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1'
RUN='mf_position_20261125_v2'
START_UTC='2026-11-25T06:39:10+00:00'

def verify(manifest):
    for group in ['source_sha256','input_sha256']:
        for p,h in manifest[group].items():
            if sha(p)!=h:raise ValueError('frozen_changed:'+p)

def guard(dest,begun,budget=2700):
    if time.monotonic()-begun>budget:raise TimeoutError('wall_budget')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise OSError('disk_below_20GiB')
    if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())>128*1024**2:raise OSError('storage_budget')

def prepare():
    DATA.mkdir(exist_ok=False)
    old_release=read(BASE/'main_experiments/causal_retrieval_full_001/release.json')
    r1release=read(R1/'release.json');transfer=read(TRANSFER/'release.json')
    assert len(r1release['scenes'])==38
    queries=[r for r in lines(BASE/'main_experiments/causal_retrieval_full_001/queries.jsonl') if r['method']=='existing_object_joint_rule']
    assert len(queries)==1115 and sum(r['positive'] for r in queries)==733
    tasks=[];inputs={str(R1/'release.json'):sha(R1/'release.json'),str(TRANSFER/'queries.jsonl'):sha(TRANSFER/'queries.jsonl')}
    scenes=[]
    for scene in r1release['scenes']:
        name=scene['scene'];scenes.append(scene)
        r1path=R1/name/'frames.jsonl.gz';inputs[str(r1path)]=sha(r1path)
        expected={r['frame_idx']:r for r in lines(r1path)}
        oldpath=BASE/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl';inputs[str(oldpath)]=sha(oldpath)
        for row in lines(oldpath):
            det=expected[row['frame_idx']]['input']['detections']
            for p in [row['rgb'],row['depth'],det]:inputs[str(p)]=sha(p)
            assert inputs[det]==expected[row['frame_idx']]['input']['detections_sha256']
            tasks.append(dict(scene=name,frame_idx=row['frame_idx'],timestamp_s=row['camera_timestamp_s'],
                rgb=row['rgb'],depth=row['depth'],detections=det,intrinsic=row['intrinsic'],camera_pose=row['camera_pose']))
    assert len(tasks)==1508
    path=DATA/'method_frames.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in tasks));inputs[str(path)]=sha(path)
    # 标签单独留在既有隔离来源；方法指令包不包含正负/可接受实例。
    qpath=DATA/'method_queries.jsonl';qpath.write_text(''.join(json.dumps({k:r[k] for k in ['scene','frame_idx','query_index','query_text']},ensure_ascii=False)+'\n' for r in queries));inputs[str(qpath)]=sha(qpath)
    input_query=BASE/'main_experiments/causal_retrieval_full_001/queries.jsonl'
    sources=[Path(__file__),ROOT/'conceptgraph/navigation/mf_position_20261125_v2.py',
        ROOT/'conceptgraph/navigation/navstage2_repair_diagnosis_20261124.py',
        ROOT/'conceptgraph/completion/selection_map_transfer.py',ROOT/'conceptgraph/completion/evidence.py',
        ROOT/'conceptgraph/navigation/semantic.py',ROOT/'conceptgraph/navigation/types.py',
        ROOT/'conceptgraph/scenegraph/relations.py',ROOT/'conceptgraph/project_paths.py']+[
        ROOT/'conceptgraph/slam'/n for n in ['dense_refinement.py','quality_mapper.py','sparse_observations.py',
        'audited_mapper.py','evidence_mapper.py','streamlined_mapping.py','utils.py','mapping.py']]+[
        ROOT/'conceptgraph/completion'/n for n in ['motion.py','association.py','depth.py','stage9.py']]
    old_manifest=read(ROOT/'dataset/processed/navstage2_repair_20261124_v1/manifest.json')
    assert old_manifest['scene_overlap']==[] and old_manifest['log_overlap']==[]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),task_started_utc=START_UTC,
        git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=17,policy=POLICY,
        scenes=scenes,frames=1508,queries=1115,positive=733,negative=382,split='seen_validation_debugging_38_scenes_11_logs',
        configurations=1,no_parameter_search=True,historical81_used=False,scene_log_overlap_with81=False,
        technical_revision='v1未评价；v2只记录前阶段dense可行但未分配的未决竞争，并保留unknown；原分配/预测/全部门限不变',
        budget=dict(total_wall_s=2700,new_bytes=128*1024**2,GPU_calls=0,minimum_disk_free_bytes=20*1024**3),
        source_sha256={str(p):sha(p) for p in sources},input_sha256=inputs,
        evaluation_only_sha256={str(input_query):sha(input_query)},config=r1release['config'],
        methods=['R1_original_rule','R1_multiframe_position_rule'],
        semantics='只扩无形状稀疏节点的任务级位置候选；原密集门、语义、关系、距离和排序保留；不补造深度/形状；不进行车辆授权或规划',
        evaluation='完整1115同任务733/382；原3m同类一对一；误选/拒选/选对/候选覆盖/逐log并列。GT在选择返回后隔离评价',
        safety='unknown shape never vehicle_safety_authorized; covariance is heuristic, not calibrated probability')
    dump(DATA/'manifest.json',manifest);print(json.dumps(dict(manifest=str(DATA/'manifest.json'),sha256=sha(DATA/'manifest.json'),frames=1508,queries=1115)))

def export():
    manifest=read(DATA/'manifest.json');verify(manifest)
    dest=experiment_directory('nuscenes','validation',RUN,create=True);shutil.copy2(DATA/'manifest.json',dest/'manifest.json')
    for p in manifest['source_sha256']:
        target=dest/'source_snapshot'/Path(p).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    begun=time.monotonic();tasks=list(lines(DATA/'method_frames.jsonl'));counts=Counter();per_scene=[]
    try:
        with gzip.open(dest/'frames.jsonl.gz','wt',compresslevel=6) as out:
            for scene in manifest['scenes']:
                name=scene['scene'];mapper=LedgerR1Mapper(OmegaConf.create(manifest['config']),CLASSES,scene=name)
                expected={r['frame_idx']:r for r in lines(R1/name/'frames.jsonl.gz')};scount=Counter()
                for row in [r for r in tasks if r['scene']==name]:
                    guard(dest,begun)
                    with gzip.open(row['detections'],'rb') as f:gobs=pickle.load(f)
                    idx=row['frame_idx'];np.random.seed(17+idx)
                    frame=MappingFrame(idx,np.asarray(Image.open(row['rgb']).convert('RGB')),
                        np.asarray(Image.open(row['depth']),float)/1000.,np.asarray(row['intrinsic']),
                        np.asarray(row['camera_pose']),copy.deepcopy(gobs),row['rgb'],timestamp_s=row['timestamp_s'])
                    mapper.update(frame)
                    predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
                    if json.loads(json.dumps(predictions,allow_nan=False))!=expected[idx]['predictions']['R1']:
                        raise AssertionError(f'original_prediction_changed:{name}:{idx}')
                    ledger=mapper.ledger(row['timestamp_s'])
                    record=dict(scene=name,log=scene['log_token'],frame_idx=idx,timestamp_s=row['timestamp_s'],
                        original_R1_prediction_exact=True,predictions=predictions,ledger=ledger,
                        GT_in_method=False,vehicle_safety_authorized=False)
                    out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    counts['frames_exact']+=1;counts['raw_observations']+=len(ledger['raw_observations'])
                    counts['sparse_state_frames']+=len(ledger['sparse_states']);scount['frames']+=1
                    for s in ledger['sparse_states']:
                        counts['confirmed_sparse_state_frames']+=int(s['confirmed'])
                        for reason in s['reasons']:counts['reject:'+reason]+=1
                    for e in ledger['events']:counts['event:'+e['kind']]+=1
                per_scene.append(dict(scene=name,counts=dict(scount)));del mapper;out.flush()
                print(json.dumps(dict(scene=name,completed_frames=counts['frames_exact'],elapsed_s=time.monotonic()-begun)),flush=True)
        assert counts['frames_exact']==1508
        verify(manifest)
        dump(dest/'export_summary.json',dict(counts=dict(counts),per_scene=per_scene,elapsed_s=time.monotonic()-begun,
            predictions_exact_all1508=True,GT_in_method=False,GPU_calls=0,
            frame_file_sha256=sha(dest/'frames.jsonl.gz'),source_and_inputs_unchanged=True))
    except BaseException as e:
        dump(dest/'interrupted.json',dict(error=repr(e),counts=dict(counts),elapsed_s=time.monotonic()-begun));raise

def evaluate():
    manifest=read(DATA/'manifest.json');verify(manifest)
    dest=experiment_directory('nuscenes','validation',RUN);assert read(dest/'export_summary.json')['predictions_exact_all1508']
    qgroups=defaultdict(list)
    for q in lines(DATA/'method_queries.jsonl'):qgroups[q['scene'],q['frame_idx']].append(q)
    # 原评价资料单独加载；只在两个选择函数返回之后取对应标注打分。
    for p,h in manifest['evaluation_only_sha256'].items():assert sha(p)==h
    truths={(q['scene'],q['frame_idx'],q['query_index']):q for q in lines(BASE/'main_experiments/causal_retrieval_full_001/queries.jsonl') if q['method']=='existing_object_joint_rule'}
    previous={(q['scene'],q['frame_idx'],q['query_index']):q for q in lines(TRANSFER/'queries.jsonl') if q['method']=='R1_map_original_rule'}
    original={};methodframes={(r['scene'],r['frame_idx']):r for r in lines(DATA/'method_frames.jsonl')}
    for scene in manifest['scenes']:
        for f in lines(R1/scene['scene']/'frames.jsonl.gz'):
            if (f['scene'],f['frame_idx']) in qgroups:original[f['scene'],f['frame_idx']]=f
    summaries=defaultdict(Counter);bylog=defaultdict(lambda:defaultdict(Counter));rows=[];deltas=[];begun=time.monotonic()
    with gzip.open(dest/'node_confirmation.jsonl.gz','wt') as evidenceout:
        for frame in lines(dest/'frames.jsonl.gz'):
            key=frame['scene'],frame['frame_idx']
            if key not in qgroups:continue
            guard(dest,begun)
            pred=frame['predictions'];pose=np.asarray(methodframes[key]['camera_pose'])
            confirms={r['node_id']:r for r in frame['ledger']['sparse_states']}
            for c in confirms.values():evidenceout.write(json.dumps(dict(scene=key[0],frame_idx=key[1],**c),ensure_ascii=False,allow_nan=False)+'\n')
            for query in qgroups[key]:
                baseline=select_frozen_map(query['query_text'],pred,pose,key[1])
                negative_control,_=select_position_candidates(query['query_text'],pred,pose,key[1],{})
                assert [c.node_id for c in negative_control]==[c.node_id for c in baseline]
                assert [c.score for c in negative_control]==[c.score for c in baseline]
                candidate,info=select_position_candidates(query['query_text'],pred,pose,key[1],confirms)
                # GT首次用于当前返回结果的评分；从未进入候选/位置确认函数。
                qid=(*key,query['query_index']);label=truths[qid];gtframe=original[key];truth=gtframe['evaluation_only_official']
                binding={m['prediction_index']:truth[m['truth_index']]['instance_token'] for m in gtframe['metrics']['R1']['matches']}
                accepted=set(label['acceptable_instances']);pair=[]
                for method,ranked in [('R1_original_rule',baseline),('R1_multiframe_position_rule',candidate)]:
                    selected=ranked[0].node_id if ranked else None;identity=binding.get(selected)
                    if method=='R1_original_rule':
                        old=previous[qid]
                        assert old['candidate_ids']==[c.node_id for c in ranked] and old['selected_node_id']==selected and old['evaluation_identity']==identity
                    pos=bool(label['positive']);correct=identity in accepted if pos else None
                    rawcovered=any(v in accepted for v in binding.values()) if pos else None
                    covered=any(binding.get(c.node_id) in accepted for c in ranked) if pos else None
                    sparse=selected is not None and pred[selected].get('shape_available') is False
                    record=dict(method=method,scene=key[0],frame_idx=key[1],query_index=query['query_index'],
                        query_text=query['query_text'],log=frame['log'],positive=pos,selected_index=selected,
                        selected_node_id=None if selected is None else pred[selected]['node_id'],selected_identity=identity,
                        candidate_ids=[c.node_id for c in ranked],candidate_node_ids=[pred[c.node_id]['node_id'] for c in ranked],
                        raw_node_covered=rawcovered,candidate_covered=covered,selected_correct=correct,
                        negative_false_selection=bool(not pos and selected is not None),position_only_selected=bool(sparse),
                        semantic_position_candidate=bool(sparse),vehicle_safety_authorized=False,
                        actual_arrival_credit=False,GT_in_method=False,
                        selected_confirmation=confirms.get(pred[selected]['node_id']) if sparse else None)
                    pair.append(record);rows.append(record)
                    increments=dict(tasks=1,positive=int(pos),negative=int(not pos),positive_selected_correct=int(bool(correct)),
                        positive_raw_covered=int(bool(rawcovered)),positive_candidate_covered=int(bool(covered)),
                        negative_false_selection=int(not pos and selected is not None),rejected=int(selected is None),
                        position_only_selected=int(bool(sparse)),vehicle_safety_authorized=0)
                    summaries[method].update(increments);bylog[frame['log']][method].update(increments)
                if pair[0]['selected_index']!=pair[1]['selected_index']:
                    deltas.append(dict(scene=key[0],frame_idx=key[1],query_index=query['query_index'],baseline=pair[0],candidate=pair[1]))
    assert len(rows)==2230 and all(r['tasks']==1115 and r['positive']==733 and r['negative']==382 for r in summaries.values())
    (dest/'queries.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    dump(dest/'changed_cases.json',deltas);verify(manifest)
    dump(dest/'selection_summary.json',dict(methods={k:dict(v) for k,v in summaries.items()},
        by_log={k:{m:dict(v) for m,v in methods.items()} for k,methods in bylog.items()},
        changed_cases=len(deltas),baseline_exact=True,disabled_new_confirmation_exact=True,
        configuration_count=1,parameter_search=False,historical81_accessed=False,GT_in_method=False,
        GPU_calls=0,planner_calls=0,vehicle_safety_authorized=False,elapsed_s=time.monotonic()-begun,
        queries_sha256=sha(dest/'queries.jsonl'),frames_sha256=sha(dest/'frames.jsonl.gz')))
    print(json.dumps({k:dict(v) for k,v in summaries.items()},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','export','evaluate']);args=p.parse_args()
    {'prepare':prepare,'export':export,'evaluate':evaluate}[args.action]()
