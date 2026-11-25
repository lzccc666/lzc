"""只读旧诊断并等价重放冻结L1，保留全部生命周期/前视未匹配来源证据。"""
from __future__ import annotations

from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper
from scripts.multiview_20261123_run_graph import load_frame


P='mvrepair_lifecycle_20261123_'
REPORT=PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'
BASE=PROJECT/'outputs/results/nuscenes/validation/iterations'
L1=BASE/(P+'candidate_L1_v1')


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def csv_write(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def main():
    started=time.monotonic()
    freeze=json.loads((L1/(P+'release.json')).read_text())
    assert all(digest(p)==v['sha256'] for p,v in freeze['source_sha256'].items())
    run=experiment_directory('nuscenes','validation',P+'L1_provenance_v1',create=True)
    shutil.copy2(Path(__file__),run/Path(__file__).name)
    dump(run/'release.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(Path(__file__).resolve()):digest(__file__)},frozen_L1_release_sha256=digest(L1/(P+'release.json')),
        purpose='exact frozen L1 replay with passive real observation/source provenance; not a new candidate',GT_in_inference=False))
    manifest=json.loads((PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json').read_text())
    all_source, original_rows, retirement_audits=[],{},[]
    for scene in manifest['scenes']:
        name=scene['scene']
        old=list(map(json.loads,gzip.open(BASE/'mviter_20261123_diagnosis_v1'/('mviter_20261123_'+name+'.jsonl.gz'),'rt')))
        expected=list(map(json.loads,gzip.open(L1/name/(P+'predictions.jsonl.gz'),'rt')))
        mapper=LifecycleRetiringMapper(OmegaConf.create(freeze['original_config']),CLASSES)
        observer=mapper.lifecycle.observe_group
        captured=[]
        def observe(group_id,timestamp_s,observations):
            captured[:]=copy.deepcopy(observations)
            return observer(group_id,timestamp_s,observations)
        mapper.lifecycle.observe_group=observe
        with gzip.open(run/(name+'.jsonl.gz'),'wt') as out:
            for group,reference,original in zip(scene['groups'],expected,old):
                assert time.monotonic()-started<1200 and shutil.disk_usage(PROJECT).free>20*2**30
                np.random.seed(17+group['frame_idx'])
                frames=[(c,load_frame(group,c)) for c in group['views']]
                mapper.capture_times={c:group['views'][c]['camera_timestamp_s'] for c,_ in frames}
                mapper.update_group(frames)
                predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frames[0][1])
                assert predictions==reference['predictions']['L1_lifecycle']
                assert mapper.lifecycle_audit==reference['audit']['L1_lifecycle']
                states=[*mapper.position_layer.tracks,*mapper.position_layer.dense_states.values()]
                def readonly_root(node):
                    parent=mapper.lifecycle.parent
                    while node in parent and parent[node]!=node:node=parent[node]
                    return node
                evidence={}
                for p in predictions:
                    members=[s for s in states if readonly_root(mapper.position_layer.aliases.get(s['node_id'],s['node_id']))==p['node_id']]
                    evidence[p['node_id']]=dict(source_object_ids=[o['stable_object_id'] for o in mapper.objects
                        if readonly_root(mapper.position_layer.aliases.get(o['stable_object_id'],o['stable_object_id']))==p['node_id']],
                        observation_keys=[k for s in members for k in s.get('_observation_keys',[])],
                        source_cameras=[c for s in members for c in s.get('_source_cameras',[])],
                        state_timestamp_s=max((s['timestamp'] for s in members),default=None),
                        raw_state_ids=[s['node_id'] for s in members])
                item=serial(dict(scene=name,frame_idx=group['frame_idx'],reference_timestamp_s=group['reference_timestamp_s'],
                    predictions=predictions,evidence=evidence,current_real_observations=captured,internal=mapper.internal_snapshot()))
                out.write(json.dumps(item,allow_nan=False)+'\n')
                all_source.append(item)
                original_rows[name,group['frame_idx']]=original
                for retired in mapper.lifecycle_audit['retired_states']:
                    accepted=[w for w in mapper.lifecycle_audit['registry']['witnesses'] if w['accepted']]
                    relevant=[w for w in accepted if retired['logical_id'] in [w['track_a'],w['track_b']]]
                    checks=[]
                    for w in relevant:
                        a=next(o for o in captured if o['track_id']==w['track_a'])
                        b=next(o for o in captured if o['track_id']==w['track_b'])
                        pa,pb=np.asarray(a['real_points']),np.asarray(b['real_points'])
                        small,large=(pa,pb) if len(pa)<=len(pb) else(pb,pa)
                        count=int(np.sum(cKDTree(large).query(small)[0]<=.2))
                        assert count==w['shared_points'] and count/len(small)==w['shared_fraction']
                        checks.append(dict(witness=w,independent_shared_count=count,independent_smaller_count=len(small),
                            raw_source_keys=[a['source_key'],b['source_key']],source_real_points_recomputed=True))
                    assert checks
                    retirement_audits.append(dict(scene=name,frame_idx=group['frame_idx'],retired=retired,checks=checks))
    # 以下只读评价结果：官方身份不进入上面的候选/退休过程。
    output_rows,group_rows,alias_rows=[],[],[]
    totals=Counter()
    for item in all_source:
        key=(item['scene'],item['frame_idx']);original=original_rows[key]
        for method,record in [*original['methods'].items(),('L1_lifecycle',item)]:
            states=record['internal']['states'];aliases=record['internal']['aliases']
            counts=Counter(e['node_id'] for e in record['internal']['position_events'])
            totals[method,'same_group_multi_events']+=sum(v>1 for v in counts.values())
            for raw,target in aliases.items():alias_rows.append(dict(scene=key[0],frame_idx=key[1],method=method,raw_id=raw,alias=target,
                raw_state_live=any(s['node_id']==raw for s in states),alias_target_live_as_raw_state=any(s['node_id']==target for s in states)))
            for p in record['predictions']:
                ev=record['evidence'].get(p['node_id'],{});now=original['reference_timestamp_s'];st=ev.get('state_timestamp_s')
                row=dict(scene=key[0],frame_idx=key[1],method=method,node_id=p['node_id'],category=p['category'],
                    motion_state=p.get('motion_state'),num_detections=p.get('num_detections'),last_seen_frame=p.get('last_seen_frame'),
                    group_timestamp_s=now,object_timestamp_s=p.get('timestamp_s'),state_timestamp_s=st,
                    prediction_age_s=p.get('prediction_age_s'),source_cameras=ev.get('source_cameras',[]),
                    observation_keys=ev.get('observation_keys',[]),source_object_ids=ev.get('source_object_ids',[]))
                output_rows.append(row);totals[method,'outputs']+=1
                totals[method,'motion_unknown_singleton']+=p.get('motion_state')=='unknown' and p.get('num_detections')==1
                totals[method,'category_unknown']+=p['category']=='unknown'
                totals[method,'object_state_clock_disagreement']+=st is not None and p.get('timestamp_s') is not None and abs(st-p['timestamp_s'])>1e-6
                totals[method,'old_without_prediction_age']+=p.get('timestamp_s') is not None and now-p['timestamp_s']>.75 and p.get('prediction_age_s') is None
    dump(REPORT/'L1_three_retirements_independent_audit.json',retirement_audits)
    csv_write(REPORT/'all_output_lifecycle_sources.csv',output_rows);csv_write(REPORT/'all_alias_lifecycle.csv',alias_rows)
    src={(r['scene'],r['frame_idx']):r for r in all_source}
    unmatched,paired=[],[]
    for scene in manifest['scenes']:
        name=scene['scene']
        for ev in map(json.loads,gzip.open(L1/name/(P+'evaluation.jsonl.gz'),'rt')):
            frame=ev['frame_idx'];gt=ev['evaluation_only_official']['front'];parts={}
            for method in ['A_front','C_three_dedup','L1_lifecycle']:
                result=ev['result'][method]['front'];preds=result['predictions'];matches={m['prediction_index']:m for m in result['metrics']['matches']}
                record=src[name,frame] if method=='L1_lifecycle' else original_rows[name,frame]['methods'][method]
                part=[]
                for i,p in enumerate(preds):
                    e=record['evidence'][p['node_id']];keys={tuple(k) for k in e['observation_keys']}
                    nearest=[(float(np.linalg.norm(np.asarray(p['center'][:2])-o['center'][:2])),o['instance_token']) for o in gt if o['category']==p['category']]
                    row=dict(scene=name,frame_idx=frame,method=method,prediction_index=i,node_id=p['node_id'],category=p['category'],center=p['center'],
                        matched=i in matches,matched_instance=gt[matches[i]['truth_index']]['instance_token'] if i in matches else None,
                        nearest_same_category_gt_xy_m=min(nearest)[0] if nearest else None,
                        source_cameras=e['source_cameras'],source_object_ids=e['source_object_ids'],observation_keys=e['observation_keys'],
                        last_key_camera=e['observation_keys'][-1][0] if e['observation_keys'] else None,
                        object_timestamp_s=p.get('timestamp_s'),state_timestamp_s=e.get('state_timestamp_s'),prediction_age_s=p.get('prediction_age_s'),
                        motion_state=p.get('motion_state'),num_detections=p.get('num_detections'))
                    part.append((row,keys))
                    if i not in matches:unmatched.append(row)
                parts[method]=part
            # 只按实测sourcekey共享配对身份流；没有共享sourcekey的全部另列，绝不按GT创造推理配对。
            pairs=[]
            for i,(a,ka) in enumerate(parts['A_front']):
                for j,(b,kb) in enumerate(parts['L1_lifecycle']):
                    overlap=len(ka&kb)
                    if a['category']==b['category'] and overlap:pairs.append((overlap,i,j))
            used_a,used_l=set(),set()
            for overlap,i,j in sorted(pairs,reverse=True):
                if i in used_a or j in used_l:continue
                used_a.add(i);used_l.add(j);a=parts['A_front'][i][0];b=parts['L1_lifecycle'][j][0]
                paired.append(dict(scene=name,frame_idx=frame,kind='shared_source_stream',shared_source_keys=overlap,
                    A=a,L1=b,unmatched_delta=int(not b['matched'])-int(not a['matched'])))
            for method,used in [('A_front',used_a),('L1_lifecycle',used_l)]:
                for i,(r,_) in enumerate(parts[method]):
                    if i not in used:paired.append(dict(scene=name,frame_idx=frame,kind='source_stream_only_in_'+method,
                        **{method:r},unmatched_delta=(1 if method=='L1_lifecycle' else -1)*int(not r['matched'])))
            group_rows.append(dict(scene=name,frame_idx=frame,
                **{m+'_'+k:v for m in parts for k,v in ev['result'][m]['front']['metrics'].items() if k in ['predictions','true_positive','unmatched_predictions','duplicate_nodes']}))
    csv_write(REPORT/'all_front_unmatched_sources.csv',unmatched);csv_write(REPORT/'front_per_group_counts.csv',group_rows)
    dump(REPORT/'front_A_vs_L1_source_stream_pairing.json',paired)
    delta=[p for p in paired if p['unmatched_delta']]
    dump(REPORT/'front_unmatched_difference_all_cases.json',delta)
    summary=dict(run=str(run),exact_replayed_groups=40,retirement_events=len(retirement_audits),
        retirement_shared_points_independently_recomputed=True,
        all_unmatched_rows=Counter(r['method'] for r in unmatched),
        unmatched_difference_net=sum(p['unmatched_delta'] for p in paired),
        difference_case_count=len(delta),lifecycle_counts={str(k):v for k,v in totals.items()},
        wall_s=time.monotonic()-started,GT_in_inference=False,
        frozen_L1_sources_unchanged=all(digest(p)==v['sha256'] for p,v in freeze['source_sha256'].items()))
    dump(run/'report.json',summary);dump(REPORT/'L1_provenance_diagnosis_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
