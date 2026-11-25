"""Input-only rule reference. No task-family, fault flag or GT enters inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from conceptgraph.vlm.qnext_20261124 import (
    DATA,EVAL,REPORT,WORDS,canonical,dump,jsonl,now,payload,prompt,rows,score,sha,summarize,
)


def is_reliable(node):
    evidence=node.get('evidence',{})
    age=node['now_s']-node['last_seen_s']
    return (node.get('motion_state')=='static' and evidence.get('source')=='current_projected_lidar'
        and evidence.get('observations',0)>=3 and evidence.get('lidar_pixels',0)>=5
        and evidence.get('support_fraction',0)>=.4 and evidence.get('version')==node.get('version')
        and 0<=age<=2 and evidence.get('conflict') is False)


def rank_payload(observed):
    assert set(observed)=={'query_text','candidate_nodes','relation_metadata'}
    query=observed['query_text'];nodes=observed['candidate_nodes'];edges=observed['relation_metadata']
    ids=[n['node_id'] for n in nodes]
    categories=[category for category,word in WORDS.items() if word in query]
    eligible=[n for n in nodes if n['category'] in categories]
    supported=bool(categories) and ('选择' in query)
    if '证据可靠' in query or '授权' in query:eligible=[n for n in eligible if is_reliable(n)]
    if '为参照' in query:
        match=re.fullmatch('以(.+?)为参照，选择其(左侧|右侧|前方|后方)最近的(.+)',query)
        category_by_word={word:category for category,word in WORDS.items()}
        if not match or match[1] not in category_by_word or match[3] not in category_by_word:
            supported=False;eligible=[]
        else:
            anchor=category_by_word[match[1]];target=category_by_word[match[3]]
            relation={'左侧':'left_of','右侧':'right_of','前方':'in_front_of','后方':'behind'}[match[2]]
            by_id={n['node_id']:n for n in nodes}
            good={e['source'] for e in edges if e['source']!=e['target'] and e['target'] in by_id
                  and by_id[e['target']]['category']==anchor and e['relation']==relation}
            eligible=[n for n in eligible if n['category']==target and n['node_id'] in good]
    eligible=sorted(eligible,key=lambda n:(n['distance_m'],n['node_id']))
    index=1 if '第二近' in query else 0
    selected=eligible[index]['node_id'] if supported and len(eligible)>index else None
    ranking=([selected] if selected is not None else [])+[i for i in ids if i!=selected]
    return canonical(dict(schema_version='1.0.0',selected_node_id=selected,ranking=ranking,no_match=selected is None,
                          explanation='input-only category/distance/relation rule',confidence=.9))


def parse_language(query):
    """A predefined small vocabulary; not an exact sentence lookup or learned model."""
    action='park_nearest' if '车位' in query else 'follow_vehicle' if '跟' in query else 'pull_over'
    conditions=[];constraints={}
    def condition(attribute,value):conditions.append(dict(attribute=attribute,requested_value=value,grounding_required=True))
    if action=='follow_vehicle':
        target='本车道前车';condition('category','vehicle');condition('lane_relation','same_lane_ahead')
        for word,value in [('白色','white'),('黑色','black')]:
            if word in query:condition('color',value)
    else:
        target='最近空闲车位' if action=='park_nearest' else '路边停靠区域'
        condition('region_type','individual_parking_space' if action=='park_nearest' else 'stopping_region')
        if '空闲' in query or '占用' in query:condition('free_occupancy_requirement','free')
        if '允许停车' in query:condition('parking_permission_requirement','allowed')
        if '左侧' in query:constraints['side']='left'
        if '右侧' in query or '靠右' in query:constraints['side']='right'
    match=re.search('每秒(\d+(?:\.\d+)?)米',query)
    if match:constraints['maximum_speed_mps']=float(match[1])
    return canonical(dict(action=action,target_description=target,target_conditions=conditions,
                          requested_constraints=constraints,needs_clarification=False))


def main():
    source=rows(DATA/'validation_full.jsonl')+rows(DATA/'language_seen_regression.jsonl')
    dest=EVAL/'input_only_rule_v2';dest.mkdir(exist_ok=False);records=[];start=time.perf_counter()
    for row in source:
        tick=time.perf_counter()
        raw=parse_language(row['query_text']) if row['task']=='language_parse' else rank_payload(payload(row))
        record=score(row,raw)
        record.update(latency_s=time.perf_counter()-tick,prompt_sha256=hashlib.sha256(prompt(row).encode()).hexdigest())
        records.append(record)
    jsonl(dest/'raw_records.jsonl',records)
    report=dict(variant='input_only_rule_v2',completed_utc=now(),wall_s=time.perf_counter()-start,peak_gpu_mib=0,
        summary=summarize(records),inference_allowed_inputs=['query_text','candidate_nodes','relation_metadata'],
        candidate_ids_source='candidate_nodes',language_rule='predefined word/category patterns, no exact sentence/sample-id lookup; seen16 regression only',
        source_sha256=sha(Path(__file__)),same_frozen_prompt=True,independent_fresh_model_process=True)
    dump(dest/'report.json',report)
    old={r['sample_id']:r for r in rows(EVAL/'rule/raw_records.jsonl')};changes=[]
    for record in records:
        before=old[record['sample_id']];a=before['normalized_output'];b=record['normalized_output']
        keys=['selected_node_id','ranking','no_match'] if record['task']=='candidate_rerank' else list(a)
        diff={key:dict(old=a[key],new=b[key]) for key in keys if a[key]!=b[key]}
        if diff:changes.append(dict(sample_id=record['sample_id'],differences=diff))
    dump(REPORT/'input_only_rule_v2_correction.json',dict(created_utc=now(),
        reason='Original rule read hidden query_kind and controlled_fault. Disqualified original v1 from fair comparison and retained all original records.',
        original_rule_path=str(EVAL/'rule'),replacement_rule_path=str(dest),
        input_permission_corrected=True,score_selection_applied=False,semantic_changed_rows=len(changes),changes=changes,
        original_records_sha256=sha(EVAL/'rule/raw_records.jsonl'),replacement_records_sha256=sha(dest/'raw_records.jsonl'),
        protocol='Evidence reliability, second-nearest, reference category/direction are parsed from query text only; IDs derived from visible nodes.'))
    print(canonical(report))


if __name__=='__main__':main()
