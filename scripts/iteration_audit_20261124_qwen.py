"""CPU-only input checks and independent scoring of saved original Qwen answers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import random
import shutil

from conceptgraph.project_paths import PROJECT_ROOT as ROOT

OUT = ROOT / 'outputs/reports/iteration_independent_audit_20261124'
DATA = ROOT / 'dataset/qwen/qnext_20261124_v1'
EVAL = ROOT / 'outputs/results/qwen/validation/iterations/qnext_20261124_comparison_v1'
WORDS = {'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}


def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def save(name,v):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name
    with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return p
def stratum(r):
    return 'controlled_fault' if r['controlled_fault'] else 'true_no_match' if not r['official_positive'] else 'ordinary_covered' if r['candidate_target_coverage'] else 'ordinary_missing'


def witness(r):
    if r['query_kind']!='complex_relation' or r['response']['selected_node_id'] is None:return None
    q=r['query_text'];m=re.search('以(.+?)为参照',q)
    anchor=next((c for c,w in WORDS.items() if m and m[1]==w),None)
    relation=next((en for cn,en in [('左侧','left_of'),('右侧','right_of'),('前方','in_front_of'),('后方','behind')] if cn in q),None)
    by={n['node_id']:n for n in r['candidate_nodes']}
    return any(e['source']==r['response']['selected_node_id'] and e['target'] in by and by[e['target']]['category']==anchor and e['relation']==relation for e in r['relation_metadata'])


def semantic_selected(r):
    """Visible controlled task protocol, never official instance labels."""
    q=r['query_text'];cats={c for c,w in WORDS.items() if w in q};ns=[n for n in r['candidate_nodes'] if n['category'] in cats]
    if '只有当前证据可靠且目标静止时才授权' in q:
        def valid(n):
            e=n['evidence'];age=n['now_s']-n['last_seen_s']
            return n['motion_state']=='static' and e['source']=='current_projected_lidar' and e['observations']>=3 and e['lidar_pixels']>=5 and e['support_fraction']>=.4 and e['version']==n['version'] and 0<=age<=2 and not e['conflict']
        ns=[n for n in ns if valid(n)]
    if q.startswith('以'):
        m=re.search('以(.+?)为参照',q);a=next((c for c,w in WORDS.items() if m and m[1]==w),None)
        t=next((c for c,w in WORDS.items() if q.endswith(w)),None)
        axis,sign=next((i,s) for cn,i,s in [('左侧',1,1),('右侧',1,-1),('前方',0,1),('后方',0,-1)] if cn in q)
        anchors=[n for n in r['candidate_nodes'] if n['category']==a]
        def satisfies(n):
            return any(n['node_id']!=x['node_id'] and sign*(n['center_xy'][axis]-x['center_xy'][axis])>1 and math.dist(n['center_xy'],x['center_xy'])<=12 for x in anchors)
        ns=[n for n in r['candidate_nodes'] if n['category']==t and satisfies(n)]
    ns.sort(key=lambda n:(n['distance_m'],n['node_id']))
    index=1 if '第二近' in q else 0
    return ns[index]['node_id'] if len(ns)>index else None


def preflight():
    from conceptgraph.vlm.qnext_20261124 import prompt
    cfg=json.loads((DATA/'manifest.json').read_text())
    for p,h in cfg['source_sha256'].items():assert sha(p)==h,('source',p)
    for p,h in cfg['protected_sha256'].items():assert sha(p)==h,('protected',p)
    for p,h in cfg['files_sha256'].items():assert sha(DATA/p)==h,('data',p)
    source=ROOT/'dataset/qwen/real_graph_v003'
    oldtrain={r['sample_id']:r for r in rows(source/'train.jsonl')}
    pilot=rows(ROOT/'dataset/qwen/paper_qwen_20261123_pilot_v1/train.jsonl')
    train=rows(DATA/'train.jsonl');val=rows(DATA/'validation_full.jsonl');val40=rows(DATA/'checkpoint_validation.jsonl');langs=rows(DATA/'language_seen_regression.jsonl')
    assert len(train)==40 and len(val)==171 and len(val40)==40 and len(langs)==16
    assert all(canonical(r)==canonical(oldtrain[r['sample_id']]) for r in train)
    assert [canonical(r) for r in val]==[canonical(r) for r in rows(source/'validation.jsonl')]
    assert [canonical(r) for r in val40]==[canonical(r) for r in rows(ROOT/'dataset/qwen/paper_qwen_20261123_pilot_v1/validation.jsonl')]
    original_ids={r['sample_id'] for r in pilot};new_ids={r['sample_id'] for r in train}
    assert len(original_ids-new_ids)==len(new_ids-original_ids)==1
    assert Counter(stratum(r) for r in train)==Counter(ordinary_covered=20,ordinary_missing=4,true_no_match=4,controlled_fault=12)
    excluded=next(r for r in pilot if r['sample_id'] in original_ids-new_ids)
    replacement=next(r for r in train if r['sample_id'] in new_ids-original_ids)
    pool=[r for r in oldtrain.values() if r['sample_id'] not in original_ids and r['query_kind']==excluded['query_kind'] and stratum(r)==stratum(excluded) and witness(r) is True]
    pool.sort(key=lambda r:hashlib.sha256(('qnext17|'+r['sample_id']).encode()).hexdigest())
    assert witness(excluded) is False and witness(replacement) is True and replacement['sample_id']==pool[0]['sample_id']
    overlaps={}
    for key in ['log_token','scene_id','sample_id']:
        overlaps[key]=len({r[key] for r in train}&{r[key] for r in val})
        assert overlaps[key]==0
    overlaps['official_instances']=len({x for r in train for x in r['official_acceptable_instances']}&{x for r in val for x in r['official_acceptable_instances']})
    assert overlaps['official_instances']==0
    for r in train+val:
        text=prompt(r)
        assert 'official_' not in text and 'acceptable_node_ids' not in text and 'predicted_identity_binding' not in text
        assert not any(x in text for x in r['official_acceptable_instances'])
    ta=rows(DATA/'token_audit.jsonl')
    assert len(ta)==267 and max(r['full_tokens'] for r in ta)<=2048 and min(r['answer_tokens'] for r in ta)>0
    weak={}
    for role,rs in [('train',train),('validation_full',val)]:
        conflicts=[dict(sample_id=r['sample_id'],query_kind=r['query_kind'],stratum=stratum(r),supervised_selected=r['response']['selected_node_id'],visible_semantic_selected=semantic_selected(r)) for r in rs if r['response']['selected_node_id']!=semantic_selected(r)]
        weak[role]=dict(count=len(conflicts),cases=conflicts,meaning='official-object weak labels can disagree with perceptual visible query geometry; retained and not relabeled, complete validation denominator unchanged')
    freeze=json.loads((OUT/'initial_input_freeze.json').read_text())
    protected=[p for p,h in freeze['protected_sha256'].items() if sha(p)!=h]
    assert not protected
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),train_rows=40,full_validation_rows=171,checkpoint_validation_rows=40,language_seen_regression_rows=16,
        overlaps=overlaps,replacement=cfg['modifications'],replacement_verified_against_original_train=True,GT_in_prompt=False,old_validation_unchanged=True,token_audit_max=max(r['full_tokens'] for r in ta),
        original_protected_unchanged=True,weak_supervision=weak,scope='same-template cross-log candidate choice; language16 already-seen regression; no RGB/no route/no arrival evaluation',
        coordinate_source='camera_pose forward ground projection and camera XY origin; not exact ego body frame',model_calls=0,GPU_calls=0,training_calls=0,
        registration_sha256=sha(DATA/'manifest.json'),source_sha256=sha(__file__))
    print(save('qwen_preflight_v1.json',result))
    print(canonical({k:result[k] for k in ['passed','train_rows','full_validation_rows','overlaps','token_audit_max']}))


def parse_rank(raw,r):
    try:
        v=json.loads(raw)
        assert isinstance(v,dict) and set(v)=={'schema_version','selected_node_id','ranking','no_match','explanation','confidence'}
        assert v['schema_version']=='1.0.0' and type(v['no_match']) is bool
        assert isinstance(v['ranking'],list) and all(type(i) is int for i in v['ranking'])
        assert len(v['ranking'])==len(r['candidate_ids']) and set(v['ranking'])==set(r['candidate_ids'])
        assert v['selected_node_id'] is None or type(v['selected_node_id']) is int
        assert (v['selected_node_id'] is None) if v['no_match'] else bool(v['ranking']) and v['selected_node_id']==v['ranking'][0]
        assert type(v['confidence']) in [int,float] and math.isfinite(v['confidence']) and 0<=v['confidence']<=1
        assert isinstance(v['explanation'],str) and 1<=len(v['explanation'])<=512
        return v
    except Exception:return None


def rule_audit():
    from conceptgraph.vlm.qnext_20261124_input_rule import rank_payload, parse_language
    source=rows(DATA/'validation_full.jsonl')+rows(DATA/'language_seen_regression.jsonl')
    old={r['sample_id']:r for r in rows(EVAL/'rule/raw_records.jsonl')}
    new={r['sample_id']:r for r in rows(EVAL/'input_only_rule_v2/raw_records.jsonl')}
    assert len(old)==len(new)==len(source)==187
    differences=[]
    for row in source:
        if row['task']=='candidate_rerank':
            raw=rank_payload({k:row[k] for k in ['query_text','candidate_nodes','relation_metadata']})
            keys=['selected_node_id','ranking','no_match']
        else:
            raw=parse_language(row['query_text']);keys=list(row['response'])
        assert raw==new[row['sample_id']]['raw_output']
        before=json.loads(old[row['sample_id']]['raw_output']);after=json.loads(raw)
        if any(before[k]!=after[k] for k in keys):differences.append(row['sample_id'])
    correction=json.loads((ROOT/'outputs/reports/qwen_iteration_20261124/input_only_rule_v2_correction.json').read_text())
    assert sha(EVAL/'rule/raw_records.jsonl')==correction['original_records_sha256']
    assert sha(EVAL/'input_only_rule_v2/raw_records.jsonl')==correction['replacement_records_sha256']
    assert len(differences)==correction['semantic_changed_rows']==0
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        old_rule_ineligible_for_fair_comparison=True,old_inference_hidden_fields=['query_kind','controlled_fault'],
        replacement_inference_fields=['query_text','candidate_nodes','relation_metadata'],candidate_ids_derived_from_visible_nodes=True,
        inference_functions_independently_replayed_CPU=187,semantic_changed_rows=0,
        task_interface_dispatch='Separate candidate ranking and language parsing interfaces, as exposed in the common model prompt; task flag is not passed into the ranking decision.',
        old_raw_preserved=True,source_sha256=sha(ROOT/'conceptgraph/vlm/qnext_20261124_input_rule.py'),
        model_calls=0,GPU_calls=0,training_calls=0)
    print(save('qwen_input_only_rule_v2_audit.json',result))


def final():
    from jsonschema import Draft202012Validator
    cfg=json.loads((DATA/'manifest.json').read_text())
    amendment=json.loads((ROOT/'outputs/reports/qwen_iteration_20261124/compatibility_amendment_v1.json').read_text())
    assert sha(ROOT/'conceptgraph/vlm/qnext_20261124.py')==amendment['runtime_source_sha256']
    for p,h in cfg['source_sha256'].items():
        assert sha(p)==(amendment['runtime_source_sha256'] if p.endswith('/conceptgraph/vlm/qnext_20261124.py') else h)
    for p,h in cfg['protected_sha256'].items():assert sha(p)==h
    for p,h in cfg['files_sha256'].items():assert sha(DATA/p)==h
    freeze=json.loads((OUT/'initial_input_freeze.json').read_text())
    assert all(sha(p)==h for p,h in freeze['protected_sha256'].items())
    post=json.loads((ROOT/'outputs/reports/qwen_iteration_20261124/postprocessing_source_registration.json').read_text())
    post_amend=json.loads((ROOT/'outputs/reports/qwen_iteration_20261124/postprocessing_amendment_v2.json').read_text())
    assert all(sha(p)==(post_amend['report_source_sha256'] if p.endswith('/conceptgraph/vlm/qnext_20261124_report.py') else h) for p,h in post['sha256'].items())
    fingerprint=json.loads((ROOT/'outputs/reports/qwen_iteration_20261124/base_model_fingerprint.json').read_text())
    for name,h in fingerprint['config_sha256'].items():assert sha(Path(fingerprint['path'])/name)==h
    for record in fingerprint['weight_files']:
        stat=(Path(fingerprint['path'])/record['name']).stat()
        assert stat.st_size==record['size_bytes'] and stat.st_mtime_ns==record['mtime_ns']
    train_dir=ROOT/'outputs/results/qwen/train/iterations/qnext_20261124_seed17_v1'
    training=json.loads((train_dir/'training_report.json').read_text())
    select_rows={r['sample_id']:r for r in rows(DATA/'checkpoint_validation.jsonl')}
    selection=[]
    for entry in training['validation']:
        raw=rows(train_dir/f"validation_step{entry['step']}.jsonl")
        assert len(raw)==40 and {r['sample_id'] for r in raw}==set(select_rows)
        correct=negative=valid_count=0
        for rec in raw:
            row=select_rows[rec['sample_id']];parsed=parse_rank(rec['raw_output'],row);valid=parsed is not None
            valid_count+=valid
            selected=parsed['selected_node_id'] if valid else None
            correct+=bool(valid and row['official_positive'] and not row['controlled_fault'] and selected is not None and selected in row['acceptable_node_ids'])
            negative+=bool(valid and not row['official_positive'] and not row['controlled_fault'] and selected is None)
        key=[correct,negative,valid_count];assert key==entry['selection_key'];selection.append(dict(step=entry['step'],key=key))
    best=max(selection,key=lambda r:r['key'])
    assert best['step']==training['best_step']==20
    assert len(training['curve'])==40 and training['seed']==17 and training['training_process_gpu_wall_s']<=3600
    schedule=rows(train_dir/'schedule.jsonl');expected=[];rng=random.Random(17);train_rows=rows(DATA/'train.jsonl')
    for _ in range(2):
        epoch=list(train_rows);rng.shuffle(epoch);expected.extend(r['sample_id'] for r in epoch)
    assert [r['sample_id'] for r in schedule]==expected
    schema=json.loads((ROOT/'dataset/qwen/multiview_task_20261124_simple_candidates_v3/simple_intent.schema.json').read_text())
    rs=rows(DATA/'validation_full.jsonl')+rows(DATA/'language_seen_regression.jsonl');by={r['sample_id']:r for r in rs}
    hashes={r['sample_id']:r['prompt_sha256'] for r in rows(DATA/'token_audit.jsonl') if r['role'] in ['validation_full','language_seen_regression']}
    outputs={};details=[]
    for variant in ['input_only_rule_v2','base','old_adapter','new_adapter']:
        p=EVAL/variant/'raw_records.jsonl';raw=rows(p)
        assert len(raw)==187 and len({r['sample_id'] for r in raw})==187 and {r['sample_id'] for r in raw}==set(by)
        c=Counter();perlog=defaultdict(Counter)
        for rec in raw:
            r=by[rec['sample_id']];assert rec['prompt_sha256']==hashes[r['sample_id']]
            assert not rec['fallback_used']
            if r['task']=='candidate_rerank':
                parsed=parse_rank(rec['raw_output'],r);valid=parsed is not None;s=stratum(r);c[s]+=1;c['ranking_all']+=1;c['ranking_schema_valid']+=valid
                selected=parsed['selected_node_id'] if valid else None
                correct=valid and selected is not None and selected in r['acceptable_node_ids']
                null=valid and selected is None and r['response']['selected_node_id'] is None
                c[s+'_schema_valid']+=valid;c[s+'_format_invalid']+=not valid
                c[s+'_target_correct']+=correct;c[s+'_null_correct']+=null;c[s+'_selected_nonnull']+=bool(valid and selected is not None)
                assert valid==rec['schema_valid'] and correct==rec['target_correct'] and null==rec['null_correct']
                perlog[r['log_token']].update({s:1,s+'_target_correct':int(correct),s+'_null_correct':int(null)})
                details.append(dict(variant=variant,sample_id=r['sample_id'],stratum=s,schema_valid=valid,target_correct=bool(correct),null_correct=bool(null),selected=selected,visible_semantic_selected=semantic_selected(r)))
            else:
                c['language_all']+=1
                try:
                    parsed=json.loads(rec['raw_output']);Draft202012Validator(schema).validate(parsed);valid=True
                except Exception:parsed=None;valid=False
                exact=valid and parsed==r['response'];c['language_schema_valid']+=valid;c['language_exact']+=exact
                assert valid==rec['schema_valid'] and exact==rec['exact_correct']
        c['ordinary_positive_denominator']=c['ordinary_covered']+c['ordinary_missing'];c['ordinary_positive_correct']=c['ordinary_covered_target_correct']+c['ordinary_missing_target_correct']
        outcomes={s:dict(correct_rejection=c[s+'_null_correct'],valid_wrong_selection=c[s+'_selected_nonnull'],format_invalid=c[s+'_format_invalid'],denominator=c[s]) for s in ['true_no_match','controlled_fault']}
        assert all(sum(v[k] for k in ['correct_rejection','valid_wrong_selection','format_invalid'])==v['denominator'] for v in outcomes.values())
        outputs[variant]=dict(counts=dict(c),negative_outcomes=outcomes,per_log={k:dict(v) for k,v in perlog.items()},raw_sha256=sha(p))
    report_dir=ROOT/'outputs/reports/qwen_iteration_20261124'
    def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    sizes={str(p.relative_to(ROOT)):size(p) for p in [DATA,report_dir,train_dir,EVAL]}
    assert size(DATA)+size(report_dir)<=128*2**20 and size(train_dir)+size(EVAL)<=256*2**20
    assert shutil.disk_usage(ROOT).free>=20*2**30
    audit_paths=[OUT,*list((ROOT/'outputs/results/nuscenes/validation/iterations').glob('iteration_independent_audit_20261124_*')),*list((ROOT/'outputs/results/qwen/validation/iterations').glob('iteration_independent_audit_20261124_*'))]
    audit_bytes=sum(size(p) for p in audit_paths)
    assert audit_bytes<=64*2**20
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),variants=outputs,raw_records=details,
        full_ranking_denominator=171,full_language_seen_regression_denominator=16,no_fallback_credit=True,
        protected_unchanged=True,frozen_inputs_unchanged=True,runtime_source_sha256=amendment['runtime_source_sha256'],
        fair_rule_variant='input_only_rule_v2',old_rule_v1_ineligible=True,
        selection_independently_recomputed=selection,best_step=best['step'],selection_only_40=True,formal171_includes_selection40=True,
        schedule_seed17_recomputed=True,base_config_hash_and_weight_size_mtime_unchanged=True,base_weights_not_newly_hashed=True,
        old_adapter_protected_SHA_unchanged=True,training_gpu_wall_s=training['training_process_gpu_wall_s'],
        size_bytes=sizes,audit_total_bytes_before_this_report=audit_bytes,
        new_adapter_sha256={p.name:sha(p) for p in (train_dir/'best_adapter').iterdir() if p.is_file()},
        compatibility_amendment='Only eager attention backend changed before training; same across all neural variants',
        postprocessing_report_source_sha256=post_amend['report_source_sha256'],
        model_calls=0,GPU_calls=0,training_calls=0,source_sha256=sha(__file__))
    print(save('qwen_independent_final_v1.json',result))
    print(canonical({k:v['counts'] for k,v in outputs.items()}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','rule','final']);a=ap.parse_args()
    {'preflight':preflight,'rule':rule_audit,'final':final}[a.mode]()
