"""Bounded, versioned real-candidate LoRA experiment; no GT in model inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time

from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory

DATA = ROOT / 'dataset/qwen/qnext_20261124_v1'
REPORT = ROOT / 'outputs/reports/qwen_iteration_20261124'
TRAIN = experiment_directory('qwen', 'train', 'qnext_20261124_seed17_v1')
EVAL = experiment_directory('qwen', 'validation', 'qnext_20261124_comparison_v1')
MODEL = ROOT / 'checkpoint/Qwen2-VL-2B-Instruct'
OLD_ADAPTER = ROOT / 'outputs/project_v2/qwen_eval/finetune_v2/best_adapter'
SEED = 17
WORDS = {'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
RANK_KEYS = {'schema_version','selected_node_id','ranking','no_match','explanation','confidence'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,x):
    with Path(p).open('x') as f:
        for r in x:f.write(canonical(r)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()
def disk_guard():
    if shutil.disk_usage(ROOT).free < 20*1024**3: raise RuntimeError('free disk below 20GiB')


def stratum(r):
    if r.get('controlled_fault'):return 'controlled_fault'
    if not r['official_positive']:return 'true_no_match'
    return 'ordinary_covered' if r['response']['selected_node_id'] is not None else 'ordinary_missing'


def relation_witness(r):
    if r['query_kind']!='complex_relation' or r['response']['selected_node_id'] is None:return None
    q=r['query_text'];target=r['response']['selected_node_id']
    m=re.search('以(.+?)为参照',q)
    anchor=next((c for c,w in WORDS.items() if m and m[1]==w),None)
    expected=next((en for cn,en in [('左侧','left_of'),('右侧','right_of'),('前方','in_front_of'),('后方','behind')] if cn in q),None)
    by={n['node_id']:n for n in r['candidate_nodes']}
    return any(e['source']==target and e['target'] in by and by[e['target']]['category']==anchor and e['relation']==expected for e in r['relation_metadata'])


def payload(r):
    return {k:r[k] for k in ['query_text','candidate_nodes','relation_metadata']}


def prompt(r):
    if r['task']=='language_parse':
        return ('只返回JSON对象，提取普通导航指令的动作和文字要求，不判断实际环境或驾驶结果。不得添加结束条件。'
          '固定五字段action,target_description,target_conditions,requested_constraints,needs_clarification。'
          'action只允许pull_over、park_nearest、follow_vehicle、unsupported。'
          '靠边停车对应pull_over/目标描述路边停靠区域/region_type=stopping_region；'
          '最近车位停车对应park_nearest/目标描述最近空闲车位/region_type=individual_parking_space，要求空闲时另加free_occupancy_requirement=free；'
          '跟车对应follow_vehicle/目标描述本车道前车/category=vehicle，本车道前车加lane_relation=same_lane_ahead。'
          'target_conditions是数组，每项恰含attribute,requested_value,grounding_required=true；'
          'attribute可为category,color,lane_relation,region_type,free_occupancy_requirement,parking_permission_requirement。'
          '白/黑车color分别white/black；允许停车表示parking_permission_requirement=allowed；没被占用表示free。'
          'requested_constraints只写文字明确要求的side(left/right)、maximum_speed_mps、minimum_headway_s、duration_s；没有就{}。'
          'needs_clarification是布尔；target_description使用上述标准描述，不填具体坐标。'
          '\n指令：'+r['query_text'])
    ids=r['candidate_ids']
    example=dict(schema_version='1.0.0',selected_node_id=ids[0] if ids else None,ranking=ids,
                 no_match=not bool(ids),explanation='按观测证据判断',confidence=.8)
    return ('只输出一个JSON对象。任务是在真实感知候选图中提议目标编号，不代表路线可行或实际执行。'
      '位置是前视相机地面局部坐标：x沿相机光轴的水平投影向前，y向左，原点为相机地面位置，单位米；'
      'distance_m是相对该原点的水平距离；输入顺序不代表正确顺序。'
      '普通指令按类别、距离和空间关系选择；最近按distance_m，第二近需排除最近。'
      '本受控参照任务采用存在任一参照的协议：目标以外任一给定类别候选满足所述方向即可，'
      '方向轴差必须大于1米、两点距离不超过12米；满足者中按distance_m最近选择。'
      '不要假定有不可见参照，不补充输入中没有的关系、颜色或对象。'
      '仅在指令明确要求证据可靠或授权时检查以下门限：motion_state=static，'
      'observations>=3，lidar_pixels>=5，support_fraction>=0.4，source=current_projected_lidar，'
      'evidence.version等于version，0<=now_s-last_seen_s<=2，conflict=false；任一不满足则不授权该候选。'
      '没有匹配或无法确定时selected_node_id=null且no_match=true；否则no_match=false，选中ID为ranking首项。'
      'ranking必须恰好包含给定全部候选ID一次，不生成新ID。输出只有示例中的六字段。'
      '\n输出格式：'+canonical(example)+'\n输入：'+canonical(payload(r)))


def validate(r,raw):
    try:
        p=json.loads(raw)
        if r['task']=='language_parse':
            from jsonschema import Draft202012Validator
            schema=json.loads((ROOT/'dataset/qwen/multiview_task_20261124_simple_candidates_v3/simple_intent.schema.json').read_text())
            Draft202012Validator(schema).validate(p)
        else:
            if not isinstance(p,dict) or set(p)!=RANK_KEYS:raise ValueError('rank fields')
            if p['schema_version']!='1.0.0' or type(p['no_match']) is not bool:raise ValueError('version/no_match')
            if not isinstance(p['ranking'],list) or any(type(i) is not int for i in p['ranking']):raise ValueError('ranking IDs')
            if len(p['ranking'])!=len(r['candidate_ids']) or set(p['ranking'])!=set(r['candidate_ids']):raise ValueError('ranking candidate coverage')
            selected=p['selected_node_id']
            if selected is not None and type(selected) is not int:raise ValueError('selected ID')
            if p['no_match']:
                if selected is not None:raise ValueError('no_match requires null')
            elif not p['ranking'] or selected!=p['ranking'][0]:raise ValueError('selected must rank first')
            if type(p['confidence']) not in [int,float] or not 0<=p['confidence']<=1:raise ValueError('confidence')
            if not isinstance(p['explanation'],str) or not 1<=len(p['explanation'])<=512:raise ValueError('explanation')
        return True,p,None
    except Exception as e:return False,None,str(e)


def messages(r,answer=False):
    result=[dict(role='user',content=[dict(type='text',text=prompt(r))])]
    if answer:result.append(dict(role='assistant',content=[dict(type='text',text=canonical(r['response']))]))
    return result


def inputs(processor,r,training=False):
    import torch
    text=processor.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True)
    x=processor(text=[text],images=None,padding=False,return_tensors='pt')
    if not training:return x
    full=processor.apply_chat_template(messages(r,True),tokenize=False,add_generation_prompt=False)
    y=processor(text=[full],images=None,padding=False,return_tensors='pt')
    if y['input_ids'].shape[-1]>2048:raise ValueError('refuse truncation '+r['sample_id'])
    labels=y['input_ids'].clone();labels[:,:x['input_ids'].shape[-1]]=-100
    if not (labels!=-100).any():raise ValueError('empty supervision')
    y['labels']=labels
    return y


def sources():
    return [Path(__file__),ROOT/'scripts/qnext_20261124_run.py',ROOT/'conceptgraph/project_paths.py']


def prepare():
    from transformers import AutoProcessor
    disk_guard();DATA.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    original=ROOT/'dataset/qwen/real_graph_v003'
    pilot=ROOT/'dataset/qwen/paper_qwen_20261123_pilot_v1'
    train=rows(pilot/'train.jsonl');val40=rows(pilot/'validation.jsonl');full=rows(original/'validation.jsonl')
    excluded=[r for r in train if relation_witness(r) is False]
    retained=[r for r in train if r not in excluded]
    pool=rows(original/'train.jsonl');used={r['sample_id'] for r in train};replacements=[]
    for old in excluded:
        eligible=[r for r in pool if r['sample_id'] not in used and r['query_kind']==old['query_kind'] and stratum(r)==stratum(old) and relation_witness(r) is True]
        eligible.sort(key=lambda r:hashlib.sha256(('qnext17|'+r['sample_id']).encode()).hexdigest())
        if not eligible:raise RuntimeError('no same-stratum visible-witness replacement')
        selected=eligible[0];retained.append(selected);used.add(selected['sample_id'])
        replacements.append(dict(excluded_id=old['sample_id'],replacement_id=selected['sample_id'],reason='non-null relation target lacks visible typed relation witness'))
    train=retained
    langs=[]
    for r in rows(ROOT/'dataset/qwen/command_review_import_20261124_v2/approved_language_intents.jsonl'):
        langs.append(dict(task='language_parse',sample_id=r['sample_id'],query_text=r['instruction'],response=r['intent'],split='seen_language_regression',split_group_id=r['split_group_id']))
    groups={k:{r['log_token'] for r in v} for k,v in [('train',train),('validation',full)]}
    assert not groups['train']&groups['validation']
    assert not {r['scene_id'] for r in train}&{r['scene_id'] for r in full}
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in full}
    official={role:{i for r in rs for i in r.get('official_acceptable_instances',[])} for role,rs in [('train',train),('validation',full)]}
    assert not official['train']&official['validation']
    for r in train+val40+full:
        valid,_,err=validate(r,canonical(r['response']))
        if not valid:raise ValueError((r['sample_id'],err))
        assert not any(k in payload(r) for k in ['response','official_acceptable_instances','acceptable_node_ids'])
    for name,rs in [('train',train),('checkpoint_validation',val40),('validation_full',full),('language_seen_regression',langs)]:jsonl(DATA/(name+'.jsonl'),rs)
    jsonl(DATA/'training_exclusions.jsonl',[dict(sample_id=r['sample_id'],reason='no visible typed relation witness for supervised non-null target') for r in excluded])
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    token_audit=[]
    for role,rs in [('train',train),('checkpoint_validation',val40),('validation_full',full),('language_seen_regression',langs)]:
        for r in rs:
            x=inputs(processor,r,training=True)
            token_audit.append(dict(role=role,sample_id=r['sample_id'],full_tokens=int(x['input_ids'].shape[-1]),answer_tokens=int((x['labels']!=-100).sum()),prompt_sha256=hashlib.sha256(prompt(r).encode()).hexdigest()))
    jsonl(DATA/'token_audit.jsonl',token_audit)
    protected=[ROOT/'AGENTS.md',ROOT/'MEMORY.md',ROOT/'PAPER_MEMORY.md',ROOT/'paper/当前版_v4/sources/论文正文.md',ROOT/'conceptgraph/completion/semantic_contract.py',ROOT/'scripts/train_qwen_lora_semantic_delegate_v2.py']
    protected += list(original.glob('*.jsonl'))+[original/'manifest.json',pilot/'train.jsonl',pilot/'validation.jsonl',ROOT/'dataset/qwen/command_review_import_20261124_v2/manifest.json']
    protected += [p for p in OLD_ADAPTER.rglob('*') if p.is_file()]
    cfg=dict(created_utc=now(),seed=SEED,train_rows=len(train),checkpoint_validation_rows=len(val40),formal_validation_rows=len(full),language_seen_regression_rows=len(langs),
        train_strata=dict(Counter(stratum(r) for r in train)),checkpoint_strata=dict(Counter(stratum(r) for r in val40)),
        train_logs=sorted(groups['train']),validation_logs=sorted(groups['validation']),official_instance_overlap=0,scene_overlap=0,
        shared_query_texts=len({r['query_text'] for r in train}&{r['query_text'] for r in full}),
        evaluation_claim='same-template cross-log real-candidate selection; not unseen language or closed-loop navigation',
        data_source='historical real causal predicted candidate graph; not newly generated T3 graph; no RGB',
        modifications=replacements,relation_protocol='exists any distinct typed visible anchor, axis gap >1m and distance<=12m, then ego-nearest; common prompt for all variants',
        formal_validation_unchanged=True,ordinary_missing_not_true_negative=True,language_rows_not_training=True,
        hyperparameters=dict(lora_r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],learning_rate=1e-4,batch_size=1,gradient_accumulation=2,max_steps=40,validation_steps=[20,40],max_length=2048,max_new_tokens=256,do_sample=False,dtype='bfloat16'),
        checkpoint_selection='lexicographic ordinary positive correct count, true-negative correct count, schema-valid count on fixed40 only',
        budget=dict(training_gpu_seconds_max=3600,training_wall_seconds_max=2100,total_deadline_utc='2026-11-24T08:08:13+00:00',checkpoint_and_logs_mib=256,new_inputs_and_reports_mib=128),
        original_inputs={str(p):sha(p) for p in [pilot/'train.jsonl',pilot/'validation.jsonl',original/'train.jsonl',original/'validation.jsonl',ROOT/'dataset/qwen/command_review_import_20261124_v2/approved_language_intents.jsonl']},
        source_sha256={str(p):sha(p) for p in sources()},protected_sha256={str(p):sha(p) for p in protected},
        files_sha256={p.name:sha(p) for p in DATA.glob('*.jsonl')},CPU_token_audit_passed=True,max_tokens=max(r['full_tokens'] for r in token_audit),model_calls=0,test_opened=False)
    dump(DATA/'manifest.json',cfg);dump(REPORT/'registration.json',cfg)
    for split,name in [('train','qnext_20261124_seed17_v1'),('validation','qnext_20261124_comparison_v1')]:
        dest=experiment_directory('qwen',split,name,create=True);dump(dest/'registration.json',cfg)
        for p in sources():
            out=dest/'source_snapshot'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    print(canonical(dict(prepared=str(DATA),train=len(train),formal_validation=len(full),excluded=len(excluded),max_tokens=cfg['max_tokens'])),flush=True)


def load_model(adapter=None,trainable=False):
    import torch
    from transformers import Qwen2VLForConditionalGeneration
    from peft import PeftModel
    m=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='eager')
    if adapter:m=PeftModel.from_pretrained(m,str(adapter),is_trainable=trainable)
    return m


def score(r,raw):
    valid,p,err=validate(r,raw)
    result=dict(sample_id=r['sample_id'],task=r['task'],schema_valid=valid,raw_output=raw,normalized_output=p,validation_error=err,fallback_used=False)
    if r['task']=='language_parse':
        exact=valid and p==r['response'];keys=['action','target_description','target_conditions','requested_constraints','needs_clarification']
        result.update(exact_correct=bool(exact),field_correct={k:bool(valid and (sorted([canonical(v) for v in p[k]])==sorted([canonical(v) for v in r['response'][k]]) if k=='target_conditions' else p[k]==r['response'][k])) for k in keys})
    else:
        selected=p.get('selected_node_id') if valid else None
        result.update(query_kind=r['query_kind'],stratum=stratum(r),official_positive=r['official_positive'],candidate_target_coverage=r['candidate_target_coverage'],
            target_correct=bool(valid and selected is not None and selected in r['acceptable_node_ids']),
            null_correct=bool(valid and selected is None and r['response']['selected_node_id'] is None),selected_nonnull=bool(valid and selected is not None),
            relation_visible_witness=relation_witness(r),expected_selected_id=r['response']['selected_node_id'])
    return result


def summarize(records):
    ranking=[r for r in records if r['task']=='candidate_rerank'];langs=[r for r in records if r['task']=='language_parse'];c=Counter()
    for r in ranking:
        c['all']+=1;c['valid']+=r['schema_valid'];c['target_correct']+=r['target_correct']
        c[r['stratum']]+=1;c[r['stratum']+'_valid']+=r['schema_valid'];c[r['stratum']+'_selected']+=r['selected_nonnull']
        c[r['stratum']+'_target_correct']+=r['target_correct'];c[r['stratum']+'_null_correct']+=r['null_correct']
    lp=dict(total=len(langs),schema_valid=sum(r['schema_valid'] for r in langs),exact_correct=sum(r['exact_correct'] for r in langs))
    if langs:lp['fields']={k:sum(r['field_correct'][k] for r in langs) for k in langs[0]['field_correct']}
    return dict(ranking=dict(c),language_seen_regression=lp,ordinary_positive_denominator=c['ordinary_covered']+c['ordinary_missing'],
        ordinary_positive_correct=c['ordinary_covered_target_correct']+c['ordinary_missing_target_correct'],
        ordinary_covered_correct=c['ordinary_covered_target_correct'],true_no_match_wrong_selection=c['true_no_match_selected'],
        controlled_fault_wrong_selection=c['controlled_fault_selected'],fallback_credit=0)


def generate(model,processor,rs,path):
    import torch
    result=[];model.eval();model.config.use_cache=True
    with Path(path).open('x') as out:
        for i,r in enumerate(rs):
            if datetime.now(timezone.utc).isoformat()> '2026-11-24T08:08:13+00:00':raise TimeoutError('overall wall budget')
            x=inputs(processor,r).to(model.device);torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                y=model.generate(**x,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
            torch.cuda.synchronize();raw=processor.tokenizer.decode(y[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
            record=score(r,raw);record.update(latency_s=time.perf_counter()-start,input_tokens=int(x['input_ids'].shape[-1]),output_tokens=int(y.shape[-1]-x['input_ids'].shape[-1]),prompt_sha256=hashlib.sha256(prompt(r).encode()).hexdigest())
            out.write(canonical(record)+'\n');out.flush();result.append(record)
            if (i+1)%20==0:print(canonical(dict(generated=i+1,total=len(rs),file=str(path))),flush=True)
    return result


def train():
    import numpy as np
    import torch
    from transformers import AutoProcessor,get_linear_schedule_with_warmup
    from peft import LoraConfig,get_peft_model
    disk_guard();cfg=json.loads((DATA/'manifest.json').read_text())
    for name,h in cfg['files_sha256'].items():assert sha(DATA/name)==h
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    rs=rows(DATA/'train.jsonl');val=rows(DATA/'checkpoint_validation.jsonl')
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    model=get_peft_model(load_model(),LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,2,40)
    schedule=[];rng=random.Random(SEED)
    while len(schedule)<80:
        epoch=list(rs);rng.shuffle(epoch);schedule.extend(epoch)
    jsonl(TRAIN/'schedule.jsonl',[dict(index=i,sample_id=r['sample_id']) for i,r in enumerate(schedule[:80])])
    curve=[];best=None;best_step=None;validation=[];gpu_start=time.perf_counter();training_seconds=0.;stop='max_steps'
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    for step in range(1,41):
        disk_guard()
        if time.perf_counter()-gpu_start>2100:stop='training wall budget';break
        model.train();model.config.use_cache=False;losses=[];start=time.perf_counter()
        for r in schedule[(step-1)*2:step*2]:
            x=inputs(processor,r,True).to(model.device)
            loss=model(**x).loss
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            (loss/2).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();training_seconds+=time.perf_counter()-start
        curve.append(dict(step=step,loss=sum(losses)/len(losses),training_seconds=training_seconds,lr=scheduler.get_last_lr()[0]))
        print(canonical(curve[-1]),flush=True)
        if step not in [20,40]:continue
        records=generate(model,processor,val,TRAIN/f'validation_step{step}.jsonl');summary=summarize(records)
        key=(summary['ordinary_positive_correct'],summary['ranking'].get('true_no_match_null_correct',0),summary['ranking']['valid'])
        validation.append(dict(step=step,selection_key=key,summary=summary))
        if best is None or key>best:
            model.save_pretrained(TRAIN/'best_adapter',safe_serialization=True);best=key;best_step=step
    dump(TRAIN/'training_report.json',dict(completed_utc=now(),curve=curve,validation=validation,best_step=best_step,
        selection_key=best,best_adapter=str(TRAIN/'best_adapter'),stop_reason=stop,training_compute_s=training_seconds,
        training_process_gpu_wall_s=time.perf_counter()-gpu_start,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),seed=SEED,
        test_used=False,formal171_not_used_for_selection=True,processor_duplicated=False,optimizer_checkpoint_saved=False))


def reliable(n):
    e=n.get('evidence',{});age=n['now_s']-n['last_seen_s']
    return n.get('motion_state')=='static' and e.get('source')=='current_projected_lidar' and e.get('observations',0)>=3 and e.get('lidar_pixels',0)>=5 and e.get('support_fraction',0)>=.4 and e.get('version')==n.get('version') and 0<=age<=2 and e.get('conflict') is False


def rule(r):
    q=r['query_text']
    if r['task']=='language_parse':
        action='park_nearest' if '车位' in q else 'follow_vehicle' if '跟' in q else 'pull_over'
        cs=[];constraints={}
        def cond(a,v):cs.append(dict(attribute=a,requested_value=v,grounding_required=True))
        if action=='follow_vehicle':
            target='本车道前车';cond('category','vehicle');cond('lane_relation','same_lane_ahead')
            for cn,en in [('白色','white'),('黑色','black')]:
                if cn in q:cond('color',en)
        else:
            target='最近空闲车位' if action=='park_nearest' else '路边停靠区域'
            cond('region_type','individual_parking_space' if action=='park_nearest' else 'stopping_region')
            if '空闲' in q or '占用' in q:cond('free_occupancy_requirement','free')
            if '允许停车' in q:cond('parking_permission_requirement','allowed')
            if '左侧' in q:constraints['side']='left'
            if '右侧' in q or '靠右' in q:constraints['side']='right'
        m=re.search('每秒(\d+(?:\.\d+)?)米',q)
        if m:constraints['maximum_speed_mps']=float(m[1])
        return canonical(dict(action=action,target_description=target,target_conditions=cs,requested_constraints=constraints,needs_clarification=False))
    nodes=r['candidate_nodes'];cats=[c for c,w in WORDS.items() if w in q]
    eligible=[n for n in nodes if n['category'] in cats]
    if r['controlled_fault']:eligible=[n for n in eligible if reliable(n)]
    if r['query_kind']=='complex_relation':
        m=re.search('以(.+?)为参照',q);anchor=next((c for c,w in WORDS.items() if m and m[1]==w),None)
        target=next((c for c,w in WORDS.items() if q.endswith(w)),None)
        rel=next((en for cn,en in [('左侧','left_of'),('右侧','right_of'),('前方','in_front_of'),('后方','behind')] if cn in q),None)
        by={n['node_id']:n for n in nodes}
        good={e['source'] for e in r['relation_metadata'] if e['target'] in by and by[e['target']]['category']==anchor and e['relation']==rel}
        eligible=[n for n in nodes if n['category']==target and n['node_id'] in good]
    eligible=sorted(eligible,key=lambda n:(n['distance_m'],n['node_id']))
    k=1 if r['query_kind'] in ['negation','similar_instances'] else 0
    selected=eligible[k]['node_id'] if len(eligible)>k else None
    ids=r['candidate_ids'];ranking=([selected] if selected is not None else [])+[i for i in ids if i!=selected]
    return canonical(dict(schema_version='1.0.0',selected_node_id=selected,ranking=ranking,no_match=selected is None,explanation='fixed geometry/category rule',confidence=.9))


def evaluate(variant):
    disk_guard();rs=rows(DATA/'validation_full.jsonl')+rows(DATA/'language_seen_regression.jsonl')
    dest=EVAL/variant;dest.mkdir(exist_ok=False);start=time.perf_counter()
    if variant=='rule':
        records=[]
        for r in rs:
            st=time.perf_counter();rec=score(r,rule(r));rec.update(latency_s=time.perf_counter()-st,prompt_sha256=hashlib.sha256(prompt(r).encode()).hexdigest());records.append(rec)
        jsonl(dest/'raw_records.jsonl',records);peak=0
    else:
        import torch
        from transformers import AutoProcessor
        adapter=None if variant=='base' else OLD_ADAPTER if variant=='old_adapter' else TRAIN/'best_adapter'
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
        model=load_model(adapter);torch.cuda.reset_peak_memory_stats()
        records=generate(model,processor,rs,dest/'raw_records.jsonl');peak=torch.cuda.max_memory_allocated()/1024**2
    result=dict(variant=variant,completed_utc=now(),wall_s=time.perf_counter()-start,peak_gpu_mib=peak,summary=summarize(records),same_frozen_prompt=True,independent_fresh_model_process=True)
    dump(dest/'report.json',result);print(canonical(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train','evaluate']);parser.add_argument('--variant',choices=['rule','base','old_adapter','new_adapter'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='train':train()
    else:evaluate(args.variant)


if __name__=='__main__':main()
