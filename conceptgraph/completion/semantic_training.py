"""新目录内的有界真实CUDA LoRA训练；只以隔离验证集选模。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import tarfile
import time

import numpy as np
from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from conceptgraph.completion.semantic_contract import build_prompt, validate
from conceptgraph.completion.semantic_dataset import audit_rows
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.vlm.image_input import prepare_qwen_image
from scripts.train_qwen_lora_semantic_delegate_v2 import (
    _load_model, _set_seed, multimodal_collate, save_checkpoint, _restore_rng_state,
)

FIELDS=['target_categories','target_colors','anchor_categories','anchor_colors','relation',
        'prefer_nearest','prefer_front','excluded_categories','excluded_colors','requires_static']


def messages(row, image, answer=False):
    content=[]
    if image is not None:
        content.append(dict(type='image',image=image))
    content.append(dict(type='text',text=build_prompt(row['task'],row)))
    result=[dict(role='user',content=content)]
    if answer:
        result.append(dict(role='assistant',content=[dict(type='text',text=json.dumps(row['response'],ensure_ascii=False,sort_keys=True,separators=(',',':')))]))
    return result


def image_for(row):
    if row['task']!='object_attributes':
        return None
    with Image.open(row['crop_path']) as image:
        return prepare_qwen_image(image,448).copy()


def inputs_for(processor,row,training=False):
    image=image_for(row)
    kwargs=dict(images=[image] if image is not None else None,padding=False,return_tensors='pt')
    prompt=processor.apply_chat_template(messages(row,image),tokenize=False,add_generation_prompt=True)
    x=processor(text=[prompt],**kwargs)
    if not training:
        return x
    full=processor.apply_chat_template(messages(row,image,True),tokenize=False,add_generation_prompt=False)
    result=processor(text=[full],**kwargs)
    length=int(result['input_ids'].shape[-1])
    if length>2048:
        raise ValueError(f"refuse truncation: {row['sample_id']} length={length}")
    result={k:(v if k=='image_grid_thw' else v.squeeze(0)) for k,v in result.items() if torch.is_tensor(v)}
    labels=result['input_ids'].clone()
    labels[:x['input_ids'].shape[-1]]=-100
    if not (labels!=-100).any():
        raise ValueError('empty supervised sequence')
    result['labels']=labels
    return result


def evaluate(model,processor,rows,max_tokens=256):
    results=[]
    totals=defaultdict(Counter)
    model.eval()
    model.config.use_cache=True
    for row in rows:
        x=inputs_for(processor,row).to(model.device)
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**x,max_new_tokens=max_tokens,do_sample=False,temperature=None,
                top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        raw=processor.tokenizer.decode(output[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
        seconds=time.perf_counter()-start
        v=validate(row['task'],raw,allowed_candidate_ids=row.get('candidate_ids'))
        pred=v.normalized_output if v.valid else {}
        task=row['task']
        totals[task]['count']+=1
        totals[task]['schema_valid']+=v.valid
        try:
            strict_json=isinstance(json.loads(raw),dict)
        except (TypeError,ValueError):
            strict_json=False
        totals[task]['strict_json']+=strict_json
        if task=='query_parse':
            correct=v.valid and all(pred.get(f)==row['response'][f] for f in FIELDS)
            for f in FIELDS:
                totals[task][f+'_correct']+=v.valid and pred.get(f)==row['response'][f]
        elif task=='object_attributes':
            correct=pred.get('object_tag')==row['response']['object_tag']
        else:
            acceptable=row.get('acceptable_node_ids',[row['response']['selected_node_id']])
            correct=pred.get('selected_node_id') in acceptable and v.valid
        totals[task]['correct']+=correct
        results.append(dict(sample_id=row['sample_id'],scene_id=row['scene_id'],log_token=row['log_token'],
            task=task,raw_output=raw,normalized_output=pred,schema_valid=v.valid,strict_json=strict_json,
            semantic_correct=correct,validation_errors=v.validation_errors,latency_s=seconds,
            crop_sha256=row.get('crop_sha256'),backend_success=True,fallback_used=False))
    metrics={t:dict(v,accuracy=v['correct']/v['count'],schema_valid_rate=v['schema_valid']/v['count']) for t,v in totals.items()}
    score=sum(v['accuracy'] for v in metrics.values())/len(metrics)
    return dict(metrics=metrics,selection_score=score,predictions=results,
        color_accuracy=None,motion_accuracy=None,
        limitation='属性只评官方类别；未知颜色、车型、单帧运动不计准确率。解析不要求逐字复述target_text。')


def validation_subset(rows,per_task):
    groups=defaultdict(list)
    for row in rows:
        groups[row['task']].append(row)
    selected=[]
    for task,items in sorted(groups.items()):
        items=sorted(items,key=lambda r:hashlib.sha256(('validation-v1|'+r['sample_id']).encode()).hexdigest())
        selected.extend(items[:per_task] if per_task else items)
    return selected


@torch.enable_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--max-steps',type=int,default=240)
    parser.add_argument('--eval-steps',type=int,default=80)
    parser.add_argument('--grad-accum',type=int,default=4)
    parser.add_argument('--validation-per-task',type=int,default=24)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.seed not in {17,29,43} or not 1<=args.max_steps<=240 or not 1<=args.grad_accum<=8:
        raise ValueError('outside declared resource budget')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires actual bf16 CUDA GPU')
    data=OUTPUT/'semantic_dataset'/args.dataset
    if not data.resolve().is_relative_to((OUTPUT/'semantic_dataset').resolve()):
        raise ValueError('dataset outside completion namespace')
    manifest=json.loads((data/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if digest(data/name)!=h:
            raise ValueError('dataset fingerprint mismatch')
    rows={role:list(map(json.loads,(data/f'{role}.jsonl').open())) for role in ['train','validation']}
    audit_rows(rows['train']+rows['validation'])
    for row in rows['train']+rows['validation']:
        if row.get('crop_path') and digest(row['crop_path'])!=row['crop_sha256']:
            raise ValueError('crop fingerprint mismatch')
    val=validation_subset(rows['validation'],args.validation_per_task)
    dest=OUTPUT/'training'/args.run_id
    sources=[Path(__file__),Path(__file__).with_name('semantic_contract.py'),
        Path(__file__).with_name('semantic_dataset.py'),PROJECT/'conceptgraph/vlm/image_input.py',
        PROJECT/'scripts/train_qwen_lora_semantic_delegate_v2.py']
    cfg=dict(seed=args.seed,max_steps=args.max_steps,eval_steps=args.eval_steps,grad_accum=args.grad_accum,
        dataset_manifest_sha256=digest(data/'manifest.json'),dataset=args.dataset,
        validation_ids=[r['sample_id'] for r in val],train_ids=[r['sample_id'] for r in rows['train']],
        method='bf16_lora',lora_r=8,lora_alpha=16,lora_dropout=.05,lora_modules=['q_proj','v_proj'],
        learning_rate=1e-4,early_stop_patience=2,max_new_tokens=256,max_image_side=448,
        model=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),
        model_config_sha256=digest(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct/config.json'),
        sources={str(p.relative_to(PROJECT)):digest(p) for p in sources},
        test_used=False,balanced_task_sampling=True,synthetic_color_copy=False,
        note='独立新监督；首版仅解析和官方类别，候选排序训练尚未完成。')
    if args.resume:
        if json.loads((dest/'release.json').read_text())!=cfg:
            raise ValueError('resume release mismatch')
        if not args.resume.resolve().is_relative_to(dest.resolve()):
            raise ValueError('resume checkpoint must belong to this run')
    else:
        dest.mkdir(parents=True,exist_ok=False)
        dump(dest/'release.json',cfg)
        with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as archive:
            for path in sources:
                archive.add(path,arcname=str(path.relative_to(PROJECT)))
    _set_seed(args.seed)
    processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
    model=_load_model(Path(cfg['model']),args.resume,trainable=args.resume is not None)
    if not args.resume:
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],
            lora_dropout=.05,bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,round(args.max_steps*.05)),args.max_steps)
    completed=0
    curve=[]
    if args.resume:
        state=torch.load(args.resume/'trainer_state.pt',map_location='cpu')
        completed=state['completed_steps']
        curve=state['curve']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        _restore_rng_state(state['rng_state'])
    by_task=defaultdict(list)
    for row in rows['train']:
        by_task[row['task']].append(row)
    rng=random.Random(args.seed)
    schedule=[]
    tasks=sorted(by_task)
    for _ in range(args.max_steps*args.grad_accum):
        task=rng.choice(tasks)
        schedule.append(rng.choice(by_task[task]))
    dump(dest/f'schedule_from_{completed:04d}.json',dict(sample_ids=[r['sample_id'] for r in schedule],
        sha256=hashlib.sha256(json.dumps([r['sample_id'] for r in schedule]).encode()).hexdigest()))
    best=-math.inf
    checkpoints=[]
    stale=0
    if args.resume:
        for p in sorted(dest.glob('checkpoints/*/validation.json')):
            item=json.loads(p.read_text())
            score=item['selection_score']
            best=max(best,score)
            checkpoints.append(dict(path=str(p.parent),score=score))
    started=time.perf_counter()
    stop='max_steps'
    optimizer.zero_grad(set_to_none=True)
    for step in range(completed,args.max_steps):
        model.train()
        model.config.use_cache=False
        losses=[]
        t0=time.perf_counter()
        for row in schedule[step*args.grad_accum:(step+1)*args.grad_accum]:
            batch=multimodal_collate([inputs_for(processor,row,True)],processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            batch={k:v.to(model.device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            (loss/args.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed=step+1
        curve.append(dict(step=completed,loss=float(np.mean(losses)),lr=scheduler.get_last_lr()[0],
            duration_s=time.perf_counter()-t0))
        print(json.dumps(curve[-1]),flush=True)
        if completed%args.eval_steps and completed!=args.max_steps:
            continue
        path=dest/'checkpoints'/f'step_{completed:04d}'
        save_checkpoint(path,model,processor,optimizer,scheduler,completed,curve)
        result=evaluate(model,processor,val)
        dump(path/'validation.json',result)
        score=result['selection_score']
        checkpoints.append(dict(path=str(path),score=score,adapter_sha256=digest(path/'adapter_model.safetensors')))
        if score>best+1e-8:
            best=score
            stale=0
        else:
            stale+=1
        print(json.dumps(dict(step=completed,validation_score=score,best_score=best,stale=stale)),flush=True)
        if stale>=2:
            stop='validation_patience_exhausted'
            break
    selected=max(checkpoints,key=lambda c:c['score'])
    dump(dest/'training_report.json',dict(status='completed',seed=args.seed,completed_steps=completed,
        stop_reason=stop,training_curve=curve,checkpoints=checkpoints,best_checkpoint=selected,
        wall_time_s=time.perf_counter()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        gpu=torch.cuda.get_device_name(0),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        unknown_colors_not_visual_ground_truth=True,independent_test_used=False,
        stage10_complete=False,goal_complete=False))
    print(json.dumps(dict(run=str(dest),selected=selected)),flush=True)


if __name__=='__main__':
    main()
