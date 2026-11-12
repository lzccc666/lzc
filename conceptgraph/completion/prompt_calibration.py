"""只在预先固定的训练侧请求上选择原始Qwen可兼容的共同完整提示。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch

from conceptgraph.completion.semantic_contract import build_completion_prompt, validate
from conceptgraph.completion.semantic_training import FIELDS
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    data=OUTPUT/'semantic_dataset/structured_supervision_001'
    rows=list(map(json.loads,(data/'train.jsonl').open()))
    selected=[]
    for task in ['query_parse','object_attributes']:
        pool=sorted([r for r in rows if r['task']==task],key=lambda r:hashlib.sha256(('base-prompt-calibration-v1|'+r['sample_id']).encode()).hexdigest())
        selected.extend(pool[:8])
    dest=OUTPUT/'main_experiments'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    profiles=['completion_v1','completion_v2','completion_v3']
    dump(dest/'release.json',dict(training_only_sample_ids=[r['sample_id'] for r in selected],
        profiles=profiles,dataset_sha256=digest(data/'train.jsonl'),
        prompts_sha256=digest(Path(__file__).with_name('semantic_contract.py')),
        source_sha256=digest(Path(__file__)),selection_rule='原始Qwen在两个任务的等权正确率最高，平手取较短提示；不按Adapter效果选提示',
        validation_used=False,test_used=False,max_new_tokens=256,max_image_side=448,dtype='bfloat16'))
    model=QwenVLChat(str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),device='cuda:0',torch_dtype=torch.bfloat16,max_image_side=448)
    reports={}
    for profile in profiles:
        metrics={task:Counter() for task in ['query_parse','object_attributes']}
        records=[]
        prompt_lengths=[]
        for row in selected:
            images=[]
            if row['task']=='object_attributes':
                with Image.open(row['crop_path']) as image:
                    images=[image.convert('RGB').copy()]
            prompt=build_completion_prompt(row['task'],row,profile)
            prompt_lengths.append(len(prompt))
            start=time.perf_counter()
            raw=model.chat(prompt,images=images,generation_config=dict(max_new_tokens=256,do_sample=False))
            v=validate(row['task'],raw)
            pred=v.normalized_output or {}
            correct=v.valid and (all(pred.get(f)==row['response'][f] for f in FIELDS) if row['task']=='query_parse'
                else pred.get('object_tag')==row['response']['object_tag'])
            task=row['task']
            metrics[task]['count']+=1
            metrics[task]['schema_valid']+=v.valid
            metrics[task]['correct']+=correct
            records.append(dict(sample_id=row['sample_id'],task=task,profile=profile,raw_output=raw,
                schema_valid=v.valid,semantic_correct=correct,validation_errors=v.validation_errors,
                latency_s=time.perf_counter()-start,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                crop_sha256=row.get('crop_sha256')))
            with (dest/'raw_records.jsonl').open('a') as f:
                f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
        scores={t:dict(c,accuracy=c['correct']/c['count']) for t,c in metrics.items()}
        score=sum(v['accuracy'] for v in scores.values())/2
        reports[profile]=dict(metrics=scores,selection_score=score,mean_prompt_characters=sum(prompt_lengths)/len(prompt_lengths))
        dump(dest/f'{profile}.json',dict(**reports[profile],records=records))
        print(json.dumps(dict(profile=profile,**reports[profile]),ensure_ascii=False),flush=True)
    chosen=min(profiles,key=lambda p:(-reports[p]['selection_score'],reports[p]['mean_prompt_characters']))
    dump(dest/'selection.json',dict(chosen_profile=chosen,profiles=reports,
        original_model_only=True,validation_used=False,test_used=False,
        prompt_source_sha256=digest(Path(__file__).with_name('semantic_contract.py')),
        fairness_limit='仅16条训练侧固定请求，兼容性校准不代表新测试能力；首轮不兼容对照保留，不将其差距归为微调收益'))


if __name__=='__main__':
    main()
