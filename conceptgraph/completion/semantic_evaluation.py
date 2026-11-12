"""相同证据、完整契约和预算下的真实Qwen比较与冻结请求缓存复跑。"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import gc
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from conceptgraph.completion.semantic_contract import validate, build_completion_prompt
from conceptgraph.completion.semantic_training import validation_subset, FIELDS
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate


def invoke(delegate,row):
    if row['task']=='query_parse':
        return delegate.parse_query(row['query_text'])
    if row['task']=='object_attributes':
        return delegate.describe_object([row['crop_path']],row['metadata'],object_version=row['crop_sha256'])
    return delegate.rerank_candidates(row['query_text'],row['candidate_nodes'],row['relation_metadata'])


def metrics_for(records,rows):
    labels={r['sample_id']:r for r in rows}
    scores=defaultdict(Counter)
    for record in records:
        row=labels[record['sample_id']]
        task=row['task']
        count=scores[task]
        count['requests']+=1
        output=record['envelope']
        count['backend_success']+=output['raw_output'] is not None
        count['schema_valid']+=output['output_schema_valid']
        count['fallback']+=output['fallback_used']
        raw=output['raw_output']
        try:
            count['strict_json']+=isinstance(json.loads(raw),dict)
        except (TypeError,ValueError):
            pass
        validation=validate(task,raw,allowed_candidate_ids=row.get('candidate_ids'))
        pred=validation.normalized_output or {}
        valid=validation.valid and output['used_qwen'] and not output['fallback_used']
        if task=='query_parse':
            correct=valid and all(pred.get(f)==row['response'][f] for f in FIELDS)
            for field in FIELDS:
                count[field+'_correct']+=valid and pred.get(field)==row['response'][field]
        elif task=='object_attributes':
            correct=valid and pred.get('object_tag')==row['response']['object_tag']
        else:
            correct=valid and pred.get('selected_node_id') in row['acceptable_node_ids']
        count['semantic_correct']+=correct
        record['semantic_correct']=bool(correct)
    return {task:dict(c,accuracy=c['semantic_correct']/c['requests'],
        schema_valid_rate=c['schema_valid']/c['requests']) for task,c in scores.items()}


def permutation_metrics(records,rows):
    """按去掉输入顺序的同一语义证据分组；非法输出也占分母。"""
    outputs={r['sample_id']:r['envelope'] for r in records}
    groups=defaultdict(list)
    for row in rows:
        if row['task']!='candidate_rerank':
            continue
        canonical=dict(query_text=row['query_text'],
            candidate_nodes=sorted(row['candidate_nodes'],key=lambda n:n['node_id']),
            relation_metadata=row['relation_metadata'])
        key=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
        output=outputs[row['sample_id']]
        validation=validate('candidate_rerank',output['raw_output'],allowed_candidate_ids=row['candidate_ids'])
        valid=validation.valid and output['used_qwen'] and not output['fallback_used']
        selected=validation.normalized_output['selected_node_id'] if valid else 'INVALID'
        groups[key].append(dict(sample_id=row['sample_id'],order=row['candidate_ids'],
            selected=selected,valid=valid,correct=valid and selected in row['acceptable_node_ids']))
    comparisons=[]
    for key,items in groups.items():
        if len({tuple(r['order']) for r in items})<2:
            continue
        comparisons.append(dict(evidence_sha256=key,rows=len(items),
            all_schema_valid=all(r['valid'] for r in items),
            identity_consistent=all(r['valid'] for r in items) and len({r['selected'] for r in items})==1,
            consistently_correct=all(r['correct'] for r in items),observations=items))
    n=len(comparisons)
    return dict(unique_semantic_evidence=len(groups),permutation_groups=n,
        identity_consistency=sum(r['identity_consistent'] for r in comparisons)/n if n else None,
        consistently_correct=sum(r['consistently_correct'] for r in comparisons)/n if n else None,
        groups=comparisons,denominator='具有至少两个不同候选输入顺序的语义证据组；非法输出保留为失败',
        independent_real_scene_credit=0)


def base_revision():
    names={}
    for line in (OUTPUT/'source_bundle/protected_manifest.jsonl').open():
        row=json.loads(line)
        if row['path'].startswith('checkpoint/Qwen2-VL-2B-Instruct/'):
            names[row['path']]=row['sha256']
    return hashlib.sha256(json.dumps(names,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dataset',default='structured_supervision_001')
    parser.add_argument('--validation-per-task',type=int,default=24)
    parser.add_argument('--prompt-selection')
    parser.add_argument('--training-series',choices=['structured_formal','three_task_v2_formal'],default='structured_formal')
    args=parser.parse_args()
    profile='completion_v1'
    selection_sha=None
    if args.prompt_selection:
        selection_path=OUTPUT/'main_experiments'/args.prompt_selection/'selection.json'
        selection=json.loads(selection_path.read_text())
        if selection['validation_used'] or selection['test_used']:
            raise ValueError('prompt calibration must be training-only')
        if selection['prompt_source_sha256']!=digest(Path(__file__).with_name('semantic_contract.py')):
            raise ValueError('calibrated prompt source changed')
        profile=selection['chosen_profile']
        selection_sha=digest(selection_path)
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA required')
    data=OUTPUT/'semantic_dataset'/args.dataset
    manifest=json.loads((data/'manifest.json').read_text())
    if digest(data/'validation.jsonl')!=manifest['files']['validation.jsonl']:
        raise ValueError('dataset changed')
    rows=validation_subset(list(map(json.loads,(data/'validation.jsonl').open())),args.validation_per_task)
    for row in rows:
        if row.get('crop_path') and digest(row['crop_path'])!=row['crop_sha256']:
            raise ValueError('crop changed')
    variants=[('base_fullschema',None),('old_adapter_fullschema',PROJECT/'outputs/project_v2/qwen_eval/finetune_v2/best_adapter')]
    for seed in [17,29,43]:
        report=json.loads((OUTPUT/f'training/{args.training_series}_seed{seed}_001/training_report.json').read_text())
        variants.append((f'new_adapter_seed{seed}',Path(report['best_checkpoint']['path'])))
    dest=OUTPUT/'main_experiments'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'release.json',dict(dataset_manifest_sha256=digest(data/'manifest.json'),
        sample_ids=[r['sample_id'] for r in rows],prompt_profile=profile,prompt_selection_sha256=selection_sha,
        input_prompt_hashes={r['sample_id']:hashlib.sha256(build_completion_prompt(r['task'],r,profile).encode()).hexdigest() for r in rows},
        model_revision=base_revision(),variants=[dict(name=n,adapter=str(p) if p else None,
            weights_sha256=digest(p/'adapter_model.safetensors') if p else None) for n,p in variants],
        max_new_tokens=256,max_image_side=448,dtype='bfloat16',do_sample=False,cache_cold_first=True,
        source_sha256=digest(Path(__file__)),independent_test=False,
        training_series=args.training_series,ranking_task_pending=not any(r['task']=='candidate_rerank' for r in rows)))
    sources=[Path(__file__),Path(__file__).with_name('semantic_contract.py'),
        PROJECT/'conceptgraph/vlm/qwen_semantic_delegate.py',PROJECT/'conceptgraph/vlm/qwen_vl_chat.py',
        PROJECT/'conceptgraph/vlm/image_input.py',PROJECT/'conceptgraph/vlm/qwen_cache.py']
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as archive:
        for path in sources:
            archive.add(path,arcname=str(path.relative_to(PROJECT)))
    reports={}
    for name,adapter in variants:
        variant=dest/name
        variant.mkdir()
        config=QwenDelegateConfig(enabled=True,model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),
            adapter_path=adapter,model_revision=base_revision(),adapter_revision=digest(adapter/'adapter_model.safetensors') if adapter else None,
            device='cuda:0',use_qwen_query_parse=True,use_qwen_object_attributes=True,use_qwen_rerank=True,
            prompt_profile=profile,timeout_s=60,max_new_tokens=256,max_images=1,
            torch_dtype='bfloat16',max_image_side=448,
            cache_dir=OUTPUT/'cache_audit'/args.run_id/name/'cache',log_dir=variant/'audit')
        delegate=QwenSemanticDelegate(config)
        records=[]
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for row in rows:
            envelope=invoke(delegate,row)
            records.append(dict(sample_id=row['sample_id'],task=row['task'],scene_id=row['scene_id'],
                log_token=row['log_token'],envelope=envelope))
            with (variant/'raw_records.jsonl').open('a') as f:
                f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
        first_calls=delegate.backend_call_count
        cache_counts=Counter()
        for original,row in zip(records,rows):
            repeated=invoke(delegate,row)
            cache_counts[row['task']+'/hit']+=bool(repeated['cache']['hit'])
            if repeated['raw_output']!=original['envelope']['raw_output'] or repeated['normalized_output']!=original['envelope']['normalized_output']:
                raise AssertionError('same frozen request changed business content on cache replay')
        added_calls=delegate.backend_call_count-first_calls
        if added_calls or sum(cache_counts.values())!=len(rows):
            raise AssertionError('frozen request cache replay was not a zero-call hit')
        metrics=metrics_for(records,rows)
        uncached=[r['envelope']['latency']['backend_s'] for r in records]
        reports[name]=dict(metrics=metrics,backend_attempts=first_calls,cache_replay_added_calls=added_calls,
            cache_replay_counts=dict(cache_counts),elapsed_s=time.perf_counter()-started,
            candidate_permutation=permutation_metrics(records,rows),
            peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            warm_backend_latency_s_median=float(np.median(uncached[1:])),
            warm_backend_latency_s_p95=float(np.percentile(uncached[1:],95)),
            cold_first_request_s=uncached[0],fallback_receives_accuracy_credit=False)
        dump(variant/'report.json',reports[name])
        delegate.close()
        del delegate
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(dict(variant=name,report=reports[name]),ensure_ascii=False),flush=True)
    dump(dest/'report.json',dict(variants=reports,gpu=torch.cuda.get_device_name(0),sample_count=len(rows),
        same_input_evidence=True,same_full_schema_prompt=True,same_resource_budget=True,
        unmeasured=['可靠颜色','语义消融','下游导航增益','封存测试']+
            ([] if any(r['task']=='candidate_rerank' for r in rows) else ['排序','候选置换']),
        independent_test=False,stage10_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
