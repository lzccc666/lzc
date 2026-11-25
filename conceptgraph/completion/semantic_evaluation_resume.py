"""按原release、源码快照、请求与权重指纹续跑中断的公平对照；不覆盖旧报告。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from conceptgraph.completion.semantic_contract import build_completion_prompt
from conceptgraph.completion.semantic_evaluation import invoke, metrics_for, permutation_metrics, base_revision
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate


def load_prefix(path,rows):
    if not path.exists():
        return []
    contents=path.read_bytes()
    if contents and not contents.endswith(b'\n'):
        raise ValueError('incomplete raw line retained; explicit recovery required before append')
    records=[json.loads(line) for line in contents.splitlines()]
    if len(records)>len(rows):
        raise ValueError('too many records')
    for record,row in zip(records,rows):
        for key,source in [('sample_id','sample_id'),('task','task'),('scene_id','scene_id'),('log_token','log_token')]:
            if record[key]!=row[source]:
                raise ValueError('request prefix differs from release')
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wall-budget-s',type=float,default=6600.)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA required')
    started=time.monotonic()
    dest=OUTPUT/'main_experiments'/args.run_id
    if (dest/'report.json').exists():
        raise ValueError('run already complete; do not repeat or overwrite')
    release=json.loads((dest/'release.json').read_text())
    sources={}
    with tarfile.open(dest/'source_snapshot.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():
                frozen=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                current=digest(PROJECT/member.name)
                if frozen!=current:
                    raise ValueError('frozen dependency changed: '+member.name)
                sources[member.name]=current
    if sources['conceptgraph/completion/semantic_evaluation.py']!=release['source_sha256']:
        raise ValueError('original evaluator fingerprint differs')
    dataset=OUTPUT/'semantic_dataset/structured_supervision_002'
    if digest(dataset/'manifest.json')!=release['dataset_manifest_sha256']:
        raise ValueError('dataset manifest changed')
    manifest=json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'validation.jsonl')!=manifest['files']['validation.jsonl']:
        raise ValueError('validation changed')
    lookup={row['sample_id']:row for row in map(json.loads,(dataset/'validation.jsonl').open())}
    rows=[lookup[identity] for identity in release['sample_ids']]
    if len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate frozen request')
    for row in rows:
        prompt=build_completion_prompt(row['task'],row,release['prompt_profile'])
        if hashlib.sha256(prompt.encode()).hexdigest()!=release['input_prompt_hashes'][row['sample_id']]:
            raise ValueError('frozen prompt changed')
        if row.get('crop_path') and digest(row['crop_path'])!=row['crop_sha256']:
            raise ValueError('real input crop changed')
    if base_revision()!=release['model_revision']:
        raise ValueError('base model revision differs')
    resume=dest/f'resume_{time.time_ns()}'
    resume.mkdir()
    dump(resume/'audit_before.json',dict(original_release_sha256=digest(dest/'release.json'),
        original_sources_verified=sources,resume_source_sha256=digest(__file__),
        previous_file_hashes={str(p.relative_to(dest)):digest(p) for p in dest.glob('*/raw_records.jsonl')},
        wall_budget_s=args.wall_budget_s,independent_test=False,goal_complete=False))
    (resume/'resume_source.py').write_text(Path(__file__).read_text())
    reports={}
    for variant_spec in release['variants']:
        if time.monotonic()-started>args.wall_budget_s:
            raise TimeoutError('resume wall budget exhausted; completed records retained')
        name=variant_spec['name']
        adapter=Path(variant_spec['adapter']) if variant_spec['adapter'] else None
        if adapter is not None and digest(adapter/'adapter_model.safetensors')!=variant_spec['weights_sha256']:
            raise ValueError('adapter changed')
        variant=dest/name
        variant.mkdir(exist_ok=True)
        records=load_prefix(variant/'raw_records.jsonl',rows)
        if (variant/'report.json').exists():
            previous=json.loads((variant/'report.json').read_text())
            if len(records)!=len(rows) or metrics_for(records,rows)!=previous['metrics'] or previous['cache_replay_added_calls']!=0:
                raise ValueError('completed variant report does not match saved records')
            reports[name]=previous
            print(json.dumps(dict(variant=name,status='verified_complete_skipped',requests=len(records))),flush=True)
            continue
        previous_count=len(records)
        config=QwenDelegateConfig(enabled=True,model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),
            adapter_path=adapter,model_revision=release['model_revision'],adapter_revision=variant_spec['weights_sha256'],
            device='cuda:0',use_qwen_query_parse=True,use_qwen_object_attributes=True,use_qwen_rerank=True,
            prompt_profile=release['prompt_profile'],timeout_s=60,max_new_tokens=release['max_new_tokens'],max_images=1,
            torch_dtype=release['dtype'],max_image_side=release['max_image_side'],
            cache_dir=OUTPUT/'cache_audit'/args.run_id/name/'cache',log_dir=variant/'audit')
        delegate=QwenSemanticDelegate(config)
        variant_started=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            with (variant/'raw_records.jsonl').open('a') as handle:
                for row in rows[previous_count:]:
                    if time.monotonic()-started>args.wall_budget_s:
                        raise TimeoutError('resume wall budget exhausted')
                    record=dict(sample_id=row['sample_id'],task=row['task'],scene_id=row['scene_id'],
                        log_token=row['log_token'],envelope=invoke(delegate,row))
                    handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    handle.flush()
                    records.append(record)
            first_calls=delegate.backend_call_count
            cache_counts=Counter()
            for original,row in zip(records,rows):
                repeated=invoke(delegate,row)
                cache_counts[row['task']+'/hit']+=bool(repeated['cache']['hit'])
                if repeated['raw_output']!=original['envelope']['raw_output'] or repeated['normalized_output']!=original['envelope']['normalized_output']:
                    raise AssertionError('cache replay changed frozen business content')
            added=delegate.backend_call_count-first_calls
            if added or sum(cache_counts.values())!=len(rows):
                raise AssertionError('cache replay must have zero model calls')
            latency=[r['envelope']['latency']['backend_s'] for r in records]
            report=dict(metrics=metrics_for(records,rows),candidate_permutation=permutation_metrics(records,rows),
                backend_attempts_saved_first_requests=sum(not r['envelope']['cache']['hit'] for r in records),
                backend_attempts_observed_this_resume=first_calls,completed_records_before_resume=previous_count,
                cache_replay_added_calls=added,cache_replay_counts=dict(cache_counts),
                elapsed_this_resume_s=time.perf_counter()-variant_started,
                peak_gpu_memory_mib_this_resume=torch.cuda.max_memory_allocated()/2**20,
                warm_backend_latency_s_median=float(np.median(latency[1:])),
                warm_backend_latency_s_p95=float(np.percentile(latency[1:],95)),cold_first_request_s=latency[0],
                fallback_receives_accuracy_credit=False,performance_main_table_credit=False,
                performance_limit='中断前后计算时段混合；独占分阶段性能另测')
            dump(variant/'report.json',report)
            reports[name]=report
            print(json.dumps(dict(variant=name,metrics=report['metrics'],cache_added_calls=added),ensure_ascii=False),flush=True)
        finally:
            delegate.close()
            del delegate
            gc.collect()
            torch.cuda.empty_cache()
    dump(dest/'report.json',dict(variants=reports,gpu=torch.cuda.get_device_name(0),sample_count=len(rows),
        same_input_evidence=True,same_full_schema_prompt=True,same_resource_budget=True,resumed=True,
        resume_audit=str(resume/'audit_before.json'),unmeasured=['可靠颜色','语义消融','下游导航增益','封存测试'],
        independent_test=False,stage10_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
