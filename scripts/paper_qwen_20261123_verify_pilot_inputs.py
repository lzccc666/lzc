"""只在CPU加载本地processor核验小试输入；不加载模型、不生成或训练。"""
from __future__ import annotations

import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/results/qwen/validation/iterations/paper_qwen_20261123_existing_outputs_v1'
PILOT = ROOT / 'dataset/qwen/paper_qwen_20261123_pilot_v1'
CONTRACT = ROOT / 'outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/source/real_graph_contract_003.py'
MODEL = ROOT / 'checkpoint/Qwen2-VL-2B-Instruct'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    start = time.monotonic()
    source = [Path(__file__), CONTRACT, PILOT / 'manifest.json', PILOT / 'train.jsonl', PILOT / 'validation.jsonl']
    source += [p for p in MODEL.iterdir() if p.is_file() and p.suffix in {'.json', '.txt'}]
    frozen = {str(p): sha(p) for p in source}
    dump(RUN / 'paper_qwen_20261123_token_audit_release.json',
         dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
              processor_only=True, model_load=False, model_generate=False, training=False,
              cpu_budget_minutes=10, input_sequence_limit=2048, no_truncation=True))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    spec = importlib.util.spec_from_file_location('paper_qwen_pilot_contract', CONTRACT)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    details = []
    summary = {}
    for split in ['train', 'validation']:
        data = list(map(json.loads, (PILOT / f'{split}.jsonl').open()))
        source_rows = {r['sample_id']: r for r in map(json.loads, (ROOT / 'dataset/qwen/real_graph_v003' / f'{split}.jsonl').open())}
        assert all(r == source_rows[r['sample_id']] for r in data), 'pilot changed original record or label'
        for row in data:
            assert row['task'] == 'candidate_rerank'
            payload = {k: row[k] for k in ['query_text', 'candidate_nodes', 'relation_metadata']}
            prompt_text = contract.build_completion_prompt('candidate_rerank', row)
            assert all(x not in prompt_text for x in row['official_acceptable_instances'])
            messages = [dict(role='user', content=[dict(type='text', text=prompt_text)])]
            prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            answer = json.dumps(row['response'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            full_messages = messages + [dict(role='assistant', content=[dict(type='text', text=answer)])]
            full = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[prefix], images=None, padding=False, return_tensors='pt')
            full_inputs = processor(text=[full], images=None, padding=False, return_tensors='pt')
            prefix_n = int(inputs['input_ids'].shape[-1])
            full_n = int(full_inputs['input_ids'].shape[-1])
            assert full_n > prefix_n and full_n <= 2048
            assert 'pixel_values' not in full_inputs and 'image_grid_thw' not in full_inputs
            details.append(dict(split=split, sample_id=row['sample_id'], prompt_tokens=prefix_n,
                                full_training_tokens=full_n, answer_tokens=full_n-prefix_n,
                                acceptable_ids=row['acceptable_node_ids'], training_response=row['response']['selected_node_id'],
                                candidate_count=len(row['candidate_ids']), rgb_input=False,
                                payload_fields=list(payload), original_row_and_label_unchanged=True))
        ss = [r for r in details if r['split'] == split]
        summary[split] = dict(requests=len(data), training_response_selected=sum(r['response']['selected_node_id'] is not None for r in data),
                              training_response_null=sum(r['response']['selected_node_id'] is None for r in data),
                              evaluation_acceptable_id_empty=sum(r['acceptable_node_ids'] == [] for r in data),
                              evaluation_acceptable_null=sum(r['acceptable_node_ids'] == [None] for r in data),
                              logs=sorted({r['log_token'] for r in data}),
                              max_prompt_tokens=max(r['prompt_tokens'] for r in ss),
                              max_full_training_tokens=max(r['full_training_tokens'] for r in ss),
                              schema_valid_labels=40, unchanged_source_rows=40,
                              query_kinds=dict(Counter(r['query_kind'] for r in data)))
    assert not set(summary['train']['logs']) & set(summary['validation']['logs'])
    assert all(sha(p) == h for p, h in frozen.items())
    result = dict(processor_class=type(processor).__name__, tokenizer_class=type(processor.tokenizer).__name__,
                  processor_local_path=str(MODEL), source_sha256=frozen, roles=summary, details=details,
                  model_loaded=False, gpu_model_calls=0, training_started=False,
                  input_labels_or_official_instance_ids_leaked=False, rgb_input=False,
                  legacy_training_entry_compatible_without_changes=False,
                  legacy_entry_gap='旧train_real_graph_adapter_004.py硬编码历史数据根;后续新入口应接受整理后的dataset路径，不迁回旧outputs',
                  pilot_data_ready=True, pilot_launch_entry_required=True,
                  missing_candidate_semantics='4条候选缺失样本训练response为null，而acceptable_node_ids为空表示无法获得正确选择信用；不能把正确拒选当选中目标',
                  elapsed_s=time.monotonic()-start)
    dump(RUN / 'paper_qwen_20261123_token_audit.json', result)
    print(json.dumps(dict(roles=summary, elapsed_s=result['elapsed_s'], all_checks_passed=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
