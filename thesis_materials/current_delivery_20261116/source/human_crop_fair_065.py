"""真实人工审核裁剪上的开发核验；相同图像/提示，答案仅在评价阶段读取。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path('/root/autodl-tmp/lizichong')
OLD = PROJECT / 'outputs/project_v2/thesis_completion_v1'
ROOT = OLD / 'public_experiment_completion_20261115'
OUT = ROOT / 'qwen_adapter_fair_comparison/human_crop_development_065'
SEALED = ROOT / 'main_experiments/sealed_public_058'
MODEL = PROJECT / 'checkpoint/Qwen2-VL-2B-Instruct'
REFERENCE = ROOT / 'annotation/human_attribute_reference_003/attributes.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def rows(path):
    with Path(path).open() as f:
        yield from map(json.loads, f)


def safe_input(reference):
    """白名单导出，禁止人工类别/颜色/备注进入模型。"""
    return dict(sample_id=f"human-crop-{reference['sample_number']:03d}",
        scene=reference['scene'], task='object_attributes',
        crop_path=reference['crop_file'], crop_sha256=reference['crop_sha256'],
        metadata={'detection_label': reference['predicted_category']})


def score_one(prediction, reference):
    valid = prediction['schema_valid']
    pred = prediction.get('normalized_output') or {}
    known = reference['human_color'] not in {'unknown', 'multiple'}
    return dict(sample_id=prediction['sample_id'], scene=reference['scene'],
        schema_valid=valid, human_color=reference['human_color'],
        predicted_colors=pred.get('colors'), color_evaluable=known,
        color_correct=(valid and pred.get('colors') == [reference['human_color']]) if known else None,
        category_evaluable=reference['category_comparable'],
        category_correct=(valid and pred.get('object_tag') == reference['human_category'])
            if reference['category_comparable'] else None,
        motion_unknown_output=valid and pred.get('motion_state') == 'unknown',
        body_type_output=pred.get('body_type'), fine_body_type_accuracy_credit=False)


def prepare():
    from conceptgraph.completion.semantic_contract import build_completion_prompt
    OUT.mkdir(exist_ok=False)
    release = json.loads((SEALED / 'release.json').read_text())
    calibration = OLD / 'main_experiments/base_prompt_calibration_001/selection.json'
    assert json.loads(calibration.read_text())['chosen_profile'] == 'completion_v3'
    refs = json.loads(REFERENCE.read_text())['rows']
    assert len(refs) == 20 and len({r['scene'] for r in refs}) == 20
    inputs = [safe_input(r) for r in refs]
    fingerprints = {}
    for row in inputs:
        assert sha(row['crop_path']) == row['crop_sha256']
        row['prompt'] = build_completion_prompt(row['task'], row, 'completion_v3')
        row['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
    with (OUT / 'inputs.jsonl').open('x') as f:
        for row in inputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    for p in [Path(__file__), calibration, PROJECT / 'conceptgraph/completion/semantic_contract.py',
              PROJECT / 'conceptgraph/vlm/schemas.py', PROJECT / 'conceptgraph/vlm/image_input.py',
              PROJECT / 'scripts/train_qwen_lora_semantic_delegate_v2.py']:
        fingerprints[str(p)] = sha(p)
    model_hashes = {p: h for p, h in release['frozen_files_sha256'].items()
                    if str(MODEL) in p or any(v and str(v) in p for v in release['variants'].values())}
    for adapter in release['variants'].values():
        if adapter:
            for p in Path(adapter).glob('adapter*'):
                if p.is_file():
                    model_hashes[str(p)] = sha(p)
    assert any('model' in Path(p).name and p.endswith('.safetensors')
               and str(MODEL) in p for p in model_hashes), '缺少基础权重指纹'
    write(OUT / 'protocol.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        purpose='用户20张真实审核图的开发属性核验，不是独立测试或下游到达证据',
        variants=release['variants'], model=str(MODEL), model_hashes=model_hashes,
        source_hashes=fingerprints, source_reference_sha256=sha(REFERENCE),
        inference_inputs_sha256=sha(OUT / 'inputs.jsonl'), samples=20,
        profile='completion_v3', profile_chosen_before_human_review=True,
        max_input_tokens=2048, max_new_tokens=256, max_image_side=448,
        deterministic=True, seed=17, dtype='bfloat16', GPU_concurrency=1,
        each_model_budget_s=240, all_model_budget_s=1000, output_budget_bytes=16*1024**2,
        primary_color_rule='16个单一已知颜色样本，schema有效且colors列表恰为该单色才正确；无效/拒答/超时保留失败',
        category_rule='17个用户类别可比较样本严格比较object_tag；other/unknown不硬转类别',
        unknown_multiple_rule='另外4张全部推理并保留输出，但没有精确颜色集合真值，不计算颜色准确率',
        CI='20个开发场景中的16个颜色场景/17个类别场景，bootstrap4000 seed17；小样本描述统计',
        limitations=['审核包按检测最大框筛选，不是随机无偏抽样', '单人审核，没有第二人复核',
            '与历史训练/验证/开发图像可能重合，不能声称独立泛化',
            '新real_graph Adapter只训练过排序，本次不预设它改善属性识别',
            '不授予三维实例、自然指令、运动真值或到达信用',
            'CPU有并行建图，GPU单模型；耗时非端到端实时性'],
        sealed_pipeline_sources_unchanged=True, actual_new_CARLA_calls=0, goal_complete=False))
    print('prepared 20 inputs; human labels excluded from inference', flush=True)


def gpu(variant):
    # 此阶段只读取无答案输入、协议和模型，REFERENCE只在独立evaluate使用。
    import ctypes
    ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    from conceptgraph.completion.semantic_contract import validate
    from conceptgraph.vlm.image_input import prepare_qwen_image
    from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model

    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert variant in protocol['variants']
    for p, h in protocol['source_hashes'].items():
        assert sha(p) == h, p
    adapter = protocol['variants'][variant]
    for p, h in protocol['model_hashes'].items():
        if str(MODEL) in p or (adapter and str(adapter) in p):
            assert sha(p) == h, p
    assert sha(OUT / 'inputs.jsonl') == protocol['inference_inputs_sha256']
    assert not (SEALED / 'requests').exists(), '为封存GPU队列让路，保留前缀不重试'
    dest = OUT / variant
    dest.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    begun = time.monotonic()
    model = _load_model(MODEL, Path(adapter) if adapter else None, trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    load_s = time.monotonic() - begun
    count = actual = 0
    with (dest / 'outputs.jsonl').open('x') as output:
        for row in rows(OUT / 'inputs.jsonl'):
            assert time.monotonic() - begun < protocol['each_model_budget_s']
            assert not (SEALED / 'requests').exists(), '为封存GPU队列让路'
            assert sha(row['crop_path']) == row['crop_sha256']
            with Image.open(row['crop_path']) as original:
                image = prepare_qwen_image(original, 448).copy()
            content = [dict(type='image', image=image), dict(type='text', text=row['prompt'])]
            text = processor.apply_chat_template([dict(role='user', content=content)],
                tokenize=False, add_generation_prompt=True)
            x = processor(text=[text], images=[image], padding=False, return_tensors='pt')
            assert {'pixel_values', 'image_grid_thw'} <= set(x), '真实图像必须进入processor'
            # 保存CPU张量字节指纹，不改变送入模型的张量。
            tensor_hashes = {k: hashlib.sha256(v.contiguous().cpu().numpy().tobytes()).hexdigest()
                             for k, v in x.items() if torch.is_tensor(v)}
            x = x.to(model.device)
            tokens = int(x['input_ids'].shape[-1])
            raw, latency, result = None, None, None
            if tokens <= protocol['max_input_tokens']:
                torch.cuda.synchronize()
                tick = time.monotonic()
                with torch.inference_mode():
                    y = model.generate(**x, max_new_tokens=protocol['max_new_tokens'], do_sample=False,
                        temperature=None, top_p=None, top_k=None, use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id)
                torch.cuda.synchronize()
                latency = time.monotonic()-tick
                raw = processor.tokenizer.decode(y[0, tokens:], skip_special_tokens=True).strip()
                result = validate('object_attributes', raw)
                actual += 1
            record = dict(sample_id=row['sample_id'], variant=variant, input_tokens=tokens,
                prompt_sha256=row['prompt_sha256'], crop_sha256=row['crop_sha256'],
                processed_image_size=list(image.size), processor_tensor_sha256=tensor_hashes,
                raw_output=raw, GPU_latency_s=latency, actual_GPU_call=tokens <= protocol['max_input_tokens'],
                schema_valid=bool(result and result.valid),
                normalized_output=result.normalized_output if result and result.valid else None,
                errors=result.validation_errors if result else ['TOKEN_LIMIT_EXCEEDED'],
                human_truth_received=False, fallback_used=False)
            output.write(json.dumps(record, ensure_ascii=False)+'\n')
            output.flush()
            count += 1
            print(json.dumps(dict(variant=variant, completed=count, total=20)), flush=True)
    assert count == 20
    write(dest / 'report.json', dict(requests=count, actual_GPU_calls=actual,
        wall_s=time.monotonic()-begun, load_s=load_s,
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
        outputs_sha256=sha(dest / 'outputs.jsonl'), human_truth_received=False, goal_complete=False))


def evaluate():
    import csv
    import numpy as np
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(REFERENCE) == protocol['source_reference_sha256']
    refs = {safe_input(r)['sample_id']: r for r in json.loads(REFERENCE.read_text())['rows']}
    summary, detailed, common = [], [], {}
    for variant in protocol['variants']:
        dest = OUT / variant
        report = json.loads((dest / 'report.json').read_text())
        assert report['requests'] == 20 and sha(dest / 'outputs.jsonl') == report['outputs_sha256']
        predictions = list(rows(dest / 'outputs.jsonl'))
        assert len(predictions) == 20 and {p['sample_id'] for p in predictions} == set(refs)
        scores = []
        for pred in predictions:
            signature = {k: pred[k] for k in ['input_tokens', 'crop_sha256', 'prompt_sha256', 'processor_tensor_sha256']}
            if pred['sample_id'] in common:
                assert signature == common[pred['sample_id']], '跨模型多模态输入不一致'
            common[pred['sample_id']] = signature
            s = dict(variant=variant, **score_one(pred, refs[pred['sample_id']]))
            detailed.append(s)
            scores.append(s)
        row = dict(variant=variant, requests=20, actual_GPU_calls=report['actual_GPU_calls'],
            schema_valid=sum(s['schema_valid'] for s in scores), peak_GPU_MiB=report['peak_GPU_MiB'])
        for task in ['color', 'category']:
            vals = np.array([s[task+'_correct'] for s in scores if s[task+'_evaluable']], dtype=float)
            rng = np.random.default_rng(17)
            ci = np.quantile(vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(axis=1), [.025, .975])
            row.update({task+'_correct': int(sum(vals)), task+'_denominator': len(vals),
                        task+'_accuracy': float(vals.mean()), task+'_CI95_low': float(ci[0]),
                        task+'_CI95_high': float(ci[1])})
        latencies = [p['GPU_latency_s'] for p in predictions if p['actual_GPU_call']]
        row['mean_GPU_latency_s'] = float(np.mean(latencies))
        row['p95_GPU_latency_s'] = float(np.percentile(latencies, 95))
        summary.append(row)
    for name, values in [('comparison.csv', summary), ('per_sample.csv', detailed)]:
        with (OUT / name).open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader()
            w.writerows(values)
    write(OUT / 'report.json', dict(UTC=datetime.now(timezone.utc).isoformat(), results=summary,
        actual_GPU_calls=sum(s['actual_GPU_calls'] for s in summary), unique_crops=20,
        cross_model_input_tensors_equal=True, human_labels_only_used_in_evaluator=True,
        color_single_known_denominator=16, unknown_or_multiple_color_excluded_from_accuracy=4,
        source_protocol_sha256=sha(OUT / 'protocol.json'), formal_generalization_metric=False,
        human_review_3D_identity_credit=False, actual_arrival_credit=0,
        limitations=protocol['limitations'], goal_complete=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def series():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    begun = time.monotonic()
    for variant in protocol['variants']:
        assert time.monotonic()-begun < protocol['all_model_budget_s']
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout.strip()
        assert not busy, 'GPU已有进程，禁止并行或自动重试'
        with (OUT / f'{variant}.log').open('x') as log:
            worker = subprocess.run([sys.executable, str(Path(__file__)), 'gpu', '--variant', variant],
                env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                timeout=protocol['each_model_budget_s']+20)
        write(OUT / f'{variant}_receipt.json', dict(returncode=worker.returncode))
        if worker.returncode:
            raise RuntimeError(f'{variant}失败，保留前缀，停止其余模型')
        print('finished '+variant, flush=True)
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'gpu', 'series', 'evaluate'])
    parser.add_argument('--variant')
    args = parser.parse_args()
    try:
        if args.action == 'gpu':
            gpu(args.variant)
        else:
            globals()[args.action]()
    except Exception as exc:
        if OUT.exists():
            write(OUT / f'failure_{args.action}_{time.time_ns()}.json',
                  dict(error=repr(exc), goal_complete=False))
        raise
