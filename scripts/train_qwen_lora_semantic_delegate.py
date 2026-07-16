from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _prefer_conda_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = Path(conda_prefix) / "lib"
    libstdcxx = lib_dir / "libstdc++.so.6"
    if lib_dir.exists():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [str(lib_dir)] + [item for item in current.split(":") if item and item != str(lib_dir)]
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
    if libstdcxx.exists():
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)


_prefer_conda_runtime_libs()

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


TASK_FILES = {
    "object_caption": "object_caption_dataset.jsonl",
    "relation": "relation_dataset.jsonl",
    "navigation_query": "navigation_query_dataset.jsonl",
    "candidate_rerank": "candidate_rerank_dataset.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetune/evaluate Qwen2-VL as a semantic delegate.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/qwen_finetune_dataset"))
    parser.add_argument("--model", default="/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/checkpoint/qwen_lora_semantic_delegate"))
    parser.add_argument("--report-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/experiment_summary/qwen_lora_eval"))
    parser.add_argument("--tasks", nargs="*", default=["navigation_query", "relation", "candidate_rerank", "object_caption"])
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset_dir, args.tasks)
    random.shuffle(samples)
    if len(samples) < 8:
        raise RuntimeError(f"Too few samples for training/evaluation: {len(samples)}")
    eval_count = min(args.max_eval_samples, max(8, len(samples) // 5))
    eval_samples = samples[:eval_count]
    train_samples = samples[eval_count: eval_count + args.max_train_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    baseline_metrics = evaluate_model(
        args=args,
        processor=processor,
        adapter_path=None,
        eval_samples=eval_samples,
        device=device,
        output_json=args.report_dir / "baseline_predictions.json",
    )

    train_log: dict[str, Any] = {
        "model": args.model,
        "output_dir": str(args.output_dir),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "device": device,
        "baseline_metrics": baseline_metrics,
    }

    if not args.eval_only:
        started = time.perf_counter()
        train_lora(args, processor, train_samples, device)
        train_log["train_time_s"] = time.perf_counter() - started
        train_log["status"] = "trained"
    else:
        train_log["status"] = "eval_only"

    adapter_path = args.adapter or args.output_dir
    finetuned_metrics = evaluate_model(
        args=args,
        processor=processor,
        adapter_path=adapter_path if adapter_path.exists() else None,
        eval_samples=eval_samples,
        device=device,
        output_json=args.report_dir / "finetuned_predictions.json",
    )
    train_log["finetuned_metrics"] = finetuned_metrics
    train_log["delta"] = {
        key: finetuned_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
        for key in sorted(set(baseline_metrics) | set(finetuned_metrics))
        if isinstance(baseline_metrics.get(key, 0.0), (int, float)) and isinstance(finetuned_metrics.get(key, 0.0), (int, float))
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(train_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.report_dir / "comparison_metrics.json").write_text(json.dumps(train_log, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(args.report_dir / "comparison_report.md", train_log)
    print(json.dumps(train_log, indent=2, ensure_ascii=False))


def load_samples(dataset_dir: Path, tasks: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task in tasks:
        path = dataset_dir / TASK_FILES[task]
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt = build_prompt(task, row)
                response = normalize_response(row.get("response", {}))
                samples.append({"task": task, "prompt": prompt, "response": response, "row": row})
    return samples


def build_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "navigation_query":
        return f"请解析中文语义导航指令并只输出 JSON。\n指令：{row.get('query_text')}"
    if task == "relation":
        payload = {
            "source_tag": row.get("source_tag"),
            "target_tag": row.get("target_tag"),
            "source_color": row.get("source_color"),
            "target_color": row.get("target_color"),
            "distance_m": row.get("distance_m"),
            "relative_position": row.get("relative_position"),
        }
        return f"请判断两个对象的空间关系并只输出 JSON。\n输入：{json.dumps(payload, ensure_ascii=False)}"
    if task == "candidate_rerank":
        payload = {
            "query_text": row.get("query_text"),
            "candidate_scores": row.get("candidate_scores"),
            "target_validation": row.get("target_validation"),
        }
        return f"请根据导航指令和候选信息选择目标节点并只输出 JSON。\n输入：{json.dumps(payload, ensure_ascii=False)}"
    payload = {
        "detection_label": row.get("detection_label"),
        "color": row.get("color"),
        "body_type": row.get("body_type"),
        "motion_state": row.get("motion_state"),
        "caption": row.get("qwen_or_rule_caption"),
    }
    return f"请根据对象元数据输出对象语义 JSON。\n输入：{json.dumps(payload, ensure_ascii=False)}"


def normalize_response(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.strip()


class ChatDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], processor, max_length: int = 768):
        self.samples = samples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = chat_text(self.processor, sample["prompt"], add_generation_prompt=True)
        answer = sample["response"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + answer_ids)[-self.max_length:]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[-self.max_length:]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    result = {}
    for key in ("input_ids", "attention_mask", "labels"):
        values = []
        pad = -100 if key == "labels" else (0 if key == "attention_mask" else pad_token_id)
        for item in batch:
            tensor = item[key]
            if tensor.numel() < max_len:
                tensor = torch.cat([torch.full((max_len - tensor.numel(),), pad, dtype=tensor.dtype), tensor])
            values.append(tensor)
        result[key] = torch.stack(values, dim=0)
    return result


def train_lora(args: argparse.Namespace, processor, train_samples: list[dict[str, Any]], device: str) -> None:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dataset = ChatDataset(train_samples, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    losses = []
    progress = tqdm(total=args.max_steps, desc="qwen-lora")
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if (len(losses) % args.grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{sum(losses[-args.grad_accum:]) / args.grad_accum:.4f}")
                if step >= args.max_steps:
                    break
    progress.close()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    (args.output_dir / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    args: argparse.Namespace,
    processor,
    adapter_path: Path | None,
    eval_samples: list[dict[str, Any]],
    device: str,
    output_json: Path,
) -> dict[str, float]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path)).eval()
    if device != "cuda":
        model.to(device)
    predictions = []
    exact = 0
    json_ok = 0
    field_hits = 0
    field_total = 0
    max_eval = min(len(eval_samples), args.max_eval_samples)
    for sample in tqdm(eval_samples[:max_eval], desc="eval"):
        pred = generate_json(model, processor, sample["prompt"])
        target = sample["response"]
        pred_obj = parse_json_object(pred)
        target_obj = parse_json_object(target)
        if pred.strip() == target.strip():
            exact += 1
        if isinstance(pred_obj, dict):
            json_ok += 1
        hits, total = compare_json_fields(pred_obj, target_obj)
        field_hits += hits
        field_total += total
        predictions.append(
            {
                "task": sample["task"],
                "prompt": sample["prompt"],
                "target": target,
                "prediction": pred,
                "prediction_json": pred_obj,
                "target_json": target_obj,
                "field_hits": hits,
                "field_total": total,
            }
        )
    output_json.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total = max(1, max_eval)
    return {
        "num_eval": float(max_eval),
        "exact_match": exact / total,
        "json_valid_rate": json_ok / total,
        "field_accuracy": field_hits / max(1, field_total),
    }


def generate_json(model, processor, prompt: str) -> str:
    text = chat_text(processor, prompt, add_generation_prompt=True)
    inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def chat_text(processor, prompt: str, add_generation_prompt: bool) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def parse_json_object(text: str) -> Any:
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"(\{.*\})", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def compare_json_fields(pred: Any, target: Any) -> tuple[int, int]:
    if not isinstance(target, dict):
        return (int(pred == target), 1)
    if not isinstance(pred, dict):
        return (0, max(1, len(flatten_dict(target))))
    target_flat = flatten_dict(target)
    pred_flat = flatten_dict(pred)
    hits = 0
    for key, value in target_flat.items():
        if key in pred_flat and normalize_value(pred_flat[key]) == normalize_value(value):
            hits += 1
    return hits, max(1, len(target_flat))


def flatten_dict(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten_dict(item, name))
        else:
            result[name] = item
    return result


def normalize_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_markdown_report(path: Path, log: dict[str, Any]) -> None:
    b = log.get("baseline_metrics", {})
    f = log.get("finetuned_metrics", {})
    d = log.get("delta", {})
    lines = [
        "# Qwen LoRA 微调前后对比",
        "",
        f"- 模型：`{log.get('model')}`",
        f"- LoRA 输出：`{log.get('output_dir')}`",
        f"- 训练样本数：{log.get('num_train_samples')}",
        f"- 验证样本数：{log.get('num_eval_samples')}",
        f"- 训练状态：{log.get('status')}",
        "",
        "| 指标 | 微调前 | 微调后 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for key in ["json_valid_rate", "field_accuracy", "exact_match"]:
        lines.append(f"| {key} | {b.get(key, 0):.4f} | {f.get(key, 0):.4f} | {d.get(key, 0):+.4f} |")
    lines.extend(
        [
            "",
            "说明：当前数据集主要来自自动规则/Qwen 伪标签，指标反映模型对本项目结构化语义格式的拟合能力，不等价于人工语义真值精度。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
