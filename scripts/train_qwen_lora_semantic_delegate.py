from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path("/root/autodl-tmp/lizichong")
MAX_IMAGE_SIDE = 672
TASK_FILES = {
    "object_attributes": "object_attributes.jsonl",
    "query_parse": "query_parse.jsonl",
    "candidate_rerank": "candidate_rerank.jsonl",
}
LEGACY_TASK_ALIASES = {
    "object_caption": "object_attributes",
    "navigation_query": "query_parse",
}
LEGACY_TASK_FILES = {
    "object_attributes": "object_caption_dataset.jsonl",
    "query_parse": "navigation_query_dataset.jsonl",
    "candidate_rerank": "candidate_rerank_dataset.jsonl",
}


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

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402

from conceptgraph.vlm.schemas import parse_and_validate_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按scene split微调/评估Qwen2-VL语义代理。")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-random-split",
        action="store_true",
        help="仅用于旧数据迁移诊断；阶段4验收禁止使用随机scene混合。",
    )
    parser.add_argument("--model", default=str(PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project_v2/qwen_eval/sanity_adapter",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project_v2/qwen_eval/sanity_training",
    )
    parser.add_argument("--tasks", nargs="*", default=list(TASK_FILES))
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-train-samples", type=int, default=160)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--task-weight", action="append", default=[], metavar="TASK=WEIGHT")
    parser.add_argument(
        "--processor-smoke-only",
        action="store_true",
        help="只验证真实crop经过AutoProcessor，不加载生成模型。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.tasks = normalize_tasks(args.tasks)
    _guard_output_paths(args)
    manifest_path = args.split_manifest or args.dataset_dir / "split_manifest.json"
    manifest = load_split_manifest(
        manifest_path,
        allow_legacy_random_split=args.allow_legacy_random_split,
    )
    train_samples = deterministic_task_sample_limit(
        load_samples(
            args.dataset_dir,
            args.tasks,
            split="train",
            manifest=manifest,
            allow_legacy=args.allow_legacy_random_split,
        ),
        args.max_train_samples,
    )
    eval_samples = deterministic_task_sample_limit(
        load_samples(
            args.dataset_dir,
            args.tasks,
            split=args.eval_split,
            manifest=manifest,
            allow_legacy=args.allow_legacy_random_split,
        ),
        args.max_eval_samples,
    )
    if not train_samples or not eval_samples:
        raise RuntimeError(
            f"split样本不足：train={len(train_samples)}, {args.eval_split}={len(eval_samples)}"
        )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.processor_smoke_only:
        report = processor_smoke_check(processor, train_samples + eval_samples, args.max_length)
        path = args.report_dir / "multimodal_processor_smoke.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_metrics = evaluate_model(
        args=args,
        processor=processor,
        adapter_path=None,
        eval_samples=eval_samples,
        device=device,
        output_json=args.report_dir / "baseline_predictions.json",
    )
    task_weights = resolve_task_weights(args, manifest)
    train_log: dict[str, Any] = {
        "model": args.model,
        "output_dir": str(args.output_dir),
        "split_manifest": str(manifest_path),
        "train_scenes": sorted({sample["scene_id"] for sample in train_samples}),
        "eval_scenes": sorted({sample["scene_id"] for sample in eval_samples}),
        "scene_leakage": sorted(
            {sample["scene_id"] for sample in train_samples}
            & {sample["scene_id"] for sample in eval_samples}
        ),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "task_sampling_weights": task_weights,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "device": device,
        "baseline_metrics": baseline_metrics,
    }
    if train_log["scene_leakage"]:
        raise RuntimeError(f"scene split泄漏：{train_log['scene_leakage']}")

    if not args.eval_only:
        started = time.perf_counter()
        train_lora(args, processor, train_samples, device, task_weights)
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
    train_log["delta"] = metric_delta(baseline_metrics, finetuned_metrics)
    _write_json(args.output_dir / "train_log.json", train_log)
    _write_json(args.report_dir / "comparison_metrics.json", train_log)
    _write_json(args.report_dir / "per_task_metrics.json", finetuned_metrics["by_task"])
    _write_json(args.report_dir / "per_scene_metrics.json", finetuned_metrics["by_scene"])
    write_markdown_report(args.report_dir / "comparison_report.md", train_log)
    print(json.dumps(train_log, indent=2, ensure_ascii=False))


def normalize_tasks(tasks: list[str]) -> list[str]:
    result = []
    for raw_task in tasks:
        task = LEGACY_TASK_ALIASES.get(raw_task, raw_task)
        if task not in TASK_FILES:
            raise ValueError(
                f"不支持任务{raw_task!r}；阶段4任务为{sorted(TASK_FILES)}。relation旧任务不再参与严格schema训练。"
            )
        if task not in result:
            result.append(task)
    return result


def deterministic_task_sample_limit(
    samples: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["task"]].append(sample)
    result = []
    index = 0
    tasks = sorted(groups)
    while len(result) < limit:
        added = False
        for task in tasks:
            if index < len(groups[task]):
                result.append(groups[task][index])
                added = True
                if len(result) >= limit:
                    break
        if not added:
            break
        index += 1
    return result


def load_split_manifest(path: Path, *, allow_legacy_random_split: bool = False) -> dict[str, Any]:
    if not path.exists():
        if allow_legacy_random_split:
            return {"schema_version": "legacy-random-split", "scene_to_split": {}}
        raise RuntimeError(
            f"缺少split manifest：{path}。请先运行build_qwen_finetune_dataset.py；"
            "旧数据只能显式添加--allow-legacy-random-split进行迁移诊断。"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage4-split-manifest-v1":
        raise RuntimeError(f"不兼容split manifest版本：{manifest.get('schema_version')}")
    scene_to_split = manifest.get("scene_to_split")
    if not isinstance(scene_to_split, dict):
        raise RuntimeError("split manifest缺少scene_to_split")
    return manifest


def load_samples(
    dataset_dir: Path,
    tasks: list[str],
    *,
    split: str,
    manifest: dict[str, Any],
    allow_legacy: bool = False,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    scene_to_split = manifest.get("scene_to_split", {})
    strict_manifest = manifest.get("schema_version") == "stage4-split-manifest-v1"
    for task in tasks:
        path = dataset_dir / TASK_FILES[task]
        legacy = False
        if not path.exists() and allow_legacy:
            path = dataset_dir / LEGACY_TASK_FILES[task]
            legacy = path.exists()
        if not path.exists():
            raise FileNotFoundError(f"任务数据文件不存在：{path}")
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                scene_id = str(row.get("scene_id", ""))
                sample_id = row.get("sample_id", f"{scene_id}:{task}:{line_number}")
                row_split = row.get("split") or scene_to_split.get(scene_id)
                if legacy and not row_split:
                    if not allow_legacy:
                        raise RuntimeError("旧数据没有scene split")
                    row_split = _legacy_split(scene_id)
                if strict_manifest and scene_id not in scene_to_split:
                    raise RuntimeError(f"{path}:{line_number} 的scene不在split manifest中：{scene_id!r}")
                expected = scene_to_split.get(scene_id, row_split)
                if expected != row_split:
                    raise RuntimeError(
                        f"{path}:{line_number} 的split={row_split}与manifest={expected}不一致"
                    )
                if strict_manifest:
                    task_samples = manifest.get("samples", {}).get(task)
                    if not isinstance(task_samples, dict):
                        raise RuntimeError(f"split manifest缺少任务样本清单：{task}")
                    allowed_ids = set(task_samples.get(str(row_split), []))
                    if sample_id not in allowed_ids:
                        raise RuntimeError(
                            f"{path}:{line_number} 的sample不在split manifest中：{sample_id!r}"
                        )
                if row_split != split:
                    continue
                prompt = build_prompt(task, row)
                response = normalize_response(row.get("response", {}))
                samples.append(
                    {
                        "task": task,
                        "scene_id": scene_id,
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "response": response,
                        "row": row,
                    }
                )
    return sorted(samples, key=lambda sample: (sample["task"], sample["sample_id"]))


def build_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "query_parse":
        return f"请解析中文语义导航指令并只输出严格JSON。\n指令：{row.get('query_text')}"
    if task == "candidate_rerank":
        payload = {
            "query_text": row.get("query_text"),
            "candidate_nodes": row.get("candidate_nodes"),
            "relation_metadata": row.get("relation_metadata"),
        }
        return f"请只在候选集合内重排并只输出严格JSON。\n输入：{json.dumps(payload, ensure_ascii=False)}"
    return (
        "请结合真实对象裁剪图和元数据输出严格对象属性JSON。\n"
        f"元数据：{json.dumps(row.get('metadata', {}), ensure_ascii=False)}"
    )


def normalize_response(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.strip()


class ChatDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], processor, max_length: int = 2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return prepare_training_inputs(self.processor, self.samples[index], self.max_length)


def prepare_training_inputs(processor, sample: dict[str, Any], max_length: int) -> dict[str, torch.Tensor]:
    image = load_sample_image(sample)
    user_messages = build_messages(sample, image=image, include_answer=False)
    full_messages = build_messages(sample, image=image, include_answer=True)
    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    images = [image] if image is not None else None
    prompt_inputs = processor(
        text=[prompt_text],
        images=images,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_text],
        images=images,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    result = {
        key: (value if key == "image_grid_thw" else value.squeeze(0))
        for key, value in full_inputs.items()
        if torch.is_tensor(value)
    }
    prompt_length = min(prompt_inputs["input_ids"].shape[-1], result["input_ids"].shape[-1])
    labels = result["input_ids"].clone()
    labels[:prompt_length] = -100
    result["labels"] = labels
    if sample["task"] == "object_attributes":
        if image is None or "pixel_values" not in result or "image_grid_thw" not in result:
            raise RuntimeError("object_attributes样本未真正进入多模态processor")
    return result


def build_messages(
    sample: dict[str, Any],
    *,
    image: Image.Image | None,
    include_answer: bool,
) -> list[dict[str, Any]]:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": sample["prompt"]})
    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]}
        )
    return messages


def load_sample_image(sample: dict[str, Any]) -> Image.Image | None:
    if sample["task"] != "object_attributes":
        return None
    path = Path(sample["row"].get("crop_path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"对象crop不存在：{path}")
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        if max(rgb.size) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(rgb.size)
            resized = (
                max(1, round(rgb.width * scale)),
                max(1, round(rgb.height * scale)),
            )
            rgb = rgb.resize(resized, Image.Resampling.LANCZOS)
        return rgb.copy()


def multimodal_collate(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    max_length = max(item["input_ids"].numel() for item in batch)
    for key in ("input_ids", "attention_mask", "labels"):
        values = []
        pad = -100 if key == "labels" else (0 if key == "attention_mask" else pad_token_id)
        for item in batch:
            tensor = item[key]
            if tensor.numel() < max_length:
                tensor = torch.cat(
                    [tensor, torch.full((max_length - tensor.numel(),), pad, dtype=tensor.dtype)]
                )
            values.append(tensor)
        result[key] = torch.stack(values)
    for key in sorted(set().union(*(item.keys() for item in batch)) - set(result)):
        tensors = [item[key] for item in batch if key in item]
        if tensors:
            result[key] = torch.cat(tensors, dim=0)
    return result


def train_lora(
    args: argparse.Namespace,
    processor,
    train_samples: list[dict[str, Any]],
    device: str,
    task_weights: dict[str, float],
) -> None:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if device != "cuda":
        model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    dataset = ChatDataset(train_samples, processor, max_length=args.max_length)
    counts = Counter(sample["task"] for sample in train_samples)
    sample_weights = [task_weights[sample["task"]] / counts[sample["task"]] for sample in train_samples]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda batch: multimodal_collate(
            batch,
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    losses = []
    progress = tqdm(total=args.max_steps, desc="qwen-lora")
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if len(losses) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                if step >= args.max_steps:
                    break
    progress.close()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    _write_json(args.output_dir / "losses.json", losses)
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
) -> dict[str, Any]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path)).eval()
    if device != "cuda":
        model.to(device)
    predictions = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "task": defaultdict(list),
        "scene": defaultdict(list),
    }
    for sample in tqdm(eval_samples[: args.max_eval_samples], desc="eval"):
        prediction = generate_json(model, processor, sample, max_length=args.max_length)
        target = sample["response"]
        target_obj = json.loads(target)
        allowed_ids = sample["row"].get("candidate_ids")
        validation = parse_and_validate_output(
            sample["task"],
            prediction,
            allowed_candidate_ids=allowed_ids,
        )
        pred_obj = validation.normalized_output
        hits, total = compare_json_fields(pred_obj, target_obj)
        item = {
            "task": sample["task"],
            "scene_id": sample["scene_id"],
            "sample_id": sample["sample_id"],
            "target": target,
            "prediction": prediction,
            "prediction_json": pred_obj,
            "target_json": target_obj,
            "schema_valid": validation.valid,
            "validation_errors": validation.validation_errors,
            "exact_match": prediction.strip() == target.strip(),
            "field_hits": hits,
            "field_total": total,
        }
        if sample["task"] == "candidate_rerank":
            ranking = pred_obj.get("ranking", []) if isinstance(pred_obj, dict) else []
            correct = int(sample["row"]["correct_target_node_id"])
            item["rerank_top1"] = bool(ranking and ranking[0] == correct)
            item["rerank_reciprocal_rank"] = (
                1.0 / (ranking.index(correct) + 1) if correct in ranking else 0.0
            )
        predictions.append(item)
        grouped["task"][sample["task"]].append(item)
        grouped["scene"][sample["scene_id"]].append(item)
    _write_json(output_json, predictions)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    overall = summarize_prediction_group(predictions)
    overall["by_task"] = {
        key: summarize_prediction_group(value) for key, value in sorted(grouped["task"].items())
    }
    overall["by_scene"] = {
        key: summarize_prediction_group(value) for key, value in sorted(grouped["scene"].items())
    }
    return overall


def generate_json(model, processor, sample: dict[str, Any], *, max_length: int) -> str:
    image = load_sample_image(sample)
    messages = build_messages(sample, image=image, include_answer=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs.input_ids.shape[1] :]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def processor_smoke_check(processor, samples: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    sample = next((item for item in samples if item["task"] == "object_attributes"), None)
    if sample is None:
        raise RuntimeError("没有object_attributes样本可用于processor检查")
    tensors = prepare_training_inputs(processor, sample, max_length)
    supervised_tokens = int((tensors["labels"] != -100).sum().item())
    passed = (
        "pixel_values" in tensors
        and "image_grid_thw" in tensors
        and supervised_tokens > 0
    )
    if not passed:
        raise RuntimeError("多模态processor检查失败：缺少图像张量或监督token")
    return {
        "schema_version": "stage4-multimodal-processor-smoke-v1",
        "passed": passed,
        "scene_id": sample["scene_id"],
        "sample_id": sample["sample_id"],
        "crop_path": sample["row"]["crop_path"],
        "image_sha256": sample["row"]["image_sha256"],
        "supervised_token_count": supervised_tokens,
        "processor_tensor_shapes": {
            key: list(value.shape) for key, value in sorted(tensors.items())
        },
    }


def summarize_prediction_group(items: list[dict[str, Any]]) -> dict[str, float]:
    count = len(items)
    if count == 0:
        return {
            "num_eval": 0.0,
            "exact_match": 0.0,
            "schema_valid_rate": 0.0,
            "field_accuracy": 0.0,
            "rerank_top1": 0.0,
            "rerank_mrr": 0.0,
        }
    rerank = [item for item in items if "rerank_top1" in item]
    return {
        "num_eval": float(count),
        "exact_match": sum(item["exact_match"] for item in items) / count,
        "schema_valid_rate": sum(item["schema_valid"] for item in items) / count,
        "field_accuracy": sum(item["field_hits"] for item in items)
        / max(1, sum(item["field_total"] for item in items)),
        "rerank_top1": sum(item["rerank_top1"] for item in rerank) / max(1, len(rerank)),
        "rerank_mrr": sum(item["rerank_reciprocal_rank"] for item in rerank)
        / max(1, len(rerank)),
    }


def compare_json_fields(prediction: Any, target: Any) -> tuple[int, int]:
    if not isinstance(target, dict):
        return int(prediction == target), 1
    target_flat = flatten_dict(target)
    if not isinstance(prediction, dict):
        return 0, max(1, len(target_flat))
    prediction_flat = flatten_dict(prediction)
    hits = sum(
        key in prediction_flat and normalize_value(prediction_flat[key]) == normalize_value(value)
        for key, value in target_flat.items()
    )
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
        return f"{value:.6f}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def resolve_task_weights(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, float]:
    weights = {
        task: float(manifest.get("task_sampling_weights", {}).get(task, 1.0))
        for task in args.tasks
    }
    for value in args.task_weight:
        task, raw_weight = value.split("=", 1)
        task = LEGACY_TASK_ALIASES.get(task, task)
        if task not in weights:
            raise ValueError(f"任务权重引用了未启用任务：{task}")
        weights[task] = float(raw_weight)
    if any(value <= 0 for value in weights.values()):
        raise ValueError("任务权重必须为正数")
    total = sum(weights.values())
    return {task: value / total for task, value in sorted(weights.items())}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(after[key]) - float(before[key])
        for key in ("schema_valid_rate", "field_accuracy", "exact_match", "rerank_top1", "rerank_mrr")
    }


def write_markdown_report(path: Path, log: dict[str, Any]) -> None:
    before = log["baseline_metrics"]
    after = log["finetuned_metrics"]
    lines = [
        "# 阶段4 Qwen LoRA微调前后对比",
        "",
        f"- 模型：`{Path(log['model']).name}`",
        f"- 训练样本数：{log['num_train_samples']}",
        f"- 评估样本数：{log['num_eval_samples']}",
        f"- scene泄漏：{len(log['scene_leakage'])}",
        f"- 训练状态：{log['status']}",
        "",
        "| 指标 | 微调前 | 微调后 |",
        "|---|---:|---:|",
    ]
    for key in ("schema_valid_rate", "field_accuracy", "exact_match", "rerank_top1", "rerank_mrr"):
        lines.append(f"| {key} | {before[key]:.4f} | {after[key]:.4f} |")
    lines.extend(
        [
            "",
            "逐任务和逐场景指标分别保存在`per_task_metrics.json`与`per_scene_metrics.json`。",
            "自动标签属于弱监督，指标不等价于人工语义真值精度。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _guard_output_paths(args: argparse.Namespace) -> None:
    protected = (PROJECT_ROOT / "checkpoint/qwen_lora_semantic_delegate").resolve()
    output = args.output_dir.resolve()
    if output == protected or protected in output.parents:
        raise ValueError("拒绝覆盖既有LoRA adapter；请使用outputs/project_v2/qwen_eval/sanity_adapter")


def _legacy_split(scene_id: str) -> str:
    value = int.from_bytes(scene_id.encode("utf-8"), "little") % 10
    return "train" if value < 7 else ("validation" if value < 9 else "test")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
