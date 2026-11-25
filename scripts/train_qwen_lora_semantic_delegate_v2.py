from __future__ import annotations

import argparse
import copy
import ctypes
import itertools
import json
import math
import os
import random
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from conceptgraph.vlm.image_input import prepare_qwen_image


PROJECT_ROOT = Path("/root/autodl-tmp/lizichong")
DEFAULT_DATASET = PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2"
DEFAULT_MODEL = PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct"
MAX_IMAGE_SIDE = 448


def _prefer_conda_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = Path(conda_prefix) / "lib"
    libstdcxx = lib_dir / "libstdc++.so.6"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(lib_dir)] + [item for item in current.split(":") if item != str(lib_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(item for item in paths if item)
    if libstdcxx.exists():
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)


_prefer_conda_runtime_libs()

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from conceptgraph.vlm.qwen_cache import sha256_file  # noqa: E402
from conceptgraph.vlm.qwen_finetune_v2 import (  # noqa: E402
    TASK_FILES,
    audit_dataset,
    build_prompt,
    canonical_json,
    evaluate_prediction_items,
    load_dataset_rows,
    prediction_item,
    semantic_sha256,
)
from conceptgraph.vlm.schemas import CANONICAL_COLORS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2-VL导航语义LoRA v2训练与严格评估。")
    parser.add_argument("--mode", choices=("audit", "evaluate", "train"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=ADAPTER_OR_BASE",
        help="evaluate模式可重复传入；BASE表示不加载adapter。",
    )
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--task-weight", action="append", default=[], metavar="TASK=WEIGHT")
    parser.add_argument("--rare-multicolor-boost", type=float, default=64.0)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _guard_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        report = audit_dataset(args.dataset_dir)
        _write_json(args.output_root / "data_audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    _require_cuda()
    _set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    rows = load_dataset_rows(args.dataset_dir)
    if args.mode == "evaluate":
        run_evaluation(args, processor, rows)
    else:
        run_training(args, processor, rows)


def _guard_args(args: argparse.Namespace) -> None:
    if args.output_root.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"输出根必须精确为：{DEFAULT_OUTPUT}")
    if args.dataset_dir.resolve() != DEFAULT_DATASET.resolve():
        raise ValueError(f"数据集必须使用未修改的Stage4目录：{DEFAULT_DATASET}")
    if args.model.resolve() != DEFAULT_MODEL.resolve():
        raise ValueError(f"基础模型必须使用本地固定目录：{DEFAULT_MODEL}")
    if args.repeats < 1:
        raise ValueError("repeats必须为正整数")
    if args.mode == "train" and args.split != "validation":
        raise ValueError("训练和checkpoint选择只允许使用validation")
    if args.mode == "evaluate" and not args.variant:
        raise ValueError("evaluate模式至少需要一个--variant")
    if args.split == "test" and not args.allow_sealed_test:
        raise ValueError("test集仍封存；最终评估必须显式传入--allow-sealed-test")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size和grad-accum必须为正整数")
    if args.max_steps < 1 or args.eval_steps < 1:
        raise ValueError("max-steps和eval-steps必须为正整数")
    if args.rare_multicolor_boost < 1.0:
        raise ValueError("rare-multicolor-boost不能小于1")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("本目标要求真实GPU训练/推理，但PyTorch CUDA不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持所选bf16 LoRA配置")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _parse_variants(values: list[str]) -> list[tuple[str, Path | None]]:
    result = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant格式错误：{value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in names:
            raise ValueError(f"variant名称为空或重复：{name!r}")
        path = None if raw_path.strip().upper() == "BASE" else Path(raw_path).resolve()
        if path is not None and not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter不存在：{path}")
        result.append((name, path))
        names.add(name)
    return result


def _select_rows(
    rows: list[dict[str, Any]], split: str, max_samples: int
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split]
    if max_samples > 0:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_task[row["task"]].append(row)
        limited = []
        per_task = max(1, max_samples // len(TASK_FILES))
        for task in sorted(by_task):
            limited.extend(by_task[task][:per_task])
        selected = limited[:max_samples]
    return sorted(selected, key=lambda row: (row["task"], row["sample_id"]))


def _adapter_fingerprint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"kind": "base_model", "path": None, "adapter_sha256": None}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    payload = [(str(item.relative_to(path)), item.stat().st_size, sha256_file(item)) for item in files]
    return {
        "kind": "lora_adapter",
        "path": str(path),
        "file_count": len(files),
        "adapter_sha256": semantic_sha256(payload),
    }


def _load_model(model_path: Path, adapter_path: Path | None, *, trainable: bool = False):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    return model


def run_evaluation(args: argparse.Namespace, processor, rows: list[dict[str, Any]]) -> None:
    eval_rows = _select_rows(rows, args.split, args.max_eval_samples)
    output_dir = args.output_root / args.evaluation_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_marker = args.output_root / "sealed_test_release.json"
    if args.split == "test":
        if sealed_marker.exists():
            raise RuntimeError("sealed test已执行；拒绝再次评估以防根据test调参")
        _write_json(
            sealed_marker,
            {
                "schema_version": "qwen-finetune-v2-sealed-test-release-v1",
                "released_at_unix_s": time.time(),
                "dataset_manifest_sha256": sha256_file(args.dataset_dir / "split_manifest.json"),
                "variants": [name for name, _ in _parse_variants(args.variant)],
                "repeats": args.repeats,
            },
        )

    variant_reports = {}
    for name, adapter_path in _parse_variants(args.variant):
        model = _load_model(args.model, adapter_path).eval()
        model.config.use_cache = True
        repeat_reports = []
        content_hashes = []
        for repeat in range(args.repeats):
            _set_seed(args.seed)
            predictions = evaluate_rows(
                model,
                processor,
                eval_rows,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                description=f"{name}:{args.split}:run{repeat + 1}",
            )
            metrics = evaluate_prediction_items(predictions)
            content_hash = prediction_content_sha256(predictions)
            content_hashes.append(content_hash)
            _write_json(output_dir / f"{name}_run_{repeat + 1}_predictions.json", predictions)
            _write_json(output_dir / f"{name}_run_{repeat + 1}_metrics.json", metrics)
            repeat_reports.append(
                {
                    "repeat": repeat + 1,
                    "content_sha256": content_hash,
                    "metrics": metrics,
                }
            )
        variant_reports[name] = {
            "adapter": _adapter_fingerprint(adapter_path),
            "repeat_reports": repeat_reports,
            "deterministic": len(set(content_hashes)) == 1,
            "mean_metrics": aggregate_metric_reports(
                [item["metrics"] for item in repeat_reports]
            ),
        }
        del model
        torch.cuda.empty_cache()

    report = {
        "schema_version": "qwen-finetune-v2-evaluation-v1",
        "split": args.split,
        "cache_used": False,
        "raw_model_outputs_only": True,
        "sample_count": len(eval_rows),
        "model": str(args.model),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_length,
        },
        "variants": variant_reports,
    }
    _write_json(output_dir / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


class NavigationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor, max_length: int):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return prepare_training_inputs(self.processor, self.rows[index], self.max_length)


def _load_image(row: dict[str, Any]) -> Image.Image | None:
    if row["task"] != "object_attributes":
        return None
    path = Path(row.get("crop_path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"对象crop不存在：{path}")
    with Image.open(path) as image:
        image.load()
        return prepare_qwen_image(image, MAX_IMAGE_SIDE).copy()


def _messages(row: dict[str, Any], image: Image.Image | None, *, answer: bool) -> list[dict[str, Any]]:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": build_prompt(row["task"], row)})
    messages = [{"role": "user", "content": content}]
    if answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": canonical_json(row["response"])}],
            }
        )
    return messages


def prepare_training_inputs(processor, row: dict[str, Any], max_length: int) -> dict[str, torch.Tensor]:
    image = _load_image(row)
    prompt_text = processor.apply_chat_template(
        _messages(row, image, answer=False), tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        _messages(row, image, answer=True), tokenize=False, add_generation_prompt=False
    )
    images = [image] if image is not None else None
    prompt_inputs = processor(
        text=[prompt_text], images=images, padding=False, return_tensors="pt"
    )
    full_inputs = processor(text=[full_text], images=images, padding=False, return_tensors="pt")
    full_length = int(full_inputs["input_ids"].shape[-1])
    if full_length > max_length:
        raise RuntimeError(
            f"样本{row['sample_id']}长度{full_length}超过max_length={max_length}；拒绝截断监督答案"
        )
    result = {
        key: (value if key == "image_grid_thw" else value.squeeze(0))
        for key, value in full_inputs.items()
        if torch.is_tensor(value)
    }
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    labels = result["input_ids"].clone()
    labels[:prompt_length] = -100
    result["labels"] = labels
    if int((labels != -100).sum()) == 0:
        raise RuntimeError(f"样本{row['sample_id']}没有监督token")
    if row["task"] == "object_attributes" and not {
        "pixel_values",
        "image_grid_thw",
    }.issubset(result):
        raise RuntimeError("object_attributes样本没有进入多模态processor")
    return result


def multimodal_collate(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    max_length = max(item["input_ids"].numel() for item in batch)
    for key in ("input_ids", "attention_mask", "labels"):
        values = []
        pad_value = -100 if key == "labels" else (0 if key == "attention_mask" else pad_token_id)
        for item in batch:
            value = item[key]
            if value.numel() < max_length:
                padding = torch.full(
                    (max_length - value.numel(),), pad_value, dtype=value.dtype
                )
                value = torch.cat((value, padding))
            values.append(value)
        result[key] = torch.stack(values)
    other_keys = sorted(set().union(*(item.keys() for item in batch)) - set(result))
    for key in other_keys:
        values = [item[key] for item in batch if key in item]
        if values:
            result[key] = torch.cat(values, dim=0)
    return result


def _task_weights(values: list[str]) -> dict[str, float]:
    weights = {task: 1.0 for task in TASK_FILES}
    for value in values:
        task, raw_weight = value.split("=", 1)
        if task not in weights:
            raise ValueError(f"未知任务权重：{task}")
        weights[task] = float(raw_weight)
    if any(value <= 0 for value in weights.values()):
        raise ValueError("任务权重必须为正")
    total = sum(weights.values())
    return {task: value / total for task, value in sorted(weights.items())}


def augment_color_copy_contracts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = next(
        row
        for row in rows
        if row["task"] == "object_attributes"
        and len(row["response"].get("colors", [])) > 1
    )
    colors = sorted(CANONICAL_COLORS - {"unknown"})
    augmented = list(rows)
    for index, (first, second) in enumerate(itertools.permutations(colors, 2)):
        row = copy.deepcopy(source)
        phrase = f"{first}/{second} {first}/{second} {first}/{second}"
        row["sample_id"] = f"{source['sample_id']}:synthetic-color-copy-{index:03d}"
        row["metadata"]["caption"] = f"{phrase} {row['response']['object_tag']}"
        row["response"]["summary"] = row["metadata"]["caption"]
        row["response"]["colors"] = [first, second]
        row["synthetic_color_copy"] = True
        augmented.append(row)
    return augmented


def run_training(args: argparse.Namespace, processor, rows: list[dict[str, Any]]) -> None:
    original_train_rows = _select_rows(rows, "train", args.max_train_samples)
    train_rows = augment_color_copy_contracts(original_train_rows)
    val_rows = _select_rows(rows, "validation", args.max_eval_samples)
    train_scenes = {row["scene_id"] for row in train_rows}
    val_scenes = {row["scene_id"] for row in val_rows}
    if train_scenes & val_scenes:
        raise RuntimeError(f"scene泄漏：{sorted(train_scenes & val_scenes)}")
    seed_dir = args.output_root / "runs" / f"seed_{args.seed}"
    if seed_dir.exists() and args.resume_from is None:
        raise FileExistsError(f"seed输出已存在，拒绝覆盖：{seed_dir}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(seed_dir / "processor")

    task_weights = _task_weights(args.task_weight)
    resolved_config = {
        "schema_version": "qwen-finetune-v2-config-v1",
        "seed": args.seed,
        "model": str(args.model),
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest_sha256": sha256_file(args.dataset_dir / "split_manifest.json"),
        "method": "bf16_lora",
        "dtype": "bfloat16",
        "max_image_side": MAX_IMAGE_SIDE,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "early_stop_patience": args.early_stop_patience,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_weights": task_weights,
        "rare_multicolor_boost": args.rare_multicolor_boost,
        "rare_multicolor_train_samples": sum(
            row["task"] == "object_attributes"
            and not row.get("synthetic_color_copy")
            and len(row["response"].get("colors", [])) > 1
            for row in train_rows
        ),
        "training_augmentation": {
            "type": "systematic_ordered_color_pair_copy_contract",
            "synthetic_sample_count": sum(
                bool(row.get("synthetic_color_copy")) for row in train_rows
            ),
            "canonical_colors": sorted(CANONICAL_COLORS - {"unknown"}),
            "uses_validation_or_test_rows": False,
            "visual_ground_truth_claim": False,
        },
        "original_train_sample_count": len(original_train_rows),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(val_rows),
        "train_scenes": sorted(train_scenes),
        "validation_scenes": sorted(val_scenes),
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "prompt_sha256_by_task": {
            task: semantic_sha256(
                [
                    build_prompt(task, row)
                    for row in train_rows
                    if row["task"] == task
                ]
            )
            for task in sorted(TASK_FILES)
        },
    }
    _write_json(seed_dir / "resolved_config.json", resolved_config)

    if args.resume_from is None:
        model = _load_model(args.model, None)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=resolved_config["lora_target_modules"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        completed_steps = 0
    else:
        model = _load_model(args.model, args.resume_from, trainable=True)
        state_path = args.resume_from / "trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"恢复checkpoint缺少trainer_state.pt：{state_path}")
        completed_steps = int(torch.load(state_path, map_location="cpu")["completed_steps"])

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())

    dataset = NavigationDataset(train_rows, processor, args.max_length)
    counts = Counter(row["task"] for row in train_rows)
    sample_weights = []
    for row in train_rows:
        weight = task_weights[row["task"]] / counts[row["task"]]
        if (
            row["task"] == "object_attributes"
            and not row.get("synthetic_color_copy")
            and len(row["response"].get("colors", [])) > 1
        ):
            weight *= args.rare_multicolor_boost
        sample_weights.append(weight)
    micro_steps = max(1, (args.max_steps - completed_steps) * args.grad_accum)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=micro_steps * args.batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + completed_steps),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda batch: multimodal_collate(
            batch, processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        ),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(1, round(args.max_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps
    )
    if args.resume_from is not None:
        state = torch.load(args.resume_from / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng_state"])

    curve = []
    checkpoints = []
    best_score = -math.inf
    best_checkpoint = None
    stale_evaluations = 0
    stop_reason = "max_steps"
    micro_step = 0
    accumulated_loss = 0.0
    started = time.perf_counter()
    peak_memory_before = torch.cuda.max_memory_allocated()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, initial=completed_steps, desc=f"qwen-v2-seed-{args.seed}")
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        (loss / args.grad_accum).backward()
        accumulated_loss += float(loss.detach().cpu())
        micro_step += 1
        if micro_step % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        curve.append(
            {
                "step": completed_steps,
                "loss": accumulated_loss / args.grad_accum,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        accumulated_loss = 0.0
        progress.update(1)

        should_evaluate = completed_steps % args.eval_steps == 0 or completed_steps == args.max_steps
        if not should_evaluate:
            continue
        checkpoint_dir = seed_dir / "checkpoints" / f"step_{completed_steps:04d}"
        save_checkpoint(
            checkpoint_dir,
            model,
            processor,
            optimizer,
            scheduler,
            completed_steps,
            curve,
        )
        model.eval()
        model.config.use_cache = True
        predictions = evaluate_rows(
            model,
            processor,
            val_rows,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            description=f"seed{args.seed}:step{completed_steps}:validation",
        )
        metrics = evaluate_prediction_items(predictions)
        _write_json(checkpoint_dir / "validation_predictions.json", predictions)
        _write_json(checkpoint_dir / "validation_metrics.json", metrics)
        score = float(metrics["selection_score"])
        improved = score > best_score + 1e-8
        checkpoints.append(
            {
                "step": completed_steps,
                "path": str(checkpoint_dir),
                "adapter_sha256": _adapter_fingerprint(checkpoint_dir)["adapter_sha256"],
                "selection_score": score,
                "improved": improved,
                "metrics": metrics,
            }
        )
        if improved:
            best_score = score
            best_checkpoint = checkpoint_dir
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        model.config.use_cache = False
        model.train()
        if metrics["all_hard_gates_passed"]:
            stop_reason = "validation_hard_gates_passed"
            break
        if stale_evaluations >= args.early_stop_patience:
            stop_reason = "validation_patience_exhausted"
            break
        if completed_steps >= args.max_steps:
            break
    progress.close()
    duration_s = time.perf_counter() - started
    peak_memory_mib = max(peak_memory_before, torch.cuda.max_memory_allocated()) / (1024**2)
    if best_checkpoint is None:
        raise RuntimeError("训练结束但没有生成可选checkpoint")
    training_report = {
        "schema_version": "qwen-finetune-v2-training-report-v1",
        "seed": args.seed,
        "status": "completed",
        "actual_completed_steps": completed_steps,
        "early_stopped": stop_reason != "max_steps",
        "stop_reason": stop_reason,
        "training_duration_s": duration_s,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mib": peak_memory_mib,
        "trainable_parameters": trainable_parameters,
        "all_parameters": all_parameters,
        "trainable_fraction": trainable_parameters / all_parameters,
        "best_checkpoint": str(best_checkpoint),
        "best_selection_score": best_score,
        "checkpoints": checkpoints,
        "training_curve": curve,
        "resolved_config": resolved_config,
    }
    _write_json(seed_dir / "training_report.json", training_report)
    _write_json(
        seed_dir / "best_checkpoint.json",
        {
            "path": str(best_checkpoint),
            "selection_score": best_score,
            "adapter_sha256": _adapter_fingerprint(best_checkpoint)["adapter_sha256"],
        },
    )
    print(json.dumps(training_report, ensure_ascii=False, indent=2))


def save_checkpoint(
    path: Path,
    model,
    processor,
    optimizer,
    scheduler,
    completed_steps: int,
    curve: list[dict[str, Any]],
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    torch.save(
        {
            "completed_steps": completed_steps,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": _rng_state(),
            "curve": curve,
        },
        path / "trainer_state.pt",
    )
    _write_json(
        path / "checkpoint_manifest.json",
        {
            "schema_version": "qwen-finetune-v2-checkpoint-v1",
            "completed_steps": completed_steps,
            "recoverable": True,
            "contains_optimizer": True,
            "contains_scheduler": True,
            "contains_rng_state": True,
        },
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def evaluate_rows(
    model,
    processor,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    max_new_tokens: int,
    description: str,
) -> list[dict[str, Any]]:
    predictions = []
    for row in tqdm(rows, desc=description):
        started = time.perf_counter()
        raw_output = generate(model, processor, row, max_length, max_new_tokens)
        predictions.append(
            prediction_item(
                row=row, raw_output=raw_output, latency_s=time.perf_counter() - started
            )
        )
    return predictions


def generate(model, processor, row: dict[str, Any], max_length: int, max_new_tokens: int) -> str:
    image = _load_image(row)
    text = processor.apply_chat_template(
        _messages(row, image, answer=False), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt",
    )
    if int(inputs["input_ids"].shape[-1]) > max_length:
        raise RuntimeError(f"评估样本{row['sample_id']}输入超过max_length={max_length}")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def prediction_content_sha256(items: list[dict[str, Any]]) -> str:
    semantic_items = [
        {key: value for key, value in item.items() if key != "latency_s"} for item in items
    ]
    return semantic_sha256(semantic_items)


def aggregate_metric_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "overall_schema_valid_rate": lambda item: item["overall_schema_valid_rate"],
        "minimum_task_schema_valid_rate": lambda item: min(
            item["per_task_schema_valid_rate"].values()
        ),
        "query_macro_f1": lambda item: item["query_parse"]["semantic_macro_f1"],
        "query_semantic_exact_match": lambda item: item["query_parse"][
            "semantic_exact_match"
        ],
        "object_category_macro_f1": lambda item: item["object_attributes"][
            "category_macro_f1"
        ],
        "object_color_macro_f1": lambda item: item["object_attributes"]["color_macro_f1"],
        "object_motion_macro_f1": lambda item: item["object_attributes"][
            "motion_macro_f1"
        ],
        "rerank_top1": lambda item: item["candidate_rerank"]["top1"],
        "rerank_mrr": lambda item: item["candidate_rerank"]["mrr"],
        "rerank_candidate_legality": lambda item: item["candidate_rerank"][
            "candidate_legality_rate"
        ],
        "fallback_rate": lambda item: item["schema_or_illegal_fallback_rate"],
        "selection_score": lambda item: item["selection_score"],
    }
    result = {}
    for name, getter in paths.items():
        values = [float(getter(report)) for report in reports]
        result[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "worst": min(values) if name != "fallback_rate" else max(values),
            "values": values,
        }
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def promote_best_adapter(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖best_adapter：{destination}")
    shutil.copytree(source, destination)


if __name__ == "__main__":
    main()
