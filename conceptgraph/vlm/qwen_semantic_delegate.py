from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from conceptgraph.vlm.qwen_cache import QwenDiskCache, semantic_version, sha256_file
from conceptgraph.vlm.image_input import IMAGE_PREPROCESS_VERSION
from conceptgraph.vlm.schemas import (
    SCHEMA_VERSION,
    normalize_category,
    normalize_color,
    normalize_motion_state,
    normalize_relation,
    parse_and_validate_output,
)


Backend = Callable[..., str]
PROMPT_VERSION = "stage4-compact-json-v2"
FINETUNE_V2_PROMPT_VERSION = "finetune-v2-json-v1"


@dataclass
class QwenDelegateConfig:
    enabled: bool = False
    model_name_or_path: str | None = None
    adapter_path: str | Path | None = None
    model_revision: str | None = None
    adapter_revision: str | None = None
    device: str = "cuda:0"
    use_qwen_caption: bool = False
    use_qwen_object_attributes: bool = False
    use_qwen_relation: bool = False
    use_qwen_query_parse: bool = False
    use_qwen_rerank: bool = False
    timeout_s: float = 30.0
    max_new_tokens: int = 192
    max_images: int = 4
    do_sample: bool = False
    cache_dir: str | Path | None = None
    log_dir: str | Path | None = None
    async_workers: int = 2
    prompt_profile: str = "stage4"
    torch_dtype: str | None = None
    max_image_side: int | None = None

    def __post_init__(self) -> None:
        if self.use_qwen_caption:
            self.use_qwen_object_attributes = True
        if self.do_sample:
            raise ValueError("阶段4要求确定性生成，do_sample 必须为 false")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_images <= 0:
            raise ValueError("max_images must be positive")
        if self.async_workers <= 0:
            raise ValueError("async_workers must be positive")
        if self.prompt_profile not in {"stage4", "finetune_v2", "completion_v1", "completion_v2", "completion_v3"}:
            raise ValueError("unsupported prompt_profile")
        if self.torch_dtype not in {None, "float16", "bfloat16", "float32"}:
            raise ValueError("torch_dtype must be float16, bfloat16, float32, or null")
        if self.max_image_side is not None and self.max_image_side <= 0:
            raise ValueError("max_image_side must be positive")


class QwenSemanticDelegate:
    """带严格schema、缓存、超时和规则回退的惰性Qwen语义代理。"""

    def __init__(
        self,
        config: QwenDelegateConfig | None = None,
        *,
        backend: Backend | Any | None = None,
        cache: QwenDiskCache | None = None,
    ):
        self.config = config or QwenDelegateConfig()
        self.log_dir = Path(self.config.log_dir) if self.config.log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        if self.cache is None and self.config.cache_dir is not None:
            self.cache = QwenDiskCache(self.config.cache_dir)
        self._backend = backend
        self._backend_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._counter = 0
        self._backend_calls = 0
        self._sync_executor = self._new_sync_executor()
        self._retired_executors: list[ThreadPoolExecutor] = []
        self._async_executor = ThreadPoolExecutor(
            max_workers=self.config.async_workers,
            thread_name_prefix="qwen-semantic-async",
        )
        self._active_async: set[asyncio.Future] = set()
        self._closed = False
        self.model_identifier = _artifact_identifier(
            self.config.model_name_or_path,
            explicit_revision=self.config.model_revision,
            default_name="Qwen2-VL-2B-Instruct",
        )
        self.adapter_identifier = _artifact_identifier(
            self.config.adapter_path,
            explicit_revision=self.config.adapter_revision,
            default_name=None,
        )

    @property
    def backend_call_count(self) -> int:
        with self._counter_lock:
            return self._backend_calls

    @property
    def pending_async_count(self) -> int:
        return sum(not future.done() for future in self._active_async)

    def describe_object(
        self,
        crop_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        object_version: str | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        fallback = self._rule_object_attributes(metadata)
        paths = [Path(item) for item in (crop_paths or [])[: self.config.max_images]]
        image_hashes: list[str] = []
        missing = []
        for path in paths:
            try:
                image_hashes.append(sha256_file(path))
            except OSError:
                missing.append(path.name)
        enabled = self.config.enabled and self.config.use_qwen_object_attributes
        if enabled and (not paths or missing):
            return self._fallback_envelope(
                task="object_attributes",
                fallback=fallback,
                failure_code="IMAGE_UNAVAILABLE",
                fallback_reason=(
                    "object crop is required and must be readable"
                    if not missing
                    else f"unreadable object crop names: {missing}"
                ),
                validation_errors=[],
                cache_info=self._empty_cache_info(),
                latency_s=0.0,
            )
        if self.config.prompt_profile.startswith("completion_"):
            from conceptgraph.completion.semantic_contract import build_completion_prompt

            prompt = build_completion_prompt("object_attributes", {"metadata": _semantic_metadata(metadata)},self.config.prompt_profile)
        elif self.config.prompt_profile == "finetune_v2":
            from conceptgraph.vlm.qwen_finetune_v2 import build_prompt

            prompt = build_prompt(
                "object_attributes",
                {"metadata": _semantic_metadata(metadata)},
            )
        else:
            prompt = (
                "请结合对象裁剪图和元数据，只输出一行JSON，不要输出JSON Schema、Markdown或解释。"
                "必须且只能包含：schema_version(固定为1.0.0)、summary、object_tag、possible_tags、"
                "colors、body_type、motion_state、confidence。"
                "示例：{\"schema_version\":\"1.0.0\",\"summary\":\"gray car\","
                "\"object_tag\":\"car\",\"possible_tags\":[\"car\"],\"colors\":[\"gray\"],"
                "\"body_type\":\"sedan\",\"motion_state\":\"static\",\"confidence\":0.8}"
                f"\n元数据：{json.dumps(_semantic_metadata(metadata), ensure_ascii=False, sort_keys=True)}"
            )
        normalized_input = {
            "prompt_version": self._prompt_version,
            "metadata": _semantic_metadata(metadata),
        }
        version = object_version or str(metadata.get("object_version") or semantic_version(normalized_input))
        return self._execute(
            task="object_attributes",
            enabled=enabled,
            prompt=prompt,
            normalized_input=normalized_input,
            fallback=fallback,
            image_paths=paths,
            image_hashes=image_hashes,
            object_version=version,
        )

    def parse_query(
        self,
        query_text: str,
        *,
        default_near_distance_m: float = 8.0,
        rule_result: Any | None = None,
    ) -> dict[str, Any]:
        fallback = self._rule_query_parse(
            query_text,
            default_near_distance_m=default_near_distance_m,
            rule_result=rule_result,
        )
        if self.config.prompt_profile.startswith("completion_"):
            from conceptgraph.completion.semantic_contract import build_completion_prompt

            prompt = build_completion_prompt("query_parse", {"query_text": query_text},self.config.prompt_profile)
        elif self.config.prompt_profile == "finetune_v2":
            from conceptgraph.vlm.qwen_finetune_v2 import build_prompt

            prompt = build_prompt("query_parse", {"query_text": query_text})
        else:
            prompt = (
                "请解析中文语义导航指令，只输出一行JSON，不要输出JSON Schema、Markdown或解释。"
                "必须且只能包含：schema_version(固定为1.0.0)、target_text、target_categories、"
                "target_colors、anchor_text、anchor_categories、anchor_colors、relation、"
                "prefer_nearest、prefer_front、confidence。"
                "示例：{\"schema_version\":\"1.0.0\",\"target_text\":\"小汽车\","
                "\"target_categories\":[\"car\"],\"target_colors\":[],\"anchor_text\":null,"
                "\"anchor_categories\":[],\"anchor_colors\":[],\"relation\":\"near\","
                "\"prefer_nearest\":false,\"prefer_front\":false,\"confidence\":0.8}"
                f"\n指令：{query_text.strip()}"
            )
        return self._execute(
            task="query_parse",
            enabled=self.config.enabled and self.config.use_qwen_query_parse,
            prompt=prompt,
            normalized_input={
                "prompt_version": self._prompt_version,
                "query_text": " ".join(query_text.strip().split()),
            },
            fallback=fallback,
        )

    def rerank_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
        *,
        edge_version: str | None = None,
    ) -> dict[str, Any]:
        candidate_ids = _candidate_ids(candidates)
        if not candidate_ids:
            return self._fallback_envelope(
                task="candidate_rerank",
                fallback={
                    "schema_version": SCHEMA_VERSION,
                    "selected_node_id": None,
                    "ranking": [],
                    "explanation": "no_verified_candidates",
                    "confidence": 0.0,
                },
                failure_code="NO_CANDIDATES",
                fallback_reason="no verified candidates were supplied",
                validation_errors=[],
                cache_info=self._empty_cache_info(),
                latency_s=0.0,
            )
        fallback = {
            "schema_version": SCHEMA_VERSION,
            "selected_node_id": candidate_ids[0],
            "ranking": candidate_ids,
            "explanation": "rule_order_fallback",
            "confidence": 0.5,
        }
        semantic_edges = [_semantic_edge(item) for item in (edges or [])]
        version = edge_version or semantic_version(semantic_edges)
        if self.config.prompt_profile.startswith("completion_"):
            from conceptgraph.completion.semantic_contract import build_completion_prompt

            prompt = build_completion_prompt("candidate_rerank", dict(query_text=query_text,
                candidate_ids=candidate_ids, candidate_nodes=candidates, relation_metadata=semantic_edges),self.config.prompt_profile)
        elif self.config.prompt_profile == "finetune_v2":
            from conceptgraph.vlm.qwen_finetune_v2 import build_prompt

            prompt = build_prompt(
                "candidate_rerank",
                {
                    "query_text": query_text,
                    "candidate_ids": candidate_ids,
                    "candidate_nodes": candidates,
                    "relation_metadata": semantic_edges,
                },
            )
        else:
            prompt = (
                "请只在输入的已验证候选节点集合内重排，只输出一行JSON，不要输出JSON Schema、"
                "Markdown或解释文字。不得返回新节点，不得判断可达性、碰撞或路径。"
                "必须且只能包含：schema_version(固定为1.0.0)、selected_node_id(整数且等于ranking首项)、"
                "ranking(包含所有输入候选ID且不重复)、explanation、confidence(0到1)。"
                f"\n指令：{query_text.strip()}"
                f"\n已验证候选：{json.dumps(candidates, ensure_ascii=False, sort_keys=True)}"
                f"\n结构化关系元数据：{json.dumps(semantic_edges, ensure_ascii=False, sort_keys=True)}"
            )
        return self._execute(
            task="candidate_rerank",
            enabled=self.config.enabled and self.config.use_qwen_rerank,
            prompt=prompt,
            normalized_input={
                "prompt_version": self._prompt_version,
                "query_text": " ".join(query_text.strip().split()),
            },
            fallback=fallback,
            allowed_candidate_ids=candidate_ids,
            candidates=candidates,
            edge_version=version,
        )

    def infer_relation(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        *,
        structured_fact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """兼容旧入口；阶段3结构化几何事实始终具有最高权威。"""
        if structured_fact is not None:
            relation = structured_fact.get("relation", "near")
            try:
                relation = normalize_relation(relation)
            except (TypeError, ValueError):
                relation = "near"
            normalized = {
                "relation": relation,
                "confidence": _bounded_confidence(structured_fact.get("confidence", 1.0)),
                "explanation": "stage3_structured_fact",
            }
            return self._compatibility_relation_envelope(
                normalized,
                fallback_reason="阶段3结构化几何关系为权威事实",
                failure_code="STRUCTURED_RELATION_AUTHORITATIVE",
            )
        normalized = self._rule_relation(source, target)
        return self._compatibility_relation_envelope(
            normalized,
            fallback_reason="关系推理由阶段3几何规则负责",
            failure_code="GEOMETRIC_RELATION_AUTHORITATIVE",
        )

    async def parse_query_async(self, *args, **kwargs) -> dict[str, Any]:
        return await self._run_async(self.parse_query, *args, **kwargs)

    async def describe_object_async(self, *args, **kwargs) -> dict[str, Any]:
        return await self._run_async(self.describe_object, *args, **kwargs)

    async def rerank_candidates_async(self, *args, **kwargs) -> dict[str, Any]:
        return await self._run_async(self.rerank_candidates, *args, **kwargs)

    async def aclose(self) -> None:
        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executors = [self._async_executor, self._sync_executor, *self._retired_executors]
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, _shutdown_executor, executor, True) for executor in executors)
        )
        self._active_async.clear()

    def close(self, *, wait: bool = True) -> None:
        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executors = [self._async_executor, self._sync_executor, *self._retired_executors]
        for executor in executors:
            _shutdown_executor(executor, wait)
        self._active_async.clear()

    async def _run_async(self, function, *args, **kwargs):
        if self._closed:
            raise RuntimeError("QwenSemanticDelegate is closed")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._async_executor, lambda: function(*args, **kwargs))
        self._active_async.add(future)
        future.add_done_callback(self._active_async.discard)
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _execute(
        self,
        *,
        task: str,
        enabled: bool,
        prompt: str,
        normalized_input: dict[str, Any],
        fallback: dict[str, Any],
        image_paths: list[Path] | None = None,
        image_hashes: list[str] | None = None,
        object_version: str | None = None,
        allowed_candidate_ids: list[int] | None = None,
        candidates: Any = None,
        edge_version: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        normalized_input = {**normalized_input, "inference_contract": {
            "prompt_sha256": semantic_version(prompt),
            "image_preprocess": IMAGE_PREPROCESS_VERSION,
            "max_image_side": self.config.max_image_side,
            "max_new_tokens": self.config.max_new_tokens,
            "max_images": self.config.max_images,
            "torch_dtype": self.config.torch_dtype,
            "do_sample": self.config.do_sample,
        }}
        if self._closed:
            raise RuntimeError("QwenSemanticDelegate is closed")
        cache_info = self._empty_cache_info()
        if not enabled:
            return self._fallback_envelope(
                task=task,
                fallback=fallback,
                failure_code="QWEN_DISABLED",
                fallback_reason="Qwen task is disabled",
                validation_errors=[],
                cache_info=cache_info,
                latency_s=time.perf_counter() - started,
            )

        key = None
        lookup_diagnostic = None
        if self.cache is not None:
            key = self.cache.build_key(
                task=task,
                schema_version=SCHEMA_VERSION,
                model_id=self.model_identifier,
                adapter_id=self.adapter_identifier,
                normalized_input=normalized_input,
                image_hashes=image_hashes,
                object_version=object_version,
                candidates=candidates,
                edge_version=edge_version,
            )
            lookup = self.cache.get(key)
            cache_info = {
                "enabled": True,
                "hit": lookup.hit,
                "key": key,
                "status": lookup.status,
                "diagnostic": lookup.diagnostic,
            }
            lookup_diagnostic = lookup.diagnostic
            if lookup.hit and lookup.payload is not None:
                envelope = dict(lookup.payload)
                envelope["latency"] = {
                    "total_s": float(time.perf_counter() - started),
                    "backend_s": 0.0,
                }
                envelope["cache"] = cache_info
                self._flatten_output(envelope)
                self._write_audit(task, normalized_input, envelope)
                return envelope

        raw_output: str | dict[str, Any] | None = None
        backend_started = time.perf_counter()
        failure_code = None
        fallback_reason = None
        try:
            raw_output = self._call_backend(task, prompt, image_paths or [])
        except FutureTimeoutError:
            failure_code = "BACKEND_TIMEOUT"
            fallback_reason = f"backend exceeded {self.config.timeout_s:.3f}s timeout"
        except Exception as exc:  # noqa: BLE001 - delegate must isolate backend failures
            failure_code = "BACKEND_EXCEPTION"
            fallback_reason = _safe_exception_message(exc)
        backend_s = float(time.perf_counter() - backend_started)

        if failure_code is None:
            validator = parse_and_validate_output
            if self.config.prompt_profile.startswith("completion_"):
                from conceptgraph.completion.semantic_contract import validate

                validator = validate
            validation = validator(
                task,
                raw_output,
                allowed_candidate_ids=allowed_candidate_ids,
            )
            if validation.valid:
                normalized_output = validation.normalized_output
                validation_errors: list[dict[str, Any]] = []
                used_qwen = True
                fallback_used = False
            else:
                normalized_output = fallback
                validation_errors = validation.validation_errors
                used_qwen = False
                fallback_used = True
                error_codes = {item["code"] for item in validation_errors}
                if "empty_output" in error_codes:
                    failure_code = "EMPTY_OUTPUT"
                    fallback_reason = "backend returned an empty output"
                elif "illegal_candidate_id" in error_codes:
                    failure_code = "ILLEGAL_CANDIDATE_ID"
                    fallback_reason = "backend returned a node outside the verified candidates"
                else:
                    failure_code = "SCHEMA_VALIDATION_FAILED"
                    fallback_reason = "backend output did not satisfy the strict task schema"
        else:
            normalized_output = fallback
            validation_errors = []
            used_qwen = False
            fallback_used = True

        envelope = {
            "task": task,
            "schema_version": SCHEMA_VERSION,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "validation_errors": validation_errors,
            "output_schema_valid": bool(used_qwen),
            "used_qwen": used_qwen,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "failure_code": failure_code,
            "latency": {
                "total_s": float(time.perf_counter() - started),
                "backend_s": backend_s,
            },
            "cache": cache_info,
            "model_id": self.model_identifier,
            "adapter_id": self.adapter_identifier,
        }
        if lookup_diagnostic and envelope["fallback_reason"] is None:
            envelope["cache"]["diagnostic"] = lookup_diagnostic
        self._flatten_output(envelope)
        if self.cache is not None and key is not None:
            cache_payload = {k: v for k, v in envelope.items() if k not in {"latency", "cache"}}
            for output_key in normalized_output:
                cache_payload.pop(output_key, None)
            self.cache.put(key, cache_payload)
            envelope["cache"]["status"] = f"{cache_info['status']}_written"
        self._write_audit(task, normalized_input, envelope)
        return envelope

    def _call_backend(self, task: str, prompt: str, image_paths: list[Path]) -> str:
        images = []
        if image_paths:
            from PIL import Image

            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
        backend = self._load_backend()
        with self._counter_lock:
            self._backend_calls += 1
        with self._executor_lock:
            executor = self._sync_executor
            future = executor.submit(
                _invoke_backend,
                backend,
                task,
                prompt,
                images,
                {
                    "do_sample": False,
                    "max_new_tokens": self.config.max_new_tokens,
                },
            )
        try:
            return future.result(timeout=self.config.timeout_s)
        except FutureTimeoutError:
            future.cancel()
            with self._executor_lock:
                if self._sync_executor is executor and not self._closed:
                    self._retired_executors.append(executor)
                    executor.shutdown(wait=False, cancel_futures=True)
                    self._sync_executor = self._new_sync_executor()
            raise

    def _load_backend(self):
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    import torch

                    from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

                    torch_dtype = (
                        None
                        if self.config.torch_dtype is None
                        else getattr(torch, self.config.torch_dtype)
                    )
                    self._backend = QwenVLChat(
                        self.config.model_name_or_path,
                        self.config.device,
                        adapter_path=self.config.adapter_path,
                        torch_dtype=torch_dtype,
                        max_image_side=self.config.max_image_side,
                    )
        return self._backend

    def _new_sync_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-semantic-backend")

    def _fallback_envelope(
        self,
        *,
        task: str,
        fallback: dict[str, Any],
        failure_code: str,
        fallback_reason: str,
        validation_errors: list[dict[str, Any]],
        cache_info: dict[str, Any],
        latency_s: float,
    ) -> dict[str, Any]:
        envelope = {
            "task": task,
            "schema_version": SCHEMA_VERSION,
            "raw_output": None,
            "normalized_output": fallback,
            "validation_errors": validation_errors,
            "output_schema_valid": False,
            "used_qwen": False,
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "failure_code": failure_code,
            "latency": {"total_s": float(latency_s), "backend_s": 0.0},
            "cache": cache_info,
            "model_id": self.model_identifier,
            "adapter_id": self.adapter_identifier,
        }
        self._flatten_output(envelope)
        self._write_audit(task, {}, envelope)
        return envelope

    def _compatibility_relation_envelope(
        self,
        normalized: dict[str, Any],
        *,
        fallback_reason: str,
        failure_code: str,
    ) -> dict[str, Any]:
        envelope = {
            **normalized,
            "task": "relation_compatibility",
            "schema_version": SCHEMA_VERSION,
            "raw_output": None,
            "normalized_output": normalized,
            "validation_errors": [],
            "output_schema_valid": True,
            "used_qwen": False,
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "failure_code": failure_code,
            "latency": {"total_s": 0.0, "backend_s": 0.0},
            "cache": self._empty_cache_info(),
            "model_id": self.model_identifier,
            "adapter_id": self.adapter_identifier,
        }
        self._write_audit("relation_compatibility", {}, envelope)
        return envelope

    @staticmethod
    def _flatten_output(envelope: dict[str, Any]) -> None:
        output = envelope.get("normalized_output")
        if isinstance(output, dict):
            for key, value in output.items():
                envelope.setdefault(key, value)

    def _empty_cache_info(self) -> dict[str, Any]:
        return {
            "enabled": self.cache is not None,
            "hit": False,
            "key": None,
            "status": "disabled" if self.cache is None else "not_checked",
            "diagnostic": None,
        }

    @property
    def _prompt_version(self) -> str:
        if self.config.prompt_profile.startswith("completion_"):
            return "completion-fullschema-" + self.config.prompt_profile.removeprefix("completion_")
        if self.config.prompt_profile == "finetune_v2":
            return FINETUNE_V2_PROMPT_VERSION
        return PROMPT_VERSION

    def _write_audit(self, task: str, normalized_input: dict[str, Any], envelope: dict[str, Any]) -> None:
        if not self.log_dir:
            return
        with self._counter_lock:
            self._counter += 1
            counter = self._counter
        payload = {
            "audit_version": "stage4-v1",
            "task": task,
            "input_sha256": semantic_version(normalized_input),
            "result": envelope,
        }
        path = self.log_dir / f"{counter:06d}_{task}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _rule_object_attributes(metadata: dict[str, Any]) -> dict[str, Any]:
        raw_tag = str(metadata.get("object_tag") or metadata.get("detection_label") or "unknown")
        try:
            tag = normalize_category(raw_tag)
        except (TypeError, ValueError):
            tag = "unknown"
        possible_tags = []
        for raw in metadata.get("possible_tags", [tag]) or [tag]:
            try:
                canonical = normalize_category(raw)
            except (TypeError, ValueError):
                continue
            if canonical not in possible_tags:
                possible_tags.append(canonical)
        if not possible_tags:
            possible_tags = [tag]
        colors = []
        for raw in metadata.get("colors", []) or []:
            try:
                canonical = normalize_color(raw)
            except (TypeError, ValueError):
                continue
            if canonical not in colors:
                colors.append(canonical)
        body_type = metadata.get("body_type")
        if body_type not in {"bus", "hatchback", "motorcycle", "sedan", "suv", "truck", "unknown", "van"}:
            body_type = None
        try:
            motion_state = normalize_motion_state(metadata.get("motion_state", "unknown"))
        except (TypeError, ValueError):
            motion_state = "unknown"
        summary = str(metadata.get("caption") or f"{tag} observed in object crop").strip()
        return {
            "schema_version": SCHEMA_VERSION,
            "summary": summary[:512] or "unknown object",
            "object_tag": tag,
            "possible_tags": possible_tags[:8],
            "colors": colors[:4],
            "body_type": body_type,
            "motion_state": motion_state,
            "confidence": 0.5,
        }

    @staticmethod
    def _rule_query_parse(
        query_text: str,
        *,
        default_near_distance_m: float,
        rule_result: Any | None,
    ) -> dict[str, Any]:
        if rule_result is None:
            rule_result = _basic_rule_query(query_text)
        getter = (
            (lambda key, default=None: rule_result.get(key, default))
            if isinstance(rule_result, dict)
            else (lambda key, default=None: getattr(rule_result, key, default))
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "target_text": str(getter("target_text", query_text) or query_text)[:256],
            "target_categories": list(getter("target_categories", []) or [])[:4],
            "target_colors": list(getter("target_colors", []) or [])[:4],
            "anchor_text": getter("anchor_text"),
            "anchor_categories": list(getter("anchor_categories", []) or [])[:4],
            "anchor_colors": list(getter("anchor_colors", []) or [])[:4],
            "relation": getter("relation"),
            "prefer_nearest": bool(getter("prefer_nearest", False)),
            "prefer_front": bool(getter("prefer_front", False)),
            "confidence": 0.5,
        }

    @staticmethod
    def _rule_relation(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        sx, sy = source.get("xy", source.get("bbox_center", [0.0, 0.0]))[:2]
        tx, ty = target.get("xy", target.get("bbox_center", [0.0, 0.0]))[:2]
        dx, dy = float(tx) - float(sx), float(ty) - float(sy)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > 8.0:
            relation = "near"
            confidence = 0.0
        elif abs(dx) > abs(dy) * 1.35:
            relation = "right_of" if dx > 0 else "left_of"
            confidence = 1.0 - distance / 8.0
        elif abs(dy) > abs(dx) * 1.35:
            relation = "in_front_of" if dy > 0 else "behind"
            confidence = 1.0 - distance / 8.0
        else:
            relation = "near"
            confidence = 1.0 - distance / 8.0
        return {
            "relation": relation,
            "confidence": _bounded_confidence(confidence),
            "explanation": "geometric_rule",
        }


def robust_json_parse(text: str) -> Any:
    """保留旧导入路径；新代码应使用 parse_and_validate_output。"""
    from conceptgraph.vlm.schemas import extract_json_object

    value, _ = extract_json_object(text)
    return value


def _invoke_backend(
    backend: Any,
    task: str,
    prompt: str,
    images: list[Any],
    generation_config: dict[str, Any],
) -> str:
    target = backend.generate if hasattr(backend, "generate") else backend
    if not callable(target):
        raise TypeError("backend must be callable or expose generate()")
    parameters = inspect.signature(target).parameters
    kwargs = {}
    if "task" in parameters:
        kwargs["task"] = task
    if "prompt" in parameters:
        kwargs["prompt"] = prompt
    elif "query" in parameters:
        kwargs["query"] = prompt
    if "images" in parameters:
        kwargs["images"] = images or None
    if "generation_config" in parameters:
        kwargs["generation_config"] = generation_config
    if kwargs and ("prompt" in kwargs or "query" in kwargs):
        return str(target(**kwargs))
    return str(target(prompt, images=images or None, generation_config=generation_config))


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for candidate in candidates:
        node_id = candidate.get("node_id")
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            continue
        if node_id not in result:
            result.append(node_id)
    return result


def _semantic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "attributes",
        "bbox_center",
        "bbox_extent",
        "body_type",
        "caption",
        "colors",
        "detection_label",
        "motion_state",
        "object_id",
        "object_tag",
        "observation_count",
        "possible_tags",
    }
    return {key: metadata[key] for key in sorted(metadata) if key in allowed}


def _semantic_edge(edge: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "confidence",
        "coordinate_frame",
        "directed",
        "distance_m",
        "first_seen_frame",
        "last_seen_frame",
        "observation_count",
        "relation",
        "relation_type",
        "source",
        "target",
        "version",
    }
    return {key: edge[key] for key in sorted(edge) if key in allowed}


def _artifact_identifier(
    value: str | Path | None,
    *,
    explicit_revision: str | None,
    default_name: str | None,
) -> str | None:
    if value is None:
        return default_name
    path = Path(value)
    name = path.name or str(value)
    if explicit_revision:
        return f"{name}@{explicit_revision}"
    if not path.exists():
        return name
    entries = []
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        stat = item.stat()
        relative = item.name if path.is_file() else str(item.relative_to(path))
        entries.append((relative, stat.st_size, stat.st_mtime_ns))
    return f"{name}@{semantic_version(entries)[:12]}"


def _bounded_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)[:240]
    absolute_path = re.compile(r"(?<![\w:])/(?:[^\s:]+/)*[^\s:]*")
    sanitized = absolute_path.sub(
        lambda match: f"<path:{Path(match.group(0)).name or 'root'}>",
        message,
    )
    return f"{type(exc).__name__}: {sanitized}"


def _basic_rule_query(query_text: str) -> dict[str, Any]:
    category_terms = [
        ("traffic light", ("交通信号灯", "红绿灯", "traffic light")),
        ("traffic sign", ("交通标志", "路牌", "traffic sign")),
        ("pedestrian", ("行人", "路人", "pedestrian", "person")),
        ("motorcycle", ("摩托车", "motorcycle")),
        ("bicycle", ("自行车", "bicycle")),
        ("truck", ("卡车", "货车", "truck")),
        ("bus", ("公交车", "巴士", "bus")),
        ("car", ("小汽车", "轿车", "car")),
        ("vehicle", ("车辆", "机动车", "vehicle")),
    ]
    color_terms = [
        ("black", ("黑色", "black")),
        ("white", ("白色", "white")),
        ("gray", ("灰色", "grey", "gray")),
        ("red", ("红色", "red")),
        ("blue", ("蓝色", "blue")),
        ("silver", ("银色", "silver")),
        ("yellow", ("黄色", "yellow")),
        ("green", ("绿色", "green")),
    ]
    normalized = query_text.lower()
    categories = [name for name, terms in category_terms if any(term in normalized for term in terms)]
    if "car" in categories and "vehicle" in categories:
        categories.remove("vehicle")
    colors = [name for name, terms in color_terms if any(term in normalized for term in terms)]
    relation = "near" if any(term in normalized for term in ("附近", "旁边", "near", "beside")) else None
    return {
        "target_text": query_text.strip(),
        "target_categories": categories[:4],
        "target_colors": colors[:4],
        "anchor_text": None,
        "anchor_categories": [],
        "anchor_colors": [],
        "relation": relation,
        "prefer_nearest": any(term in normalized for term in ("最近", "nearest", "closest")),
        "prefer_front": any(term in normalized for term in ("前方", "front", "ahead")),
    }


def _shutdown_executor(executor: ThreadPoolExecutor, wait: bool) -> None:
    executor.shutdown(wait=wait, cancel_futures=True)
