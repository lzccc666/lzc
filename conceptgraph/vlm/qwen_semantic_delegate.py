from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenDelegateConfig:
    enabled: bool = False
    model_name_or_path: str | None = None
    device: str = "cuda:0"
    use_qwen_caption: bool = False
    use_qwen_relation: bool = False
    use_qwen_query_parse: bool = False
    use_qwen_rerank: bool = False
    log_dir: str | Path | None = None


class QwenSemanticDelegate:
    """Robust semantic delegate with rule fallback.

    The class is deliberately lazy: Qwen is loaded only when an enabled task is
    called. This keeps mapping/navigation usable on machines without enough GPU
    memory and records every prompt/response for thesis reproducibility.
    """

    def __init__(self, config: QwenDelegateConfig | None = None):
        self.config = config or QwenDelegateConfig()
        self.log_dir = Path(self.config.log_dir) if self.config.log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._chat = None
        self._counter = 0

    def describe_object(self, crop_paths: list[str] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        fallback = self._rule_object_description(metadata)
        if not (self.config.enabled and self.config.use_qwen_caption):
            return fallback
        prompt = (
            "请根据目标裁剪图和元数据输出严格 JSON："
            "{\"summary\":...,\"object_tag\":...,\"possible_tags\":[...],\"color\":...,"
            "\"body_type\":...,\"static_dynamic_hint\":...,\"confidence\":0到1}。"
            f"元数据：{json.dumps(metadata, ensure_ascii=False)}"
        )
        return self._ask_json("object_caption", prompt, fallback, image_paths=crop_paths or [])

    def infer_relation(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        fallback = self._rule_relation(source, target)
        if not (self.config.enabled and self.config.use_qwen_relation):
            return fallback
        prompt = (
            "请判断两个场景图对象的空间关系，输出严格 JSON："
            "{\"relation\":\"near/front/behind/left/right/none\",\"confidence\":0到1,\"explanation\":...}。"
            f"对象A：{json.dumps(source, ensure_ascii=False)}\n对象B：{json.dumps(target, ensure_ascii=False)}"
        )
        return self._ask_json("relation", prompt, fallback)

    def parse_query(self, query_text: str) -> dict[str, Any]:
        fallback = self._rule_query_parse(query_text)
        if not (self.config.enabled and self.config.use_qwen_query_parse):
            return fallback
        prompt = (
            "请解析中文语义导航指令，输出严格 JSON："
            "{\"target_category\":...,\"target_color\":...,\"anchor_category\":...,"
            "\"anchor_color\":...,\"relation\":...,\"prefer_nearest\":true/false,"
            "\"prefer_front\":true/false}。"
            f"指令：{query_text}"
        )
        return self._ask_json("query_parse", prompt, fallback)

    def rerank_candidates(self, query_text: str, candidates: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        fallback = {"selected_node_id": candidates[0].get("node_id") if candidates else None, "ranking": candidates, "used_qwen": False}
        if not (self.config.enabled and self.config.use_qwen_rerank and candidates):
            return fallback
        prompt = (
            "请根据中文导航指令、候选节点和关系边重新排序，输出严格 JSON："
            "{\"selected_node_id\":整数,\"ranking\":[节点id列表],\"explanation\":...}。"
            f"指令：{query_text}\n候选：{json.dumps(candidates, ensure_ascii=False)}\n关系边：{json.dumps(edges or [], ensure_ascii=False)}"
        )
        return self._ask_json("rerank", prompt, fallback)

    def _ask_json(self, task: str, prompt: str, fallback: dict[str, Any], image_paths: list[str] | None = None) -> dict[str, Any]:
        raw = ""
        error = None
        try:
            chat = self._load_chat()
            images = []
            if image_paths:
                from PIL import Image

                for path in image_paths[:4]:
                    p = Path(path)
                    if p.exists():
                        images.append(Image.open(p).convert("RGB"))
            raw = chat(prompt, images=images or None)
            parsed = robust_json_parse(raw)
            if isinstance(parsed, dict):
                parsed["used_qwen"] = True
                self._log(task, prompt, raw, parsed, error=None)
                return parsed
        except Exception as exc:  # keep main experiment alive
            error = repr(exc)
        result = dict(fallback)
        result["used_qwen"] = False
        if error:
            result["fallback_error"] = error
        self._log(task, prompt, raw, result, error=error)
        return result

    def _load_chat(self):
        if self._chat is None:
            from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

            self._chat = QwenVLChat(self.config.model_name_or_path, self.config.device)
        return self._chat

    def _log(self, task: str, prompt: str, raw_response: str, parsed: dict[str, Any], error: str | None) -> None:
        if not self.log_dir:
            return
        self._counter += 1
        payload = {
            "task": task,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "error": error,
        }
        (self.log_dir / f"{self._counter:06d}_{task}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _rule_object_description(metadata: dict[str, Any]) -> dict[str, Any]:
        tag = str(metadata.get("object_tag") or metadata.get("detection_label") or "object").lower()
        colors = metadata.get("colors") or []
        color = colors[0] if isinstance(colors, list) and colors else None
        return {
            "summary": f"{tag} observed in nuScenes object crops",
            "object_tag": tag,
            "possible_tags": metadata.get("possible_tags", [tag]),
            "color": color,
            "body_type": metadata.get("body_type"),
            "static_dynamic_hint": metadata.get("motion_state", "unknown"),
            "confidence": 0.5,
            "used_qwen": False,
        }

    @staticmethod
    def _rule_relation(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        sx, sy = source.get("xy", source.get("bbox_center", [0.0, 0.0]))[:2]
        tx, ty = target.get("xy", target.get("bbox_center", [0.0, 0.0]))[:2]
        dx, dy = float(tx) - float(sx), float(ty) - float(sy)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 8.0:
            rel = "none"
        elif abs(dx) > abs(dy) * 1.35:
            rel = "right" if dx > 0 else "left"
        elif abs(dy) > abs(dx) * 1.35:
            rel = "front" if dy > 0 else "behind"
        else:
            rel = "near"
        return {"relation": rel, "confidence": max(0.0, 1.0 - dist / 8.0), "explanation": "rule_fallback", "used_qwen": False}

    @staticmethod
    def _rule_query_parse(query_text: str) -> dict[str, Any]:
        colors = [c for c in ["黑色", "白色", "灰色", "红色", "蓝色", "银色", "黄色", "绿色"] if c in query_text]
        relation = "near" if ("旁边" in query_text or "附近" in query_text) else None
        return {
            "target_category": None,
            "target_color": colors[0] if colors else None,
            "anchor_category": None,
            "anchor_color": colors[1] if len(colors) > 1 else None,
            "relation": relation,
            "prefer_nearest": "最近" in query_text,
            "prefer_front": "前方" in query_text,
            "used_qwen": False,
        }


def robust_json_parse(text: str) -> Any:
    text = str(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"(\{.*\})", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None
