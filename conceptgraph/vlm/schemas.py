from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"
SUPPORTED_TASKS = {"query_parse", "object_attributes", "candidate_rerank"}

CANONICAL_RELATIONS = {"near", "in_front_of", "behind", "left_of", "right_of"}
CANONICAL_COLORS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "red",
    "silver",
    "white",
    "yellow",
    "unknown",
}
CANONICAL_CATEGORIES = {
    "barrier",
    "bicycle",
    "building",
    "bus",
    "car",
    "construction vehicle",
    "fence",
    "motorcycle",
    "object",
    "parking area",
    "pedestrian",
    "pole",
    "road",
    "sidewalk",
    "traffic cone",
    "traffic light",
    "traffic sign",
    "trailer",
    "tree",
    "truck",
    "unknown",
    "van",
    "vehicle",
}
CANONICAL_BODY_TYPES = {
    "bus",
    "hatchback",
    "motorcycle",
    "sedan",
    "suv",
    "truck",
    "unknown",
    "van",
}
CANONICAL_MOTION_STATES = {"dynamic", "static", "unknown"}

RELATION_ALIASES = {
    "near": "near",
    "nearby": "near",
    "next_to": "near",
    "next to": "near",
    "beside": "near",
    "附近": "near",
    "旁边": "near",
    "front": "in_front_of",
    "in_front": "in_front_of",
    "in front of": "in_front_of",
    "in_front_of": "in_front_of",
    "前方": "in_front_of",
    "behind": "behind",
    "back": "behind",
    "后方": "behind",
    "left": "left_of",
    "left of": "left_of",
    "left_of": "left_of",
    "左侧": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "right_of": "right_of",
    "右侧": "right_of",
}
COLOR_ALIASES = {
    "black": "black",
    "黑": "black",
    "黑色": "black",
    "blue": "blue",
    "蓝": "blue",
    "蓝色": "blue",
    "brown": "brown",
    "棕": "brown",
    "棕色": "brown",
    "gray": "gray",
    "grey": "gray",
    "灰": "gray",
    "灰色": "gray",
    "green": "green",
    "绿": "green",
    "绿色": "green",
    "orange": "orange",
    "橙": "orange",
    "橙色": "orange",
    "red": "red",
    "红": "red",
    "红色": "red",
    "silver": "silver",
    "银": "silver",
    "银色": "silver",
    "white": "white",
    "白": "white",
    "白色": "white",
    "yellow": "yellow",
    "黄": "yellow",
    "黄色": "yellow",
    "unknown": "unknown",
    "未知": "unknown",
}
CATEGORY_ALIASES = {
    "automobile": "car",
    "auto": "car",
    "bike": "bicycle",
    "lorry": "truck",
    "person": "pedestrian",
    "sign": "traffic sign",
    "小汽车": "car",
    "汽车": "car",
    "车辆": "vehicle",
    "卡车": "truck",
    "货车": "truck",
    "公交车": "bus",
    "巴士": "bus",
    "摩托车": "motorcycle",
    "自行车": "bicycle",
    "行人": "pedestrian",
    "交通标志": "traffic sign",
    "交通信号灯": "traffic light",
    "红绿灯": "traffic light",
    "道路": "road",
    "人行道": "sidewalk",
    "建筑": "building",
    "树": "tree",
    "围栏": "fence",
    "杆": "pole",
    "未知": "unknown",
}
BODY_TYPE_ALIASES = {
    "sport utility vehicle": "suv",
    "轿车": "sedan",
    "越野车": "suv",
    "面包车": "van",
    "未知": "unknown",
}
MOTION_ALIASES = {
    "moving": "dynamic",
    "potential_dynamic": "dynamic",
    "运动": "dynamic",
    "动态": "dynamic",
    "stationary": "static",
    "静止": "static",
    "静态": "static",
    "未知": "unknown",
}
BOOLEAN_ALIASES = {
    "false": False,
    "no": False,
    "否": False,
    "不是": False,
    "true": True,
    "yes": True,
    "是": True,
}


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryParseSchema(_StrictSchema):
    schema_version: Literal[SCHEMA_VERSION]
    target_text: str = Field(min_length=1, max_length=256)
    target_categories: list[str] = Field(default_factory=list, max_length=4)
    target_colors: list[str] = Field(default_factory=list, max_length=4)
    anchor_text: str | None = Field(default=None, min_length=1, max_length=256)
    anchor_categories: list[str] = Field(default_factory=list, max_length=4)
    anchor_colors: list[str] = Field(default_factory=list, max_length=4)
    relation: str | None = None
    prefer_nearest: bool
    prefer_front: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("target_categories", "anchor_categories")
    @classmethod
    def validate_categories(cls, value: list[str]) -> list[str]:
        return _validate_unique_values(value, CANONICAL_CATEGORIES, "category")

    @field_validator("target_colors", "anchor_colors")
    @classmethod
    def validate_colors(cls, value: list[str]) -> list[str]:
        return _validate_unique_values(value, CANONICAL_COLORS, "color")

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str | None) -> str | None:
        if value is not None and value not in CANONICAL_RELATIONS:
            raise ValueError(f"unsupported relation: {value}")
        return value


class ObjectAttributesSchema(_StrictSchema):
    schema_version: Literal[SCHEMA_VERSION]
    summary: str = Field(min_length=1, max_length=512)
    object_tag: str
    possible_tags: list[str] = Field(min_length=1, max_length=8)
    colors: list[str] = Field(default_factory=list, max_length=4)
    body_type: str | None = None
    motion_state: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("object_tag")
    @classmethod
    def validate_object_tag(cls, value: str) -> str:
        if value not in CANONICAL_CATEGORIES:
            raise ValueError(f"unsupported category: {value}")
        return value

    @field_validator("possible_tags")
    @classmethod
    def validate_possible_tags(cls, value: list[str]) -> list[str]:
        return _validate_unique_values(value, CANONICAL_CATEGORIES, "category")

    @field_validator("colors")
    @classmethod
    def validate_colors(cls, value: list[str]) -> list[str]:
        return _validate_unique_values(value, CANONICAL_COLORS, "color")

    @field_validator("body_type")
    @classmethod
    def validate_body_type(cls, value: str | None) -> str | None:
        if value is not None and value not in CANONICAL_BODY_TYPES:
            raise ValueError(f"unsupported body type: {value}")
        return value

    @field_validator("motion_state")
    @classmethod
    def validate_motion_state(cls, value: str) -> str:
        if value not in CANONICAL_MOTION_STATES:
            raise ValueError(f"unsupported motion state: {value}")
        return value


class CandidateRerankSchema(_StrictSchema):
    schema_version: Literal[SCHEMA_VERSION]
    selected_node_id: int = Field(ge=0)
    ranking: list[int] = Field(min_length=1, max_length=64)
    explanation: str | None = Field(default=None, max_length=512)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ranking(self) -> CandidateRerankSchema:
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("ranking contains duplicate candidate IDs")
        if self.selected_node_id != self.ranking[0]:
            raise ValueError("selected_node_id must equal the first ranking entry")
        return self


TASK_SCHEMAS: dict[str, type[_StrictSchema]] = {
    "query_parse": QueryParseSchema,
    "object_attributes": ObjectAttributesSchema,
    "candidate_rerank": CandidateRerankSchema,
}


@dataclass(frozen=True)
class SchemaValidationResult:
    task: str
    raw_output: Any
    parsed_output: dict[str, Any] | None
    normalized_output: dict[str, Any] | None
    validation_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.normalized_output is not None and not self.validation_errors


def parse_and_validate_output(
    task: str,
    raw_output: str | dict[str, Any] | None,
    *,
    allowed_candidate_ids: list[int] | None = None,
) -> SchemaValidationResult:
    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unsupported schema task: {task}")
    parsed, parse_errors = extract_json_object(raw_output)
    if parsed is None:
        return SchemaValidationResult(task, raw_output, None, None, parse_errors)
    try:
        normalized = normalize_task_payload(task, parsed)
    except (TypeError, ValueError) as exc:
        return SchemaValidationResult(
            task,
            raw_output,
            parsed,
            None,
            [_error("normalization_error", (), str(exc))],
        )
    try:
        validated = TASK_SCHEMAS[task].model_validate(normalized)
    except ValidationError as exc:
        return SchemaValidationResult(
            task,
            raw_output,
            parsed,
            None,
            [
                _error(str(item["type"]), tuple(item["loc"]), str(item["msg"]))
                for item in exc.errors(include_url=False, include_input=False)
            ],
        )
    output = validated.model_dump(mode="json")
    if task == "candidate_rerank" and allowed_candidate_ids is not None:
        allowed = {int(item) for item in allowed_candidate_ids}
        returned = set(output["ranking"])
        illegal = sorted(returned - allowed)
        if output["selected_node_id"] not in allowed or illegal:
            return SchemaValidationResult(
                task,
                raw_output,
                parsed,
                None,
                [
                    _error(
                        "illegal_candidate_id",
                        ("ranking",),
                        f"candidate IDs are outside the verified input set: {illegal}",
                    )
                ],
            )
        missing = [item for item in allowed_candidate_ids if item not in returned]
        output["ranking"].extend(missing)
    return SchemaValidationResult(task, raw_output, parsed, output, [])


def extract_json_object(
    raw_output: str | dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if isinstance(raw_output, dict):
        return dict(raw_output), []
    if raw_output is None or not str(raw_output).strip():
        return None, [_error("empty_output", (), "model output is empty")]
    text = str(raw_output).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value, []
    if value is not None:
        return None, [_error("wrong_top_level_type", (), "top-level JSON value must be an object")]

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates: list[dict[str, Any]] = []
    for fragment in fenced:
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
    unique = {_stable_json(candidate): candidate for candidate in candidates}
    if len(unique) == 1:
        return next(iter(unique.values())), []
    if len(unique) > 1:
        return None, [_error("ambiguous_json", (), "model output contains multiple JSON objects")]
    return None, [_error("invalid_json", (), "no valid JSON object found in model output")]


def normalize_task_payload(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if task == "query_parse":
        _promote_singular_to_list(normalized, "target_category", "target_categories")
        _promote_singular_to_list(normalized, "target_color", "target_colors")
        _promote_singular_to_list(normalized, "anchor_category", "anchor_categories")
        _promote_singular_to_list(normalized, "anchor_color", "anchor_colors")
        normalized["target_categories"] = _normalize_list(
            normalized.get("target_categories", []), normalize_category
        )
        normalized["anchor_categories"] = _normalize_list(
            normalized.get("anchor_categories", []), normalize_category
        )
        normalized["target_colors"] = _normalize_list(
            normalized.get("target_colors", []), normalize_color
        )
        normalized["anchor_colors"] = _normalize_list(
            normalized.get("anchor_colors", []), normalize_color
        )
        if "relation" in normalized and normalized["relation"] is not None:
            normalized["relation"] = normalize_relation(normalized["relation"])
        for key in ("prefer_nearest", "prefer_front"):
            if key in normalized:
                normalized[key] = normalize_boolean(normalized[key])
    elif task == "object_attributes":
        if "color" in normalized and "colors" not in normalized:
            color = normalized.pop("color")
            normalized["colors"] = [] if color is None else [color]
        normalized["object_tag"] = normalize_category(normalized.get("object_tag"))
        tags = normalized.get("possible_tags") or [normalized["object_tag"]]
        normalized["possible_tags"] = _normalize_list(tags, normalize_category)
        normalized["colors"] = _normalize_list(normalized.get("colors", []), normalize_color)
        if normalized.get("body_type") is not None:
            normalized["body_type"] = normalize_body_type(normalized["body_type"])
        motion = normalized.pop("static_dynamic_hint", normalized.get("motion_state", "unknown"))
        normalized["motion_state"] = normalize_motion_state(motion)
    elif task == "candidate_rerank":
        if "ranking" in normalized and isinstance(normalized["ranking"], list):
            converted = []
            for item in normalized["ranking"]:
                if isinstance(item, dict) and set(item) == {"node_id"}:
                    item = item["node_id"]
                converted.append(item)
            normalized["ranking"] = converted
    else:
        raise ValueError(f"Unsupported schema task: {task}")
    return normalized


def normalize_relation(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    key = _normalize_token(value)
    if key not in RELATION_ALIASES:
        raise ValueError(f"unsupported relation: {value}")
    return RELATION_ALIASES[key]


def normalize_color(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("color must be a string")
    key = _normalize_token(value)
    if key not in COLOR_ALIASES:
        raise ValueError(f"unsupported color: {value}")
    return COLOR_ALIASES[key]


def normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("category must be a string")
    key = _normalize_token(value)
    canonical = CATEGORY_ALIASES.get(key, key)
    if canonical not in CANONICAL_CATEGORIES:
        raise ValueError(f"unsupported category: {value}")
    return canonical


def normalize_body_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("body_type must be a string")
    key = _normalize_token(value)
    canonical = BODY_TYPE_ALIASES.get(key, key)
    if canonical not in CANONICAL_BODY_TYPES:
        raise ValueError(f"unsupported body type: {value}")
    return canonical


def normalize_motion_state(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("motion_state must be a string")
    key = _normalize_token(value)
    canonical = MOTION_ALIASES.get(key, key)
    if canonical not in CANONICAL_MOTION_STATES:
        raise ValueError(f"unsupported motion state: {value}")
    return canonical


def normalize_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = _normalize_token(value)
        if key in BOOLEAN_ALIASES:
            return BOOLEAN_ALIASES[key]
    raise TypeError("boolean value must be bool or an accepted boolean word")


def schema_contract(task: str) -> dict[str, Any]:
    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unsupported schema task: {task}")
    return TASK_SCHEMAS[task].model_json_schema()


def _promote_singular_to_list(payload: dict[str, Any], singular: str, plural: str) -> None:
    if singular not in payload:
        return
    if plural in payload:
        raise ValueError(f"both {singular} and {plural} were provided")
    value = payload.pop(singular)
    payload[plural] = [] if value is None else [value]


def _normalize_list(value: Any, normalizer) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("value must be a list")
    result: list[str] = []
    for item in value:
        normalized = normalizer(item)
        if normalized not in result:
            result.append(normalized)
    return result


def _validate_unique_values(value: list[str], allowed: set[str], label: str) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError(f"duplicate {label} values are not allowed")
    invalid = sorted(set(value) - allowed)
    if invalid:
        raise ValueError(f"unsupported {label} values: {invalid}")
    return value


def _normalize_token(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", "_").split())


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(code: str, location: tuple[Any, ...], message: str) -> dict[str, Any]:
    return {"code": code, "location": [str(item) for item in location], "message": message}
