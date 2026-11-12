"""本轮统一语义契约：显式否定、无匹配与单帧未知属性。"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from conceptgraph.vlm.schemas import (
    CANONICAL_CATEGORIES, CANONICAL_COLORS, CANONICAL_RELATIONS,
    QueryParseSchema, SchemaValidationResult, extract_json_object,
    normalize_task_payload, parse_and_validate_output,
)

PROMPT_VERSION = "completion-fullschema-v1"


class CompletionQuery(QueryParseSchema):
    excluded_categories: list[str] = Field(default_factory=list, max_length=4)
    excluded_colors: list[str] = Field(default_factory=list, max_length=4)
    requires_static: bool = False

    @field_validator("excluded_categories", "excluded_colors")
    @classmethod
    def check_exclusions(cls, value, info):
        allowed = CANONICAL_CATEGORIES if info.field_name == "excluded_categories" else CANONICAL_COLORS
        if len(set(value)) != len(value) or any(v not in allowed for v in value):
            raise ValueError("invalid exclusion values")
        return value


class CompletionRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0.0"]
    selected_node_id: int | None = Field(default=None, ge=0)
    ranking: list[int] = Field(min_length=1, max_length=64)
    no_match: bool
    explanation: str = Field(min_length=1, max_length=512)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.ranking)) != len(self.ranking) or any(i < 0 for i in self.ranking):
            raise ValueError("invalid ranking")
        if self.no_match and self.selected_node_id is not None:
            raise ValueError("no_match requires null selected_node_id")
        if not self.no_match and self.selected_node_id != self.ranking[0]:
            raise ValueError("selected_node_id must equal first ranking entry")
        return self


def validate(task, raw, *, allowed_candidate_ids=None):
    if task == "object_attributes":
        return parse_and_validate_output(task, raw)
    parsed, errors = extract_json_object(raw)
    if parsed is None:
        return SchemaValidationResult(task, raw, None, None, errors)
    try:
        if task == "query_parse":
            output = CompletionQuery.model_validate(normalize_task_payload(task, parsed)).model_dump()
        elif task == "candidate_rerank":
            output = CompletionRanking.model_validate(parsed).model_dump()
            if allowed_candidate_ids is not None and set(output["ranking"]) != set(allowed_candidate_ids):
                raise ValueError("ranking must contain exactly the verified candidate set")
        else:
            raise ValueError("unsupported completion task")
    except (ValueError, TypeError, ValidationError) as exc:
        return SchemaValidationResult(task, raw, parsed, None,
            [{"code": "completion_schema_invalid", "loc": [], "message": str(exc)}])
    return SchemaValidationResult(task, raw, parsed, output, [])


def build_prompt(task, row):
    common = ("只输出一个JSON对象，不要Markdown、说明或额外字段。"
              "不得编造对象、几何、颜色、运动或可达性。schema_version固定为1.0.0。\n")
    if task == "query_parse":
        template = dict(schema_version="1.0.0", target_text="小汽车", target_categories=["car"],
            target_colors=[], anchor_text=None, anchor_categories=[], anchor_colors=[], relation=None,
            prefer_nearest=False, prefer_front=False, excluded_categories=[], excluded_colors=[],
            requires_static=False, confidence=0.9)
        return common + ("任务：解析中文导航指令。目标是被靠近的对象，锚点是用于限定目标的参照对象；"
            "例如‘公交车左侧的小汽车’的目标是小汽车、锚点是公交车、关系left_of。"
            "无锚点时anchor_text为null、anchor_categories和anchor_colors为空。"
            "仅‘附近/靠近目标’不是锚点关系。否定类别/颜色写入excluded字段，禁止当作正目标。"
            "‘停着/静止’要求requires_static=true；不要从当前图像推断指令中未给出的颜色。\n"
            f"字段模板：{json.dumps(template,ensure_ascii=False)}\n"
            f"类别：{json.dumps(sorted(CANONICAL_CATEGORIES),ensure_ascii=False)}\n"
            f"颜色：{json.dumps(sorted(CANONICAL_COLORS))}\n关系：null或{json.dumps(sorted(CANONICAL_RELATIONS))}\n"
            f"指令：{row['query_text']}")
    if task == "object_attributes":
        template = dict(schema_version="1.0.0", summary="object", object_tag="unknown",
            possible_tags=["unknown"], colors=["unknown"], body_type="unknown", motion_state="unknown", confidence=0.8)
        metadata = {k:v for k,v in row.get("metadata",{}).items() if k in {"detection_label","motion_state","observation_count"}}
        return common + ("任务：根据真实对象裁剪图判断类别和可见外观。看不清的颜色或车型保留unknown；"
            "单张裁剪无法可靠证明静止或运动，motion_state为unknown，多帧几何状态由独立模块维护。"
            "检测类别只是可出错的观测，不能复制为视觉真值。\n"
            f"字段模板：{json.dumps(template)}\n"
            f"类别：{json.dumps(sorted(CANONICAL_CATEGORIES))}\n颜色：{json.dumps(sorted(CANONICAL_COLORS))}\n"
            f"元数据：{json.dumps(metadata,ensure_ascii=False)}")
    if task == "candidate_rerank":
        ids = row.get("candidate_ids") or [c['node_id'] for c in row['candidate_nodes']]
        template = dict(schema_version="1.0.0", selected_node_id=ids[0], ranking=ids,
            no_match=False, explanation="semantic evidence", confidence=0.9)
        payload = {k:row.get(k) for k in ['query_text','candidate_nodes','relation_metadata']}
        return common + ("任务：在已经通过确定性安全检查的候选内按语义排序，安全检查的范围以候选证据为准。"
            "不得修改地图、位置、碰撞检查或路径。输入顺序不代表正确程度。"
            "ranking必须恰好包含全部候选ID一次；没有语义匹配时no_match=true且selected_node_id=null，"
            "否则no_match=false且selected_node_id等于ranking首项。只按当前和过去的观测。\n"
            f"字段模板：{json.dumps(template)}\n输入：{json.dumps(payload,ensure_ascii=False)}")
    raise ValueError(task)


def build_completion_prompt(task,row,profile='completion_v1'):
    if profile=='completion_v1' or task=='candidate_rerank':
        return build_prompt(task,row)
    if profile not in {'completion_v2','completion_v3'}:
        raise ValueError('unknown completion prompt')
    common='请直接完成任务，只返回答案JSON。示例是输出格式，必须根据本次输入填写，禁止复制示例的属性值。\n'
    if task=='object_attributes':
        example=dict(schema_version='1.0.0',summary='car',object_tag='car',possible_tags=['car'],
            colors=['unknown'],body_type='unknown',motion_state='unknown',confidence=.9)
        metadata={k:v for k,v in row.get('metadata',{}).items() if k in {'detection_label','observation_count'}}
        instruction=("观察提供的真实裁剪图，识别其中对象。主要类别为car、truck、bus、bicycle、motorcycle、pedestrian；"
            "没有证据时可为unknown。object_tag和possible_tags必须与summary识别的类别一致。"
            "detection_label是检测器的观测，可结合图像使用；图像明显不符时纠正它。"
            "颜色若无法确定则unknown，body_type若无法确定则unknown。单帧不能证明运动，motion_state固定unknown。\n")
        examples='输出格式例子：'+json.dumps(example,separators=(',',':'))+'\n'
        if profile=='completion_v3':
            for category in ['truck','bicycle']:
                value={**example,'summary':category,'object_tag':category,'possible_tags':[category]}
                examples+=f'如果图中对象为{category}，输出：'+json.dumps(value,separators=(',',':'))+'\n'
        return common+instruction+examples+'本次观测元数据：'+json.dumps(metadata,ensure_ascii=False)+'\n本次裁剪图的答案JSON：'
    if task=='query_parse':
        instruction=("解析中文指令中的目标和参照物。target_text为目标短语，target_categories为目标类别列表；"
            "target_colors为明确要求的颜色，未提颜色就为空。anchor_text和anchor_categories只填参照物，"
            "没有参照物时anchor_text=null、anchor_categories=[]。anchor_colors同理。"
            "relation表示目标相对于参照物的位置，只能为left_of、right_of、in_front_of、behind、near或null。"
            "仅‘靠近/附近’一个目标没有参照物，relation=null。"
            "prefer_nearest表示最近，prefer_front表示自车前方（锚点前方不等于自车前方）。"
            "否定的类别放excluded_categories，否定颜色放excluded_colors；没有否定就空列表。"
            "要求停着/静止时requires_static=true，否则false。"
            "类别用car、truck、bus、bicycle、motorcycle、pedestrian或vehicle；颜色用英文单词。"
            "所有字段都必须返回，不能输出数组形式的relation。\n")
        examples=[('去小汽车附近',dict(schema_version='1.0.0',target_text='小汽车',target_categories=['car'],
            target_colors=[],anchor_text=None,anchor_categories=[],anchor_colors=[],relation=None,
            prefer_nearest=False,prefer_front=False,excluded_categories=[],excluded_colors=[],requires_static=False,confidence=.9))]
        if profile=='completion_v3':
            examples += [
                ('去公交车左侧停着的红色小汽车旁边',dict(schema_version='1.0.0',target_text='红色小汽车',target_categories=['car'],
                    target_colors=['red'],anchor_text='公交车',anchor_categories=['bus'],anchor_colors=[],relation='left_of',
                    prefer_nearest=False,prefer_front=False,excluded_categories=[],excluded_colors=[],requires_static=True,confidence=.9)),
                ('不要卡车，去最近的自行车附近',dict(schema_version='1.0.0',target_text='自行车',target_categories=['bicycle'],
                    target_colors=[],anchor_text=None,anchor_categories=[],anchor_colors=[],relation=None,
                    prefer_nearest=True,prefer_front=False,excluded_categories=['truck'],excluded_colors=[],requires_static=False,confidence=.9))]
        text=''.join(f'格式示例指令：{query}\n答案：{json.dumps(response,ensure_ascii=False,separators=(",",":"))}\n' for query,response in examples)
        return common+instruction+text+'本次指令：'+row['query_text']+'\n本次指令的答案JSON：'
    raise ValueError(task)
