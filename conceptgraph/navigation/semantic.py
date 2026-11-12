from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

import numpy as np

from conceptgraph.navigation.types import CandidateTarget, ParsedNavigationQuery, SceneGraphEdge, SceneGraphNode
from conceptgraph.scenegraph.relations import edge_relation_for_query, resolve_coordinate_frame


RELATION_SYNONYMS = {
    "near": {"near", "nearby", "near to", "close to", "next to", "beside", "附近", "靠近", "旁边", "边上"},
    "in_front_of": {"in front of", "ahead of", "前方"},
    "behind": {"behind", "后方"},
    "left_of": {"left of", "左侧", "左边"},
    "right_of": {"right of", "右侧", "右边"},
}

NEAREST_WORDS = {"nearest", "closest", "最近"}
FAR_WORDS = {"far", "farther", "farthest", "远处", "较远", "更远"}

STOPWORDS = {
    "go", "to", "the", "a", "an", "please", "take", "me", "towards", "area", "region",
    "去", "到", "一下", "附近", "靠近", "最近", "的", "区域",
}

CLASS_DISPLAY_ZH = {
    "vehicle": "车辆",
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
    "pedestrian": "行人",
    "person": "行人",
    "traffic sign": "交通标志",
    "traffic light": "交通信号灯",
    "road": "道路",
    "sidewalk": "人行道",
    "building": "建筑物",
    "tree": "树木",
    "fence": "围栏",
    "pole": "杆状物",
}

QUERY_CATEGORY_ALIASES = {
    "vehicle": {"车辆", "机动车", "交通工具", "vehicle", "vehicles"},
    "car": {
        "小汽车", "轿车", "私家车", "汽车", "小车",
        "黑车", "黑色车", "白车", "白色车", "红车", "红色车",
        "灰车", "灰色车", "蓝车", "蓝色车", "银车", "银色车",
        "黄车", "黄色车", "绿车", "绿色车", "橙车", "橙色车",
        "car", "cars", "auto", "automobile",
    },
    "truck": {"卡车", "货车", "truck", "trucks", "lorry"},
    "bus": {"公交车", "巴士", "公共汽车", "bus", "buses"},
    "motorcycle": {"摩托车", "motorcycle", "motorbike"},
    "bicycle": {"自行车", "单车", "bicycle", "bike"},
    "pedestrian": {"行人", "人", "路人", "pedestrian", "person", "people"},
    "traffic sign": {"交通标志", "路牌", "标志牌", "交通牌", "sign", "traffic sign"},
    "traffic light": {"交通信号灯", "红绿灯", "信号灯", "traffic light"},
    "road": {"道路", "路面", "road"},
    "sidewalk": {"人行道", "sidewalk"},
    "building": {"建筑", "建筑物", "building"},
    "tree": {"树", "树木", "tree"},
    "fence": {"围栏", "护栏", "fence"},
    "pole": {"杆", "杆状物", "pole"},
}

CLASS_COMPATIBILITY = {
    "vehicle": {"vehicle", "car", "truck", "bus", "motorcycle", "van"},
    "car": {"car"},
    "truck": {"truck"},
    "bus": {"bus"},
    "motorcycle": {"motorcycle"},
    "bicycle": {"bicycle"},
    "pedestrian": {"pedestrian", "person"},
    "traffic sign": {"traffic sign", "sign", "stop sign", "parking sign"},
    "traffic light": {"traffic light"},
    "road": {"road"},
    "sidewalk": {"sidewalk"},
    "building": {"building"},
    "tree": {"tree"},
    "fence": {"fence"},
    "pole": {"pole"},
}

STATIC_QUERY_WORDS = {"静态目标", "固定目标", "路侧目标", "稳定目标"}
STATIC_OBJECT_CATEGORIES = {"traffic sign", "traffic light", "road", "sidewalk", "building", "tree", "fence", "pole"}

TAG_ALIASES = {
    "parking area": {"parking area", "parking spot", "parking lot", "park area", "停车区域", "停车场", "停车位"},
    "vehicle": {"vehicle", "car", "auto", "车辆", "汽车", "小车"},
    "car": {"car", "vehicle", "auto", "车辆", "汽车", "小车"},
    "truck": {"truck", "vehicle", "lorry", "卡车", "货车", "车辆"},
    "traffic cone": {"traffic cone", "cone", "路锥", "锥桶", "路障锥"},
}

COLOR_ALIASES = {
    "white": {"white", "白色", "白"},
    "black": {"black", "黑色", "黑"},
    "blue": {"blue", "蓝色", "蓝"},
    "red": {"red", "红色", "红"},
    "gray": {"gray", "grey", "灰色", "灰"},
    "silver": {"silver", "银色", "银"},
    "green": {"green", "绿色", "绿"},
    "yellow": {"yellow", "黄色", "黄"},
    "orange": {"orange", "橙色", "橙"},
}

BRAND_ALIASES = {
    "honda": {"honda", "本田"},
    "toyota": {"toyota", "丰田"},
    "nissan": {"nissan", "日产"},
    "bmw": {"bmw", "宝马"},
    "ford": {"ford", "福特"},
    "chevrolet": {"chevrolet", "雪佛兰"},
    "mazda": {"mazda", "马自达"},
    "mitsubishi": {"mitsubishi", "三菱"},
    "subaru": {"subaru", "斯巴鲁"},
    "volvo": {"volvo", "沃尔沃"},
}

BODY_TYPE_ALIASES = {
    "suv": {"suv", "越野车", "运动型多用途车"},
    "sedan": {"sedan", "轿车", "三厢车"},
    "hatchback": {"hatchback", "掀背车"},
    "van": {"van", "面包车"},
    "bus": {"bus", "公交车"},
    "truck": {"truck", "卡车", "货车"},
    "motorcycle": {"motorcycle", "摩托车"},
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


@lru_cache(maxsize=4096)
def _tokenize_text_cached(text: str) -> tuple[str, ...]:
    ascii_tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    tokens = [token for token in ascii_tokens + chinese_chunks if token and token not in STOPWORDS]
    raw_text = str(text)
    normalized_raw = normalize_text(raw_text)
    for alias_map in (TAG_ALIASES, COLOR_ALIASES, BRAND_ALIASES, BODY_TYPE_ALIASES):
        for key, aliases in alias_map.items():
            if key in normalized_raw and key not in tokens:
                tokens.append(key)
            for alias in aliases:
                if _contains_chinese(alias) and alias in raw_text:
                    alias_chunks = re.findall(r"[\u4e00-\u9fff]+", alias)
                    tokens.extend(token for token in alias_chunks if token not in STOPWORDS and token not in tokens)
                elif alias in normalized_raw and alias not in tokens:
                    tokens.append(alias)
    for aliases in TAG_ALIASES.values():
        for alias in aliases:
            if _contains_chinese(alias) and alias in raw_text:
                alias_chunks = re.findall(r"[\u4e00-\u9fff]+", alias)
                tokens.extend(token for token in alias_chunks if token not in STOPWORDS and token not in tokens)
    if re.search(r"(黑|黑色|白|白色|红|红色|灰|灰色|蓝|蓝色|银|银色|黄|黄色|绿|绿色|橙|橙色)车", raw_text):
        tokens.append("car")
    return tuple(tokens)


def tokenize_text(text: str) -> list[str]:
    return list(_tokenize_text_cached(str(text)))


@lru_cache(maxsize=4096)
def _contains_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def parse_navigation_query(text: str, default_near_distance: float) -> ParsedNavigationQuery:
    raw_text = text
    negative = re.findall(r"(?:不要|排除|而不是|不是)\s*([^，,。；;]+)", text)
    text = re.sub(r"(?:不要|排除|而不是|不是)\s*[^，,。；;]+", "", text).strip(" ，,。；;")
    normalized = normalize_text(text)
    prefer_nearest = any(word in text or word in normalized for word in NEAREST_WORDS)
    prefer_front = "前方" in text or "front" in normalized or "ahead" in normalized

    relation = None
    matched_phrase = None
    for relation_name in ["left_of", "right_of", "in_front_of", "behind", "near"]:
        phrases = RELATION_SYNONYMS[relation_name]
        for phrase in sorted(phrases, key=lambda value: (-len(value), value)):
            if phrase in text or phrase in normalized:
                relation = relation_name
                matched_phrase = phrase
                break
        if relation is not None:
            break

    anchor_text = None
    target_text = text
    structured_relation = False
    beside_first = re.search(
        r"^(?:前往|去|到)?\s*(?:停在|停靠在|停放在)?\s*(?P<anchor>.+?)(?:旁边|边上|附近)\s*的\s*(?P<target>.+?)(?:附近|那边|位置)?$",
        text,
    )
    anchor_first = re.search(
        r"^(?:前往|去|到)?\s*(?:靠近|位于|在|停在|停靠在|停放在)\s*(?P<anchor>.+?)\s*的\s*(?P<target>.+?)(?:附近|那边|位置)?$",
        text,
    )
    if (anchor_first is not None or beside_first is not None) and relation in {None, "near"}:
        matched = beside_first if beside_first is not None else anchor_first
        relation = "near"
        anchor_text = matched.group("anchor")
        target_text = matched.group("target")
        structured_relation = True
    if not structured_relation and relation is not None and matched_phrase is not None:
        raw_parts = re.split(re.escape(matched_phrase), text, maxsplit=1)
        if len(raw_parts) == 2:
            lhs = raw_parts[0].strip(" ，,。")
            rhs = raw_parts[1].strip(" ，,。")
            if lhs and rhs:
                if _contains_chinese(matched_phrase):
                    # 汉语“锚点左侧的目标”；英语“target left of anchor”。
                    if infer_query_categories(lhs) and infer_query_categories(rhs):
                        anchor_text = lhs
                        target_text = re.sub(r"^的", "", rhs)
                    else:
                        target_text = rhs if infer_query_categories(rhs) else lhs
                        anchor_text = None
                else:
                    target_text = lhs
                    anchor_text = rhs

        if _contains_chinese(matched_phrase) and relation != "near":
            target_first = re.search(r"(?P<target>.+?)在(?P<anchor>.+?)" + re.escape(matched_phrase), text)
            if target_first and infer_query_categories(target_first.group('target')) and infer_query_categories(target_first.group('anchor')):
                target_text = target_first.group('target')
                anchor_text = target_first.group('anchor')

        if anchor_text is None and not _contains_chinese(matched_phrase):
            normalized_parts = re.split(re.escape(matched_phrase), normalized, maxsplit=1)
            if len(normalized_parts) == 2:
                lhs = normalized_parts[0].strip(" ,.")
                rhs = normalized_parts[1].strip(" ,.")
                if lhs and rhs:
                    target_text = lhs
                    anchor_text = rhs

    target_text = _clean_query_fragment(target_text)
    anchor_text = _clean_query_fragment(anchor_text) if anchor_text is not None else None
    if anchor_text == "":
        anchor_text = None
    if anchor_text is None:
        relation = None
    elif relation == "in_front_of":
        prefer_front = False

    target_categories = infer_query_categories(target_text)
    target_colors = infer_query_colors(target_text)
    anchor_colors = infer_query_colors(anchor_text or "")
    query_mode = infer_query_mode(text, target_categories)
    display_text = to_chinese_query_text(text, query_mode, target_categories, relation, anchor_text, target_text)
    approach_target = any(word in text or word in normalized for word in RELATION_SYNONYMS['near'])

    return ParsedNavigationQuery(
        raw_text=raw_text,
        target_text=target_text or text,
        anchor_text=anchor_text,
        relation=relation,
        prefer_nearest=prefer_nearest,
        prefer_front=prefer_front,
        target_categories=target_categories,
        target_colors=target_colors,
        anchor_colors=anchor_colors,
        excluded_categories=sorted({category for fragment in negative for category in infer_query_categories(fragment)}),
        excluded_colors=sorted({color for fragment in negative for color in infer_query_colors(fragment)}),
        requires_static=any(word in target_text for word in ["停着", "静止", "停放", "停靠", "parked", "stationary"]),
        display_text=display_text,
        query_mode=query_mode,
        near_distance=default_near_distance if relation == "near" else None,
        min_target_distance_m=0.0 if (approach_target or prefer_nearest or prefer_front or relation is not None) else 6.0,
        max_target_distance_m=45.0 if relation is not None else (35.0 if (prefer_nearest or prefer_front) else None),
    )


def apply_qwen_query_parse(
    rule_query: ParsedNavigationQuery,
    semantic_output: dict | None,
    default_near_distance: float,
) -> ParsedNavigationQuery:
    """把已验证的Qwen字段合并到规则结果，不改变导航几何约束。"""
    if not isinstance(semantic_output, dict):
        return rule_query
    target_categories = list(semantic_output.get("target_categories", rule_query.target_categories))
    prefer_nearest = bool(semantic_output.get("prefer_nearest", rule_query.prefer_nearest))
    prefer_front = bool(semantic_output.get("prefer_front", rule_query.prefer_front))
    relation = semantic_output.get("relation", rule_query.relation)
    if prefer_front:
        query_mode = "front"
    elif prefer_nearest:
        query_mode = "nearest"
    elif target_categories:
        query_mode = "class"
    else:
        query_mode = rule_query.query_mode
    merged = replace(
        rule_query,
        target_text=str(semantic_output.get("target_text") or rule_query.target_text),
        anchor_text=semantic_output.get("anchor_text", rule_query.anchor_text),
        relation=relation,
        prefer_nearest=prefer_nearest,
        prefer_front=prefer_front,
        target_categories=target_categories,
        target_colors=list(semantic_output.get("target_colors", rule_query.target_colors)),
        anchor_colors=list(semantic_output.get("anchor_colors", rule_query.anchor_colors)),
        excluded_categories=list(semantic_output.get("excluded_categories", rule_query.excluded_categories)),
        excluded_colors=list(semantic_output.get("excluded_colors", rule_query.excluded_colors)),
        requires_static=bool(semantic_output.get("requires_static", rule_query.requires_static)),
        query_mode=query_mode,
        near_distance=default_near_distance if relation == "near" else None,
        min_target_distance_m=0.0 if (prefer_nearest or prefer_front or relation is not None) else 6.0,
        max_target_distance_m=(
            45.0 if relation is not None else (35.0 if (prefer_nearest or prefer_front) else None)
        ),
    )
    return merged


def infer_query_colors(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches = []
    for color, aliases in COLOR_ALIASES.items():
        if any(alias in text or alias in normalized for alias in aliases):
            matches.append(color)
    return matches


def infer_query_categories(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches: list[str] = []
    # More specific classes must win over generic vehicle aliases.
    priority = [
        "traffic light",
        "traffic sign",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "truck",
        "bus",
        "car",
        "vehicle",
        "road",
        "sidewalk",
        "building",
        "tree",
        "fence",
        "pole",
    ]
    for category in priority:
        aliases = QUERY_CATEGORY_ALIASES.get(category, set())
        if any(alias in text or alias in normalized for alias in aliases):
            matches.append(category)
    if "car" in matches and "vehicle" in matches and ("小汽车" in text or "轿车" in text or "car" in normalized):
        matches = [item for item in matches if item != "vehicle"]
    if not matches and any(word in text for word in STATIC_QUERY_WORDS):
        return []
    return matches


def infer_query_mode(text: str, target_categories: list[str]) -> str:
    normalized = normalize_text(text)
    if any(word in text for word in STATIC_QUERY_WORDS):
        return "nearest_static"
    if "前方" in text or "front" in normalized or "ahead" in normalized:
        return "front"
    if "最近" in text or "nearest" in normalized or "closest" in normalized:
        return "nearest"
    if "右侧" in text or "右边" in text or "right" in normalized:
        return "right_side"
    if "停车" in text or "parking" in normalized:
        return "parking_area"
    if target_categories:
        return "class"
    return "generic"


def to_chinese_query_text(
    raw_text: str,
    query_mode: str,
    target_categories: list[str],
    relation: str | None = None,
    anchor_text: str | None = None,
    target_text: str | None = None,
) -> str:
    if re.search(r"[\u4e00-\u9fff]", str(raw_text)):
        return str(raw_text)
    if relation == "near" and anchor_text:
        target_label = _categories_to_zh(target_categories) if target_categories else str(target_text or "目标")
        return f"前往靠近{anchor_text}的{target_label}附近"
    if query_mode == "nearest" and target_categories:
        return f"前往最近的{_categories_to_zh(target_categories)}附近"
    if query_mode == "front" and target_categories:
        return f"前往前方的{_categories_to_zh(target_categories)}附近"
    if query_mode == "nearest_static":
        return "前往最近的静态目标附近"
    if query_mode == "right_side":
        return "前往道路右侧目标附近"
    if query_mode == "parking_area":
        return "前往停车区域附近"
    if target_categories:
        return f"前往{_categories_to_zh(target_categories)}附近"
    return str(raw_text)


def _categories_to_zh(categories: list[str]) -> str:
    if categories == ["vehicle"]:
        return "车辆"
    labels = [CLASS_DISPLAY_ZH.get(category, category) for category in categories]
    return "或".join(labels)


def _clean_query_fragment(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"^(go|take|bring|move)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(前往|去|到)\s*", "", cleaned)
    cleaned = re.sub(r"^(停在|停靠在|停放在|位于|靠近|在)\s*", "", cleaned)
    cleaned = re.sub(r"(附近|那边|位置)$", "", cleaned)
    cleaned = cleaned.strip(" ，,。. ")
    return cleaned


def build_relation_index(edges: list[SceneGraphEdge]) -> dict[int, list[dict]]:
    relation_index: dict[int, list[dict]] = defaultdict(list)
    for edge in edges:
        for source, target in ((edge.source, edge.target), (edge.target, edge.source)):
            relation = edge_relation_for_query(edge, source, target)
            if relation is None:
                continue
            relation_index[source].append(
                {
                    "neighbor_id": target,
                    "relation": normalize_text(relation),
                    "confidence": float(edge.confidence),
                    "distance_m": edge.distance_m,
                    "coordinate_frame": edge.coordinate_frame,
                    "edge_format": str(edge.metadata.get("edge_format", "structured")),
                    "metadata_available": bool(edge.metadata.get("metadata_available", True)),
                    "stored_source": edge.source,
                    "stored_target": edge.target,
                    "stored_relation": edge.relation,
                    "observation_count": edge.observation_count,
                    "first_seen_frame": edge.first_seen_frame,
                    "last_seen_frame": edge.last_seen_frame,
                }
            )
    for entries in relation_index.values():
        entries.sort(
            key=lambda item: (
                int(item["neighbor_id"]),
                str(item["relation"]),
                -round(float(item["confidence"]), 12),
            )
        )
    return relation_index


def score_text_similarity(query_text: str, node: SceneGraphNode) -> float:
    query_tokens = set(tokenize_text(query_text))
    if not query_tokens:
        return 0.0

    metadata = node.metadata or {}
    attributes = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
    node_texts = [
        node.object_tag,
        node.caption,
        " ".join(node.possible_tags),
        " ".join(node.metadata.get("annotation", {}).get("object_tags", "").split(",")),
        " ".join(metadata.get("raw_captions", [])) if isinstance(metadata, dict) else "",
        " ".join(attributes.get("colors", [])) if isinstance(attributes, dict) else "",
        " ".join(attributes.get("brands", [])) if isinstance(attributes, dict) else "",
        " ".join(attributes.get("body_types", [])) if isinstance(attributes, dict) else "",
    ]
    expanded_texts = list(node_texts)
    expanded_texts.extend(sorted(TAG_ALIASES.get(normalize_text(node.object_tag), set())))
    for alias_map in (COLOR_ALIASES, BRAND_ALIASES, BODY_TYPE_ALIASES):
        for key, aliases in alias_map.items():
            if key in normalize_text(" ".join(node_texts)):
                expanded_texts.extend(sorted(aliases))
    node_tokens = set()
    for text in expanded_texts:
        node_tokens.update(tokenize_text(text))

    overlap = len(query_tokens & node_tokens)
    contains_phrase = 1.0 if normalize_text(query_text) and normalize_text(query_text) in normalize_text(" ".join(node_texts)) else 0.0
    jaccard = overlap / max(len(query_tokens | node_tokens), 1)
    return min(1.0, 0.6 * jaccard + 0.4 * contains_phrase)


def node_category_terms(node: SceneGraphNode) -> set[str]:
    tag_terms = canonical_terms_from_text(node.object_tag)
    if tag_terms:
        return tag_terms

    metadata = node.metadata or {}
    annotation = metadata.get("annotation", {}) if isinstance(metadata, dict) else {}
    attributes = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
    texts = [
        node.caption,
        str(annotation.get("object_tags", "")) if isinstance(annotation, dict) else "",
        " ".join(metadata.get("raw_captions", [])) if isinstance(metadata, dict) else "",
        " ".join(attributes.get("body_types", [])) if isinstance(attributes, dict) else "",
    ]
    normalized_text = normalize_text(" ".join(str(text) for text in texts if text))
    terms = set()
    for category, compatible in CLASS_COMPATIBILITY.items():
        aliases = QUERY_CATEGORY_ALIASES.get(category, set()) | compatible
        if any(alias in normalized_text or alias in str(node.object_tag).lower() for alias in aliases):
            terms.update(compatible)
            terms.add(category)
    tag = normalize_text(node.object_tag)
    if tag:
        terms.add(tag)
    return terms


def canonical_terms_from_text(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized or normalized in {"unknown", "invalid", "fail"}:
        return set()
    for category in [
        "traffic light",
        "traffic sign",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "truck",
        "bus",
        "car",
        "vehicle",
        "road",
        "sidewalk",
        "building",
        "tree",
        "fence",
        "pole",
    ]:
        aliases = QUERY_CATEGORY_ALIASES.get(category, set()) | {category}
        if normalized in {normalize_text(alias) for alias in aliases}:
            if category == "vehicle":
                return {"vehicle"}
            if category == "traffic sign":
                return {"traffic sign", "sign"}
            return {category}
    return set()


def is_node_compatible_with_query(parsed_query: ParsedNavigationQuery, node: SceneGraphNode) -> tuple[bool, str]:
    if parsed_query.query_mode == "nearest_static":
        if node.motion_state == "dynamic" or node.is_short_term:
            return False, "dynamic_node_for_static_query"
        terms = node_category_terms(node)
        if terms & STATIC_OBJECT_CATEGORIES:
            return True, "matched_static_class"
        return False, "non_static_class_for_static_query"
    if not parsed_query.target_categories:
        return True, "no_category_constraint"

    terms = node_category_terms(node)
    for category in parsed_query.target_categories:
        allowed = CLASS_COMPATIBILITY.get(category, {category})
        if terms & allowed:
            if parsed_query.target_colors and not node_has_color(node, parsed_query.target_colors):
                return False, "color_mismatch"
            return True, f"matched_{category}"
    return False, "category_mismatch"


def node_has_color(node: SceneGraphNode, colors: list[str]) -> bool:
    if not colors:
        return True
    metadata = node.metadata or {}
    attrs = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
    node_colors = {str(c).lower() for c in attrs.get("colors", [])} if isinstance(attrs, dict) else set()
    return bool(node_colors & {str(c).lower() for c in colors})


def score_relation_constraint(
    candidate_node: SceneGraphNode,
    parsed_query: ParsedNavigationQuery,
    all_nodes: dict[int, SceneGraphNode],
    relation_index: dict[int, list[dict]],
    anchor_text_scores: dict[int, float] | None = None,
    minimum_edge_confidence: float = 0.0,
    fallback_coordinate_frame: str = "world",
    fallback_coordinate_frame_fallback: str | None = "world",
    fallback_ego_heading: np.ndarray | None = None,
    fallback_trajectory: np.ndarray | None = None,
    fallback_ego_position: np.ndarray | None = None,
) -> tuple[float, dict]:
    if parsed_query.relation is None or parsed_query.anchor_text is None:
        return 1.0, {"reason": "no_relation_constraint"}

    anchor_scores = []
    debug_matches = []
    candidate_has_structured_evidence = False
    for anchor_id, anchor_node in all_nodes.items():
        anchor_text_score = (
            float(anchor_text_scores.get(anchor_id, 0.0))
            if anchor_text_scores is not None
            else score_text_similarity(parsed_query.anchor_text, anchor_node)
        )
        if anchor_text_score <= 0.0:
            continue
        if parsed_query.anchor_colors and not node_has_color(anchor_node, parsed_query.anchor_colors):
            continue
        for entry in relation_index.get(candidate_node.node_id, []):
            if int(entry["neighbor_id"]) != anchor_id:
                continue
            if str(entry.get("edge_format", "structured")) == "legacy":
                continue
            if not bool(entry.get("metadata_available", True)):
                continue
            if float(entry.get("confidence", 0.0)) + 1e-12 < minimum_edge_confidence:
                continue
            if _relation_match_score(parsed_query.relation, str(entry["relation"])) > 0.0:
                candidate_has_structured_evidence = True
                break
        if candidate_has_structured_evidence:
            break

    for anchor_id, anchor_node in all_nodes.items():
        anchor_text_score = (
            float(anchor_text_scores.get(anchor_id, 0.0))
            if anchor_text_scores is not None
            else score_text_similarity(parsed_query.anchor_text, anchor_node)
        )
        if anchor_text_score <= 0.0:
            continue
        if parsed_query.anchor_colors and not node_has_color(anchor_node, parsed_query.anchor_colors):
            continue

        graph_relation_score = 0.0
        relation_chain = []
        fallback_reasons = []
        for entry in relation_index.get(candidate_node.node_id, []):
            if int(entry["neighbor_id"]) != anchor_id:
                continue
            edge_format = str(entry.get("edge_format", "structured"))
            metadata_available = bool(entry.get("metadata_available", True))
            confidence = float(entry.get("confidence", 0.0))
            if edge_format == "legacy" or not metadata_available:
                fallback_reasons.append("legacy_edge_missing_metadata")
                continue
            if confidence + 1e-12 < minimum_edge_confidence:
                fallback_reasons.append("structured_edge_below_confidence")
                continue
            relation_match = _relation_match_score(parsed_query.relation, str(entry["relation"]))
            score = relation_match * confidence
            if score > graph_relation_score:
                graph_relation_score = score
                relation_chain = [
                    {
                        "source": candidate_node.node_id,
                        "target": anchor_id,
                        "relation": str(entry["relation"]),
                        "stored_source": int(entry["stored_source"]),
                        "stored_target": int(entry["stored_target"]),
                        "stored_relation": str(entry["stored_relation"]),
                        "confidence": confidence,
                        "distance_m": entry.get("distance_m"),
                        "coordinate_frame": entry.get("coordinate_frame"),
                        "observation_count": int(entry.get("observation_count", 1)),
                        "first_seen_frame": entry.get("first_seen_frame"),
                        "last_seen_frame": entry.get("last_seen_frame"),
                        "edge_source": "structured_edge",
                    }
                ]

        geometry_relation_score = 0.0
        geometry_debug = None
        if graph_relation_score <= 0.0 and not candidate_has_structured_evidence:
            if not fallback_reasons:
                fallback_reasons.append("structured_edge_missing")
            geometry_relation_score, geometry_debug = _geometric_relation_score_with_debug(
                parsed_query.relation,
                candidate_node.bbox_center,
                anchor_node.bbox_center,
                parsed_query.near_distance,
                coordinate_frame=fallback_coordinate_frame,
                coordinate_frame_fallback=fallback_coordinate_frame_fallback,
                ego_heading=fallback_ego_heading,
                trajectory=fallback_trajectory,
                ego_position=fallback_ego_position,
            )
            if geometry_relation_score > 0.0:
                relation_chain = [
                    {
                        "source": candidate_node.node_id,
                        "target": anchor_id,
                        "relation": parsed_query.relation,
                        "edge_source": "geometry_fallback",
                        "fallback_reason": sorted(set(fallback_reasons)),
                        "coordinate_frame": geometry_debug.get("actual_coordinate_frame"),
                    }
                ]
        elif graph_relation_score <= 0.0:
            fallback_reasons.append("structured_edge_preferred_for_other_anchor")

        combined = max(graph_relation_score, geometry_relation_score) * anchor_text_score
        anchor_scores.append(combined)
        debug_matches.append(
            {
                "anchor_id": anchor_id,
                "anchor_tag": anchor_node.object_tag,
                "anchor_text_score": anchor_text_score,
                "graph_relation_score": graph_relation_score,
                "geometry_relation_score": geometry_relation_score,
                "combined": combined,
                "relation_chain": relation_chain,
                "edge_source": (
                    "structured_edge" if graph_relation_score > 0.0 else "geometry_fallback"
                ),
                "fallback_reason": (
                    [] if graph_relation_score > 0.0 else sorted(set(fallback_reasons))
                ),
                "geometry_debug": geometry_debug,
            }
        )

    if not anchor_scores:
        return 0.0, {"reason": "anchor_not_found", "anchor_text": parsed_query.anchor_text}
    return float(max(anchor_scores)), {"matches": sorted(debug_matches, key=lambda item: item["combined"], reverse=True)[:5]}


def geometric_relation_score(
    relation: str,
    candidate_center: np.ndarray,
    anchor_center: np.ndarray,
    near_distance: float | None,
    coordinate_frame: str = "world",
    coordinate_frame_fallback: str | None = "world",
    ego_heading: np.ndarray | None = None,
    trajectory: np.ndarray | None = None,
    ego_position: np.ndarray | None = None,
) -> float:
    score, _debug = _geometric_relation_score_with_debug(
        relation,
        candidate_center,
        anchor_center,
        near_distance,
        coordinate_frame=coordinate_frame,
        coordinate_frame_fallback=coordinate_frame_fallback,
        ego_heading=ego_heading,
        trajectory=trajectory,
        ego_position=ego_position,
    )
    return score


def _geometric_relation_score_with_debug(
    relation: str,
    candidate_center: np.ndarray,
    anchor_center: np.ndarray,
    near_distance: float | None,
    *,
    coordinate_frame: str,
    coordinate_frame_fallback: str | None,
    ego_heading: np.ndarray | None,
    trajectory: np.ndarray | None,
    ego_position: np.ndarray | None,
) -> tuple[float, dict]:
    delta = np.asarray(candidate_center, dtype=float) - np.asarray(anchor_center, dtype=float)
    planar_distance = float(np.linalg.norm(delta[:2]))

    if relation == "near":
        threshold = near_distance if near_distance is not None else 8.0
        score = float(max(0.0, 1.0 - planar_distance / max(threshold, 1e-6)))
        return score, {
            "actual_coordinate_frame": "distance_invariant",
            "fallback_reason": None,
            "planar_distance_m": planar_distance,
        }

    frame = resolve_coordinate_frame(
        coordinate_frame,
        coordinate_frame_fallback,
        ego_heading=ego_heading,
        trajectory=trajectory,
        ego_position=ego_position,
    )
    if not frame.success:
        return 0.0, {
            "actual_coordinate_frame": None,
            "fallback_reason": frame.failure_reason,
            "failure_code": frame.failure_code,
        }
    forward = float(np.dot(delta[:2], frame.forward_axis_world))
    left = float(np.dot(delta[:2], frame.left_axis_world))
    if relation == "left_of":
        score = 1.0 if left > 0 else 0.0
    elif relation == "right_of":
        score = 1.0 if left < 0 else 0.0
    elif relation == "in_front_of":
        score = 1.0 if forward > 0 else 0.0
    elif relation == "behind":
        score = 1.0 if forward < 0 else 0.0
    else:
        score = 0.0
    return score, {
        "actual_coordinate_frame": frame.actual_frame,
        "requested_coordinate_frame": frame.requested_frame,
        "fallback_reason": frame.fallback_reason,
        "forward_component_m": forward,
        "left_component_m": left,
        "planar_distance_m": planar_distance,
    }


def _relation_match_score(expected_relation: str, actual_relation_text: str) -> float:
    actual_relation_text = normalize_text(actual_relation_text)
    if expected_relation == actual_relation_text:
        return 1.0
    synonyms = RELATION_SYNONYMS.get(expected_relation, set())
    if actual_relation_text in synonyms:
        return 1.0
    if expected_relation == "near" and any(word in actual_relation_text for word in ["near", "close", "next to", "beside"]):
        return 0.8
    return 0.0


def rank_candidate_targets(
    parsed_query: ParsedNavigationQuery,
    nodes: dict[int, SceneGraphNode],
    edges: list[SceneGraphEdge],
    vehicle_xy: np.ndarray,
    lambda_text: float,
    lambda_relation: float,
    lambda_distance: float,
    minimum_edge_confidence: float = 0.0,
    relation_coordinate_frame: str = "world",
    relation_coordinate_frame_fallback: str | None = "world",
    relation_trajectory: np.ndarray | None = None,
    evidence_policy=None,
) -> list[CandidateTarget]:
    from conceptgraph.completion.evidence import target_evidence_decision
    relation_index = build_relation_index(edges)
    vehicle_xy = np.asarray(vehicle_xy, dtype=float)
    max_distance = float(np.max([np.linalg.norm(node.bbox_center[:2] - vehicle_xy) for node in nodes.values()])) if nodes else 1.0
    target_text_scores = {
        node_id: score_text_similarity(parsed_query.target_text, node)
        for node_id, node in nodes.items()
    }
    anchor_text_scores = None
    if parsed_query.relation is not None and parsed_query.anchor_text is not None:
        anchor_text_scores = {
            node_id: score_text_similarity(parsed_query.anchor_text, node)
            for node_id, node in nodes.items()
        }

    candidates: list[CandidateTarget] = []
    dynamic_query = _query_allows_dynamic_target(parsed_query.raw_text)
    heading = _estimate_query_heading(parsed_query, vehicle_xy)
    for node_id, node in nodes.items():
        if parsed_query.requires_static and (node.motion_state != "static" or node.is_short_term):
            continue
        terms = node_category_terms(node)
        if any(terms & CLASS_COMPATIBILITY.get(c, {c}) for c in parsed_query.excluded_categories):
            continue
        if parsed_query.excluded_colors:
            colors = (node.metadata or {}).get("attributes", {}).get("colors", [])
            if not colors or "unknown" in colors or any(node_has_color(node, [c]) for c in parsed_query.excluded_colors):
                continue
        eligible, evidence_reason = target_evidence_decision(node, evidence_policy)
        if not eligible:
            continue
        compatible, compatibility_reason = is_node_compatible_with_query(parsed_query, node)
        if not compatible:
            continue

        text_score = target_text_scores.get(node_id, 0.0)
        if parsed_query.target_categories:
            text_score = max(text_score, 0.95)
        elif parsed_query.query_mode == "nearest_static":
            text_score = max(text_score, 0.8)
        elif text_score <= 0.0:
            continue

        relation_score, relation_debug = score_relation_constraint(
            node,
            parsed_query,
            nodes,
            relation_index,
            anchor_text_scores=anchor_text_scores,
            minimum_edge_confidence=minimum_edge_confidence,
            fallback_coordinate_frame=relation_coordinate_frame,
            fallback_coordinate_frame_fallback=relation_coordinate_frame_fallback,
            fallback_ego_heading=heading,
            fallback_trajectory=relation_trajectory,
            fallback_ego_position=vehicle_xy,
        )
        if parsed_query.relation is not None and relation_score <= 0.05:
            continue
        node_xy = node.bbox_center[:2]
        distance = float(np.linalg.norm(node_xy - vehicle_xy))
        if parsed_query.min_target_distance_m is not None and distance < parsed_query.min_target_distance_m:
            continue
        if parsed_query.max_target_distance_m is not None and distance > parsed_query.max_target_distance_m:
            continue
        front_score = _front_score(node_xy, vehicle_xy, heading)
        if parsed_query.prefer_front and front_score <= 0.0:
            continue
        norm_distance = distance / max(max_distance, 1e-6)
        if any(word in parsed_query.raw_text or word in normalize_text(parsed_query.raw_text) for word in FAR_WORDS):
            distance_prior = 0.25 + 0.75 * norm_distance
        elif parsed_query.prefer_front:
            distance_prior = (1.0 / (1.0 + distance)) * (0.35 + 0.65 * front_score)
        else:
            distance_prior = 1.0 / (1.0 + distance)

        if parsed_query.prefer_nearest or parsed_query.prefer_front:
            score = 0.45 * text_score + 0.1 * relation_score + 0.45 * distance_prior
        else:
            score = (
                lambda_text * text_score +
                lambda_relation * relation_score +
                lambda_distance * distance_prior
            )
        dynamic_penalty = 1.0
        if (node.motion_state == "dynamic" or node.is_short_term) and not dynamic_query:
            dynamic_penalty = 0.65
            score *= dynamic_penalty
        candidates.append(
            CandidateTarget(
                node_id=node_id,
                score=float(score),
                text_score=float(text_score),
                relation_score=float(relation_score),
                distance_prior=float(distance_prior),
                debug={
                    "distance_to_vehicle": distance,
                    "relation_debug": relation_debug,
                    "object_tag": node.object_tag,
                    "caption": node.caption,
                    "motion_state": node.motion_state,
                    "target_evidence": evidence_reason,
                    "is_short_term": node.is_short_term,
                    "dynamic_penalty": dynamic_penalty,
                    "semantic_compatibility": compatibility_reason,
                    "target_categories": parsed_query.target_categories,
                    "target_colors": parsed_query.target_colors,
                    "anchor_colors": parsed_query.anchor_colors,
                    "color_match": node_has_color(node, parsed_query.target_colors),
                    "front_score": front_score,
                },
            )
        )

    candidates.sort(key=lambda item: (-round(float(item.score), 12), int(item.node_id)))
    return candidates


def _estimate_query_heading(parsed_query: ParsedNavigationQuery, vehicle_xy: np.ndarray) -> np.ndarray | None:
    raw = getattr(parsed_query, "vehicle_heading_xy", None)
    if raw is None:
        return None
    heading = np.asarray(raw, dtype=float)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-6:
        return None
    return heading / norm


def _front_score(node_xy: np.ndarray, vehicle_xy: np.ndarray, heading: np.ndarray | None) -> float:
    if heading is None:
        return 1.0
    delta = np.asarray(node_xy, dtype=float) - np.asarray(vehicle_xy, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return 0.0
    cos_angle = float(np.dot(delta / norm, heading))
    if cos_angle <= 0.35:
        return 0.0
    return min(1.0, cos_angle)


def _query_allows_dynamic_target(text: str) -> bool:
    normalized = normalize_text(text)
    dynamic_words = {
        "front vehicle",
        "vehicle ahead",
        "moving",
        "dynamic",
        "nearest vehicle",
        "最近车辆",
        "前方车辆",
        "移动车辆",
        "动态目标",
    }
    return any(word in text or word in normalized for word in dynamic_words)
