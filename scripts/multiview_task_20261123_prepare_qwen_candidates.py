"""生成72条程序模板待审核候选；不生成真人标签、不划分集合、不调用模型。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

from jsonschema import Draft202012Validator

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory

PACKAGE = "multiview_task_20261123_intent_candidates_v1"
SCHEMA_PATH = PROJECT_ROOT / "outputs/reports/multiview_semantic_tasks_20261123/tasks/prototype_final/language_intent.schema.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def condition(attribute, value):
    return dict(attribute=attribute, requested_value=value, grounding_required=True)


def families():
    rows = []

    def add(family, action, texts, *, side=None, permitted=False, free=False, color=None,
            speed=None, headway=None, duration=None, ambiguous=False):
        category = "vehicle" if action == "follow_vehicle" else None
        target = "本车道前车" if action == "follow_vehicle" else "最近空闲车位" if action == "park_nearest" else "路边停靠区域"
        if ambiguous:
            target = "待澄清的车辆" if action == "follow_vehicle" else "待澄清停靠侧的最近空闲车位" if action == "park_nearest" else "待澄清指代的路边停靠区域"
        conditions = [condition("category", category)] if category else [condition("region_type", "individual_parking_space" if action == "park_nearest" else "stopping_region")]
        if action == "follow_vehicle" and not ambiguous:
            conditions.append(condition("lane_relation", "same_lane_ahead"))
        if action == "park_nearest" or free:
            conditions.append(condition("free_occupancy_requirement", "free"))
        if permitted:
            conditions.append(condition("parking_permission_requirement", "allowed"))
        if color:
            conditions.append(condition("color", color))
        constraints = {}
        if side:
            constraints["side"] = side
        if speed is not None:
            constraints["maximum_speed_mps"] = speed
        if headway is not None:
            constraints["minimum_headway_s"] = headway
        mode = "safe_follow_duration" if duration else "user_end" if action == "follow_vehicle" else "stopped_in_allowed_region"
        draft = dict(action=action, target_description=target, target_conditions=conditions,
                     requested_constraints=constraints,
                     termination_request=dict(mode=mode, duration_s=duration), needs_clarification=ambiguous)
        assert len(texts) == 3
        for i, text in enumerate(texts, 1):
            rows.append(dict(sample_id=f"{family}__p{i}", paraphrase_family_id=family,
                instruction=text, intent_draft=draft, ambiguity_kind="unresolved_vehicle_reference" if ambiguous and action == "follow_vehicle" else "undecided_side" if ambiguous and action == "park_nearest" else "unresolved_region_reference" if ambiguous else None))

    add("PO01", "pull_over", ["请靠边停车。", "麻烦把车停到路边。", "帮我找个路边的位置停一下。"])
    add("PO02", "pull_over", ["请在右侧靠边停车。", "往右边靠一靠，把车停好。", "麻烦靠右侧路边停一下。"], side="right")
    add("PO03", "pull_over", ["请在左侧靠边停车。", "往左边靠一靠，把车停好。", "麻烦靠左侧路边停一下。"], side="left")
    add("PO04", "pull_over", ["请在右侧允许停车的路边停下。", "右边如果有允许停靠的位置，就在那里靠边停车。", "麻烦靠右停，位置要允许停车。"], side="right", permitted=True)
    add("PO05", "pull_over", ["请在左侧没有被占用的路边位置停车。", "找左边空着的路边位置靠边停好。", "靠左侧路边停车，停靠位置要空闲。"], side="left", free=True)
    add("PO06", "pull_over", ["靠右边停车，过程中速度不要超过每秒5米。", "请在右侧靠边停下，速度上限是每秒5米。", "麻烦靠右停好，行驶速度不高于5米每秒。"], side="right", speed=5)
    add("PO07", "pull_over", ["请在允许停车且空闲的路边停稳，停好就结束任务。", "找允许停靠、又没被占用的路边位置，车停好后任务结束。", "麻烦靠边停到允许停车的空闲位置，停稳便完成。"], permitted=True, free=True)
    add("PO08", "pull_over", ["请在刚才说的那个路边位置停车，先确认一下我指的是哪里。", "靠边停在那个地方，具体位置先问我确认。", "把车停到我说的路边位置；我还没说明是哪处，请先问清楚。"], ambiguous=True)

    add("PK01", "park_nearest", ["请停到最近的空闲车位。", "找最近的空车位，把车停进去。", "麻烦选最近的空闲停车位停好。"])
    add("PK02", "park_nearest", ["请停到右侧最近的空闲车位。", "找右边最近的空车位停进去。", "麻烦在右侧空闲车位里选最近的一个停好。"], side="right")
    add("PK03", "park_nearest", ["请停到左侧最近的空闲车位。", "找左边最近的空车位停进去。", "麻烦在左侧空闲车位里选最近的一个停好。"], side="left")
    add("PK04", "park_nearest", ["请在允许停车的空闲车位中，选最近的停好。", "找最近的空车位停车，那个车位必须允许停车。", "麻烦停进最近的、允许使用的空闲车位。"], permitted=True)
    add("PK05", "park_nearest", ["请停到最近空闲车位，行驶速度不超过每秒2米。", "找最近的空车位停进去，速度上限2米每秒。", "麻烦选最近空闲停车位停好，过程中不快于每秒2米。"], speed=2)
    add("PK06", "park_nearest", ["请在右侧允许停车的空闲车位里，选最近的停好。", "找右边最近的空车位泊入，车位要允许停车。", "麻烦选右侧最近的、允许使用的空闲停车位。"], side="right", permitted=True)
    add("PK07", "park_nearest", ["停到左侧最近的空闲车位，车停稳后结束任务。", "请找左边最近的空车位停好，停稳便算任务完成。", "麻烦在左侧最近空闲停车位泊车，停好就结束。"], side="left")
    add("PK08", "park_nearest", ["最近空闲车位要选左侧还是右侧，我还没决定，请先问我。", "请停进最近空车位，但左右哪边我还没选定，先确认再决定。", "找最近的空闲停车位；停左边还是右边先问清楚。"], ambiguous=True)

    add("FV01", "follow_vehicle", ["请跟随本车道前车，直到我说结束。", "跟着同车道的前车行驶，我叫停任务时再结束跟随。", "麻烦保持跟随本车道前方车辆，结束时间等我通知。"])
    add("FV02", "follow_vehicle", ["请跟随本车道前车10秒。", "跟着同车道前车行驶十秒钟。", "麻烦跟随本车道前方车辆，持续10秒。"], duration=10)
    add("FV03", "follow_vehicle", ["请跟随本车道前车30秒。", "跟着同车道前车行驶半分钟。", "麻烦跟随本车道前方车辆，持续30秒。"], duration=30)
    add("FV04", "follow_vehicle", ["请跟随本车道白色前车，直到我说结束。", "跟着同车道前方的白色车走，结束时间等我通知。", "麻烦跟随本车道那辆白色前车，我说结束时再结束任务。"], color="white")
    add("FV05", "follow_vehicle", ["请跟随本车道黑色前车20秒。", "跟着同车道前方的黑色车行驶二十秒。", "麻烦跟随本车道黑色前车，持续20秒。"], color="black", duration=20)
    add("FV06", "follow_vehicle", ["请跟随本车道前车15秒，请求的最小时距是2.5秒。", "跟着同车道前车行驶十五秒，时距不要小于两点五秒。", "麻烦跟随本车道前车，持续15秒，并把最小时距设为2.5秒。"], headway=2.5, duration=15)
    add("FV07", "follow_vehicle", ["请跟随本车道前车40秒，速度不超过每秒8米。", "跟着同车道前车行驶四十秒，行驶速度上限8米每秒。", "麻烦跟随本车道前车，持续40秒，速度不高于每秒8米。"], speed=8, duration=40)
    add("FV08", "follow_vehicle", ["跟着那辆车，先确认一下我指的是哪一辆，直到我说结束。", "请跟随我说的那辆车，我还没指明是哪辆，先问清楚；结束时间等我通知。", "麻烦跟随那台车，目标车辆先向我确认，我说结束再结束跟随。"], ambiguous=True)
    return rows


def main():
    begun = time.monotonic()
    base = PROJECT_ROOT
    folder = base / "dataset/qwen" / PACKAGE
    folder.mkdir(parents=True, exist_ok=False)
    run = experiment_directory("qwen", "preparation", PACKAGE, create=True)
    report = base / "outputs/reports/multiview_task_iteration_20261123/qwen"
    report.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    rows = families()
    assert len(rows) == 72
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        validator.validate(row["intent_draft"])
        row.update(package_id=PACKAGE, source=dict(kind="programmatic_template_candidate", author="codex_assistant", human_authored=False,
            method="24 explicitly authored template families, three paraphrases per family; slots supply deterministic intent draft, no model parsing"),
            draft_status="unreviewed_text_rule_proposal_not_ground_truth", human_review_status="pending",
            human_review=dict(reviewer=None, reviewed_at=None, decision=None, notes=None),
            scene=None, log_token=None, episode_id=None, issue_frame_idx=None, issue_time_s=None,
            split="unassigned", grounding_status="unbound_unknown", training_eligible=False,
            instruction_sha256=hashlib.sha256(row["instruction"].encode()).hexdigest())
    normalized = [re.sub(r"[\s，。！？、；：,.!?;:]", "", r["instruction"]) for r in rows]
    assert len(set(normalized)) == len(rows) == len({r["sample_id"] for r in rows})
    assert {a: sum(r["intent_draft"]["action"] == a for r in rows) for a in ("pull_over", "park_nearest", "follow_vehicle")} == dict(pull_over=24, park_nearest=24, follow_vehicle=24)
    assert all(sum(r["paraphrase_family_id"] == f for r in rows) == 3 for f in {r["paraphrase_family_id"] for r in rows})
    required = set(rows[0])
    assert all(set(r) == required and r["human_review_status"] == "pending" and not r["training_eligible"] for r in rows)
    data = folder / "candidates.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    shutil.copy2(SCHEMA_PATH, folder / "language_intent.schema.json")
    keys = ["sample_id", "paraphrase_family_id", "instruction", "action", "target_description", "target_conditions", "requested_constraints", "termination_request", "needs_clarification", "ambiguity_kind", "human_review_status", "source_kind", "scene", "log_token", "episode_id", "issue_frame_idx", "issue_time_s", "split", "training_eligible"]
    with (folder / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r["intent_draft"].get(k) if k in r["intent_draft"] and not isinstance(r["intent_draft"].get(k), (list, dict)) else json.dumps(r["intent_draft"][k], ensure_ascii=False) if k in r["intent_draft"] else r["source"]["kind"] if k == "source_kind" else r.get(k) for k in keys})
    protocol_ids = [f"{s}__{a}" for s in ("scene-0012", "scene-0013") for a in ("pull_over", "park_nearest", "follow_vehicle")]
    split_policy = dict(status="not_assigned", candidate_scene_log_episode_all_null=True,
        grouping=["scene", "log_token", "episode_id", "paraphrase_family_id"],
        rule="Before splitting, form connected components: any shared log/scene/episode/paraphrase family must stay in one split. Bind new natural episodes first; user-reviewed template rewrites remain together.",
        current_navigation_protocol_episode_ids=protocol_ids, candidates_bound_to_protocol_episodes=0,
        known_pilot_scope="scene-0012/0013 share one log; cannot split the two as independent train and validation",
        existing_seen_test_policy="Keep original seen-test/regression identity; never relabel as unseen. Existing 38 development scenes remain validation/debug.",
        training_data_export_enabled=False)
    dump(folder / "group_split_policy.json", split_policy)
    manifest = dict(package_id=PACKAGE, created_utc=now, rows=72, families=24, task_counts=dict(pull_over=24, park_nearest=24, follow_vehicle=24),
        source_kind="programmatic_template_candidate", human_reviewed_rows=0, actual_training_rows=0, model_calls=0, GPU_calls=0,
        candidates_sha256=digest(data), intent_schema_sha256=digest(folder / "language_intent.schema.json"),
        schema_source=str(SCHEMA_PATH), schema_source_sha256=digest(SCHEMA_PATH),
        source_script_sha256=digest(__file__), data_split="unassigned", episode_bindings=0,
        limits=dict(wall_s=2100, additional_bytes=64*1024**2, minimum_free_bytes=20*1024**3),
        intent_only="requested color/free/permission are language conditions requiring grounding; they are not observed attributes or granted permission",
        draft_assumptions=["unspecified parking completion maps to stopped_in_allowed_region as a proposed intent contract; human confirmation required", "follow without duration uses user_end only where text explicitly says end by user", "needs_clarification describes explicit language reference/side uncertainty; map-level ambiguity is still unknown", "free occupancy and parking permission request do not assert any real candidate satisfies them"],
        unchanged_legacy="No old training data, Qwen model, v4 manuscript or previous UI default modified")
    dump(folder / "manifest.json", manifest)
    ui_source = base / "conceptgraph/annotation/multiview_task_20261123_intent_review.html"
    bundle = dict(manifest=manifest, manifest_sha256=digest(folder / "manifest.json"), schema=schema, rows=rows)
    encoded = json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c")
    (folder / "index.html").write_text(ui_source.read_text().replace("__BUNDLE_JSON__", encoded))
    instructions = """先打开index.html。默认每家族只看首条，共24条；这只是减轻第一轮阅读量，不会批准家族其余改写。填写姓名或代号，核对原句和动作/条件/终止/澄清摘要。可以改说法并写明哪里有误；‘通过’前勾选已核对字段，无法确认选待修改或保留pending。每条保存才形成审核事件。浏览器保存只在本机。导出审核JSON后发回军师聊天；未审核条目和其他改写继续pending。勿把生成器检查/测试导出当人工审核。

不用设计实验、不用写JSON、不用标具体车位/许可。本页只审核文字意图；若希望提供真实指令，可直接用自己的说法改写，保留来源记录，但仍需按连续episode发出时刻和真实画面另行绑定。颜色只作文字请求，不是看图颜色真值。

不使用UI也可回传‘sample_id → 修改说法/正确动作/条件/何时结束/是否先问清楚’；我方转换后再供确认。不要直接编辑candidates.jsonl覆盖原包。

后续微调启动条件：至少有人审通过并确认意图字段的样本；若称真实自然指令数据，还要有用户实际提供的原句与来源。固定分组方案后完成训练/验证独立性检查。同一家族、同scene/log/episode禁止跨集合，现2场景同1log不能两边各分一半。本轮不启动训练；后续待标签审核、分组及预算/资源满足后，在用户既有授权范围内安排。涉及新的外部资源或超出已授权范围时再确认。本包当前training_eligible全false，不存在可直接启动训练的导出。
"""
    (folder / "审核说明.txt").write_text(instructions)
    source = run / "source_snapshot"
    source.mkdir()
    for path in (Path(__file__), ui_source, SCHEMA_PATH, base / "conceptgraph/project_paths.py"):
        shutil.copy2(path, source / path.name)
    checks = dict(passed=True, rows=72, task_counts=manifest["task_counts"], families=24, exact_or_punctuation_duplicate_rows=0,
        schema_valid_rows=72, unexpected_intent_fields=0, unexpected_candidate_fields=0,
        human_reviewed_rows=0, pending_rows=72, training_eligible_rows=0, allocated_scene_log_episode_rows=0,
        models_called=0, gpu_called=0, assigned_split_rows=0,
        source_sha256={str(p): digest(p) for p in (Path(__file__), ui_source, SCHEMA_PATH)},
        input_sha256={str(p): digest(p) for p in folder.iterdir() if p.is_file()},
        wall_s=time.monotonic()-begun, new_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
        minimum_free_bytes_after=shutil.disk_usage(folder).free)
    assert checks["new_bytes"] < 64*1024**2 and checks["minimum_free_bytes_after"] >= 20*1024**3
    dump(run / "preparation_verification.json", checks)
    dump(report / "preparation_verification.json", checks)
    with zipfile.ZipFile(report / "Qwen任务意图_72条待审核离线包.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir():
            if path.is_file():
                z.write(path, arcname=path.name)
    report_text = f"""# Qwen任务意图候选包\n\n已准备72条程序模板候选：靠边、最近空闲车位、跟车各24条，每任务8个家族×3种措辞。不是真人指令、人工真值或模型成绩；人审0、待审72、训练可用0、调用模型/GPU/训练均0。所有scene/log/episode/发出帧时刻为null，split=unassigned，不把导航6个协议episode扩写成72次真实任务。\n\n审核入口：{folder / 'index.html'}\n\n便携包：{report / 'Qwen任务意图_72条待审核离线包.zip'}\n\nJSONL与CSV：{data}；{folder / 'candidates.csv'}\n\n{instructions}\n\n建议后续冻结专项指标（本轮无分数）：动作准确率；目标条件集合micro/macro F1与每条集合完全正确；请求侧向/速度/时距解析准确；终止模式准确与明确数字时长准确；需澄清召回、误澄清率及unknown口径；严格六字段全对/N_all。N_all包含空答、非法格式、未知字段和拒绝输出，不能只算合法JSON子集。原始模型原答与回退后系统输出分别统计。target_description采用人审冻结的规范表述；条件集合按顺序无关比较，其余字段精确比较，规范化规则须评价前冻结。颜色仅为文本请求，不发布RGB可见属性准确率或目标选对率。\n\n验证：schema72/72、文字去空白标点后重复0、未知字段0；这只是工程检查，不是Qwen正确率。所有草案保留pending，未触发训练导出。来源、冻结指纹、分组政策与逐条审核导出保留。新包对旧训练数据/论文v4/UI默认均无覆盖。\n"""
    (report / "候选包交付与最小人审.md").write_text(report_text)
    print(json.dumps(dict(package=str(folder), run=str(run), verification=str(report / "preparation_verification.json"), rows=72, human_reviewed=0, training_eligible=0, wall_s=checks["wall_s"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
