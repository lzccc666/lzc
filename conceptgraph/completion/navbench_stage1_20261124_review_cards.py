"""Add real-image review cards and semantic certificate checks without relabelling."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from conceptgraph.completion.navbench_stage1_20261124 import (
    DATA, EPISODES, REFERENCE, REGIONS, REPORT, ROOT, canonical, content_sha,
    dump, jsonl, now, read, rows, sha, source_ref,
)


def certificate_matches_task(task, certificate, acceptable):
    """A certificate for another action or operating contract is never transferable."""
    if task['action'] != 'object_approach' or task['source_family'] != 'historical_conditional_reference':
        return False
    if certificate['scene'] != task['scene'] or certificate['frame_idx'] != task['issue_frame_idx']:
        return False
    if certificate['instance_token'] not in acceptable:
        return False
    contract = task['contract']
    fixed = {'initial_speed_mps': 0., 'initial_steering_rad': 0., 'allowed_motion': 'forward_only',
             'prediction_horizon_s': 90., 'terminal_hold_s': 2., 'terminal_speed_max_mps': .02}
    if any(contract.get(key) != value or certificate.get(key) != value for key, value in fixed.items()):
        return False
    if contract.get('output_mode') != 'timed_trajectory' or contract.get('goal_reference_point') != 'rear_axle':
        return False
    if contract.get('vehicle_model') != certificate['vehicle']:
        return False
    if contract.get('terminal_heading_tolerance_rad') != certificate['region']['max_heading_error_rad']:
        return False
    if contract.get('obstacle_temporal_scope') != certificate['temporal_scope']:
        return False
    if contract.get('acceptable_target_set_sha256') != content_sha(acceptable):
        return False
    return certificate['task_contracts'].get(task['task_id']) == task['contract_sha256']


def boundary_checks():
    registry = rows(DATA / 'task_registry.jsonl')
    certificates = {c['certificate_id']: c for c in rows(DATA / 'evaluation_only/certificates.jsonl')}
    labels = {r['task_id']: r for r in rows(DATA / 'evaluation_only/conditional_reference_labels.jsonl')}
    static = {(r['scene'], r['frame_idx'], r['instance_token']): r for r in rows(REGIONS / 'targets.jsonl')}
    counts = Counter()
    for task in registry:
        if task['source_family'] != 'historical_conditional_reference':
            continue
        label = labels[task['task_id']]
        acceptable = label['source_query']['accepted_instance_tokens']
        assert acceptable == label['reachability_audit']['accepted_instance_tokens']
        for cert_id in task['certificate_ids']:
            cert = certificates[cert_id]
            assert certificate_matches_task(task, cert, acceptable)
            witness = read(cert['artifacts']['witness']['path'])
            region = static[(cert['scene'], cert['frame_idx'], cert['instance_token'])]['regions'][witness['region_index']]
            assert witness['region'] == region
            assert {k: region[k] for k in cert['region']} == cert['region']
            initial = read(cert['artifacts']['initial_pose']['path'])
            assert initial['scene'] == task['scene'] and initial['frame_idx'] == task['issue_frame_idx']
            assert initial['rear_axle_reference_world_pose'] == cert['initial_pose']
            for key in ['pull_over', 'park_nearest', 'follow_vehicle']:
                changed = dict(task, action=key)
                assert not certificate_matches_task(changed, cert, acceptable)
                counts['cross_action_rejected'] += 1
            for key, value in [('allowed_motion', 'reverse_allowed'), ('prediction_horizon_s', 30.),
                               ('terminal_speed_max_mps', .2), ('initial_speed_mps', 1.),
                               ('goal_reference_point', 'camera_center')]:
                changed = dict(task, contract={**task['contract'], key: value})
                assert not certificate_matches_task(changed, cert, acceptable)
                counts['changed_contract_rejected'] += 1
            assert not certificate_matches_task(task, cert, [])
            counts['independent_task_certificate_bindings'] += 1
        if task['feasibility_status'] == 'pending':
            assert not task['certificate_ids'] and not task['main_table_eligible']
            counts['unknown_not_promoted_to_infeasible'] += 1
    result = dict(passed=True, created_utc=now(), counts=dict(counts),
        registry_sha256=sha(DATA / 'task_registry.jsonl'), historical_truth_and_regions_unchanged=True,
        source_sha256={str(Path(__file__)): sha(__file__)}, model_calls=0, planner_calls=0, GPU_calls=0,
        checks_scope='CPU certificate semantics and negative transfer cases; no new feasibility search or algorithm score')
    dump(DATA / 'certificate_boundary_verification_v1.json', result)
    return result


def prepare_cards():
    from PIL import Image, ImageDraw
    destination = DATA / 'review_cards'
    destination.mkdir(exist_ok=False)
    manifest_path = ROOT / 'dataset/processed/multiview_20261123_pilot_v1/manifest.json'
    manifests = {s['scene']: s for s in read(manifest_path)['scenes']}
    images = {}
    card_paths = []
    for ep in read(EPISODES)['rows']:
        scene = ep['scene']
        frame = next(g for g in manifests[scene]['groups'] if g['frame_idx'] == ep['issue_frame_idx'])
        if scene not in images:
            refs, panels = [], []
            for channel in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']:
                view = frame['views'][channel]
                path = Path(view['rgb'])
                image = Image.open(path).convert('RGB')
                image.thumbnail((960, 540))
                panel = Image.new('RGB', (960, 570), 'white')
                panel.paste(image, (0, 30))
                ImageDraw.Draw(panel).text((10, 8), channel + ' | frame 0 | original image, no labels', fill='black')
                panels.append(panel)
                refs.append(dict(channel=channel, rgb=str(path), sha256=sha(path),
                    camera_timestamp_s=view['camera_timestamp_s'], frame_idx=ep['issue_frame_idx'],
                    inference_time_reference_s=frame['reference_timestamp_s'],
                    image_width=Image.open(path).width, image_height=Image.open(path).height))
            canvas = Image.new('RGB', (960, 1710), 'white')
            for i, panel in enumerate(panels):
                canvas.paste(panel, (0, 570 * i))
            preview = destination / f'{scene}_issue_frame0_three_views.jpg'
            canvas.save(preview, quality=80, optimize=True)
            images[scene] = dict(preview=str(preview), preview_sha256=sha(preview), originals=refs)
        questions = {
            'pull_over': ['这段画面是否适合表达“靠边停车”？不能确认可填unknown。',
                          '如适用，请指出期望停靠侧和可见的候选停靠区域；只标图像中可确认部分。'],
            'park_nearest': ['这段画面是否看得到具体单个车位？看不到请填not_visible或unknown。',
                            '如看得到，请指出所有可确认候选车位及可见占用状态；不要把停车场区域当单车位。'],
            'follow_vehicle': ['发出指令时，你指的是哪辆前车？如没有或不唯一请如实填写。',
                              '如可确认，请指出前车图像框/颜色/所在视角；连续身份和距离由我们再核。'],
        }[ep['action']]
        card = dict(card_id='review:' + ep['episode_id'], task_id='episode:' + ep['episode_id'],
            scene=scene, log_token=ep['log_token'], instruction=ep['instruction'], action=ep['action'],
            issue_frame_idx=ep['issue_frame_idx'], issue_time_s=ep['issue_time_s'],
            images=images[scene], review_status='pending', reviewer=None, reviewed_utc=None,
            questions=questions,
            answer=dict(instruction_applicable=None, visible_target_or_regions=None, stopping_side=None,
                target_camera_channel=None, pixel_polygon_or_box=None, visible_occupancy=None,
                ambiguity_or_occlusion=None, notes=None),
            human_not_required=['判断路径是否有解', '三维坐标/车身尺寸', '规划器参数或后轴变换',
                                '实际开过去/停车遥测', '无法从画面确认的法律或遮挡事实'],
            engineering_work_we_do=['冻结VehicleSpec/参考点变换', '像素到独立几何校验与区域绑定',
                '道路/车道/障碍与整车扫掠', '独立参考路径/轨迹认证，有限搜索失败保持unknown'],
            extra_source_if_missing=['具体车位/路缘不可见时需要其他适用时窗或独立区域标注来源',
                '跟车的连续身份/纵向状态由官方评价资料及过去因果观测核查，不靠单帧猜测'],
            prior_review_ui=str(ROOT / 'dataset/annotations/natural_graph_v004/episode_scene0012_20261120_001/index.html')
                if scene == 'scene-0012' else None,
            prior_ui_limit='旧页显示旧前视图/旧节点，仅作浏览参考，不能自动沿用其图或当作新任务标签',
            simple_language_review_ui=str(ROOT / 'dataset/qwen/multiview_task_20261124_simple_candidates_v3/index.html'),
            language_ui_limit='这里只审文字；文字通过不等于场景已绑定或轨迹正确',
            independent_feasibility_status='pending', human_answers_grant_training_or_success=False,
            source_ref=source_ref(EPISODES))
        filename = destination / f"{ep['episode_id']}.json"
        dump(filename, card)
        card_paths.append(str(filename))
        lines = [f"# {ep['episode_id']}：{ep['instruction']}", '',
            f"![三视角原图缩略]({images[scene]['preview']})", '',
            *[f"- {r['channel']}：[原图]({r['rgb']})" for r in images[scene]['originals']], '',
            *[f'{i+1}. {q}' for i, q in enumerate(questions)], '',
            '你只需指认画面里的目标/区域，不能确认就说不知道。路径是否有解由独立参考检查，不需要你判断。', '',
            f'[待审核JSON卡]({filename})。卡片为空，尚未记录任何人工结论。']
        (destination / f"{ep['episode_id']}.md").write_text('\n'.join(lines) + '\n')
    size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    assert size <= 16 * 1024 ** 2
    dump(DATA / 'review_cards_manifest.json', dict(created_utc=now(), cards=card_paths,
        source_manifest=source_ref(manifest_path), source_episode=source_ref(EPISODES),
        bytes_before_this_manifest=size, human_answers_created=0, no_existing_UI_modified=True,
        files={str(p.relative_to(DATA)): sha(p) for p in destination.rglob('*') if p.is_file()},
        source_sha256={str(Path(__file__)): sha(__file__)}, model_calls=0, GPU_calls=0))
    summary = ('# 三类任务的具体补标清单\n\n'
        '已为原6个episode生成真实发令帧三视角缩略、原图链接与空白JSON卡；同场景共用原图，未挑成功画面。'
        '这不是新UI，也没有写入人工通过记录。\n\n'
        '|任务|你只需反馈|我们自行完成|确实缺输入时再补|\n|---|---|---|---|\n'
        '|靠边|这段是否适用、期望侧、图中候选区域|工程车身/后轴、区域几何、道路与独立参考|不可见路缘/区域需适用画面或独立区域标注|\n'
        '|最近车位|是否有具体车位、可见候选与占用|绑定全部可接受车位、冻结最近定义、独立路径成本|没有具体车位时需要独立单车位来源|\n'
        '|跟车|发令时指哪辆前车、是否歧义|跨帧身份/过去运动、同车道、参考跟随轨迹|状态不足时补连续观测来源|\n\n'
        '不要求你判断规划是否有解，不等实际开到；无法从画面确认的内容保持unknown。'
        '原scene0012旧审核页只作浏览，不能把旧图标签自动迁到新任务。\n\n' +
        '\n'.join(f'- [{Path(p).stem}]({Path(p).with_suffix(".md")})' for p in card_paths) + '\n')
    (REPORT / '三类任务补标清单.md').write_text(summary)
    return dict(cards=6, image_scenes=len(images), bytes=size, manual_answers=0)


def main():
    checks = boundary_checks()
    cards = prepare_cards()
    compatibility = REPORT / 'independent/existing30_predicted_graph_compatibility_v1.json'
    if compatibility.exists():
        dump(DATA / 'readiness_amendment_v1.json', dict(created_utc=now(),
            applies_to='all 878 historical conditional reference queries',
            original_registry_sha256=sha(DATA / 'task_registry.jsonl'),
            compatibility_evidence=source_ref(compatibility),
            existing_predictions_found=True, compatibility_checked=True,
            method_inputs_still_to_be_exported_for_actual_evaluation=True,
            feasibility_and_denominators_unchanged=True,
            empty_predicted_graphs_not_excluded=True,
            previous_readiness='historical_input_compatibility_pending',
            effective_readiness='historical_prediction_compatibility_audited; method-input export/evaluation pending'))
    print(canonical(dict(boundary_checks=checks, review_cards=cards)))


if __name__ == '__main__':
    main()
