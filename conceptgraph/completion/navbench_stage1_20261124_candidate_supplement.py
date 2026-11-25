"""Read-only fixed-frame candidate browser from already cached RGB images."""
from __future__ import annotations

import html
from pathlib import Path
import shutil
import time

from PIL import Image, ImageDraw

from conceptgraph.completion.navbench_stage1_20261124 import (
    DATA, REPORT, ROOT, canonical, dump, now, read, sha, source_ref,
)
from conceptgraph.project_paths import experiment_directory

RUN_ID = 'navbench_stage1_candidates_20261124_supplement_v1'
DESTINATION = ROOT / 'dataset/processed' / RUN_ID
MANIFEST = ROOT / 'dataset/processed/mv136_20261123_sixscene_v1/manifest.json'
SCENES = ['scene-0012', 'scene-0013', 'scene-0019', 'scene-0035', 'scene-0042', 'scene-0047']
FRAMES = [0, 10, 19]
CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']


def main():
    started = time.monotonic()
    assert shutil.disk_usage(ROOT).free > 20 * 1024 ** 3
    original_files = {str(p): sha(p) for p in DATA.rglob('*') if p.is_file()}
    run = experiment_directory('nuscenes', 'validation', RUN_ID, create=True)
    DESTINATION.mkdir(exist_ok=False)
    scenes = {s['scene']: s for s in read(MANIFEST)['scenes']}
    records, md_sections, html_sections = [], [], []
    for scene_name in SCENES:
        scene = scenes[scene_name]
        groups = {g['frame_idx']: g for g in scene['groups']}
        sheet = Image.new('RGB', (1920, 3 * 400), 'white')
        draw = ImageDraw.Draw(sheet)
        sources = []
        for row_index, frame_idx in enumerate(FRAMES):
            frame = groups[frame_idx]
            for column, channel in enumerate(CHANNELS):
                entry = frame['views'][channel]
                path = Path(entry['rgb'])
                assert path.is_file(), f'No cached RGB: {path}'
                with Image.open(path) as original:
                    image = original.convert('RGB')
                    image.thumbnail((640, 360))
                    width, height = original.size
                sheet.paste(image, (column * 640, row_index * 400 + 40))
                draw.text((column * 640 + 8, row_index * 400 + 10),
                    f'{scene_name} | frame {frame_idx} | {channel}', fill='black')
                sources.append(dict(frame_idx=frame_idx, channel=channel,
                    sample_token=frame['sample_token'], rgb=str(path), sha256=sha(path),
                    width=width, height=height, camera_timestamp_s=entry['camera_timestamp_s']))
        sheet_path = DESTINATION / f'{scene_name}_fixed_0_10_19.jpg'
        sheet.save(sheet_path, quality=85, optimize=True)
        existing_viewer = ROOT / 'dataset/annotations/natural_graph_v004/episode_scene0012_20261120_001/index.html' if scene_name == 'scene-0012' else None
        record = dict(candidate_scene=scene_name, log_token=scene['log_token'],
            source_data_role=scene['data_role'], fixed_frames=FRAMES, channels=CHANNELS,
            contact_sheet=str(sheet_path), contact_sheet_sha256=sha(sheet_path), original_images=sources,
            candidate_only=True, semantic_review_status='not_reviewed', feasibility_status='not_assessed',
            main_table_eligible=False, labels_created=False,
            asks=['是否能确认具体单个车位及可见占用？', '是否能指出期望靠边区域及停靠侧？',
                  '是否有明确希望跟随的前车？请指出帧号/视角/外观；无法确认就说 unknown。'],
            does_not_ask=['路径是否有解', '是否满足实际执行停车', '推断不可见的车位/法规'],
            carpark_area_policy='如后来展示官方 carpark_area，仅为候选辅助，不是具体车位、空闲状态或停车许可真值；本补充未新增地图扫描。',
            existing_viewer=str(existing_viewer) if existing_viewer else None,
            existing_viewer_limit='仅 scene0012 有现成旧前视/旧节点审核页；只作浏览，不自动继承标签。')
        record_path = DESTINATION / f'{scene_name}.json'
        dump(record_path, record)
        page = f'''# {scene_name}：固定帧候选浏览

![左前／前／右前；从上到下为0、10、19帧]({sheet_path})

这里只浏览候选，尚未标注，也没有判定任何任务可解。请反馈“帧号、视角、目标/区域”；无法确认可填 unknown。

- 停车：是否看到具体单个车位，以及可见占用？
- 靠边：希望停在画面哪侧、哪块区域？
- 跟车：希望跟哪辆前车，是否歧义？

路径可解性、三维几何、车身和参考规划由我们检查。carpark_area 只能辅助找候选，不能充当具体车位真值。

[原图与时间/哈希索引]({record_path})。
'''
        if existing_viewer:
            page += f'\n[已有旧审核页]({existing_viewer})仅供浏览旧前视/节点，不是本候选的新标签。\n'
        (DESTINATION / f'{scene_name}.md').write_text(page)
        records.append(record)
        md_sections.append(f'- [{scene_name}：0/10/19 帧三前视]({DESTINATION / (scene_name + ".md")})')
        original_links = ' '.join(f'<a href="{html.escape(x["rgb"])}">f{x["frame_idx"]} {x["channel"]}</a>' for x in sources)
        html_sections.append(f'<section id="{scene_name}"><h2>{scene_name}</h2>'
            f'<img style="max-width:100%;height:auto" src="{sheet_path}" alt="固定0/10/19三前视候选">'
            f'<p>{original_links}</p><p>候选未审核，不代表任务可解。反馈帧号、视角和目标/区域即可。</p></section>')
    index = ('# 六场景固定帧候选浏览\n\n'
        '仅使用现有缓存，六场景均固定抽取 0/10/19 帧、左前/前/右前，不按成功结果挑图。'
        '这是独立补充候选，不改原 6 episode、不改任何冻结标签或分母。\n\n'
        '请只指认可见车位/停靠区域/前车；不知道就说 unknown。我们负责几何及独立可解性核验。\n\n' +
        '\n'.join(md_sections) + '\n')
    (DESTINATION / 'index.md').write_text(index)
    (DESTINATION / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8">'
        '<title>六场景候选浏览</title><body style="font-family:sans-serif;max-width:1600px;margin:auto">'
        '<h1>停车位／靠边／前车：六场景固定帧候选</h1>'
        '<p>只读浏览；每场景固定0/10/19帧，左前／前／右前。未创建任何人工结论或可解标签。'
        '请反馈场景、帧号、视角和可确认对象；unknown可以。路径可解性由我们核验。</p>' +
        ''.join(html_sections) + '</body></html>')
    for path, digest in original_files.items():
        assert sha(path) == digest
    size = sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file())
    assert size < 16 * 1024 ** 2 and time.monotonic() - started < 600
    sources = [Path(__file__), ROOT / 'scripts/navbench_stage1_20261124_candidate_supplement.py']
    for source in sources:
        target = run / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    summary = dict(created_utc=now(), seed=17, selection='all named six scenes; fixed frames 0,10,19; three front cameras',
        source_manifest=source_ref(MANIFEST), scenes=SCENES, frames=FRAMES, original_images=54,
        contact_sheets=6, output_bytes_before_manifest=size, time_budget_s=600,
        elapsed_s=round(time.monotonic() - started, 3), storage_budget_bytes=16*1024**2,
        model_calls=0, GPU_calls=0, planner_calls=0, new_raw_scan=False,
        original_frozen_package_file_hashes_unchanged=True, candidate_only=True,
        human_labels_created=0, feasible_labels_created=0, main_table_denominator_changed=False,
        files={str(p.relative_to(DESTINATION)):sha(p) for p in DESTINATION.rglob('*') if p.is_file()},
        source_sha256={str(p):sha(p) for p in sources}, source_frozen_manifest=source_ref(DATA/'stage1_closeout_manifest_v1.json'))
    dump(DESTINATION / 'manifest.json', summary)
    dump(run / 'candidate_preparation_summary.json', dict(summary, dataset=str(DESTINATION)))
    (REPORT / '六场景候选浏览补充.md').write_text(index + f'\n[只读浏览页]({DESTINATION/"index.html"})。\n\n'
        f'共 54 张既有原图、6 张 contact sheet，生成文件 {size:,} 字节；0 GPU/模型/规划。'
        '候选均未审核，不自动新增任务或改已有分母。\n')
    print(canonical(dict(passed=True, destination=str(DESTINATION), **summary)))


if __name__ == '__main__':
    main()
