"""真实RGB、框、mask的只读诊断入口，不生成身份标签或修改预测。"""
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
from PIL import Image, ImageDraw

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory


def main():
    root = PROJECT_ROOT
    manifest = json.loads((root/'dataset/processed/multiview_identity_repair_20261123_validation_v1/manifest.json').read_text())
    views = {(s['scene'], c, v['camera_timestamp_s']): (g['frame_idx'], v) for s in manifest['scenes'] for g in s['groups'] for c, v in g['views'].items()}
    diagnostic = root/'outputs/reports/multiview_identity_repair_20261123/validation/L4_crosslog_identity_v2_identity_diagnostics.json'
    data = json.loads(diagnostic.read_text())
    selected = [c for c in data['chains'] if c['GT_instance'] in ['39da398639e8474b85792a7127238ed0', 'e5d5e07287724eacb4103b9e94d3b828']]
    run = experiment_directory('nuscenes', 'validation', 'mvrepair_identity_visual_sources_20261123_v2', create=True)
    shutil.copy2(__file__, run/Path(__file__).name)
    records, images = [], {}
    for chain in selected:
        for method, seq in [('A', chain['A_chain']), ('L4', chain['candidate_chain'])]:
            for item in seq:
                for camera, timestamp, index in item['source']['keys']:
                    key = chain['scene'], camera, timestamp, index
                    source_frame, view = views[key[:3]]
                    if key not in images:
                        with gzip.open(view['detections'], 'rb') as stream:
                            det = pickle.load(stream)
                        image = Image.open(view['rgb']).convert('RGB')
                        box = np.asarray(det['xyxy'][index], float)
                        mask = np.asarray(det['mask'][index], bool)
                        assert mask.shape == (image.height, image.width)
                        tint = np.asarray(image).copy()
                        tint[mask] = (tint[mask]*.55 + np.array([255, 70, 30])*.45).astype(np.uint8)
                        tinted = Image.fromarray(tint)
                        ImageDraw.Draw(tinted).rectangle(tuple(box), outline=(0, 255, 120), width=4)
                        crop_box = (max(0, int(box[0])-24), max(0, int(box[1])-24), min(image.width, int(box[2])+24), min(image.height, int(box[3])+24))
                        crop = tinted.crop(crop_box)
                        crop.thumbnail((700, 500))
                        context = tinted.copy()
                        context.thumbnail((1000, 563))
                        slug = f'{chain["scene"]}_f{source_frame:02}_{camera}_d{index}'
                        crop_path, context_path = run/(slug+'_crop.jpg'), run/(slug+'_context.jpg')
                        crop.save(crop_path, quality=90)
                        context.save(context_path, quality=88)
                        images[key] = dict(source_frame=source_frame, camera=camera, timestamp_s=timestamp, detection_index=index,
                            RGB=view['rgb'], RGB_sha256=hashlib.sha256(Path(view['rgb']).read_bytes()).hexdigest(),
                            detections=view['detections'], class_name=det['classes'][int(det['class_id'][index])],
                            confidence=float(det['confidence'][index]), bbox_xyxy=box.tolist(), mask_pixels=int(mask.sum()),
                            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), crop=str(crop_path), context=str(context_path))
                    records.append(dict(GT_diagnostic_token=chain['GT_instance'], scene=chain['scene'], method=method,
                        prediction_frame=item['frame_idx'], node_id=item['node_id'],
                        provenance_status=item['source']['status'],
                        exact_output_source_verified=item['source']['status'] == 'explicit_front_position_source',
                        observation_center_exact_match_only=item['source']['status'] == 'exact_saved_observation_center_match',
                        registry_witness_does_not_prove_output_source=item['source']['status'].startswith('latest_registry'),
                        **images[key]))
    (run/'sources.json').write_text(json.dumps(dict(source=str(diagnostic), automatic_GT_pairing_only=True,
        human_identity_verified=False, records=records, no_new_model_mapping_calls=True), ensure_ascii=False, indent=2)+'\n')
    blocks = ['<!doctype html><meta charset="utf-8"><title>真实输入身份链诊断</title><style>body{max-width:1100px;margin:auto;font:18px sans-serif}img{max-width:100%}article{border-top:1px solid #aaa;margin-top:25px}code{overflow-wrap:anywhere}</style>',
        '<h1>真实RGB／检测框／mask诊断入口 v2</h1><p>橙色为真实检测mask，绿色为真实检测框；图未新增对象。GT token仅用于自动3m配对诊断，不代表人工确认身份。registry见证候选只能证明曾参与该ID的见证，不证明当帧输出精确来自这一路。精确输出归属须另核深拷贝状态和关联矩阵；未使用未来帧。</p>']
    for rec in records:
        title = f'{rec["scene"]} {rec["method"]} 预测f{rec["prediction_frame"]} ID={rec["node_id"]}；待核图f{rec["source_frame"]} {rec["camera"]} 检测{rec["detection_index"]}'
        blocks += [f'<article><h2>{html.escape(title)}</h2><p>来源证据级别：{rec["provenance_status"]}；精确输出来源已证：{rec["exact_output_source_verified"]}。registry候选不得作实际归属结论。</p><p>{rec["class_name"]} conf={rec["confidence"]:.3f}；box={rec["bbox_xyxy"]}</p>',
            f'<a href="{html.escape(rec["RGB"])}">完整原图</a><p><code>{html.escape(rec["RGB"])}</code></p>',
            f'<img src="{Path(rec["context"]).name}"><img src="{Path(rec["crop"]).name}"></article>']
    (run/'index.html').write_text('\n'.join(blocks))
    report = root/'outputs/reports/multiview_identity_repair_20261123/validation'
    (report/'身份链实际输入入口_v2.md').write_text(f'# 身份链实际输入入口 v2\n\n[离线图框mask入口]({run}/index.html)\n\n[完整来源JSON]({run}/sources.json)\n\n共{len(images)}个真实source、{len(records)}个引用。橙色mask和绿色box直接取缓存，不新增模型运行。3m自动配对只是诊断，未人工确认身份。registry见证候选不是输出精确来源，f12–14须与main的深拷贝状态/关联矩阵核对；A旧输出缺失精确来源的帧没有猜测补全。逐帧已保存JSON只向过去搜索，无未来帧来源。旧v1的source标题过强，保留文件并以本v2明确限制。\n')
    print(json.dumps(dict(index=str(run/'index.html'), sources=len(images), references=len(records), human_identity_verified=False)))


if __name__ == '__main__':
    main()
