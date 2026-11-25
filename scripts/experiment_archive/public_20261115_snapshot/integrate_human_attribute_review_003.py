"""真实用户属性记录转为评价侧参考；不将标准答案喂给感知/语义输入。"""
from collections import Counter
import csv
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    imported=R/'annotation/offline_imports/user_lzc_20261116_001'
    effective=imported/'effective_attribute_records_001.jsonl'
    manifest=R/'annotation/quick_human_review_002/manifest.json'
    cards={(c['scene'],c['frame_idx'],c['detection_index']):c for c in json.loads(manifest.read_text())['cards']}
    reviews=[json.loads(line) for line in effective.read_text().splitlines()]
    out=R/'annotation/human_attribute_reference_003';out.mkdir(exist_ok=False)
    rows=[]
    for review in reviews:
        card=cards[(review['scene'],review['frame_idx'],review['detection_index'])]
        path=Path(card['source_evidence']);assert digest(path)==card['evidence_file_sha256']
        detection=json.loads(path.read_text())['detections'][review['detection_index']]
        assert digest(detection_path:=path.parent/detection['actual_clip_crop'])==card['actual_crop_sha256']
        supported=review['observed_category'] not in {'other','unknown'}
        rows.append(dict(sample_number=card['number'],scene=review['scene'],frame_idx=review['frame_idx'],
            detection_index=review['detection_index'],reviewer=review['reviewer'],
            user_submitted_at=review['submitted_at'],local_record_id=review['local_record_id'],
            predicted_category=detection['category'],human_category=review['observed_category'],
            human_color=review['observed_color'],human_body_type=review['body_type'],
            category_comparable=supported,category_agreement=(detection['category']==review['observed_category']) if supported else None,
            taxonomy_resolution_status='待统一厢式货车与主类别的定义；保留用户other' if review['observed_category']=='other' else ('看不清，保留未知' if review['observed_category']=='unknown' else '可按当前主类别比较'),
            note=review['note'],crop_file=str(detection_path),crop_sha256=card['actual_crop_sha256'],
            evidence_file=str(path),evidence_sha256=card['evidence_file_sha256'],
            explicit_user_chat_correction=review.get('category_correction_origin')=='explicit_user_chat_correction',
            not_3D_identity_ground_truth=True,not_natural_instruction=True))
    assert len(rows)==20 and len({r['sample_number'] for r in rows})==20
    with (out/'attributes.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'attributes.json',dict(rows=rows,automatic_labels_created=0,derived_from_real_user_reviews=True))
    comparable=[r for r in rows if r['category_comparable']]
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_user_records=38,unique_images=20,distinct_scenes=len({r['scene'] for r in rows}),
        effective_colors=dict(Counter(r['human_color'] for r in rows)),
        known_single_color_images=sum(r['human_color'] not in ['unknown','multiple'] for r in rows),
        multiple_color_images=sum(r['human_color']=='multiple' for r in rows),
        unknown_color_images=sum(r['human_color']=='unknown' for r in rows),
        fine_body_types=dict(Counter(r['human_body_type'] for r in rows)),
        category_comparable_images=len(comparable),category_agreements=sum(r['category_agreement'] for r in comparable),
        taxonomy_unresolved_images=sum(r['human_category']=='other' for r in rows),
        category_unknown_images=sum(r['human_category']=='unknown' for r in rows),
        color_accuracy=None,color_accuracy_reason='源证据中的color=unknown是待人工标注状态；不能把占位字段当模型颜色预测并计算准确率。',
        formal_generalization_metric=False,whole_project_sealed_test=False,
        sampling='20场景各一张，以预测车辆/行人及最大框选择的开发审核包；不是随机无偏抽样。',
        human_3D_identity_records=0,human_natural_instructions=0,independent_second_review=False,
        labels_added_to_inference_inputs=False,training_or_test_splits_changed=False,
        source_sha256={str(p):digest(p) for p in [Path(__file__),manifest,effective,imported/'original_user_export.json',imported/'user_chat_correction_001.json']},
        generated_files_sha256={p.name:digest(p) for p in [out/'attributes.csv',out/'attributes.json']},goal_complete=False)
    dump(out/'report.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
