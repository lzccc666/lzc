"""导入用户实际上传的离线自然表达；保存原文、观看历史及未核验状态。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from conceptgraph.completion.natural_collection import validate_collection
from review_server_with_natural_071 import STORAGE, collection_status, save_submission

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError('不接受非有限JSON数值：'+value)


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(source, run_id):
    source = Path(source)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('非法独立run-id')
    payload = json.loads(source.read_text(), parse_constant=reject_constant)
    assert payload['format'] == 'thesis_human_natural_offline_087'
    assert payload['campaign'] == 'human_natural_collection_001'
    assert isinstance(payload['records'], list) and payload['records']
    checked = [validate_collection(r, output=ROOT.parent) for r in payload['records']]
    assert all(r['campaign_sha256'] == payload['campaign_sha256'] for r in checked)
    identities = [(r['scene'], r['normalized_command']) for r in checked]
    assert len(set(identities)) == len(identities), '文件内部有重复指令，先保留原文件核查'
    before = collection_status()
    destination = ROOT/'annotation/offline_imports'/run_id
    destination.mkdir(exist_ok=False)
    raw = destination/'original_user_export.json'
    with raw.open('xb') as stream:
        stream.write(source.read_bytes())
    assert sha(raw) == sha(source)
    imported, duplicates = [], []
    existing = { (r['scene'], r['normalized_command']): r for p in STORAGE.glob('*.json')
                 for r in [json.loads(p.read_text())] }
    for index, (record, validation) in enumerate(zip(payload['records'], checked)):
        key = validation['scene'], validation['normalized_command']
        viewing = record.get('observed_frames_before_submission', [])
        if not isinstance(viewing, list):
            raise ValueError('观看历史必须为列表')
        future = [v for v in viewing if v.get('scene') == record['scene']
                  and isinstance(v.get('frame_idx'), int) and v['frame_idx'] > record['frame_idx']]
        provenance = dict(source_record_index=index, scene=record['scene'], frame_idx=record['frame_idx'],
            command_original=record['command'], note_original=record['note'], author=record['author'],
            camera_token=record['camera_token'], rgb_sha256=record['rgb_sha256'],
            author_target_point_normalized=record['author_target_point_normalized'],
            observed_frames_before_submission=viewing,
            author_viewed_frames_later_than_query_anchor=bool(future),
            maximum_viewed_frame_idx_in_same_scene=max((v['frame_idx'] for v in viewing
                if v.get('scene') == record['scene'] and isinstance(v.get('frame_idx'), int)), default=None),
            inference_must_not_receive_future_observations=True,
            temporal_protocol_review_required=bool(future),
            original_query_anchor_not_reassigned=True,
            original_user_export_sha256=sha(raw), original_user_export=str(raw),
            automatic_target_identity_assigned=False, qualifies_for_sealed_test=False,
            label_review_status='原始人工表达，尚待对象身份/运动/歧义/可达性与任务范围核验')
        if key in existing:
            duplicates.append(dict(provenance, submission_id=existing[key]['submission_id'],
                                   action='保留原队列记录，本文件作为来源补充，不重复计数'))
            continue
        result = save_submission(record)
        saved = Path(result['path'])
        stored = json.loads(saved.read_text())
        assert stored['command'] == record['command'].strip()
        assert stored['note'] == record['note'].strip()
        assert stored['target_identity'] is None and not stored['qualifies_for_sealed_test']
        imported.append(dict(provenance, submission_id=result['submission_id'], saved_record=str(saved),
                             saved_record_sha256=sha(saved)))
    after = collection_status()
    assert after['human_authored_submissions'] - before['human_authored_submissions'] == len(imported)
    write(destination/'source_binding_report.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        original_user_export_sha256=sha(raw), source_sha256=sha(Path(__file__)),
        input_record_count=len(checked), imported=imported, duplicates=duplicates,
        status_before=before, status_after=after, automatic_labels_created=0,
        independent_label_review_completed=False, new_sealed_test_results_created=False,
        old_attribute_review_records_modified=False, actual_new_GPU_calls=0,
        actual_new_CARLA_calls=0, goal_complete=False))
    print(json.dumps(dict(imported=len(imported), duplicates=len(duplicates),
        raw_records_total=after['human_authored_submissions'], qualifying_labels=0,
        future_author_viewing_requires_temporal_review=sum(r['author_viewed_frames_later_than_query_anchor'] for r in imported),
        output=str(destination), goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    main(args.source, args.run_id)
