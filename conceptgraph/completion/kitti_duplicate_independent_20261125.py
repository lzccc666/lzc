"""只读复核保存匹配索引与重复代理观察账本，不调用matcher/模型。"""
from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/results/kitti/validation/iterations/kitti_local_all_20261125_tracking_v1'
REPORT=ROOT/'outputs/reports/kitti_local_all_20261125'


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    tick=time.monotonic();summary=json.loads((REPORT/'duplicate_proxy_diagnosis_v1.json').read_text())
    occurrences=list(rows(REPORT/'duplicate_proxy_occurrences_v1.jsonl'))
    indexed={(r['sequence'],r['frame_idx'],r['method'],r['prediction_index']):r for r in occurrences}
    assert len(indexed)==len(occurrences)
    inputs=list(rows(ROOT/'dataset/processed/kitti_local_all_20261125_v1/method_frames.jsonl'))
    seen=defaultdict(set);counts=defaultdict(Counter);unique=defaultdict(set);checked=0;frames=0;total_records=0
    for frame,evaluation,method_input in zip_longest(rows(RUN/'mapping_frames.jsonl.gz'),rows(RUN/'evaluation_frames.jsonl.gz'),inputs):
        assert frame is not None and evaluation is not None and method_input is not None
        assert frame['sequence']==evaluation['sequence']==method_input['sequence']
        assert frame['original_frame_id']==evaluation['original_frame_id']==method_input['original_frame_id']
        assert frame['frame_idx']==evaluation['frame_idx']
        assert frame['frame_idx']==method_input['frame_idx']
        frames+=1;truth=evaluation['official_truth']
        for name,predictions in frame['predictions'].items():
            saved=evaluation['metrics'][name];matches=saved['matches']
            matched={r['prediction_index'] for r in matches}
            assert len(matched)==len(matches)
            truth_indices={r['truth_index'] for r in matches};assert len(truth_indices)==len(matches)
            for pair in matches:
                prediction=predictions[pair['prediction_index']];target=truth[pair['truth_index']]
                assert prediction['category']==target['category']
                assert math.dist(prediction['center'][:2],target['center'][:2])<=3.
            duplicate_count=0;history=seen[frame['sequence'],name]
            for index,prediction in enumerate(predictions):
                neighbors=[(math.dist(prediction['center'][:2],gt['center'][:2]),j) for j,gt in enumerate(truth)
                    if prediction['category']==gt['category'] and math.dist(prediction['center'][:2],gt['center'][:2])<=3.]
                key=(frame['sequence'],frame['frame_idx'],name,index)
                if index in matched or not neighbors:
                    assert key not in indexed
                    continue
                duplicate_count+=1;row=indexed[key];total_records+=1
                assert row['node_id']==prediction['node_id'] and row['last_seen_frame']==prediction.get('last_seen_frame')
                assert row['original_frame_id']==frame['original_frame_id']
                distance,j=min(neighbors)
                assert row['nearest_official_instance']==truth[j]['instance_token']
                assert abs(row['nearest_center_distance_xy_m']-distance)<1e-9
                assert row['same_category_GT_within3m']==len(neighbors)
                last=prediction.get('last_seen_frame')
                update='stale_explicit_last_seen' if isinstance(last,int) and last<frame['frame_idx'] else 'current_update' if last==frame['frame_idx'] else 'unknown_update_time'
                emission='previously_exported_ID' if prediction['node_id'] in history else 'first_exported_ID_not_proven_new_creation'
                assert row['node_update_state']==update and row['export_history']==emission
                counts[name].update(duplicate_occurrences=1);counts[name][update]+=1;counts[name][emission]+=1
                unique[name].add((frame['sequence'],prediction['node_id']))
                checked+=1
            assert duplicate_count==saved['duplicate_nodes']
            counts[name].update(prediction_node_frames=len(predictions),frames=1)
            history.update(p['node_id'] for p in frame['all_predictions'][name])
    assert frames==1030 and total_records==len(occurrences)
    for name,count in counts.items():
        assert all(summary['totals'][name][key]==value for key,value in count.items())
        assert summary['totals'][name]['unique_exported_IDs_with_duplicate_proxy']==len(unique[name])
    result=dict(passed=True,frames=frames,duplicate_occurrences=checked,methods={k:dict(v) for k,v in counts.items()},
        saved_match_indices_valid=True,last_seen_unit='local per-sequence mapping frame_idx; not original sensor frame number',
        history='all_predictions ledger from strictly previous frames; includes full exported graph, not only currently visible filtered predictions',
        meaning='unmatched same-class center within3m of a GT is a duplicate proxy; no human physical-duplicate truth',
        limits=['First exported ID does not establish creation time.','Current-update flag does not prove correct/wrong association.','Integer maximum-cardinality scores already independently checked; this replay checks saved matching indices, not a new optimizer.'],
        new_models_or_matcher_calls=0,elapsed_s=time.monotonic()-tick,audit_source_sha256=sha(__file__),
        diagnosis_sha256=sha(REPORT/'duplicate_proxy_diagnosis_v1.json'),occurrences_sha256=sha(REPORT/'duplicate_proxy_occurrences_v1.jsonl'))
    target=ROOT/'outputs/reports/local_scope_audit_20261125/tracking_duplicate_independent_v1.json'
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
