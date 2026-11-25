"""只核查冻结R1与原选择任务的时间/身份及字段接口，不运行新选择实验。"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from diagnose_selection_validation import RETRIEVAL, dump, jsonl, read, sha  # noqa: E402

R1 = PROJECT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    query_keys = {(r['scene'],r['frame_idx']) for r in jsonl(RETRIEVAL/'queries.jsonl')}
    original = {}
    release = read(RETRIEVAL/'release.json')
    for filename in release['source_frames_sha256']:
        for frame in jsonl(filename):
            key=(frame['scene'],frame['frame_idx'])
            if key in query_keys:
                original[key]=frame
    fields = Counter()
    fields_sparse, fields_dense = Counter(),Counter()
    counts = Counter()
    mismatches=[]
    fingerprints = {}
    required = ('node_id','category','center','extent','num_detections','motion_state',
                'timestamp_s','depth_evidence','velocity_mps','last_seen_frame','feature')
    for path in sorted(R1.glob('scene-*/frames.jsonl.gz')):
        fingerprints[str(path)] = sha(path)
        with gzip.open(path,'rt') as handle:
            for line in handle:
                frame=json.loads(line)
                key=(frame['scene'],frame['frame_idx'])
                if key not in query_keys:
                    continue
                old=original[key]
                counts['queried_frames_present']+=1
                same_time=frame['camera_timestamp_s']==old['camera_timestamp_s']
                same_rgb=Path(frame['input']['rgb']).resolve()==Path(old['rgb']).resolve()
                same_truth=frame['evaluation_only_official']==old['evaluation_only_official']
                counts['exact_timestamp']+=same_time
                counts['same_rgb']+=same_rgb
                counts['same_official_truth']+=same_truth
                if not (same_time and same_rgb and same_truth):
                    mismatches.append(dict(scene=key[0],frame_idx=key[1],
                        same_time=same_time,same_rgb=same_rgb,same_truth=same_truth))
                for prediction in frame['predictions']['R1']:
                    sparse=prediction.get('position_source')=='confirmed_sparse_lidar'
                    counts['sparse_predictions' if sparse else 'dense_predictions']+=1
                    for field in required:
                        present=field in prediction
                        fields[field]+=present
                        (fields_sparse if sparse else fields_dense)[field]+=present
                    counts['dense_depth_nonempty']+=bool(not sparse and prediction.get('depth_evidence'))
                    counts['dense_motion_static']+=bool(not sparse and prediction.get('motion_state')=='static')
    dump(args.destination/'r1_interface_audit.json',dict(
        query_frames=len(query_keys),counts=counts,fields_present=fields,
        sparse_fields_present=fields_sparse,dense_fields_present=fields_dense,
        mismatches=mismatches,source_frames_sha256=fingerprints,
        original_queries_sha256=sha(RETRIEVAL/'queries.jsonl'),
        new_selection_experiment_run=False,gt_used_to_create_candidates=False,
        feasibility='原查询、位姿、GT可按相同scene/frame/timestamp绑定；稠密节点可直接适配；稀疏节点缺运动/形状/深度证据，须保留unknown且不授予授权，或同配置回放补出真实元数据',
        feature_limit='保存输出未导出CLIP特征；规则选择不需要，CLIP或裁剪/图像候选对照需要从真实mapper/检测绑定补导出，不能用官方ID回填'))
    print(json.dumps(dict(query_frames=len(query_keys),counts=counts,
        sparse_fields_present=fields_sparse,dense_fields_present=fields_dense,mismatches=len(mismatches)),ensure_ascii=False))


if __name__=='__main__':
    main()
