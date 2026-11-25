"""开发集官方运动属性隔离标签，供评价，禁止传入在线推理。"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, HISTORY, dump, digest, available_development
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    out=OUTPUT/'annotation'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    scenes={r['scene'] for r in available_development()}
    frames=[]
    for root in [HISTORY,HISTORY/'prospective_supplement_v2']:
        frames.extend(r for r in map(json.loads,(root/'ground_truth_frames.jsonl').open()) if r['scene'] in scenes)
    required={o['annotation_token'] for f in frames for o in f['objects']}
    meta=PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    attributes={r['token']:r['name'] for r in iter_records(meta/'attribute.json')}
    labels={}
    static={'vehicle.parked','vehicle.stopped','pedestrian.standing','pedestrian.sitting_lying_down','cycle.without_rider'}
    dynamic={'vehicle.moving','pedestrian.moving'}
    for r in iter_records(meta/'sample_annotation.json'):
        if r['token'] not in required:
            continue
        names=[attributes[t] for t in r['attribute_tokens']]
        motion='dynamic' if set(names)&dynamic else 'static' if set(names)&static else 'unknown'
        labels[r['token']]=dict(official_attributes=names,official_motion=motion,
            motion_source='official_annotation_attributes',color='unknown',human_verified=False)
    counts=Counter()
    with (out/'evaluation_only_motion.jsonl').open('x') as f:
        for frame in frames:
            row=dict(frame)
            row['objects']=[dict(o,**labels[o['annotation_token']]) for o in frame['objects']]
            for o in row['objects']:
                counts[o['official_motion']]+=1
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(out/'manifest.json',dict(scenes=len(scenes),frames=len(frames),unique_annotations=len(required),
        matched_annotations=len(labels),object_observation_groups=dict(counts),
        source_sha256={str(p):digest(p) for p in [meta/'attribute.json',meta/'sample_annotation.json']},
        labels_sha256=digest(out/'evaluation_only_motion.jsonl'),ground_truth_in_inference=False,
        human_annotation_count=0,limitations='官方属性不是颜色标签；cycle.with_rider无可靠速度，保留unknown；静止不等于合法可接近'))


if __name__=='__main__':
    main()
