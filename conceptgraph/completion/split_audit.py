"""划分审计只读传感器/位姿，不调用测试方法或读取模型成绩。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import OUTPUT, HISTORY, dump, digest, available_development


def dhash(path):
    with Image.open(path) as image:
        pixels=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=(pixels[:,1:]>pixels[:,:-1]).reshape(-1)
    return int(''.join('1' if b else '0' for b in bits),2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--reserved-inputs',default='reserved_inputs_001')
    parser.add_argument('--selection',default='reserved_test_scenes_v1.json')
    args=parser.parse_args()
    dest=OUTPUT/'splits'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    history=[]
    original={}
    forbidden_logs=set()
    for root in [HISTORY,HISTORY/'prospective_supplement_v2']:
        rows=json.loads((root/'scene_selection_manifest.json').read_text())['scenes']
        for r in rows:
            history.append(dict(r,root=str(root/'converted')))
            if r.get('original_split'):
                original[r['scene']]=r['original_split']
            if r['role']=='sealed_test' or r.get('original_split')=='test':
                forbidden_logs.add(r['log_token'])
    selection=OUTPUT/'splits'/args.selection
    reserved=json.loads(selection.read_text())['scenes']
    reserved_logs={r['log_token'] for r in reserved}
    assert not reserved_logs & {r['log_token'] for r in history}
    dev=available_development()
    eligible=[r for r in dev if r['log_token'] not in forbidden_logs|reserved_logs]
    logs=sorted({r['log_token'] for r in eligible},key=lambda log:hashlib.sha256(('seed17|'+log).encode()).hexdigest())
    validation_logs=set(logs[:2])
    roles={r['scene']:('validation' if r['log_token'] in validation_logs else 'train') for r in eligible}
    dump(dest/'train_validation_scenes.json',dict(scenes=[dict(r,role=roles[r['scene']]) for r in eligible],
        original11_unchanged=original,forbidden_historical_test_logs=sorted(forbidden_logs),
        reserved_test_logs=sorted(reserved_logs),excluded_development_scenes=[r for r in dev if r not in eligible],
        train_logs=sorted(set(logs)-validation_logs),validation_logs=sorted(validation_logs),
        expression_families_frozen=False,geographic_split_audited_below=True,
        rule='全量排除任何历史测试log；新训练和验证再按log隔离；不改变原11场景身份'))
    historical_sha={}
    historical_dhash=[]
    spatial=defaultdict(list)
    times=defaultdict(list)
    for r in history:
        scene=r['scene']
        source=Path(r['root'])/scene
        pose=np.loadtxt(source/'poses.txt').reshape(-1,4,4)
        spatial[r['strata']['location']].extend(pose[:,:2,3])
        alignment=json.loads((source/'frame_alignment.json').read_text())
        times[r['strata']['location']].extend(row['timestamp']/1e6 for row in alignment)
        for p in sorted((source/'color').glob('*.jpg')):
            historical_sha.setdefault(digest(p),[]).append(str(p))
            historical_dhash.append((dhash(p),str(p)))
    trees={loc:cKDTree(points) for loc,points in spatial.items()}
    proximity=[]
    exact=[]
    approximate=[]
    reserved_root=OUTPUT/'data_manifest'/args.reserved_inputs/'converted'
    hashes=[]
    for r in reserved:
        source=reserved_root/r['scene']
        pose=np.loadtxt(source/'poses.txt').reshape(-1,4,4)
        distance,_=trees[r['strata']['location']].query(pose[:,:2,3])
        alignment=json.loads((source/'frame_alignment.json').read_text())
        timestamp=np.asarray([f['timestamp']/1e6 for f in alignment])
        oldtime=np.asarray(times[r['strata']['location']])
        proximity.append(dict(scene=r['scene'],log_token=r['log_token'],location=r['strata']['location'],
            min_distance_to_any_historical_camera_m=float(distance.min()),
            frames_within_20m=int((distance<20).sum()),frame_count=len(pose),
            min_time_separation_s=float(np.min(np.abs(timestamp[:,None]-oldtime[None,:])))))
        for p in sorted((source/'color').glob('*.jpg')):
            h=digest(p)
            if h in historical_sha:
                exact.append(dict(input=str(p),historical=historical_sha[h],sha256=h))
            d=dhash(p)
            nearest=min(((d^old).bit_count(),path) for old,path in historical_dhash)
            if nearest[0]<=3:
                approximate.append(dict(input=str(p),historical=nearest[1],dhash_hamming=nearest[0],
                    status='自动近似全图候选，非已确认泄漏；需内容核验'))
            hashes.append(dict(path=str(p),sha256=h,dhash64=f'{d:016x}'))
    dump(dest/'report.json',dict(history_scenes=len(history),reserved_scenes=len(reserved),reserved_logs=len(reserved_logs),
        exact_rgb_duplicates=exact,approximate_full_image_candidates=approximate,geographic_temporal_proximity=proximity,
        input_sha256_manifest=hashes,geographic_independence_proven=False,
        nearby_road_policy='按同城历史相机轨迹距离记录；log隔离不等于地理路线隔离，主表须披露并分组',
        crop_approximate_audit='未完成：新测试尚未推理；须在隔离标注/冻结评测中核验',
        expression_family_audit='尚未建立新训练和测试指令，不能宣称无表达泄漏',
        source_sha256=digest(selection),test_opened=False,goal_complete=False))
    print(json.dumps(dict(reserved_scenes=len(reserved),reserved_logs=len(reserved_logs),
        exact_rgb_duplicates=len(exact),approximate_candidates=len(approximate),
        train_scenes=sum(v=='train' for v in roles.values()),validation_scenes=sum(v=='validation' for v in roles.values()),
        training_logs=len(logs)-len(validation_logs),validation_logs=len(validation_logs)),ensure_ascii=False))


if __name__=='__main__':
    main()
