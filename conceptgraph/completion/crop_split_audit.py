"""封存前只读划分审计：官方框图像crop与实际训练crop，不调用待评测方法。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

from conceptgraph.completion.inspection import projected_box
from conceptgraph.completion.stage9 import OUTPUT, HISTORY, digest, dump


def crop_hashes(image):
    rgb=image.convert('RGB')
    gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    coefficients=dctn(gray,type=2,norm='ortho')[:8,:8].ravel()
    median=float(np.median(coefficients[1:]))
    bits=coefficients>median
    bits[0]=False
    small=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    difference=(small[:,1:]>small[:,:-1]).ravel()
    def integer(values):
        return int(''.join('1' if b else '0' for b in values),2)
    return dict(pixel_sha256=hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest(),
        dhash64=f'{integer(difference):016x}',phash64=f'{integer(bits):016x}',size=list(rgb.size),
        grayscale_std=float(gray.std()),mean_rgb=np.asarray(rgb.resize((1,1))).ravel().tolist())


def crop_box(obj,pose,intrinsic,image_size):
    projected=projected_box(obj,pose,intrinsic)
    if projected is None or not np.all(np.isfinite(projected)):
        return None
    x0,y0,x1,y1=projected
    width,height=image_size
    result=(max(0,int(np.floor(x0))),max(0,int(np.floor(y0))),
        min(width,int(np.ceil(x1))),min(height,int(np.ceil(y1))))
    return result if min(result[2]-result[0],result[3]-result[1])>=28 else None


def official_crop_records(frames,source_kind,counts,deadline):
    for frame in frames:
        if time.monotonic()>deadline:
            raise TimeoutError('crop split audit exceeded declared wall budget')
        with Image.open(frame['rgb']) as image:
            pose=np.asarray(frame['camera_pose'])
            intrinsic=np.asarray(frame['intrinsic'])
            counts[source_kind+'/frames']+=1
            for obj in frame['objects']:
                counts[source_kind+'/official_objects']+=1
                bbox=crop_box(obj,pose,intrinsic,image.size)
                if bbox is None:
                    counts[source_kind+'/unprojectable_or_small']+=1
                    continue
                record=dict(source_kind=source_kind,scene=frame['scene'],frame_idx=frame['frame_idx'],
                    rgb=frame['rgb'],instance_token=obj['instance_token'],category=obj['category'],
                    crop_xyxy=list(bbox),**crop_hashes(image.crop(bbox)))
                counts[source_kind+'/hashed_crops']+=1
                yield record


def historical_frames():
    for root in [HISTORY,HISTORY/'prospective_supplement_v2']:
        calibration={}
        for line in (root/'ground_truth_frames.jsonl').open():
            row=json.loads(line)
            scene=row['scene']
            if scene not in calibration:
                source=root/'converted'/scene
                poses=np.loadtxt(source/'poses.txt').reshape(-1,4,4)
                alignment=json.loads((source/'frame_alignment.json').read_text())
                calibration[scene]=(source,poses,alignment)
            source,poses,alignment=calibration[scene]
            index=row['frame_idx']
            yield dict(row,rgb=str(source/'color'/f'{index:06d}.jpg'),
                camera_pose=poses[index],intrinsic=alignment[index]['intrinsic'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wall-budget-s',type=float,default=600.)
    args=parser.parse_args()
    dest=OUTPUT/'splits'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    annotation=OUTPUT/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl'
    trainset=OUTPUT/'semantic_dataset/structured_supervision_002'
    dump(dest/'release.json',dict(source_sha256=digest(__file__),annotation_sha256=digest(annotation),
        dataset_manifest_sha256=digest(trainset/'manifest.json'),wall_budget_s=args.wall_budget_s,
        minimum_crop_side_px=28,approximate_threshold=dict(dhash_hamming=4,phash_hamming=4),
        methods_called=False,new_test_opened=False,source_immutability_required=True))
    counts=Counter()
    deadline=time.monotonic()+args.wall_budget_s
    reference=list(official_crop_records(historical_frames(),'historical_official',counts,deadline))
    for split in ['train','validation']:
        for line in (trainset/f'{split}.jsonl').open():
            row=json.loads(line)
            if row['task']=='object_attributes':
                path=Path(row['crop_path'])
                if digest(path)!=row['crop_sha256']:
                    raise ValueError('training/validation crop changed')
                with Image.open(path) as image:
                    reference.append(dict(source_kind='actual_'+split+'_crop',scene=row['scene_id'],
                        frame_idx=row['frame_idx'],instance_token=row['instance_token'],
                        category=row['response']['object_tag'],crop_path=str(path),**crop_hashes(image)))
                counts['actual_'+split+'_crop']+=1
    with (dest/'reference_hashes.jsonl').open('x') as handle:
        for row in reference:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    exact_index={}
    for i,row in enumerate(reference):
        exact_index.setdefault(row['pixel_sha256'],[]).append(i)
    dhashes=np.asarray([int(r['dhash64'],16) for r in reference],dtype=np.uint64)
    phashes=np.asarray([int(r['phash64'],16) for r in reference],dtype=np.uint64)
    bitcounts=np.asarray([i.bit_count() for i in range(256)],dtype=np.uint8)
    def hamming(values,value):
        return bitcounts[(values^np.uint64(int(value,16))).view(np.uint8).reshape(-1,8)].sum(axis=1)
    exact,approximate=[],[]
    scenes=set()
    with (dest/'reserved_hashes.jsonl').open('x') as handle:
        frames=(json.loads(line) for line in annotation.open())
        for row in official_crop_records(frames,'reserved_official',counts,deadline):
            scenes.add(row['scene'])
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            identity=dict(scene=row['scene'],frame_idx=row['frame_idx'],instance_token=row['instance_token'],
                rgb=row['rgb'],crop_xyxy=row['crop_xyxy'])
            matches=exact_index.get(row['pixel_sha256'],[])
            if matches:
                exact.append(dict(reserved=identity,reference_indices=matches,pixel_sha256=row['pixel_sha256']))
            dh,ph=hamming(dhashes,row['dhash64']),hamming(phashes,row['phash64'])
            indices=np.flatnonzero((dh<=4)&(ph<=4))
            for i in indices:
                approximate.append(dict(reserved=identity,reference_index=int(i),
                    dhash_hamming=int(dh[i]),phash_hamming=int(ph[i]),
                    low_texture=row['grayscale_std']<8 or reference[i]['grayscale_std']<8,
                    status='自动相似候选；尚无人审，不直接判为泄漏'))
    dump(dest/'report.json',dict(counts=dict(counts),reserved_scenes_with_projectable_crop=len(scenes),
        historical_scenes=len({r['scene'] for r in reference if r['source_kind']=='historical_official'}),
        exact_pixel_duplicates=exact,approximate_candidates=approximate,
        approximate_candidates_human_verified=0,all_candidate_leaks_resolved=not approximate and not exact,
        files={p.name:digest(p) for p in dest.glob('*.jsonl')},
        scope='官方框投影crop及实际训练/验证crop；未运行测试检测，不等于最终检测crop已全部核验',
        limitations=['稀疏、小于28像素或跨近裁面对象未纳入crop哈希，数量单列',
            '低纹理相似或同类外观不等于数据泄漏；若存在候选必须核对图像、身份和采集时空',
            'crop尺度和视角变化可能漏检，不能以哈希无相似候选证明全视觉独立'],
        methods_called=False,new_test_opened=False,goal_complete=False))
    print(json.dumps(dict(counts=dict(counts),exact=len(exact),approximate=len(approximate)),ensure_ascii=False))


if __name__=='__main__':
    main()
