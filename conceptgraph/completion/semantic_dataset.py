"""独立官方类别裁剪与显式构造的指令监督；不复用旧模型排序或颜色弱标签。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image

from conceptgraph.completion.inspection import projected_box, box_iou
from conceptgraph.completion.semantic_contract import build_prompt, validate
from conceptgraph.completion.stage9 import OUTPUT, dump, digest

CATEGORIES = dict(car='小汽车', truck='卡车', bus='公交车', bicycle='自行车',
                  motorcycle='摩托车', pedestrian='行人')
COLORS = dict(red='红色', white='白色', blue='蓝色', black='黑色', gray='灰色', green='绿色')
RELATIONS = dict(left_of='左侧', right_of='右侧', in_front_of='前方', behind='后方', near='附近')
FAMILIES = {
    'train': ['前往{target}附近', '去{anchor}{rel}的{target}附近',
              '到最近的{target}旁边', '不要{exclude}，去{target}附近',
              '靠近停着的{target}', '去{anchor}{rel}那辆{target}旁边'],
    'validation': ['请寻找{target}并靠近', '以{anchor}为参照，寻找其{rel}的{target}',
                   '距离当前位置最短的{target}是我要靠近的对象',
                   '排除{exclude}，寻找{target}并靠近', '寻找静止的{target}并靠近'],
    'reserved_test': ['目标是在{anchor}{rel}的{target}，请接近它',
                      '将车辆驶向{target}旁的合法区域',
                      '观察范围内若有{target}，请安全接近',
                      '本次接近{target}，而不是{exclude}'],
}


def query_response(target, color=None, anchor=None, relation=None, nearest=False,
                   excluded=None, static=False):
    text = (COLORS.get(color, '') + CATEGORIES[target])
    return dict(schema_version='1.0.0', target_text=text, target_categories=[target],
        target_colors=[color] if color else [], anchor_text=CATEGORIES[anchor] if anchor else None,
        anchor_categories=[anchor] if anchor else [], anchor_colors=[], relation=relation,
        prefer_nearest=nearest, prefer_front=False,
        excluded_categories=[excluded] if excluded else [], excluded_colors=[],
        requires_static=static, confidence=1.0)


def make_queries(role, count, scenes):
    """文本指令的生成器即标注规范，不把其颜色请求当作图像颜色真值。"""
    options=[]
    for target in CATEGORIES:
        for color in [None, *COLORS]:
            for family, pattern in enumerate(FAMILIES[role]):
                anchors = list(CATEGORIES) if '{anchor}' in pattern else [None]
                relations = list(RELATIONS) if '{rel}' in pattern else [None]
                excludes = [c for c in CATEGORIES if c != target] if '{exclude}' in pattern else [None]
                for anchor in anchors:
                    for relation in relations:
                        for excluded in excludes:
                            response=query_response(target,color,anchor,relation,
                                nearest=('最近' in pattern or '最短' in pattern),
                                excluded=excluded, static=('停着' in pattern or '静止' in pattern))
                            query=pattern.format(target=response['target_text'],
                                anchor=CATEGORIES.get(anchor,''),rel=RELATIONS.get(relation,''),
                                exclude=CATEGORIES.get(excluded,''))
                            options.append((family,query,response))
    rng=random.Random(17 if role=='train' else 29)
    rng.shuffle(options)
    rows=[]
    for i,(family,query,response) in enumerate(options[:count]):
        scope=scenes[i%len(scenes)]
        rows.append(dict(task='query_parse',sample_id=f'{role}-parse-{i:05d}',split=role,
            scene_id=scope['scene'], log_token=scope['log_token'], query_text=query,
            expression_family=f'{role}-{family}', response=response,
            supervision_source='声明语法生成的结构化文本监督，非人工自然指令',
            visual_grounding_claim=False, scene_count_credit=False))
    return rows


def perceptual_hash(image):
    a=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    return int(''.join('1' if x else '0' for x in (a[:,1:]>a[:,:-1]).ravel()),2)


def audit_rows(rows):
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('duplicate sample IDs')
    logs=defaultdict(set)
    texts=defaultdict(set)
    families=defaultdict(set)
    for r in rows:
        logs[r['split']].add(r['log_token'])
        if r['task']=='query_parse':
            texts[r['split']].add(r['query_text'])
            families[r['split']].add(r['expression_family'])
        v=validate(r['task'],r['response'],allowed_candidate_ids=r.get('candidate_ids'))
        if not v.valid:
            raise ValueError(f"invalid label: {r['sample_id']}: {v.validation_errors}")
    for collection in [logs,texts,families]:
        if collection['train'] & collection['validation']:
            raise ValueError('train/validation leakage')
    return dict(counts=dict(Counter(r['split']+'/'+r['task'] for r in rows)),
        independent_logs={k:sorted(v) for k,v in logs.items()},
        unique_query_texts={k:len(v) for k,v in texts.items()},
        expression_families={k:sorted(v) for k,v in families.items()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--split-run',default='split_audit_002')
    parser.add_argument('--mapping-run',default='robust_motion_full_001')
    parser.add_argument('--train-queries',type=int,default=1400)
    parser.add_argument('--validation-queries',type=int,default=180)
    args=parser.parse_args()
    dest=OUTPUT/'semantic_dataset'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    splitpath=OUTPUT/'splits'/args.split_run/'train_validation_scenes.json'
    split=json.loads(splitpath.read_text())
    scenes={r['scene']:r for r in split['scenes']}
    rows=[]
    for role,count in [('train',args.train_queries),('validation',args.validation_queries)]:
        rows.extend(make_queries(role,count,[s for s in scenes.values() if s['role']==role]))
    crop_manifest=[]
    instances=Counter()
    pixel_roles={}
    rejected=Counter()
    for framepath in sorted((OUTPUT/'mapping_evaluation'/args.mapping_run).glob('scene-*/frames.jsonl')):
        scene=framepath.parent.name
        if scene not in scenes:
            continue
        meta=scenes[scene]
        for frame in map(json.loads,framepath.open()):
            if frame['frame_idx']%4:
                continue
            truth=frame['evaluation_only_official']
            boxes=[projected_box(o,np.asarray(frame['camera_pose']),np.asarray(frame['intrinsic'])) for o in truth]
            with gzip.open(frame['detections'],'rb') as f:
                gobs=pickle.load(f)
            for i,crop in enumerate(gobs.get('image_crops',[])):
                if not isinstance(crop,Image.Image):
                    crop=Image.fromarray(np.asarray(crop,dtype='uint8'))
                if min(crop.size)<28:
                    rejected['small_crop']+=1
                    continue
                scores=sorted([(box_iou(gobs['xyxy'][i],box),j) for j,box in enumerate(boxes) if box is not None],reverse=True)
                if not scores or scores[0][0]<.5 or (len(scores)>1 and scores[0][0]-scores[1][0]<.2):
                    rejected['weak_or_ambiguous_projection']+=1
                    continue
                score,j=scores[0]
                obj=truth[j]
                key=obj['instance_token']
                if instances[key]>=2:
                    rejected['instance_budget']+=1
                    continue
                pixels=hashlib.sha256(crop.convert('RGB').tobytes()+str(crop.size).encode()).hexdigest()
                if pixels in pixel_roles:
                    rejected['duplicate_crop_pixels']+=1
                    continue
                sample=f"{scene}-f{frame['frame_idx']:03d}-d{i:03d}"
                path=dest/'crops'/f'{sample}.png'
                path.parent.mkdir(exist_ok=True)
                crop.save(path)
                instances[key]+=1
                pixel_roles[pixels]=meta['role']
                detection=gobs['classes'][int(gobs['class_id'][i])]
                row=dict(task='object_attributes',sample_id=sample,split=meta['role'],scene_id=scene,
                    log_token=meta['log_token'],crop_path=str(path),
                    metadata=dict(detection_label=detection,motion_state='unknown',observation_count=1),
                    response=dict(schema_version='1.0.0',summary=obj['category'],object_tag=obj['category'],
                        possible_tags=[obj['category']],colors=['unknown'],body_type='unknown',
                        motion_state='unknown',confidence=1.0),
                    supervised_metrics=['object_tag','possible_tags'],
                    unknown_fields=['colors','body_type','motion_state'],
                    annotation_token=obj['annotation_token'],instance_token=key,
                    supervision_source='nuScenes官方类别，当前相机三维框投影与检测框唯一IoU匹配；非人工',
                    projection_iou=score,source_frame=str(framepath),source_frame_sha256=digest(framepath),
                    source_detection_sha256=digest(frame['detections']),crop_sha256=digest(path),
                    crop_pixel_sha256=pixels,frame_idx=frame['frame_idx'])
                rows.append(row)
                crop_manifest.append(dict(path=str(path),split=meta['role'],sha256=digest(path),
                    pixel_sha256=pixels,dhash64=f'{perceptual_hash(crop):016x}',instance_token=key))
    audit=audit_rows(rows)
    approximate=[]
    train=[r for r in crop_manifest if r['split']=='train']
    val=[r for r in crop_manifest if r['split']=='validation']
    for a in val:
        for b in train:
            h=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count()
            if h<=3:
                approximate.append(dict(validation=a['path'],train=b['path'],hamming=h,status='需内容核验，非确认泄漏'))
    for role in ['train','validation']:
        with (dest/f'{role}.jsonl').open('x') as f:
            for row in rows:
                if row['split']==role:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(dest/'expression_families.json',FAMILIES)
    dump(dest/'manifest.json',dict(audit=audit,split_sha256=digest(splitpath),
        files={p.name:digest(p) for p in dest.glob('*.jsonl')},
        source_sha256=digest(Path(__file__)),prompt_source_sha256=digest(Path(__file__).with_name('semantic_contract.py')),
        crops=crop_manifest,rejected=dict(rejected),unique_official_instances=len(instances),
        approximate_cross_split_crop_candidates=approximate,
        source='只用允许训练/验证log；原11身份不改；不读历史测试/预留测试标签',
        independent_new_test_used=False,manual_review_count=0,
        limitations=['文本监督由声明语法构造，非自然指令分布',
            '颜色与细属性没有可靠标签，不计算其准确率；unknown不是人工颜色真值',
            '当前训练版尚无可信安全候选排序样本，排序训练未完成',
            '图像类别对应是独立官方投影自动匹配，仍需人工抽查'],
        train_row_count=sum(r['split']=='train' for r in rows),
        validation_row_count=sum(r['split']=='validation' for r in rows),
        prompt_hashes={task:hashlib.sha256(build_prompt(task,next(r for r in rows if r['task']==task)).encode()).hexdigest()
            for task in sorted({r['task'] for r in rows})}))
    print(json.dumps(dict(audit=audit,unique_official_instances=len(instances),
        approximate_cross_split_crop_candidates=len(approximate)),ensure_ascii=False))


if __name__=='__main__':
    main()
