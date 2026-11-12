"""同一真实前端的开发检索基线；稀疏CLIP网格是明确简化的VLMaps式适配。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
import tarfile
import time

import numpy as np
from PIL import Image
import torch

from conceptgraph.completion.depth import depth_support
from conceptgraph.completion.navigation_diagnostics import queries_for_frame, to_nodes
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.scenegraph.relations import build_relations

CATEGORIES={'car','truck','bus','bicycle','motorcycle','pedestrian'}
CHECKPOINT=PROJECT/'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors'


def query_english(text):
    """仅确定性解析中文文字，未读取正确解析标签；英文模板固定后共用。"""
    parsed=parse_navigation_query(text,8.)
    target=' '.join([*parsed.target_colors,*parsed.target_categories]) or parsed.target_text
    sentence='a '+('parked ' if parsed.requires_static else '')+target
    if parsed.relation and parsed.anchor_text:
        anchor=parse_navigation_query('去'+parsed.anchor_text+'附近',8.)
        name=' '.join([*parsed.anchor_colors,*anchor.target_categories]) or parsed.anchor_text
        relation={'left_of':'to the left of','right_of':'to the right of','in_front_of':'in front of','behind':'behind','near':'near'}[parsed.relation]
        sentence+=' '+relation+' a '+name
    return sentence,parsed


def observation_objects(depth,intrinsic,pose,gobs):
    """输入只允许当前传感器、标定、位姿和冻结前端；没有对象GT。"""
    detections=[]
    for index,mask in enumerate(gobs['mask']):
        category=gobs['classes'][int(gobs['class_id'][index])]
        if category not in CATEGORIES:
            continue
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=depth.shape:
            raise ValueError('front-end mask/depth size mismatch')
        valid,evidence=depth_support(depth,mask,robust=True)
        if not evidence['sufficient_support']:
            continue
        v,u=np.nonzero(valid)
        z=depth[v,u]
        camera=np.column_stack([(u-intrinsic[0,2])*z/intrinsic[0,0],
            (v-intrinsic[1,2])*z/intrinsic[1,1],z])
        world=camera@pose[:3,:3].T+pose[:3,3]
        feature=np.asarray(gobs['image_feats'][index],dtype=np.float32)
        if not np.isfinite(feature).all() or np.linalg.norm(feature)<1e-6:
            continue
        feature/=np.linalg.norm(feature)
        detections.append(dict(category=category,center=np.median(world,axis=0),
            feature=feature,points=world,source_detection_index=index,depth_evidence=evidence))
    return detections


class SparseFeatureGrid:
    def __init__(self,resolution=.5,ttl_s=5.):
        self.resolution,self.ttl_s=resolution,ttl_s
        self.cells={}
        self.last_timestamp=None

    def update(self,detections,timestamp):
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp<=self.last_timestamp):
            raise ValueError('sensor timestamps must strictly increase in seconds')
        self.last_timestamp=timestamp
        for detection in detections:
            keys=np.floor(detection['points'][:,:2]/self.resolution).astype(int)
            unique=np.unique(keys,axis=0)
            for raw in unique:
                key=tuple(map(int,raw))
                point=np.median(detection['points'][np.all(keys==raw,axis=1)],axis=0)
                if key not in self.cells:
                    self.cells[key]=dict(feature=detection['feature'].copy(),center=point,count=1,
                        classes=Counter([detection['category']]),timestamp=timestamp)
                else:
                    cell=self.cells[key]
                    n=cell['count']
                    cell['feature']=(cell['feature']*n+detection['feature'])/(n+1)
                    cell['center']=(cell['center']*n+point)/(n+1)
                    cell['count']=n+1
                    cell['classes'][detection['category']]+=1
                    cell['timestamp']=timestamp
        self.cells={k:v for k,v in self.cells.items() if timestamp-v['timestamp']<=self.ttl_s}

    def visible(self,pose,intrinsic,width,height):
        rows=[]
        for key,cell in sorted(self.cells.items()):
            xyz=pose[:3,:3].T@(cell['center']-pose[:3,3])
            if not 1<xyz[2]<65:
                continue
            uv=intrinsic@xyz
            if 0<=uv[0]/uv[2]<width and 0<=uv[1]/uv[2]<height:
                rows.append(dict(category=cell['classes'].most_common(1)[0][0],center=cell['center'],
                    feature=cell['feature']/max(np.linalg.norm(cell['feature']),1e-6),grid_key=key,
                    observations=cell['count']))
        return rows


def associated_identity(candidate,truth):
    if candidate is None:
        return None,'no_selection'
    eligible=[(float(np.linalg.norm(candidate['center'][:2]-np.asarray(obj['center'][:2]))),obj['instance_token'])
        for obj in truth if obj['category']==candidate['category']]
    eligible=sorted((d,i) for d,i in eligible if d<=3.)
    if not eligible:
        return None,'outside_3m_or_category_unmatched'
    if len(eligible)>1 and eligible[1][0]-eligible[0][0]<.5:
        return None,'ambiguous_independent_association'
    return eligible[0][1],'associated'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--mapping-run',default='robust_motion_full_001')
    parser.add_argument('--scene',action='append')
    parser.add_argument('--wall-budget-s',type=float,default=900.)
    args=parser.parse_args()
    started=time.monotonic()
    source=OUTPUT/'mapping_evaluation'/args.mapping_run
    paths=[p for p in sorted(source.glob('scene-*/frames.jsonl')) if not args.scene or p.parent.name in args.scene]
    if not paths:
        raise ValueError('no development source')
    labelpath=OUTPUT/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
    labels={(r['scene'],r['frame_idx']):{o['annotation_token']:o['official_motion'] for o in r['objects']}
        for r in map(json.loads,labelpath.open())}
    dest=OUTPUT/'main_experiments'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'release.json',dict(mapping_run=args.mapping_run,source_frames_sha256={str(p):digest(p) for p in paths},
        checkpoint_sha256=digest(CHECKPOINT),source_sha256=digest(__file__),wall_budget_s=args.wall_budget_s,
        grid_resolution_m=.5,grid_observation_ttl_s=5.,query_translation='fixed rule-derived English; no oracle parse',
        evaluation_association='same detected class and <=3m ground center; nearest-two margin >=0.5m else unknown',
        clip_threshold=None,no_match_rejection='no eligible observation only; no tuned similarity threshold',
        external_source='https://arxiv.org/abs/2210.05714',
        adaptation='VLMaps式简化：实际crop CLIP特征+稀疏LiDAR+0.5m二维格；未复现LSeg密集像素特征、完整三维地图或原规划器',
        frontend_shared=True,inference_uses_gt=False,independent_test=False,performance_credit=False))
    with tarfile.open(dest/'source_snapshot.tar.gz','x:gz') as archive:
        for name in [Path(__file__),PROJECT/'conceptgraph/completion/depth.py',
                     PROJECT/'conceptgraph/navigation/semantic.py',PROJECT/'conceptgraph/completion/navigation_diagnostics.py']:
            archive.add(name,arcname=str(name.relative_to(PROJECT)))
    # 先收集开发指令文字；标签只用于评价，编码器仅接收固定翻译后的字符串。
    texts=set()
    for path in paths:
        for frame in map(json.loads,path.open()):
            if frame['frame_idx']%4==0:
                texts.update(query_english(q['query_text'])[0] for q in queries_for_frame(frame,labels[(frame['scene'],frame['frame_idx'])]))
    import open_clip
    model,_,_=open_clip.create_model_and_transforms('ViT-H-14',pretrained=str(CHECKPOINT),device='cpu')
    model.eval()
    tokenizer=open_clip.get_tokenizer('ViT-H-14')
    embeddings={}
    ordered=sorted(texts)
    with torch.inference_mode():
        for offset in range(0,len(ordered),16):
            batch=ordered[offset:offset+16]
            feature=model.encode_text(tokenizer(batch)).float()
            feature/=feature.norm(dim=-1,keepdim=True)
            embeddings.update(zip(batch,feature.cpu().numpy()))
    del model
    np.savez_compressed(dest/'actual_clip_text_embeddings.npz',texts=np.asarray(ordered),features=np.stack([embeddings[t] for t in ordered]))
    counts=defaultdict(Counter)
    scene_counts=defaultdict(lambda:defaultdict(Counter))
    number_frames=0
    with (dest/'queries.jsonl').open('x') as handle:
        for path in paths:
            grid=SparseFeatureGrid()
            for frame in map(json.loads,path.open()):
                if time.monotonic()-started>args.wall_budget_s:
                    raise TimeoutError('development baseline wall budget; partial raw results retained')
                pose=np.asarray(frame['camera_pose'])
                intrinsic=np.asarray(frame['intrinsic'])
                depth=np.asarray(Image.open(frame['depth']),dtype=float)/1000.
                with gzip.open(frame['detections'],'rb') as f:
                    gobs=pickle.load(f)
                detections=observation_objects(depth,intrinsic,pose,gobs)
                timestamp=frame['camera_timestamp_s']
                grid.update(detections,timestamp)
                number_frames+=1
                if frame['frame_idx']%4:
                    continue
                cells=grid.visible(pose,intrinsic,depth.shape[1],depth.shape[0])
                predictions=frame['predictions']['one_to_one_depth_quality']
                nodes=to_nodes(predictions)
                forward=pose[:2,2]
                edges=build_relations(nodes,dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),
                    frame_index=frame['frame_idx'],ego_heading=forward).edges
                truth=frame['evaluation_only_official']
                association={m['prediction_index']:truth[m['truth_index']]['instance_token']
                    for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                for qidx,q in enumerate(queries_for_frame(frame,labels[(frame['scene'],frame['frame_idx'])])):
                    english,parsed=query_english(q['query_text'])
                    vector=embeddings[english]
                    parsed.vehicle_heading_xy=forward
                    category=[d for d in detections if d['category'] in parsed.target_categories]
                    nearest=min(category,key=lambda d:np.linalg.norm(d['center'][:2]-pose[:2,3])) if category else None
                    crop=max(detections,key=lambda d:float(d['feature']@vector)) if detections else None
                    cell=max(cells,key=lambda d:float(d['feature']@vector)) if cells else None
                    ranked=rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
                        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,
                        evidence_policy=dict(enabled=True,minimum_observations=3,minimum_lidar_pixels=5,minimum_depth_support_fraction=.4))
                    joint=ranked[0].node_id if ranked else None
                    results=[]
                    for method,candidate in [('category_nearest_current_detection',nearest),
                        ('clip_current_crop',crop),('vlmaps_style_sparse_clip_grid',cell)]:
                        identity,status=associated_identity(candidate,truth)
                        results.append((method,candidate is not None,identity,status,
                            float(candidate['feature']@vector) if candidate else None))
                    results.append(('existing_object_joint_rule',joint is not None,association.get(joint),
                        'associated' if joint in association else 'unknown_or_no_selection',None))
                    acceptable=set(q['acceptable_instances'])
                    positive=bool(acceptable)
                    for method,selected,identity,status,score in results:
                        row=dict(scene=frame['scene'],frame_idx=frame['frame_idx'],query_index=qidx,query_text=q['query_text'],
                            clip_text=english,query_type=q['query_type'],method=method,selected=selected,
                            evaluation_identity=identity,association_status=status,clip_score=score,
                            positive=positive,acceptable_instances=sorted(acceptable),
                            selected_correct=identity in acceptable if positive else None,
                            false_selection_on_no_match=selected if not positive else None,
                            gt_in_inference=False,actual_arrival_evaluated=False)
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        increments=dict(queries=1,positive_queries=int(positive),negative_queries=int(not positive),
                            positive_selected_correct=int(positive and identity in acceptable),
                            no_match_false_selection=int(not positive and selected),
                            association_unknown=int(selected and identity is None),no_selection=int(not selected))
                        counts[method].update(increments)
                        scene_counts[frame['scene']][method].update(increments)
                handle.flush()
    metrics={name:dict(c,positive_selection_rate=c['positive_selected_correct']/c['positive_queries'] if c['positive_queries'] else None,
        no_match_false_selection_rate=c['no_match_false_selection']/c['negative_queries'] if c['negative_queries'] else None)
        for name,c in counts.items()}
    dump(dest/'report.json',dict(methods=metrics,scenes=scene_counts,frames=number_frames,
        scene_count=len(paths),elapsed_s=time.monotonic()-started,actual_clip_text_encoder=True,encoder_device='cpu',
        actual_saved_image_clip_features=True,only_current_and_past_observations=True,
        limitations=['VLMaps式网格仅是稀疏crop特征的二维适配，不是完整复现或论文数字横比',
            '官方中心关联不充分时保留unknown；网格读出是位置关联，非已验证跨帧实例身份',
            'CLIP不提供可靠运动状态/无目标判别；当前crop与网格都可能选择动态对象',
            '中文经相同固定规则翻译，非GT正确解析；本实验只评价检索，不产生路径/锁定/到达信用'],
        independent_test=False,stage12_complete=False,goal_complete=False))
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':
    main()
