"""KITTI逐帧对象关系与固定类别查询诊断；不产生闭环导航或实际到达信用。"""
import csv,json,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from conceptgraph.completion.kitti_tracking_experiment import eligible_truth
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.navigation.semantic import parse_navigation_query,rank_candidate_targets
from conceptgraph.navigation.state_machine import NavigationStateMachine
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.relations import RelationNode
ROOT=Path(__file__).resolve().parents[1];METHOD='one_to_one_depth_robust_motion';QUERIES={'car':'去最近的汽车附近','truck':'去最近的卡车附近','pedestrian':'去最近的行人附近'}
POLICY=dict(enabled=True,minimum_observations=3,minimum_lidar_pixels=5,minimum_depth_support_fraction=.4)
def main():
 out=ROOT/'kitti_tracking_expanded/graph_query_001';out.mkdir(exist_ok=False);source=ROOT/'mapping_evaluation/kitti_mapping_full_001/frames.jsonl';truthpath=ROOT/'data_manifest/kitti_tracking_expanded_006/evaluation_only/truth.json';summary=json.loads((source.parent/'summary.json').read_text());assert summary['frame_count']==1000 and digest(source)==summary['frames_sha256'];truth={(r['sequence'],r['original_frame_id']):r['objects'] for r in json.loads(truthpath.read_text())['frames']};graphcfg=dict(coordinate_frame='ego_heading',coordinate_frame_fallback=None,directional_mode='all_supported_axes',max_distance_m=12,near_distance_m=6)
 dump(out/'release.json',dict(source_sha256={str(p):digest(p) for p in [Path(__file__),source,truthpath]},method=METHOD,fixed_queries=QUERIES,evidence_policy=POLICY,graph_config=graphcfg,query_selection_uses_GT=False,GT_only_evaluation=True,actual_arrival_evaluated=False,independent_test=False,goal_complete=False))
 seq=None;counts=defaultdict(Counter);failures=[];index=[];seqframes=Counter();start=time.monotonic()
 with (out/'frames_graph_query.jsonl').open('x') as f,(out/'evaluation_only_query_metrics.jsonl').open('x') as ef:
  for line in source.open():
   row=json.loads(line);objects=row['predictions'][METHOD]
   if seq!=row['sequence']:
    seq=row['sequence'];registry={};graph=IncrementalSceneGraph(graphcfg);machines={k:NavigationStateMachine() for k in QUERIES}
   seqframes[seq]+=1;pose=np.asarray(row['pose']);forward=pose[:2,2];forward/=np.linalg.norm(forward);xy=pose[:2,3]
   for o in objects:registry.setdefault(o['node_id'],len(registry))
   nodes={registry[o['node_id']]:SceneGraphNode(registry[o['node_id']],o['category'],[o['category']],o['category'],np.asarray(o['center']),np.asarray(o['extent']),motion_state=o['motion_state'],is_short_term=o['motion_state']=='dynamic',metadata=dict(num_detections=o['num_detections'],depth_evidence=o['depth_evidence'])) for o in objects}
   relation_nodes={i:RelationNode(i,n.bbox_center,dynamic=n.motion_state=='dynamic',observation_count=n.metadata['num_detections'],last_seen_frame=row['frame_idx']) for i,n in nodes.items()};edges=graph.update_snapshot(relation_nodes,frame_index=row['frame_idx'],ego_heading=forward).edges
   selected={};queries=[]
   for category,text in QUERIES.items():
    parsed=parse_navigation_query(text,8);parsed.vehicle_heading_xy=forward;ranked=rank_candidate_targets(parsed,nodes,edges,xy,.55,.3,.25,relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,evidence_policy=POLICY)
    observed=machines[category].observe(row['frame_idx'],[(r.node_id,r.score) for r in ranked],valid_target_ids=set(nodes));selected[category]=dict(selected_node_id=ranked[0].node_id if ranked else None,locked_node_id=machines[category].locked_target_id,candidates=[dict(node_id=r.node_id,score=r.score,debug=r.debug) for r in ranked]);queries.append(dict(category=category,text=text,**selected[category]))
   # 选择和锁定完成后才读取官方标签评估。GT从未传入ranker、graph和machine。
   from conceptgraph.completion.quality import object_frame_metrics
   raw=truth[seq,row['original_frame_id']];eligible,excluded=eligible_truth(raw,row['image_wh']);metrics=object_frame_metrics(objects,eligible);pairs={registry[objects[m['prediction_index']]['node_id']]:eligible[m['truth_index']]['instance_token'] for m in metrics['matches']}
   for category in QUERIES:
    gt=[o for o in eligible if o['category']==category];dist=[np.linalg.norm(np.asarray(o['center'][:2])-xy) for o in gt];acceptable={o['instance_token'] for o,d in zip(gt,dist) if d<=min(dist)+.5} if dist else set();s=selected[category];has=bool(acceptable);chosen=pairs.get(s['selected_node_id']);lock=pairs.get(s['locked_node_id']);key=(seq,category);counts[key]['frame_queries']+=1;counts[key]['positive']+=has;counts[key]['no_match']+=not has;counts[key]['correct_selection']+=has and chosen in acceptable;counts[key]['false_selection_on_no_match']+=not has and s['selected_node_id'] is not None;counts[key]['correct_lock']+=has and lock in acceptable
    ev=dict(sequence=seq,frame_idx=row['frame_idx'],category=category,positive=has,acceptable_instances=sorted(acceptable),selected_instance=chosen,locked_instance=lock,correct_selection=has and chosen in acceptable,correct_lock=has and lock in acceptable,excluded_labels=len(excluded));ef.write(json.dumps(ev,ensure_ascii=False)+'\n')
   record=dict(sequence=seq,frame_idx=row['frame_idx'],original_frame_id=row['original_frame_id'],rgb=row['rgb'],depth=row['depth'],detections=row['detections'],detection_visualization=row['detection_visualization'],objects=objects,nodes=[dict(node_id=i,object_identity=objects[j]['node_id'],category=n.object_tag,center=n.bbox_center.tolist()) for j,(i,n) in enumerate(nodes.items())],edges=[e.to_dict() for e in edges],queries=queries,gt_in_inference=False,actual_arrival=False)
   f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');index.append(dict(sequence=seq,frame_idx=row['frame_idx'],original_frame_id=row['original_frame_id'],rgb=row['rgb'],depth=row['depth'],detection_pickle=row['detections'],detection_visualization=row['detection_visualization'],mapping_record=str(source),graph_query_record=str(out/'frames_graph_query.jsonl'),row_1based=len(index)+1))
 with (out/'frame_index.csv').open('x') as h:
  w=csv.DictWriter(h,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
 rows=[dict(sequence=s,category=c,**v) for (s,c),v in counts.items()]
 dump(out/'sequence_queries_summary.json',dict(rows=rows,frames=len(index),sequences=dict(seqframes),query_templates=3,frame_queries=sum(v['frame_queries'] for v in counts.values()),independent_sequence_query_units=len(rows),denominator_warning='3000次帧查询高度相关，不当作3000条独立自然语言；30序列-固定类别查询单元，无人工颜色标签',failed_frames=failures,wall_s=time.monotonic()-start,source_sha256=digest(source),records_sha256=digest(out/'frames_graph_query.jsonl'),evaluation_sha256=digest(out/'evaluation_only_query_metrics.jsonl'),navigation_arrivals=0,goal_complete=False))
 print('KITTI graph/query completed',len(index),flush=True)
if __name__=='__main__':main()
