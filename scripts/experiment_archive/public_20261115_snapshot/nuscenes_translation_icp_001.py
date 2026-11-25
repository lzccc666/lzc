"""固定38开发场景的单模块对象/关系图消融；不读取封存测试，不授予语义或导航信用。"""
import argparse,copy,csv,gzip,importlib.util,json,pickle,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,digest,dump,predict,available_development
from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics,identity_metrics,relations_from_centers,relation_metrics
from conceptgraph.completion.temporal_identity_evaluation import prepare_sequence,identity_counts
from conceptgraph.completion.relations_audit import expand
from conceptgraph.scenegraph.relations import RelationNode
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
ROOT=Path(__file__).resolve().parents[1]
SETTINGS={'translation_icp_motion':{}}
GRAPH=dict(coordinate_frame='ego_heading',coordinate_frame_fallback=None,max_distance_m=12.,near_distance_m=6.,position_update_threshold_m=.1,directional_mode='all_supported_axes',minimum_direction_component_m=1.)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--condition',choices=SETTINGS,required=True);p.add_argument('--scene',action='append');p.add_argument('--wall-budget-s',type=float,default=1800);args=p.parse_args()
 dest=ROOT/'nuscenes_ablation'/args.run_id;dest.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
 scenes=[s for s in available_development() if not args.scene or s['scene'] in args.scene]
 if not scenes or (args.scene and set(args.scene)!={s['scene'] for s in scenes}):raise ValueError('必须是非空已授权开发场景清单；不允许空跑计完成')
 cfg=mapping_config();cfg.assignment_cardinality='one_to_one'
 for k,v in SETTINGS[args.condition].items():setattr(cfg,k,v)
 source_files=[OLD/'mapping_evaluation/causal_mapping_full_001'/s['scene']/'frames.jsonl' for s in scenes]
 files=[Path(__file__).resolve(),ROOT/'source/mapping_assignment_isolated_001.py',ROOT/'source/translation_icp_motion_001.py',PROJECT/'conceptgraph/slam/streamlined_mapping.py',PROJECT/'conceptgraph/slam/mapping.py',PROJECT/'conceptgraph/slam/utils.py',PROJECT/'conceptgraph/completion/fault_robustness.py',PROJECT/'conceptgraph/completion/quality.py',PROJECT/'conceptgraph/completion/temporal_identity_evaluation.py',PROJECT/'conceptgraph/scenegraph/incremental_graph.py',PROJECT/'conceptgraph/scenegraph/relations.py',PROJECT/'conceptgraph/completion/association.py',PROJECT/'conceptgraph/completion/depth.py',PROJECT/'conceptgraph/completion/motion.py']
 frozen={str(x):digest(x) for x in files}
 release=dict(condition=args.condition,changed_config=SETTINGS[args.condition],removed_graph_module=args.condition=='without_relations',config=OmegaConf.to_container(cfg),graph_config=GRAPH,scenes=scenes,frames={str(x):digest(x) for x in source_files},source_sha256=frozen,seed=17,threshold_m=3.,wall_budget_s=args.wall_budget_s,minimum_free_gib=20,new_outputs_budget_gib=10,bootstrap_seed=17,bootstrap_repetitions=4000,GT_in_mapper=False,semantic_module_evaluated=False,new_test_opened=False,goal_complete=False)
 from translation_icp_motion_001 import PARAMS,install,COUNTERS
 release['motion_registration_params']=PARAMS
 release['controlled_change']='仅运动估计：当前/过去点云平移ICP，有效几何/拟合门限不足时回退相同稳健XY质心；原证据门限、关联、TTL、标签均不变'
 release['timing_shared_CPU_with_CARLA']=True
 dump(dest/'release.json',release)
 # 仅在隔离模块内改变匹配基数；TTL、稳定ID、门限和动态更新仍走同一个one_to_one分支。
 spec=importlib.util.spec_from_file_location('isolated_mapping',ROOT/'source/mapping_assignment_isolated_001.py');isolated=importlib.util.module_from_spec(spec);spec.loader.exec_module(isolated);install(isolated)
 import conceptgraph.slam.streamlined_mapping as sm
 sm.merge_detections_to_objects=isolated.merge_detections_to_objects
 reports=[]
 try:
  for scene,sourcepath in zip(scenes,source_files):
   frames=[json.loads(x) for x in sourcepath.open()];gpath=Path(frames[0]['detections']).parents[2]/'gsa_classes_streamlined_yolo.json';classes=json.loads(gpath.read_text());mapper=sm.StreamlinedMapper(copy.deepcopy(cfg),classes);graph=IncrementalSceneGraph(GRAPH);registry={};saved=[];identities=[];times=[];rtot=dict(true_positive=0,predicted=0,truth=0,unresolved_edges=0)
   case=dest/scene['scene'];case.mkdir();handle=(case/'frames.jsonl').open('x')
   for source in frames:
    import shutil
    if time.monotonic()-begun>args.wall_budget_s:raise TimeoutError('frozen wall budget')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise OSError('free space under 20GiB')
    idx=source['frame_idx'];image=np.asarray(Image.open(source['rgb']).convert('RGB'));depth=np.asarray(Image.open(source['depth']),float)/1000.;pose=np.asarray(source['camera_pose']);K=np.asarray(source['intrinsic'])
    with gzip.open(source['detections'],'rb') as f:gobs=pickle.load(f)
    np.random.seed(17+idx);tick=time.monotonic();mapper.update(sm.MappingFrame(idx,image,depth,K,pose,copy.deepcopy(gobs),source['rgb'],timestamp_s=source['camera_timestamp_s']));objects=predict(mapper.objects,classes,pose=pose,intrinsic=K)
    for o in objects:registry.setdefault(o['node_id'],len(registry))
    nodes={registry[o['node_id']]:RelationNode(registry[o['node_id']],o['center'],dynamic=o['motion_state']=='dynamic',observation_count=o['num_detections'],last_seen_frame=o['last_seen_frame'],metadata={'timestamp_s':o.get('timestamp_s')}) for o in objects}
    forward=pose[:2,2]
    edges=[] if args.condition=='without_relations' else graph.update_snapshot(nodes,frame_index=idx,ego_heading=forward).edges
    elapsed=time.monotonic()-tick;times.append(elapsed)
    # 真值只在上述推理完成后进入本地独立评价，未传到mapper或图维护器。
    truth=source['evaluation_only_official'];m=object_frame_metrics(objects,truth);mapping={registry[objects[x['prediction_index']]['node_id']]:truth[x['truth_index']]['instance_token'] for x in m['matches']}
    pe=expand(edges);resolved={(mapping[a],r,mapping[b]) for a,r,b in pe if a in mapping and b in mapping};unresolved=sum(a not in mapping or b not in mapping for a,r,b in pe);gt=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in truth],forward_xy=forward,max_distance_m=12.,near_m=6.)
    rm=relation_metrics(resolved,gt);rm['unresolved_edges']=unresolved
    for k in rtot:rtot[k]+=rm[k]
    row=dict(scene=scene['scene'],frame_idx=idx,camera_timestamp_s=source['camera_timestamp_s'],rgb=source['rgb'],depth=source['depth'],detections=source['detections'],predictions={args.condition:objects},evaluation_only_official=truth,metrics={args.condition:m},nodes={str(k):dict(center=v.center.tolist(),dynamic=v.dynamic) for k,v in nodes.items()},edges=[e.to_dict() for e in edges],relations=rm,elapsed_s=elapsed,gt_in_inference=False)
    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();saved.append(row);identities.append((idx,[(objects[x['prediction_index']]['node_id'],truth[x['truth_index']]['instance_token']) for x in m['matches']]))
   handle.close();count=identity_counts(prepare_sequence(saved,args.condition),'all_gated')['counts'];tot={k:sum(r['metrics'][args.condition][k] for r in saved) for k in ['true_positive','truth','predictions','duplicate_nodes']};errors=[x['distance_m'] for r in saved for x in r['metrics'][args.condition]['matches']];tot['center_error_sum_m']=sum(errors);tot.update(identity_metrics(identities));report=dict(scene=scene['scene'],log=scene['log_token'],condition=args.condition,frames=len(saved),object=tot,identity=count,relations=rtot,mapping_relation_time_s=times,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,records_sha256=digest(case/'frames.jsonl'));dump(case/'report.json',report);reports.append(report);print(json.dumps(dict(scene=scene['scene'],condition=args.condition,frames=len(saved),TP=tot['true_positive'])),flush=True)
   del mapper,saved
  if any(digest(k)!=v for k,v in frozen.items()):raise ValueError('frozen source changed')
  dump(dest/'summary.json',dict(condition=args.condition,scenes=len(reports),frames=sum(r['frames'] for r in reports),reports=reports,wall_s=time.monotonic()-begun,semantic_module_evaluated=False,actual_arrival_evaluated=False,motion_registration_counters=dict(COUNTERS),goal_complete=False))
 except BaseException as e:dump(dest/'interrupted.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_scenes=len(reports),goal_complete=False));raise
if __name__=='__main__':main()
