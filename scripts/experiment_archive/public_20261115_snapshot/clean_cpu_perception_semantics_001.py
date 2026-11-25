"""全新依赖和展开源码实际CPU检测、分割、特征、三帧建图、关系及Qwen；不冒称物理闭环。"""
import argparse,copy,gc,gzip,hashlib,json,os,pickle,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while x:=f.read(1024*1024):h.update(x)
 return h.hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);a=ap.parse_args();release=json.loads(a.release.read_text());O=a.release.parent;checkout=Path(release['checkout']);assert sys.prefix!=sys.base_prefix
 assert all(sha(p)==h for p,h in release['source_and_weight_sha256'].items());assert not any('/envs/conceptgraph/lib/python3.10/site-packages' in p for p in sys.path)
 import numpy as np
 import torch
 from PIL import Image
 import conceptgraph.scripts.streamlined_detections as frontend
 from conceptgraph.slam.streamlined_mapping import MappingFrame,StreamlinedMapper
 from conceptgraph.completion.stage9 import predict
 from conceptgraph.completion.fault_robustness import mapping_config
 from conceptgraph.scenegraph.relations import RelationNode
 from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
 assert Path(frontend.__file__).resolve().is_relative_to(checkout)
 torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);assert not torch.cuda.is_available();begun=time.monotonic();classes=release['classes'];cfg=SimpleNamespace(device='cpu');models=frontend.load_detection_models(cfg)
 for model in models[:2]:model.to('cpu');model.overrides['device']='cpu'
 mapper=StreamlinedMapper(mapping_config(),classes);graph=IncrementalSceneGraph(release['resolved_graph_config']);registry={};records=[];all_edges=[]
 with (O/'frames.jsonl').open('x') as raw:
  for frame in release['frames']:
   started=time.monotonic();gobs,annotated=frontend.detect_rgb_file(cfg,models,frame['rgb'],classes);p=O/f"frame{frame['frame_idx']:03d}_fresh_frontend.pkl.gz"
   with gzip.open(p,'xb') as f:pickle.dump(gobs,f)
   image=np.asarray(Image.open(frame['rgb']).convert('RGB'));depth=np.asarray(Image.open(frame['depth']),dtype=float)/1000;pose=np.asarray(frame['camera_pose']);K=np.asarray(frame['intrinsic']);np.random.seed(17+frame['frame_idx']);mapper.update(MappingFrame(frame['frame_idx'],image,depth,K,pose,copy.deepcopy(gobs),frame['rgb'],timestamp_s=frame['camera_timestamp_s']));objects=predict(mapper.objects,classes,pose=pose,intrinsic=K)
   for obj in objects:registry.setdefault(obj['node_id'],len(registry)+100)
   nodes={registry[obj['node_id']]:RelationNode(registry[obj['node_id']],obj['center'],dynamic=obj['motion_state']=='dynamic',observation_count=obj['num_detections'],last_seen_frame=obj['last_seen_frame'],metadata={'timestamp_s':obj.get('timestamp_s')}) for obj in objects};all_edges=graph.update_snapshot(nodes,frame_index=frame['frame_idx'],ego_heading=pose[:2,2]).edges
   row=dict(frame_idx=frame['frame_idx'],new_CPU_frontend=True,detections=len(gobs['xyxy']),masks=int(len(gobs['mask'])),feature_shape=list(gobs['image_feats'].shape),objects=objects,edges=[e.to_dict() for e in all_edges],wall_s=time.monotonic()-started,front_raw_path=str(p),front_raw_sha256=sha(p),actual_GPU_calls=0,GT_read=False);raw.write(json.dumps(row,ensure_ascii=False)+'\n');raw.flush();records.append(row);print(json.dumps({k:row[k] for k in ['frame_idx','detections','masks','wall_s']},ensure_ascii=False),flush=True)
 del models;gc.collect();sys.path.insert(0,str(checkout/release['embedded_source_relative']))
 from real_graph_contract_003 import build_completion_prompt,validate
 from real_graph_cooperation_001 import choose
 from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
 from transformers import AutoProcessor
 origin=pose[:2,3];forward=pose[:2,2].copy();forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]]);semantic=[]
 for obj in objects:
  delta=np.asarray(obj['center'][:2])-origin;e=(obj.get('depth_evidence') or [{}])[-1]
  semantic.append(dict(node_id=registry[obj['node_id']],category=obj['category'],center_xy=[round(float(delta@forward),3),round(float(delta@left),3)],distance_m=round(float(np.linalg.norm(delta)),3),motion_state=obj['motion_state'],version=frame['frame_idx']+1,now_s=frame['camera_timestamp_s'],last_seen_s=obj.get('timestamp_s',-1),evidence=dict(source=e.get('source','unknown'),version=frame['frame_idx']+1,observations=obj['num_detections'],lidar_pixels=int(e.get('retained_pixels',0)),support_fraction=float(e.get('support_fraction',0)),conflict=not bool(e.get('sufficient_support',False)))))
 semantic=sorted(semantic,key=lambda n:(n['distance_m'],n['node_id']))[:8];ids={n['node_id'] for n in semantic};relations=[dict(source=e.source,target=e.target,relation=e.relation) for e in all_edges if e.source in ids and e.target in ids][:16]
 processor=AutoProcessor.from_pretrained(release['base_model'],local_files_only=True,trust_remote_code=True);model=_load_model(Path(release['base_model']),Path(release['adapter']),trainable=False);model.eval();decisions=[]
 with (O/'semantic_outputs.jsonl').open('x') as out:
  for query in release['queries']:
   row=dict(task='candidate_rerank',query_text=query,candidate_nodes=semantic,candidate_ids=[n['node_id'] for n in semantic],relation_metadata=relations);prompt=build_completion_prompt('candidate_rerank',row);messages=[dict(role='user',content=[dict(type='text',text=prompt)])];text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=processor(text=[text],images=None,padding=False,return_tensors='pt');assert inputs['input_ids'].shape[-1]<=2048;started=time.monotonic()
   with torch.inference_mode():y=model.generate(**inputs,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
   raw=processor.tokenizer.decode(y[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip();valid=validate('candidate_rerank',raw,allowed_candidate_ids=row['candidate_ids']);decision=choose(query,semantic,relations,dict(schema_valid=valid.valid,normalized_output=valid.normalized_output if valid.valid else None));entry=dict(input=row,raw_output=raw,schema_valid=valid.valid,decision=decision,latency_s=time.monotonic()-started,actual_CPU_Qwen_calls=1,actual_GPU_calls=0,ground_truth_used=False,target_correctness_or_arrival_credit=False);out.write(json.dumps(entry,ensure_ascii=False)+'\n');out.flush();decisions.append(entry);print(json.dumps(dict(query=query,schema_valid=valid.valid,CPU_generation_s=entry['latency_s']),ensure_ascii=False),flush=True)
 save(O/'report.json',dict(actual_fresh_CPU_frontend_frames=len(records),actual_new_mapper_updates=len(records),actual_new_relation_updates=len(records),actual_new_CPU_Qwen_generations=len(decisions),actual_new_GPU_calls=0,actual_physical_CARLA_episodes=0,format_valid=sum(x['schema_valid'] for x in decisions),wall_s=time.monotonic()-begun,new_environment_site_packages_used=True,isolated_source_checkout=True,model_and_sensor_files_referenced_read_only=True,source_release_sha256=sha(a.release),output_sha256={p.name:sha(p) for p in O.glob('*.jsonl')},limitations=['前三帧开发场景CPU链路功能复现，非1000/1508帧完整重跑','CPU依赖版本差异见registration；未声称数值/原GPU字节一致或性能可比','实际完成检测/分割/CLIP/建图/关系/模型生成，未执行车辆规划控制或官方物理闭环','三句固定诊断语法，不计新人工指令、封存分母、找对率、锁定或到达'],whole_clean_environment_navigation_passed=False,goal_complete=False));print('actual fresh CPU perception-through-semantics completed',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args();save(a.release.parent/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
