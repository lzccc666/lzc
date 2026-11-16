"""同一因果前端历史原始结果复核；内部结构比较，不冒充外部论文复现。"""
import csv,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,dump,digest,available_development
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.temporal_identity_evaluation import prepare_sequence,identity_counts
R=Path(__file__).resolve().parents[1];O=R/'method_comparison/mapping_same_input_001'
METHODS=['single_frame','original_fusion','one_to_one_timestamp','one_to_one_depth_quality','one_to_one_depth_robust_motion','geometry_graph_full']
def measure(rows):
 c={k:sum(r[k] for r in rows) for k in ['true_positive','truth','predictions','duplicates','center_error_sum','idtp','idfp','idfn','id_switches']}
 return dict(recall=c['true_positive']/c['truth'] if c['truth'] else None,matched_prediction_fraction=c['true_positive']/c['predictions'] if c['predictions'] else None,center_error_m=c['center_error_sum']/c['true_positive'] if c['true_positive'] else None,duplicates=c['duplicates'],IDF1=2*c['idtp']/(2*c['idtp']+c['idfp']+c['idfn']) if 2*c['idtp']+c['idfp']+c['idfn'] else None,id_switches=c['id_switches'])
def main():
 O.mkdir(parents=True,exist_ok=False);scenes=available_development();source=OLD/'mapping_evaluation/causal_mapping_full_001';paths=[source/s['scene']/'frames.jsonl' for s in scenes];release=dict(methods=METHODS,source_sha256={str(p):digest(p) for p in paths},scenes=scenes,threshold_m=3,bootstrap_cluster='log_token',bootstrap_seed=17,bootstrap_repetitions=4000,new_inference_calls=0,source_script_sha256=digest(__file__),method_kind='project_internal_structure_comparisons',scope='mapping geometry and association; semantic and arrival not evaluated',goal_complete=False);dump(O/'release.json',release);reports=[]
 for scene,p in zip(scenes,paths):
  frames=list(map(json.loads,p.open()))
  for method in METHODS:
   m='one_to_one_depth_robust_motion' if method=='geometry_graph_full' else method;saved=[];counts=dict(true_positive=0,truth=0,predictions=0,duplicates=0,center_error_sum=0.)
   for f in frames:
    z=object_frame_metrics(f['predictions'][m],f['evaluation_only_official']);old=f['metrics'][m]
    for key in ['true_positive','truth','predictions','duplicate_nodes']:assert z[key]==old[key],(scene,m,key)
    for key in ['true_positive','truth','predictions']:counts[key]+=z[key]
    counts['duplicates']+=z['duplicate_nodes'];counts['center_error_sum']+=sum(q['distance_m'] for q in z['matches']);saved.append(f)
   ids=identity_counts(prepare_sequence(saved,m),'all_gated')['counts'];counts.update(idtp=ids['idtp'],idfp=ids['idfp'],idfn=ids['idfn'],id_switches=ids['id_switches']);reports.append(dict(scene=scene['scene'],log=scene['log_token'],method=method,frames=len(frames),**counts))
 logs=sorted({s['log_token'] for s in scenes});rng=np.random.default_rng(17);selections=rng.integers(0,len(logs),size=(4000,len(logs)));table=[]
 for method in METHODS:
  rr=[r for r in reports if r['method']==method];vals=measure(rr);bylog={l:[r for r in rr if r['log']==l] for l in logs};boots=defaultdict(list)
  for choice in selections:
   v=measure([r for i in choice for r in bylog[logs[i]]])
   for k,x in v.items():
    if x is not None:boots[k].append(x)
  for metric,value in vals.items():
   lo,hi=np.percentile(boots[metric],[2.5,97.5]);table.append(dict(method=method,metric=metric,value=value,ci95_low=lo,ci95_high=hi,scenes=len(rr),logs=len(logs),frames=sum(r['frames'] for r in rr),truth_observations=sum(r['truth'] for r in rr),predictions=sum(r['predictions'] for r in rr),TP=sum(r['true_positive'] for r in rr),per_method_time_s=None,peak_memory_mib=None))
 for filename,records in [('comparison.csv',table),('source_data_by_scene.csv',reports)]:
  with (O/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 dump(O/'report.json',dict(source_records_verified=True,new_mapper_calls=0,methods=METHODS,scene_count=len(scenes),frames=len(frames),unique_frames=sum(r['frames'] for r in reports if r['method']==METHODS[0]),scope=release['scope'],historical_full_mapping_exact_match=True,geometry_graph_full_note='与运动约束方法共享相同对象结果；图维护不改变对象计数。该行不宣称Qwen或语义模块已执行。',missing=['历史逐方法独立计时/峰值内存不可从总运行耗时拆分，填空保留','外部公开方法尚未完成同输入复现','最终语义合作模块另测'],output_sha256={n:digest(O/n) for n in ['comparison.csv','source_data_by_scene.csv']},goal_complete=False))
 (O/'说明.md').write_text('同一38场景、11日志、1508帧因果输入的内部结构比较。重读并复核历史真实推理结果，本次新增建图调用为0。\n\nmatched_prediction_fraction是匹配预测比例，不冒充官方检测AP；IDF1为3米类别/几何门限诊断。CI按11日志聚类。全几何图行复用完整对象链路，不授予Qwen执行信用。逐方法独立历史计时缺失，保留空值。外部论文复现与实时闭环均未据此完成。\n')
 print(json.dumps({'methods':len(METHODS),'scenes':len(scenes),'unique_frames':sum(r['frames'] for r in reports if r['method']==METHODS[0])}))
if __name__=='__main__':main()
