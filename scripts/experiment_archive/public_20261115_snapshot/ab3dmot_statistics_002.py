"""固定共享前端的官方跟踪器适配结果：复核计数、按日志/序列配对自助统计。"""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from conceptgraph.completion.quality import identity_metrics,object_frame_metrics
R=Path(__file__).resolve().parents[1]
S=R/'method_comparison/ab3dmot_shared_full_001'
O=R/'method_comparison/ab3dmot_statistics_002'
KEYS=['TP','GT','PRED','DUP','ERROR','IDTP','IDFP','IDFN','switches','transitions']
def measure(rows):
 c={k:sum(r[k] for r in rows) for k in KEYS}
 return dict(recall=c['TP']/c['GT'] if c['GT'] else None,matched_prediction_fraction=c['TP']/c['PRED'] if c['PRED'] else None,center_error_m=c['ERROR']/c['TP'] if c['TP'] else None,duplicates=c['DUP'],IDF1=2*c['IDTP']/(2*c['IDTP']+c['IDFP']+c['IDFN']) if 2*c['IDTP']+c['IDFP']+c['IDFN'] else None,adjacent_ID_switches=c['switches'],adjacent_matched_transitions=c['transitions'])
def main():
 O.mkdir(exist_ok=False);summary=json.loads((S/'summary.json').read_text());reports=summary['reports'];assert summary['unique_frames']==2508;methods=sorted({r['method'] for r in reports});input_hashes={};timings=[]
 for dataset in ['nuScenes','KITTI']:
  for scene in sorted({r['scene'] for r in reports if r['dataset']==dataset}):
   p=S/dataset/scene/'frames.jsonl';input_hashes[str(p)]=digest(p);frames=list(map(json.loads,p.open()));assert frames
   for method in methods:
    r=next(r for r in reports if r['dataset']==dataset and r['scene']==scene and r['method']==method);pairs=[];sums=dict(TP=0,GT=0,PRED=0,DUP=0,ERROR=0.)
    for f in frames:
     metrics=object_frame_metrics(f['predictions'][method],f['evaluation_only_official']);old=f['metrics'][method]
     for key in ['true_positive','truth','predictions','duplicate_nodes']:assert metrics[key]==old[key]
     for key,mkey in [('TP','true_positive'),('GT','truth'),('PRED','predictions'),('DUP','duplicate_nodes')]:sums[key]+=metrics[mkey]
     sums['ERROR']+=sum(m['distance_m'] for m in metrics['matches'])
     pairs.append((f['frame_idx'],[(f['predictions'][method][m['prediction_index']]['node_id'],f['evaluation_only_official'][m['truth_index']]['instance_token']) for m in metrics['matches']]))
    for k,v in sums.items():assert np.isclose(r[k],v),(dataset,scene,method,k)
    identity=identity_metrics(pairs);r.update(switches=identity['identity_switches'],transitions=identity['adjacent_matched_transitions'])
   timings.append(dict(dataset=dataset,scene=scene,frames=len(frames),shared_projection_s=sum(f['shared_projection_s'] for f in frames),official_tracking_s=sum(f['AB3DMOT_tracking_s'] for f in frames)))
 tables=[];deltas=[]
 for dataset,expected in [('nuScenes',dict(TP=1720,GT=5723,PRED=3905)),('KITTI',dict(TP=2739,GT=3257,PRED=6995))]:
  rr=[r for r in reports if r['dataset']==dataset];units=sorted({r['unit'] for r in rr});draw=np.random.default_rng(17).integers(0,len(units),(4000,len(units)));values={};boots={}
  ours=[r for r in rr if r['method']=='ours_geometry_full']
  for k,v in expected.items():assert sum(r[k] for r in ours)==v,(dataset,k)
  for method in methods:
   rows=[r for r in rr if r['method']==method];byunit={u:[r for r in rows if r['unit']==u] for u in units};v=measure(rows);b=defaultdict(list)
   for choice in draw:
    z=measure([r for idx in choice for r in byunit[units[idx]]])
    for metric,x in z.items():b[metric].append(x if x is not None else np.nan)
   values[method]=v;boots[method]=b
   for metric,x in v.items():
    low,high=np.nanpercentile(b[metric],[2.5,97.5]);tables.append(dict(dataset=dataset,method=method,metric=metric,value=x,ci95_low=low,ci95_high=high,scenes=len(rows),independent_units=len(units),frames=sum(r['frames'] for r in rows),TP=sum(r['TP'] for r in rows),GT=sum(r['GT'] for r in rows),PRED=sum(r['PRED'] for r in rows),new_GPU_calls=0))
  for method in methods:
   if method=='ours_geometry_full':continue
   for metric,x in values['ours_geometry_full'].items():
    baseline=values[method][metric];arr=np.asarray(boots['ours_geometry_full'][metric])-np.asarray(boots[method][metric]);lo,hi=np.nanpercentile(arr,[2.5,97.5]);deltas.append(dict(dataset=dataset,comparison='ours_geometry_full minus '+method,metric=metric,absolute_delta=x-baseline,relative_change=(x-baseline)/abs(baseline) if baseline else None,ci95_low=lo,ci95_high=hi,paired_units=len(units),bootstrap_repetitions=4000))
 for filename,rows in [('comparison.csv',tables),('paired_differences.csv',deltas),('source_data_by_scene.csv',reports),('timing_by_scene.csv',timings)]:
  with (O/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(O/'report.json',dict(unique_frames=2508,method_frame_outputs=2508*3,verified_shared_geometry_full_counts=True,bootstrap=dict(unit='nuScenes log / KITTI sequence',seed=17,repetitions=4000,paired=True),source_sha256={**input_hashes,str(S/'release.json'):digest(S/'release.json'),str(S/'summary.json'):digest(S/'summary.json'),str(Path(__file__)):digest(__file__)},scope='官方2020 AB3DMOT核心+共享当前LiDAR深度质量前端/AABB适配；未复现作者原检测器、榜单或导航任务',limits=['作者默认max_age=2帧，nuScenes约2Hz、KITTI名义10Hz；未按结果调参','输入框是局部可见点云AABB且yaw固定0；不提供GT朝向或完整遮挡框','预测匹配比例非官方检测AP；XY中心3m门限，IDF1是局部协议','所有负结果和失败场景完整保留；共享当前帧前端不同于历史未过滤single_frame配置','AB3DMOT独立跟踪耗时和共享投影耗时有记录；ours历史未保存逐方法独立计时，不能伪造速度优势'],new_GPU_calls=0,goal_complete=False))
 (O/'说明.md').write_text('实际运行官方AB3DMOT固定2020版本核心；仅为相同前端的跟踪模块适配对照。38个nuScenes场景1508帧、10个KITTI连续序列1000帧。绝对值、分母、置信区间和配对差值均已导出；不能称为官方排行榜成绩或完整外部导航复现。\n\n每帧按类别和XY中心3米门限匹配。ID切换只计算相邻且均有匹配的帧；不能将其与官方长时身份切换完全等同。CI按nuScenes的11条日志/KITTI的10个序列聚类。\n',encoding='utf-8')
 print(json.dumps([r for r in tables if r['metric'] in ['recall','IDF1','duplicates']],ensure_ascii=False))
if __name__=='__main__':main()
