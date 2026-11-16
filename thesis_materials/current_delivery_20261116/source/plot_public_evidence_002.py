"""从已核验实验绘制可追溯图，不将开发结果写成封存或到达。"""
import csv,ctypes,hashlib,json
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree import ElementTree as ET
R=Path(__file__).resolve().parents[1];O=R/'figures/public_evidence_002'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.65,'legend.frameon':False,'savefig.facecolor':'white'})
COLORS=['#8B939B','#B4BBC1','#507C9E','#A49E91','#B28D52','#758F89','#716C8D'];SOURCES={};EXPORTS=[];QA=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 SOURCES[str(p)]=sha(p)
 return json.loads(p.read_text())
def read(p):
 SOURCES[str(p)]=sha(p)
 return list(csv.DictReader(p.open()))
def writecsv(name,rows):
 with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def make(rows,keys,titles,name,footnote,limits=None):
 # Horizontal comparisons retain a common zero and label the denominators.
 n=len(keys);fig,axes=plt.subplots(2,2,figsize=(183/25.4,132/25.4)) if n==4 else plt.subplots(1,n,figsize=(183/25.4,92/25.4))
 axes=np.asarray(axes).ravel();labels=[r['label'] for r in rows];y=np.arange(len(rows))
 for i,(key,title,ax) in enumerate(zip(keys,titles,axes)):
  v=np.array([r[key] for r in rows]);ax.barh(y,v,color=[COLORS[j%len(COLORS)] for j in range(len(rows))],height=.6)
  if key+'_low' in rows[0]:
   low=np.array([r[key+'_low'] for r in rows]);high=np.array([r[key+'_high'] for r in rows]);ax.errorbar(v,y,xerr=[np.maximum(0,v-low),np.maximum(0,high-v)],fmt='none',color='#343A40',linewidth=.7,capsize=2)
  ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(chr(97+i)+'  '+title,loc='left',fontweight='bold');ax.set_xlim(0,limits[i] if limits else 1.08);ax.grid(axis='x',color='#E6E8EA',linewidth=.5);ax.set_axisbelow(True)
  cap=ax.get_xlim()[1]
  for yi,value in enumerate(v):ax.text(cap*.985,yi,f'{value:.3f}',ha='right',va='center',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',pad=.5))
 fig.tight_layout(rect=(0,.13,1,.99),h_pad=2.4,w_pad=2.0);fig.text(.04,.035,footnote,va='bottom',fontsize=6.3)
 export(fig,name)
def export(fig,name):
 fig.canvas.draw();width,height=fig.canvas.get_width_height();renderer=fig.canvas.get_renderer();outside=[]
 for text in fig.findobj(matplotlib.text.Text):
  if not text.get_visible() or not text.get_text():continue
  box=text.get_window_extent(renderer)
  if box.x0<-.5 or box.y0<-.5 or box.x1>width+.5 or box.y1>height+.5:outside.append(text.get_text())
 for suffix,kwargs in [('svg',{}),('pdf',{}),('tiff',dict(dpi=600,pil_kwargs={'compression':'tiff_lzw'})),('png',dict(dpi=180))]:
  p=O/f'{name}.{suffix}';assert not p.exists();fig.savefig(p,**kwargs);EXPORTS.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
 root=ET.parse(O/f'{name}.svg').getroot();texts=root.findall('.//{http://www.w3.org/2000/svg}text');assert texts
 with Image.open(O/f'{name}.tiff') as im:dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-600)<1
 QA.append(dict(figure=name,text_outside_canvas=outside,editable_SVG_text_elements=len(texts),TIFF_dpi=[float(v) for v in dpi],dimension_mm=[fig.get_figwidth()*25.4,fig.get_figheight()*25.4],visual_inspection_pending=True))
 plt.close(fig)
def main():
 contract=load(O/'figure_contract.json');assert contract['backend']=='Python, matplotlib only'
 data=read(R/'nuscenes_ablation/statistics_001/metrics.csv');relations=read(R/'nuscenes_ablation/graph_near6_correction_001/comparison.csv');ns=[]
 names={'full':'完整方案','without_depth_quality':'去深度质量','without_one_to_one':'去一对一关联','without_dynamic_maintenance':'去动态维护','without_short_term_expiry':'去短期过期','without_relations':'去关系边'}
 for c,label in names.items():
  row=dict(condition=c,label=label,frames=1508,scenes=38,logs=11)
  for key in ['object_recall','IDF1','center_error_m']:
   m=next(m for m in data if m['condition']==c and m['metric']==key);row[key]=float(m['value']);row[key+'_low']=float(m['ci95_low']);row[key+'_high']=float(m['ci95_high'])
  m=next(m for m in relations if m['condition']==c and m['metric']=='f1' and m['version']=='corrected');row['relation_f1']=float(m['value']);row['relation_f1_low']=float(m['ci95_low']);row['relation_f1_high']=float(m['ci95_high']);ns.append(row)
 writecsv('nuscenes_module_ablation_source.csv',ns)
 make(ns,['object_recall','IDF1','center_error_m','relation_f1'],['对象召回率 ↑','跨帧身份 IDF1 ↑','中心误差（米）↓','关系 F1 ↑（实际邻近半径6米）'],'nuscenes_module_ablation','38个开发场景 / 11个采集日志，每配置1508帧；误差线：日志聚类95%区间（4000次重采样）。\n关系使用新6米图；对象指标复用同一批对象结果。含训练/验证日志，不是最终封存测试。',[.6,.6,2.,.5])
 k=load(R/'statistics/kitti_expanded_full_001/report.json');scenes=load(R/'statistics/kitti_expanded_full_001/scene_reports.json');seqs=sorted(scenes);draw=np.random.default_rng(17).integers(0,len(seqs),(4000,len(seqs)));kt=[]
 for method,label in [('single_frame','单帧建图'),('original_fusion','原融合'),('one_to_one_depth_robust_motion','完整对象融合')]:
  m=k['object']['methods'][method];row=dict(method=method,label=label,frames=1000,sequences=10)
  for key,top,bot in [('recall','true_positive','truth'),('center_error_m','center_error_sum_m','true_positive'),('duplicate_fraction','duplicate_nodes','predictions')]:
   arr=np.array([[scenes[s][method]['object'][v] for v in [top,bot]] for s in seqs]);boot=arr[draw].sum(1);values=boot[:,0]/boot[:,1];row[key]=m['metrics'][key];row[key+'_low'],row[key+'_high']=map(float,np.percentile(values,[2.5,97.5]))
  a=k['identity']['methods'][method]['all_gated']['metrics']['IDF1'];row.update(IDF1=a['value'],IDF1_low=a['ci95'][0],IDF1_high=a['ci95'][1]);row['duplicate_count']=m['counts']['duplicate_nodes'];row['predicted_count']=m['counts']['predictions'];kt.append(row)
 writecsv('kitti_mapping_tradeoffs_source.csv',kt)
 make(kt,['recall','IDF1','center_error_m','duplicate_fraction'],['对象召回率 ↑','跨帧身份 IDF1 ↑','中心误差（米）↓','重复节点 / 全部预测节点 ↓'],'kitti_mapping_tradeoffs','KITTI Tracking 10个连续序列 / 1000帧；误差线：序列聚类95%区间（4000次重采样）。\nGT为官方筛选后的可见对象，中心距离配对；重复计数42 / 75 / 307。仅离线诊断，无实际到达信用。',[1.,1.,2.,.12])
 base=R/'qwen_adapter_fair_comparison/real_graph_fair_gpu_001';qw=[];baseline_ids=None
 for variant,label in [('base','基础Qwen'),('old_adapter','旧Adapter'),('historical_three_task_seed17','历史三任务Adapter'),('oracle_graph_seed17','真值图训练Adapter'),('real_graph_seed17','新Adapter seed17'),('real_graph_seed29','新Adapter seed29'),('real_graph_seed43','新Adapter seed43')]:
  report=load(base/variant/'report.json');p=base/variant/'raw_records.jsonl';SOURCES[str(p)]=sha(p);records=list(map(json.loads,p.open()));ids=[x['sample_id'] for x in records]
  if baseline_ids is None:baseline_ids=ids
  assert ids==baseline_ids and len(records)==108
  pp=[x for x in records if x['official_positive'] and not x['controlled_fault']];assert len(pp)==60
  row=dict(variant=variant,label=label,queries=108,positive_queries=60,logs=2,correct=sum(x['semantic_correct'] for x in records),positive_correct=sum(x['semantic_correct'] for x in pp),format_valid=sum(x['schema_valid'] for x in records));assert row['correct']==report['correct']
  logs=sorted({x['log_token'] for x in records});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
  for key,subset,field in [('overall',records,'semantic_correct'),('positive',pp,'semantic_correct'),('format',records,'schema_valid')]:
   row[key]=sum(x[field] for x in subset)/len(subset);arr=np.array([[sum(x[field] for x in subset if x['log_token']==log),sum(x['log_token']==log for x in subset)] for log in logs]);b=arr[draw].sum(1);row[key+'_low'],row[key+'_high']=map(float,np.percentile(b[:,0]/b[:,1],[2.5,97.5]))
  qw.append(row)
 writecsv('qwen_fair_target_vs_format_source.csv',qw)
 make(qw,['overall','positive','format'],['全部任务正确率 ↑','普通正目标选对率 ↑','输出格式有效率 ↑'],'qwen_fair_target_vs_format','相同108条开发请求 / 2日志；其中60条普通正目标；误差线为日志聚类95%区间，仅2日志代表性有限。\n三种子分别展示，输入和推理预算相同。格式/拒答改善不能解释为找对目标；真值图训练是辅助诊断。',[1.04,.45,1.06])
 pilot=load(R/'carla_official/acceleration_feedback_pilot_003/report.json');rows=[]
 for m in pilot['results']:rows.append(dict(label=('直线' if m['case']=='car20' else '弯道')+(' / 原控制' if m['policy']=='legacy_speed_slew' else ' / 加速度反馈'),maximum_acceleration=m['maximum_acceleration_mps2'],distance=m['path_distance_m'],actual_physical_ticks=m['ticks'],pass_all=m['combined_calibration_pass'],actual_arrival_credit=False))
 writecsv('official_actuator_diagnostic_source.csv',rows)
 # A threshold line makes violations visible without changing the declared acceptance.
 fig,axs=plt.subplots(1,2,figsize=(183/25.4,88/25.4));y=np.arange(4)
 for i,(key,title,limit) in enumerate([('maximum_acceleration','最大实际加速度（米/秒²）↓',1.45),('distance','实际行驶路程（米）',15.)]):
  axs[i].barh(y,[x[key] for x in rows],color=COLORS[:4]);axs[i].set_yticks(y,[x['label'] for x in rows]);axs[i].invert_yaxis();axs[i].set_xlim(0,limit);axs[i].set_title(chr(97+i)+'  '+title,loc='left',fontweight='bold');axs[i].grid(axis='x',color='#E6E8EA');axs[i].set_axisbelow(True)
  for j,x in enumerate(rows):axs[i].text(limit*.985,j,f"{x[key]:.3f}",ha='right',va='center',bbox=dict(facecolor='white',edgecolor='none',pad=.5))
 axs[0].axvline(1.,ls='--',lw=.8,color='#A36E37');axs[0].text(1.01,-.48,'约束上限1.0',fontsize=6)
 fig.tight_layout(rect=(0,.22,1,.98),w_pad=2);fig.text(.04,.04,'官方CARLA 0.9.16，2条冻结转向序列 × 2种控制，每组仅1次、510记录帧，未画重复试验误差线。\n原控制2/2满足标定条件，反馈控制1/2满足；低速Tesla扭矩限25 N·m，无传感器前端与实际到达信用。',fontsize=6.3);export(fig,'official_actuator_diagnostic')
 with (O/'automated_qa_001.json').open('x') as f:json.dump(dict(figures=QA,exports=EXPORTS,sources_sha256=SOURCES,script_sha256=sha(__file__),thesis_prose_written=False,goal_complete=False),f,ensure_ascii=False,indent=2)
 print(json.dumps(dict(figures=len(QA),files=len(EXPORTS),outside_text=[x['figure'] for x in QA if x['text_outside_canvas']]),ensure_ascii=False))
if __name__=='__main__':main()
