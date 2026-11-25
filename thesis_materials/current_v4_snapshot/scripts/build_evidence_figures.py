from pathlib import Path
import json,csv,hashlib,gzip,pickle,re
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from PIL import Image
O=Path(__file__).resolve().parents[1];B=O.parents[1]; C=B/'source_bundle/github_sync_001/checkout'
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
T={};F={};S=[]
def j(p):
 path=B/p; S.append({'type':'报告','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});return json.loads(path.read_text())
def table(key,title,headers,rows,source,note=''):
 T[key]=dict(title=title,headers=headers,rows=rows,source=source,note=note)
 with (O/'figure_data'/f'{key}.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(headers);w.writerows(rows)
def pct(x):return '未定义' if x is None else f'{100*x:.2f}'
N={'single_frame':'单帧','original_fusion':'原融合','timestamp_only':'只记录时间','one_to_one_no_expiry':'一对一无过期','one_to_one_timestamp':'一对一带时间','one_to_one_depth_quality':'深度质量','one_to_one_depth_robust_motion':'深度＋稳健运动'}
FIGURE_FILENAMES={
 'pipeline':'Figure3-1_公开录制数据研究流程',
 'case':'Figure4-1_nuScenes_scene-0039第2帧检测分割对象定位案例',
 'relations_case':'Figure4-2_nuScenes_scene-0039第2帧关系几何案例',
 'mapping':'Figure6-1_nuScenes对象建图比较',
 'retrieval':'Figure6-2_目标检索选对与误选权衡',
 'qwen':'Figure6-3_Qwen专项五模型比较',
 'maps':'Figure6-4_地图约束离线诊断',
}
def save(fig,key,source):
 stem=FIGURE_FILENAMES[key]
 fig.savefig(O/'figures'/f'{stem}.png',dpi=350,bbox_inches='tight');fig.savefig(O/'figures'/f'{stem}.pdf',bbox_inches='tight');plt.close(fig);F[key]={'path':str(O/'figures'/f'{stem}.png'),'source':source,'export':'350 dpi PNG + vector PDF','filename':stem,'generator':'scripts/build_evidence_figures.py','backend':'Python matplotlib'}
p='statistics/causal_mapping_full_001/report.json';m=j(p)['methods'];order=list(N)
table('mapping','表6-4 nuScenes对象建图结果',['条件','配对/5723','预测数','找全率/%','精确率/%','误差/m'],[[N[k],m[k]['all']['true_positive'],m[k]['all']['predictions'],pct(m[k]['all']['recall']['value']),pct(m[k]['all']['matched_prediction_precision']['value']),f"{m[k]['all']['center_error_m_mean']['value']:.3f}"] for k in order],p,'同类、地面XY中心≤3 m一对一匹配；误差仅匹配内。')
fig,axs=plt.subplots(1,3,figsize=(8.6,3.25))
for ax,metric,title,mul in zip(axs,['recall','matched_prediction_precision','center_error_m_mean'],['位置找全率 / %','匹配精确率 / %','匹配中心误差 / m'],[100,100,1]):
 for i,k in enumerate(order):
  v=m[k]['all'][metric];est=v['value']*mul;lo,hi=np.array(v['ci95'])*mul;ax.errorbar(est,i,xerr=[[est-lo],[hi-est]],fmt='o',color='#315b80',capsize=2)
 ax.set_yticks(range(len(order)),[N[k] for k in order] if ax is axs[0] else []);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'mapping',p)
p='statistics/temporal_identity_full_001/report.json';ids=j(p)['methods']
table('identity','表6-5 全序列身份及官方列表返回',['条件','IDTP','IDF1/%','相邻变化/连续匹配','返回同ID/变ID/未知'],[[N[k],ids[k]['all_gated']['counts']['IDTP'],pct(ids[k]['all_gated']['metrics']['IDF1']['value']),f"{ids[k]['all_gated']['counts']['adjacent_identity_changes']}/{ids[k]['all_gated']['counts']['adjacent_matched_transitions']}",'/'.join(str(ids[k]['all_gated']['counts']['official_return_'+s]) for s in ['same_id','changed_id','unknown_id'])] for k in order],p,'返回机会各167次；未关联端点为未知，不是人工遮挡标注。')
p='mapping_evaluation/causal_relations_full_001/summary.json';rel=j(p); rr=[]
for k,name in [('direct_geometry','直接几何'),('structured','结构化'),('incremental','增量')]:
 counts={v:sum(s['methods'][k][v] for s in rel['scenes']) for v in ['true_positive','predicted','truth','unresolved_edges']};a,b,c,u=[counts[x] for x in ['true_positive','predicted','truth','unresolved_edges']];rr.append([name,a,b,c,u,pct(a/b),pct(a/c),pct(2*a/(b+c)),pct(a/(b+u))])
table('relations','表6-6 对象关系评价',['方式','正确','可关联预测','参考','未解','精确率/%','找全率/%','F1/%','下界/%'],rr,p,'参考为官方几何规则关系；下界把全部未解关系计错。')
p='statistics/causal_retrieval_full_001/report.json';ret=j(p)['methods'];names=['类别最近','当前裁剪CLIP','对象图联合规则','稀疏CLIP网格'];keys=list(ret)
table('retrieval','表6-7 公开数据目标检索',['方法','选对/733','选对率/%','误选/382','误选率/%'],[[name,ret[k]['positive_selection']['numerator'],pct(ret[k]['positive_selection']['estimate']),ret[k]['no_match_false_selection']['numerator'],pct(ret[k]['no_match_false_selection']['estimate'])] for name,k in zip(names,keys)],p,'构造开发指令；关联协议略有差异；网格为VLMaps风格近似实现。')
fig,axs=plt.subplots(1,2,figsize=(8.1,3.1))
for ax,metric,title in zip(axs,['positive_selection','no_match_false_selection'],['正查询选对率 / %','无匹配误选率 / %（越低越好）']):
 for i,k in enumerate(keys):
  d=ret[k][metric];v=100*d['estimate'];lo,hi=np.array(d['ci95'])*100;ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#315b80')
 ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlabel(title);ax.set_xlim(left=0);ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'retrieval',p)
p='statistics/fair_three_task_full_validation_001/report.json';q=j(p)['tasks'];qkeys=list(q['query_parse']['models']);qn=['基础Qwen','旧Adapter','新Adapter17','新Adapter29','新Adapter43']
table('qwen','表6-8 Qwen专项五模型比较',['模型','解析正确/180','解析/%','类别正确/243','类别/%','排序正确/180','排序/%'],[[name,*sum(([q[t]['models'][k]['correct'],pct(q[t]['models'][k]['accuracy'])] for t in ['query_parse','object_attributes','candidate_rerank']),[])] for k,name in zip(qkeys,qn)],p,'解析所有字段正确且无回退；类别不含颜色/细属性；排序为符号bank。')
fig,axs=plt.subplots(1,3,figsize=(8.5,3.2))
for ax,t,title in zip(axs,['query_parse','object_attributes','candidate_rerank'],['解析（180）','对象类别（243）','候选排序（180）']):
 vals=[q[t]['models'][k]['accuracy']*100 for k in qkeys];ax.barh(qn,vals,color=['#b7c7d3','#8fa6bb','#537c9b','#537c9b','#537c9b']);ax.invert_yaxis();ax.set_xlim(0,107);ax.set_title(title);ax.set_xlabel('正确率 / %')
 for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=8)
fig.tight_layout();save(fig,'qwen',p)
p='statistics/causal_modules_full_001/report.json';m=j(p)['conditions'];table('modules','表6-9 对象维护模块消融',['条件','找全率/%','精确率/%','匹配误差/m','相邻变化率/%'],[[name,pct(m[k]['recall']['estimate']),pct(m[k]['precision']['estimate']),f"{m[k]['center_error_m_mean']['estimate']:.3f}",pct(m[k]['identity_switch_rate']['estimate'])] for k,name in [('full','完整'),('without_dynamic_maintenance','去动态维护'),('without_short_term_expiry','去短期过期')]],p,'相邻变化率仅连续匹配内；完整为深度＋稳健运动。')
p='ablations/causal_selection_ablation_full_001/report.json';m=j(p)['conditions'];table('selection_ablation','表6-10 选择证据与关系边消融',['条件','选对/733','误选/382','不选择/1115','关联未知'],[[name,v['positive_selected_correct'],v['no_match_false_selection']['numerator'],v['no_selection'],v['association_unknown']] for (k,v),name in zip(m.items(),['完整联合规则','去选择端深度门限','去关系边（保留几何回退）'])],p,'固定对象图，共3345条记录。')
p='ablations/causal_goals_full_001/report.json';m=j(p)['conditions'];table('goals','表6-11 目标与接近保护消融',['条件','选对/733','误选/382','路线/1115','道路通过/路线','车身通过/路线','≥2m/路线'],[[name,v['positive_selection']['numerator'],v['no_match_false_selection']['numerator'],v['route_generated']['count'],f"{v['full_road_legal']['count']}/{v['selected']}",f"{v['static_body_pose_legal']['count']}/{v['selected']}",f"{v['endpoint_2m_from_predicted_bbox']['count']}/{v['selected']}"] for (k,v),name in zip(m.items(),['首个可达/同池','联合安全接近','去接近保护'])],p,'≥2m相对预测bbox；无车辆执行，不代表真实净空。')
p='statistics/causal_maps_full_001/report.json';m=j(p);rows=[];mapnames=['自由空间','可行驶区','车道区域','车道＋方向','车道＋连接','车道＋方向＋连接']
for (k,v),name in zip(m['conditions'].items(),mapnames):
 n=m['statuses'][k]['PATH_GENERATED'];road=round(v['full_road_legal_among_routes']['estimate']*n);lane=round(v['full_lane_legal_among_routes']['estimate']*n);body=round(v['static_rectangle_legal_among_routes']['estimate']*n);rows.append([name,f'{n}/212',f'{lane}/{n}',f'{road}/{n}',f'{body}/{n}',m['statuses'][k].get('PLANNING_WALL_BUDGET',0)])
table('maps','表6-12 地图约束离线诊断',['约束','生成路线','车道通过/路线','道路通过/路线','车身通过/路线','预算耗尽'],rows,p,'给定正确对象；212查询×6条件，16次共同投影失败保留。')
fig,axs=plt.subplots(1,2,figsize=(8.6,3.8));x=np.arange(6)
axs[0].barh(x-.16,[v['route_generated']['estimate']*100 for v in m['conditions'].values()],height=.3,label='生成路线/全部查询',color='#9cb5c9');axs[0].barh(x+.16,[v['full_road_legal_all_queries']['estimate']*100 for v in m['conditions'].values()],height=.3,label='道路通过/全部查询',color='#315b80');axs[0].set_yticks(x,mapnames);axs[0].invert_yaxis();axs[0].legend(fontsize=7);axs[0].set_xlabel('占212查询 / %')
vals=[v['static_rectangle_legal_among_routes']['estimate']*100 for v in m['conditions'].values()];axs[1].barh(x,vals,color='#b08060');axs[1].set_yticks(x,mapnames);axs[1].invert_yaxis();axs[1].set_xlabel('静态车身通过 / 已生成路线 / %');axs[1].set_xlim(0,3.5)
for i,r in enumerate(rows):axs[1].text(vals[i]+.05,i,r[4],va='center',fontsize=8)
fig.tight_layout();save(fig,'maps',p)
p='statistics/kitti_object_cpu_full_001/report.json';m=j(p)['metrics'];table('kitti_object','表6-13 KITTI Object单帧诊断',['深度方式','2D匹配/319','3D匹配/319','全检测精确率/%','共同误差/m','匹配内误差/m','深度未知'],[[name,v['matches_2d'],v['matches_3d'],pct(v['precision_3d_all_detections']),f"{v['conditional_center_error_m']:.3f}",f"{v['matched_center_error_m']:.3f}",v['unknown_depth_predictions']] for (k,v),name in zip(m.items(),['普通均值','稳健均值'])],p,'365检测；二维IoU≥0.5，同类3 m中心匹配；共同支持251对；非官方AP。')
p='statistics/kitti_tracking_full_001/report.json';m=j(p);table('kitti_tracking','表6-14 KITTI Tracking短窗口诊断',['条件','配对/74','预测数','找全率/%','匹配预测/%','误差/m','IDF1/%'],[[N[k],v['counts']['true_positive'],v['counts']['predictions'],pct(v['metrics']['recall']),pct(v['metrics']['matched_prediction_fraction']),f"{v['metrics']['center_error_m']:.3f}",pct(m['identity']['methods'][k]['all_gated']['metrics']['IDF1']['value'])] for k,v in m['object']['methods'].items()],p,'全部支持GT集中在0005；保留465条不支持/DontCare。')
p='statistics/causal_mapping_parameters_full_001/report.json';m=j(p)['conditions'];table('mapping_parameters','表6-15 关联分数门限敏感性',['推理门限','找全率/%','精确率/%','误差/m','相邻变化率/%'],[[k,pct(v['recall']['estimate']),pct(v['precision']['estimate']),f"{v['center_error_m_mean']['estimate']:.3f}",pct(v['identity_switch_rate']['estimate'])] for k,v in m.items()],p,'114场景条件，评价3 m固定；基线复用38条件。')
p='performance/causal_selection_parameters_full_001/report.json';m=j(p)['conditions'];table('selection_parameters','表6-16 目标选择参数敏感性',['参数条件','选对/733','误选/382','Top-K覆盖/733'],[[k.replace('minimum_observations_','最少观测').replace('minimum_depth_support_fraction_','深度比例').replace('rejection_score_','拒绝分数').replace('baseline','基线'),v['positive_selection']['count'],v['no_match_false_selection']['count'],v['positive_topk_coverage']['count']] for k,v in m.items()],p,'10035条条件记录，不能视作独立场景；未用开发最优值改历史版本。')
table('interfaces','表3-1 公开数据模块接口',['模块','主要输入','输出与失败状态'],[['观测整理','RGB、过去雷达、标定、已知位姿','MappingFrame、时间差、缺测'],['对象建图','掩膜、稀疏深度、历史对象','对象快照、内部ID、深度支持'],['关系维护','对象中心、参考方向、历史边','关系类型、时效、未解端点'],['语义与检索','指令、裁剪、候选、关系','结构化条件、候选、拒绝原因'],['离线规划','预测目标/诊断参考、地图、障碍','接近位置、路线、预算/搜索失败'],['独立评价','冻结GT及方法保存输出','配对、选对、几何检查']], '对应代码接口，见代码索引')
table('supervision','表3-2 标准答案及监督性质',['来源','用途','审核与边界'],[['官方类别/中心/实例','对象与身份评价','官方标注，非本项目人审'],['官方中心＋固定几何规则','关系评价','自动几何辅助参考'],['裁剪与官方类别自动匹配','属性训练/验证','弱监督，仅支持类别'],['构造表达＋结构化规则','解析训练/验证','非全部自然人工指令'],['受控符号bank','排序训练/验证','非真实感知候选标注'],['独立GT参考对象/点','给定目标规划诊断','目标作为诊断输入，区域未核验安全']], '数据manifest与各评价协议')
table('qwen_data','表5-1 专项监督规模',['任务','训练','验证','答案性质'],[['指令解析',1400,180,'构造表达/结构化答案'],['对象属性（类别）',278,243,'真实裁剪＋官方类别弱监督'],['候选排序',720,180,'受控符号bank/候选置换'],['合计',2398,603,'非新增道路场景']], 'semantic_dataset/structured_supervision_002/manifest.json')
env=j('source_bundle/github_sync_001/runtime_environment.json');pk={x['name']:x['version'] for x in env['packages']};table('platform','表6-1 科研运行环境与关键配置',['项目','记录'],[['历史GPU','NVIDIA GeForce RTX 4090，24564 MiB（环境快照）'],['Python/系统',env['python'].split('(')[0]+' / Linux 5.15'],['PyTorch / Transformers',pk.get('torch','未知')+' / '+pk.get('transformers','未知')],['PEFT / CUDA构建',pk.get('peft','未知')+' / PyTorch构建依历史环境'],['视觉前端','既有YOLO/SAM/CLIP，固定RGB结果复用'],['专项模型','Qwen2-VL-2B-Instruct；bf16 LoRA；3种子×240步'],['处理范围','CPU对象重建/离线诊断，CUDA专项验证；共享主机']], 'source_bundle/environment_before.json；source_bundle/github_sync_001/runtime_environment.json')
table('datasets','表6-2 数据规模及使用状态',['数据','规模','使用与限制'],[['nuScenes开发','38场景 / 11log / 1508帧','1470有过去雷达＋38缺测首帧'],['nuScenes预留','30场景 / 14log / 1196帧','新封存模型测试未执行'],['KITTI Object','64单帧 / 33drive','单帧识别定位'],['KITTI Tracking','3序列 / 90帧','74次支持GT集中于0005的4实例'],['Qwen真实图像来源','训练25场景/7log；验证11场景/2log','nuScenes开发子集'],['Qwen专项','2398训练 / 603验证','含构造表达与符号排序'],['Talk2Car已准备','439英文指令 / 360不同图像帧','没有正式模型成绩']], '各数据manifest与实验报告')
table('metrics','表6-3 指标定义与未知处理',['指标','分子/分母或计算条件','未知处理'],[['位置找全率','同类XY≤3m一对一配对 / 全GT观测','缺测、漏检不删分母'],['匹配精确率','同样配对 / 全预测观测','未配对保留'],['匹配中心误差/m','成功配对中心距离均值','不替代覆盖指标'],['IDF1','2IDTP/(2IDTP+IDFP+IDFN)','整段序列身份分配'],['关系P/R/F1','可关联预测、全部参考分别作分母','未解边单列＋下界'],['目标选对率','正确目标 / 733正查询','未知与不选不计成功'],['无匹配误选率','仍选择 / 382无匹配查询','越低越好'],['路线生成率','生成路线 / 全冻结查询','预算/搜索失败保留'],['道路/车身通过率','检查通过 / 已生成路线','同时给全部查询分母']], '评价代码与报告口径')
# 真实图像、掩膜、保存三维预测；不进行模型推理。
p=B/'mapping_evaluation/causal_mapping_full_001/scene-0039/frames.jsonl';frame=json.loads(p.read_text().splitlines()[2]);rgb=np.asarray(Image.open(frame['rgb']));
with gzip.open(frame['detections'],'rb') as f:det=pickle.load(f)
fig,axs=plt.subplots(2,2,figsize=(8.4,5.2));axs[0,0].imshow(rgb);axs[0,0].set_title('(a) 原始图像');axs[0,1].imshow(rgb);overlay=np.zeros((*rgb.shape[:2],4));colors=plt.get_cmap('tab10')
for i,mask in enumerate(det['mask']):overlay[mask]=[*colors(i%10)[:3],.45]
axs[0,1].imshow(overlay)
for i,box in enumerate(det['xyxy']):
 x,y,x2,y2=box;axs[0,1].add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor=colors(i%10),linewidth=1));axs[0,1].text(x,y,str(det['classes'][int(det['class_id'][i])]),color='white',fontsize=6,bbox={'facecolor':'black','alpha':.5,'pad':1})
axs[0,1].set_title('(b) 已有检测框与实例掩膜')
for ax in axs[0]:ax.axis('off')
dep=np.asarray(Image.open(frame['depth']))/1000.;ys,xs=np.where(dep>0);axs[1,0].imshow(rgb);sc=axs[1,0].scatter(xs,ys,c=dep[ys,xs],s=.5,cmap='viridis',vmin=0,vmax=65);axs[1,0].axis('off');axs[1,0].set_title('(c) 保存的过去LiDAR稀疏投影');fig.colorbar(sc,ax=axs[1,0],fraction=.025,label='m')
origin=np.asarray(frame['camera_pose'])[:2,3];pred=frame['predictions']['one_to_one_depth_robust_motion'];gt=frame['evaluation_only_official'];a=axs[1,1]
for objects,marker,label,col in [(gt,'x','官方参考','#b08060'),(pred,'o','预测中心','#315b80')]:
 xy=np.array([v['center'][:2] for v in objects])-origin;a.scatter(xy[:,0],xy[:,1],marker=marker,label=label,c=col,s=25)
for match in frame['metrics']['one_to_one_depth_robust_motion']['matches']:
 p0=np.asarray(pred[match['prediction_index']]['center'][:2])-origin;p1=np.asarray(gt[match['truth_index']]['center'][:2])-origin;a.plot([p0[0],p1[0]],[p0[1],p1[1]],color='#aaaaaa',linewidth=.8)
a.set_aspect('equal',adjustable='datalim');a.set_xlabel('相对世界X / m');a.set_ylabel('相对世界Y / m');a.legend(fontsize=7);a.set_title('(d) 4/7对象配对，3次未找全');fig.tight_layout();save(fig,'case',[str(p),frame['rgb'],frame['depth'],frame['detections']]);(O/'figure_data/case_frame.json').write_text(json.dumps(frame,ensure_ascii=False,indent=2))
# 关系图：只重绘保存中心的直接几何规则，不重算模型结果或统计分数。
pred=frame['predictions']['one_to_one_depth_quality'];fwd=np.array(frame['camera_pose'])[:2,2];fwd/=np.linalg.norm(fwd);left=np.array([-fwd[1],fwd[0]]);xy=np.array([x['center'][:2] for x in pred]);xy-=origin;fig,ax=plt.subplots(figsize=(7.4,3.9));ax.scatter(xy[:,0],xy[:,1],c='#315b80',s=55)
relations=[]
for i in range(len(pred)):
 ax.annotate(pred[i]['node_id'].replace('track-000000','id'),xy[i]+[.3,.2],fontsize=8)
 for k in range(i+1,len(pred)):
  delta=xy[k]-xy[i];d=np.linalg.norm(delta)
  if d>12:continue
  df,dl=delta@fwd,delta@left;terms=['近' if d<=6 else '远'];terms+=['前' if df>0 else '后'] if abs(df)>=1 else [];terms+=['左' if dl>0 else '右'] if abs(dl)>=1 else [];label='/'.join(terms);ax.annotate('',xy[k],xy[i],arrowprops={'arrowstyle':'->','color':'#8298aa'});mid=(xy[i]+xy[k])/2;ax.text(*mid,label,fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9});relations.append({'source':pred[i]['node_id'],'target':pred[k]['node_id'],'distance_m':float(d),'relations':terms})
ax.quiver(0,0,*fwd,color='#b08060',angles='xy',scale_units='xy',scale=.25);ax.text(0,0,'相机前向',fontsize=8);ax.set_xlabel('相对世界X / m');ax.set_ylabel('相对世界Y / m');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);fig.tight_layout();save(fig,'relations_case',[str(p),'mapping_evaluation/causal_relations_full_001/release.json']);(O/'figure_data/relations_case.json').write_text(json.dumps(relations,ensure_ascii=False,indent=2))
# 无实验图像的流程示意。
fig,ax=plt.subplots(figsize=(8.5,3.25));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off');boxes=[(.1,2.5,'RGB＋过去LiDAR\n标定＋已知位姿'),(3.4,2.5,'对象图与关系图\n深度/关联/时效'),(6.8,2.5,'语言目标候选\n证据不足/拒绝'),(6.8,.3,'接近位置与离线路线\n道路与车身检查'),(3.4,.3,'独立评价\n对象/关系/选对/几何')]
for x,y,label in boxes:ax.add_patch(FancyBboxPatch((x,y),2.7,1,boxstyle='round,pad=.08',facecolor='#e4ecf2',edgecolor='#537c9b'));ax.text(x+1.35,y+.5,label,ha='center',va='center')
for a,b in [((2.9,3),(3.3,3)),((6.2,3),(6.7,3)),((8.1,2.4),(8.1,1.4)),((6.7,.8),(6.2,.8)),((4.7,2.4),(4.7,1.4))]:ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#315b80','lw':1.4})
ax.text(.2,.8,'GT只进入独立评价\n图中不包含车辆控制回路',fontsize=9,color='#6c6259');save(fig,'pipeline','本论文第3章方法定义')
# 证据与代码位置索引；已读报告、逐帧索引全量解析。
for pth in ['statistics/causal_modules_motion_full_001/report.json','semantic_dataset/structured_supervision_002/manifest.json','training/three_task_v2_formal_seed17_001/release.json','main_experiments/fair_three_task_full_validation_001/release.json','main_experiments/fair_three_task_full_validation_001/report.json','validation_report.json','最终代码与实验交付回执_031.json']:j(pth)
for p in [C/'conceptgraph/completion/development_causal_inputs.py',C/'conceptgraph/slam/streamlined_mapping.py',C/'conceptgraph/completion/depth.py',C/'conceptgraph/completion/association.py',C/'conceptgraph/completion/motion.py',C/'conceptgraph/scenegraph/relations.py',C/'conceptgraph/scenegraph/incremental_graph.py',C/'conceptgraph/navigation/semantic.py',C/'conceptgraph/completion/stage9.py']:
 S.append({'type':'代码','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for key,v in F.items():
 for source in v['source'] if isinstance(v['source'],list) else [v['source']]:
  p=Path(source);p=p if p.is_absolute() else B/p
  if p.exists():S.append({'type':'图像/图表来源','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'figure':key})
index=B/'dataset_results_guide_20261115/逐帧结果位置索引.csv';records=list(csv.DictReader(index.open(encoding='utf-8-sig')));(O/'evidence/逐帧索引核对.json').write_text(json.dumps({'rows':len(records),'datasets':{k:sum(r['dataset']==k for r in records) for k in sorted({r['dataset'] for r in records})},'source':str(index),'sha256':hashlib.sha256(index.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
table('source_index','表B-1 主要证据入口（均相对历史交付根目录）',['内容','报告/记录入口'],[['对象建图','statistics/causal_mapping_full_001/report.json'],['逐帧对象','mapping_evaluation/causal_mapping_full_001/scene-*/frames.jsonl'],['连续身份','statistics/temporal_identity_full_001/report.json'],['关系','mapping_evaluation/causal_relations_full_001/summary.json'],['检索','statistics/causal_retrieval_full_001/report.json'],['专项对比','statistics/fair_three_task_full_validation_001/report.json'],['维护/运动','statistics/causal_modules[_motion]_full_001/report.json'],['地图','statistics/causal_maps_full_001/report.json'],['选择/接近','ablations/causal_selection_ablation_full_001；causal_goals_full_001'],['KITTI','statistics/kitti_object_cpu_full_001；kitti_tracking_full_001'],['参数','statistics/causal_mapping_parameters_full_001；performance/causal_selection_parameters_full_001'],['代码说明','source_bundle/github_sync_001/checkout/thesis_materials/']], '完整绝对位置见随稿CSV。')
(O/'sources/tables.json').write_text(json.dumps(T,ensure_ascii=False,indent=2));(O/'sources/figures.json').write_text(json.dumps(F,ensure_ascii=False,indent=2));(O/'evidence/source_hashes.json').write_text(json.dumps(S,ensure_ascii=False,indent=2))
with (O/'实验_图表_代码_原始结果索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['type','path','sha256','figure'],extrasaction='ignore');w.writeheader();w.writerows(S)
print('tables',len(T),'figures',len(F),'indexed sources',len(S))
