from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib,re,zipfile,xml.etree.ElementTree as ET
O=Path(__file__).resolve().parents[1];B=O.parents[1];P=B.parents[2];checks=[];idx=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,kind,field=''):
 p=Path(p);p=p if p.is_absolute() else B/p
 assert p.exists(),p
 idx.append({'type':kind,'path':str(p),'field_or_lines':field,'sha256':digest(p)});return p
def check(name,actual,expected):
 passed=actual==expected;checks.append({'name':name,'actual':actual,'expected':expected,'passed':passed});assert passed,(name,actual,expected)
def load(p):return json.loads((B/p).read_text())
# 回到实际逐帧记录汇总，不改变报告和标准答案。
sums=defaultdict(Counter);frames=0
for p in sorted((B/'mapping_evaluation/causal_mapping_full_001').glob('scene-*/frames.jsonl')):
 record(p,'逐帧对象与评价','每行frame_idx；predictions/metrics');
 for line in p.open():
  r=json.loads(line);frames+=1
  for method,m in r['metrics'].items():
   for key in ['predictions','truth','true_positive','identity_switches']:
    if key in m and m[key] is not None:sums[method][key]+=m[key]
report=load('statistics/causal_mapping_full_001/report.json');check('对象帧数',frames,1508)
for k,v in sums.items():
 for key in ['predictions','truth','true_positive']:check('对象/'+k+'/'+key,v[key],report['methods'][k]['all'][key])
rp=record('mapping_evaluation/causal_relations_full_001/frames.jsonl','逐帧关系','scores各方法；端点未知与一致性');ss=defaultdict(Counter);n=0;consistent=0
for line in rp.open():
 r=json.loads(line);n+=1;consistent+=bool(r['scores']['incremental_full_content_consistency'])
 for method in ['direct_geometry','structured','incremental']:
  for key in ['true_positive','predicted','truth','unresolved_edges']:ss[method][key]+=r['scores'][method][key]
check('关系帧数',n,1508);check('关系一致帧',consistent,1396);summary=load('mapping_evaluation/causal_relations_full_001/summary.json')
for method,c in ss.items():
 for key,v in c.items():check('关系/'+method+'/'+key,v,sum(s['methods'][method][key] for s in summary['scenes']))
rp=record('main_experiments/causal_retrieval_full_001/queries.jsonl','逐查询检索','positive/selected_correct/false_selection_on_no_match');ss=defaultdict(Counter)
for line in rp.open():
 r=json.loads(line);c=ss[r['method']];c['rows']+=1;c['positive']+=bool(r['positive']);c['correct']+=r['selected_correct'] is True;c['false_selection']+=r['false_selection_on_no_match'] is True
report=load('statistics/causal_retrieval_full_001/report.json')
for k,c in ss.items():
 check('检索/'+k+'/rows',c['rows'],1115);check('检索/'+k+'/positive',c['positive'],733);check('检索/'+k+'/correct',c['correct'],report['methods'][k]['positive_selection']['numerator']);check('检索/'+k+'/false',c['false_selection'],report['methods'][k]['no_match_false_selection']['numerator'])
for run,expected in [('ablations/causal_selection_ablation_full_001/queries.jsonl',3345),('ablations/causal_goals_full_001/queries.jsonl',3345),('performance/causal_selection_parameters_full_001/queries.jsonl',10035),('ablations/causal_maps_full_001/cases.jsonl',1272)]:
 p=record(run,'逐查询/条件原始记录');check(run+'/rows',sum(1 for _ in p.open()),expected)
# 从地图逐条审计重核2/94及94/212。
p=B/'ablations/causal_maps_full_001/cases.jsonl';ss=defaultdict(Counter)
for line in p.open():
 r=json.loads(line);c=ss[r['condition']];e=r['evaluation'];c['queries']+=1;c['routes']+=e['path_generated'] is True;c['body']+=e.get('body_pose_legal') is True;c['road']+=e['full_lane_legal'] is True and e.get('continuous_point_drivable') is True
check('完整地图查询',ss['lane_direction_connection']['queries'],212);check('完整地图路线',ss['lane_direction_connection']['routes'],94);check('完整地图静态车身',ss['lane_direction_connection']['body'],2)
# 公平五模型逐请求评分及原始模型响应位置。
p=record('statistics/fair_three_task_full_validation_001/paired_request_scores.csv','逐请求语义评分','semantic_correct/schema_valid/fallback');ss=defaultdict(Counter);rows=list(csv.DictReader(p.open()))
for r in rows:ss[(r['model'],r['task'])]['correct']+=r['semantic_correct']=='True';ss[(r['model'],r['task'])]['requests']+=1
report=load('statistics/fair_three_task_full_validation_001/report.json');check('五模型逐请求数',len(rows),3015)
for (model,task),c in ss.items():
 check(model+'/'+task+'/correct',c['correct'],report['tasks'][task]['models'][model]['correct']);check(model+'/'+task+'/requests',c['requests'],report['tasks'][task]['requests'])
for model in ['base_fullschema','old_adapter_fullschema','new_adapter_seed17','new_adapter_seed29','new_adapter_seed43']:
 p=record('main_experiments/fair_three_task_full_validation_001/'+model+'/raw_records.jsonl','原始模型响应','envelope.raw_output与normalized_output');check(model+'/raw rows',sum(1 for _ in p.open()),603)
# 补充入口、代码、输入指南，完整读取并记录SHA。
for p in ['ablations/causal_modules_full_001/cases.jsonl','performance/causal_mapping_parameters_full_001/cases.jsonl','statistics/kitti_object_cpu_full_001/records.json','statistics/kitti_object_cpu_full_001/common_supported_pairs.json','statistics/kitti_tracking_full_001/0000_frames.jsonl','statistics/kitti_tracking_full_001/0005_frames.jsonl','statistics/kitti_tracking_full_001/0013_frames.jsonl','statistics/temporal_identity_full_001/scene_method_support.jsonl','semantic_dataset/structured_supervision_002/train.jsonl','semantic_dataset/structured_supervision_002/validation.jsonl']:
 record(p,'原始观测/监督/条件')
required=['工作交接_代码与实验.md','validation_report.json','最终代码与实验交付回执_031.json','已完成实验与结果目录索引_20261115.md','dataset_results_guide_20261115/数据集与指标结果详单.md','dataset_results_guide_20261115/逐帧结果位置索引.csv']
materials=B/'source_bundle/github_sync_001/checkout/thesis_materials'
for p in [*(B/x for x in required),*materials.rglob('*.md'),materials/'代码符号索引.csv',P/'论文1-3章初稿.md',P/'初稿.docx']:
 record(p,'已读取交接/原稿/代码说明');
 if p.suffix=='.json':json.loads(p.read_text())
 elif p.suffix in ['.md','.csv']:p.read_text(encoding='utf-8-sig')
 else:
  z=zipfile.ZipFile(p);ET.fromstring(z.read('word/document.xml'))
record(B/'source_bundle/github_sync_001/checkout/conceptgraph/scripts/streamlined_detections.py','冻结前端装载代码','load_detection_models；YOLO-World/MobileSAM/OpenCLIP')
# 原有Git索引和原稿的保护基准；只读校验，不刷新Git索引。
protected=[P/'初稿.docx',P/'论文1-3章初稿.md',P/'.git/index',B/'validation_report.json']
pp=[{'path':str(p),'sha256':digest(p)} for p in protected if p.exists()];saved=O/'qa/protected_sources.json'
if saved.exists():check('受保护文件保持',pp,json.loads(saved.read_text()))
else:saved.write_text(json.dumps(pp,ensure_ascii=False,indent=2))
# 索引中所有非空文件引用核对，不把索引作为数值来源。
p=B/'dataset_results_guide_20261115/逐帧结果位置索引.csv';rs=list(csv.DictReader(p.open(encoding='utf-8-sig')));missing=[]
for r in rs:
 for key in ['rgb','depth','detection_and_mask','preview','mapping','relation']:
  if r[key] and not Path(r[key]).exists():missing.append(r[key])
check('逐帧位置索引行数',len(rs),1662);check('逐帧引用缺失',len(missing),0)
allidx=json.loads((O/'evidence/source_hashes.json').read_text())+idx
with (O/'实验_图表_代码_原始结果索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['type','path','field_or_lines','sha256','figure'],extrasaction='ignore');w.writeheader();w.writerows(allidx)
(O/'qa/evidence_audit.json').write_text(json.dumps({'checks':checks,'passed':all(c['passed'] for c in checks),'raw_maps':ss if False else '已逐条核对','indexed_sources':len(allidx),'no_models_executed':True},ensure_ascii=False,indent=2));print('证据检查',len(checks),'全部通过，索引',len(allidx))
