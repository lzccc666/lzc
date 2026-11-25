"""最终独立分母/分log/成本和保护交付；只读已完成结果，不新增实验。"""
import csv,json,shutil
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from conceptgraph.project_paths import PROJECT_ROOT,experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump
from scripts.multiview_identity_repair_20261123_independent_scores import write_csv

def main():
 root=PROJECT_ROOT;r=root/'outputs/reports/multiview_viewcount_20261123';ind=r/'independent';sensor=r/'sensors';results=root/'outputs/results/nuscenes/validation/iterations';allrows=[];scene=[];validation=[]
 def read(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
 fields={'TP':'true_positive','GT':'truth','predictions':'predictions','FP':'unmatched_predictions','FN':'false_negative','duplicates':'duplicate_nodes','IDTP':'IDTP','IDFN':'IDFN','IDFP':'IDFP'}
 for batch,tag in [('main','main_v1'),('sharedscan','sharedscan_v1')]:
  independent=read(ind/(tag+'_main.csv'));reference=json.loads((r/'mapping'/tag/'mv136_20261123_result.json').read_text());official={(x['method'],x['scope']):x for x in reference['results']['node_main']}
  for row in independent:
   method=row['method'].split(':',1)[1];given=official[method,row['scope']]
   assert all(int(row[k])==given[v] for k,v in fields.items())
   assert all(abs(float(row[k])-given[k])<1e-12 for k in ['precision','recall','F1','IDF1']);allrows.append(dict(batch=batch,**row))
  assert len(independent)==18;validation.append(dict(batch=batch,all18_independent_rows_equal_mapper=True,run=reference['run'],mapper_wall_s=reference['wall_s'],mapper_costs_s=reference['cost_total_s']))
  scene.extend(dict(batch=batch,**row) for row in read(ind/(tag+'_scene.csv')))
 groups=defaultdict(list)
 for row in scene:groups[row['batch'],row['method'],row['scope'],row['log_token']].append(row)
 logs=[]
 sumkeys=['TP','GT','predictions','FP','FN','duplicates','IDTP','IDFN','IDFP','identity_GT','identity_predictions','adjacent_matched_transitions','adjacent_ID_changes']
 for (batch,method,scope,log),rr in groups.items():
  row=dict(batch=batch,method=method,scope=scope,log_token=log,scenes=';'.join(x['scene'] for x in rr),**{k:sum(int(x[k]) for x in rr) for k in sumkeys});row.update(precision=row['TP']/row['predictions'] if row['predictions'] else None,recall=row['TP']/row['GT'],F1=2*row['TP']/(row['GT']+row['predictions']),IDF1=2*row['IDTP']/(row['identity_GT']+row['identity_predictions']));logs.append(row)
 assert len(allrows)==36 and len(scene)==216 and len(logs)==180;write_csv(ind/'完整主与同scan主表.csv',allrows);write_csv(ind/'完整逐scene表.csv',scene);write_csv(ind/'完整逐log表.csv',logs)
 cost=read(sensor/'subset_model_cost.csv');mapper=json.loads((r/'mapping/main_v1/mv136_20261123_result.json').read_text())['cost_total_s'];costrows=[]
 for row in cost:
  n=row['views'];costrows.append(dict(views=n,required_images=int(row['required_total_image_calls']),historical_reused_images=int(row['historical_front_cache_reused']),new_rear_images=int(row['new_rear_image_calls']),recorded_image_model_total_s=float(row['sum_actual_per_image_model_wall_s']),mapper_update_total_s=mapper['T3_'+n],sum_image_model_plus_mapper_s=float(row['sum_actual_per_image_model_wall_s'])+mapper['T3_'+n],sum_cost_per_group_s=(float(row['sum_actual_per_image_model_wall_s'])+mapper['T3_'+n])/120,interpretation='additive historical per-image infer + current CPU mapper timing, excludes loading/io and model-load, not measured end-to-end or realtime/parallel latency'))
 write_csv(ind/'模型与建图分列成本.csv',costrows)
 run=experiment_directory('nuscenes','validation','mv136_20261123_independent_delivery_v1',create=True);sources={}
 for p in sorted((root/'scripts').glob('mv136_20261123_*.py')):
  target=run/'source_snapshot'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(p)]=dict(sha256=digest(p),snapshot=str(target))
 protect=json.loads((sensor/'delivery_verification.json').read_text());ownroots=[root/'dataset/raw/mv136_20261123_sixscene_v1',root/'dataset/processed/mv136_20261123_sixscene_v1',root/'dataset/processed/mv136_20261123_sharedscan_v1',sensor,ind,results/'mv136_20261123_frontend_v1',results/'mv136_20261123_sharedscan_inputs_v1',results/'mv136_20261123_sensors_delivery_v1',run];ownroots += list(results.glob('mv136_independent_*'));files={p.resolve() for base in ownroots for p in base.rglob('*') if p.is_file()};size=sum(p.stat().st_size for p in files);free=shutil.disk_usage(root).free
 verify=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_snapshots=sources,comparisons=validation,registered_protection_reference=str(sensor/'delivery_verification.json'),registered_protected_unchanged=protect['protected_unchanged'],protected_count=13,input_manifest_sha256=protect['manifest_sha256'],GT_sha256=protect['GT_sha256'],rows=dict(pooled=36,scene=216,log=180),identity_aggregation='independent one-to-one identity matching per original scene; sums across logs/scenes, no global remapping',scoped_new_bytes=size,scoped_new_MiB=size/2**20,budget_new_MiB=640,free_GiB=free/2**30,new_model_calls=360,new_GPU_wall_s=47.86386010423303,reused_front_views=360,sharedscan_model_calls=0,sharedscan_reused_views=720,training_calls=0,test_opened=False,method_tuning_after_results=False,stage_started_utc='2026-11-23T14:39:00Z',deadline_utc='2026-11-23T15:24:00Z')
 assert size<640*2**20 and free>20*2**30;dump(ind/'最终独立交付核验.json',verify)
 by={(x['batch'],x['method'].split(':',1)[1],x['scope']):x for x in allrows}
 lines=['# 1/3/6视角固定方案独立结论','','六视角明显扩大全周覆盖，同时降低精确率、增加重复；当前前向语义任务宜以三视角作为主要方案，六视角作为全周/后向覆盖对照。该建议基于本轮已看验证/回归数据，不修改默认方法或论文原稿，不宣称六视角在所有任务都更好。','','协议为6scene/5log/120时刻、同六路最新曝光clock、同六类与3m XY匹配；主分母固定全六路GT1181。front3为687、strictfront385、rear-only494。各scene先独立身份分配后累加，同一输入与唯一变化是固定T3v2可用相机子集；A/C为固定背景对照。所有时刻及缺深度均保留，非新测试、非闭环车辆实验。','','|输入/方法|TP/1181|预测|重复|P %|R %|F1 %|IDF1 %|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for batch,method in [('main','T3_1'),('main','C_3'),('main','T3_3'),('main','C_6'),('main','T3_6'),('sharedscan','T3_6')]:
  x=by[batch,method,'union6'];lines.append(f'|{batch}/{method}|{x["TP"]}|{x["predictions"]}|{x["duplicates"]}|{100*float(x["precision"]):.2f}|{100*float(x["recall"]):.2f}|{100*float(x["F1"]):.2f}|{100*float(x["IDF1"]):.2f}|')
 lines+=['','主三→六全周召回26.67→54.02%，F1 36.67→48.61%；rear-only匹配6/494→328/494。全周precision58.66→44.18%，重复11→84；相对C6，T3six得5失12、净少7个GT对象帧，尽管少112预测且重复124→84，不能只宣传精度提升。原完整基线与全部负项在36行总表，未择优删行。','','前向三路区域的三→六：TP309→310/687，precision57.54→56.57%，IDF1 45.92→44.70%，重复11→16；说明新增后视会干扰部分已有对象的身份/竞争。strictfront三/六均142/385、239预测、3重复、P59.41/ID42.63完全相同；单视角T3_1前视P61.09/ID43.23仍更高，因此前视精度/身份护栏没有全面解决。','','位置稳定性必须使用同一GT配对：仅三/六共同front3区域307对象帧/199连续对，中心误差0.84758→0.85563m、误差变化0.47419→0.49602m，微退；全周共同313帧/205对则0.86063→0.84166m、0.49934→0.48337m，小幅改善。strictfront142帧/88对两者完全相同。另1/3/6三者公共集独立保留，不能混为同一分母。误差变化为||(pred_t−GT_t)−(pred_prev−GT_prev)||，扣除真实对象位移；官方GT与异步曝光的动态时差仍存在，不能断言普遍定位更稳。','','异步LiDAR混杂：23/120组六路多scan，6个首组前3无深度，其中2组后视有点。严格同scan控制六路统一最早曝光前最近past scan、仍≤100ms，图像/特征/GT全复用，0新模型。控制后三/六全周TP、IDTP与重复总数保持；six预测1444→1435，F1 48.61→48.78%。匹配组成并非逐条全同：控制得1失1、保留637，rear-only328→327，union6口径的front3标签310→311。覆盖收益在同scan后仍成立，但异步camera曝光/ego变换未被完全消除，不能称严格同步纯相机数因果实验。','','成本为原前360图复用、后360图真实新推理；实际新GPU47.86秒，非零成本复用的历史逐图推理也计入各subset。1/3/6分别需要120/360/720图，记录模型墙钟18.96/41.99/72.82秒；本轮CPU mapper15.46/32.73/52.78秒。二者仅可作为相加诊断（每组约0.287/0.623/1.047秒，未含加载/IO），不是实测端到端延迟或闭环实时成绩。6方法同时常驻的峰值RSS也不能作为单方法内存。','','全部720真实图、mask/CLIP缓存已独立解码；720主投影与720同scan投影0像素差，未来LiDAR0，主34路缺深度、控制36路缺深度保留。旧前3的19字段/路径/哈希等同，官方K误差0、独立pose误差1.14e−13；新front3的687GT对象字段与历史原GT全同。模型权重/960/conf0.1/seed17不变，冻结后未依本轮结果改规则。','','交付入口：','']
 for label,path in [('主与同scan完整36行',ind/'完整主与同scan主表.csv'),('完整逐scene216行',ind/'完整逐scene表.csv'),('逐log180行',ind/'完整逐log表.csv'),('3/6共同位置与稳定性',ind/'main_three_vs_six_v1_paired_position.csv'),('3/6与C6逐帧得失',ind/'main_three_vs_six_v1_changes.csv'),('同scan组成交换',ind/'scan_control_v1_changes.csv'),('模型/mapper成本分列',ind/'模型与建图分列成本.csv'),('传感器原点/时钟/GT证据',sensor/'六视角传感器输入交付.md'),('全部mask/CLIP缓存',sensor/'features_independent_audit.json'),('最后预算/保护/源码',ind/'最终独立交付核验.json')]:lines.append(f'- [{label}]({path})')
 lines+=['',f'本线新增{size/2**20:.2f}MiB/640MiB，空闲{free/2**30:.2f}GiB；GPU47.86秒/300秒，训练0、新测试0。主与控制均全120完成，独立18+18行与mapper逐数一致。当前实验收口，后续不据此追加调参。']
 (ind/'六视角独立结果与主方案建议.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(passed=True,rows=36,source_files=len(sources),new_MiB=size/2**20,free_GiB=free/2**30,report=str(ind/'六视角独立结果与主方案建议.md')),ensure_ascii=False))
if __name__=='__main__':main()
