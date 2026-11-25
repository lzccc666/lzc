"""只汇总独立保存评分，去除跨run重复基线；不运行方法。"""
import csv, hashlib, json, shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_independent_scores import write_csv
R=PROJECT_ROOT/'outputs/reports/multiview_identity_repair_20261123/validation'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(n):return list(csv.DictReader((R/n).open(encoding='utf-8-sig')))
def main():
 specs=[]
 for batch,suffix in [('pilot','pilot'),('crosslog','crosslog')]:
  specs.extend((batch,f'T3_{suffix}_v1',k,v) for k,v in [('T3:A_front','A'),('T3:C_three_dedup','C'),('T3:T3_three','T3')])
  specs.extend((batch,f'P1_{suffix}_v1',k,v) for k,v in [('Afair:A_front_physical_only','A_physical'),('P1:P1_three','P1')])
  specs.append((batch,f'T2_{suffix}_v1','T2:T2_three','T2'))
 specs.extend(('nextlogs','final_nextlogs_v1',k,v) for k,v in [('P1:A_front','A'),('P1:C_three_dedup','C'),('Afair:A_front_physical_only','A_physical'),('P1:P1_three','P1'),('T2:T2_three','T2'),('T3:T3_three','T3')])
 batches=[];scenes=[]
 for batch,prefix,key,label in specs:
  for target,kind in [(batches,'main'),(scenes,'scene')]:
   rr=[r for r in read(prefix+'_'+kind+'.csv') if r['method']==key]
   assert len(rr)==(2 if kind=='main' else 4),(prefix,key,kind,len(rr))
   target.extend(dict(batch=batch,label=label,**r,evidence=str(R/(prefix+'_'+kind+'.csv'))) for r in rr)
 assert len({(r['label'],r['scope'],r['scene']) for r in scenes})==len(scenes)
 ints=['TP','GT','predictions','FP','FN','duplicates','IDTP','IDFN','IDFP','unique_gated_IDTP','identity_GT','identity_predictions','ambiguous_GT','ambiguous_predictions','adjacent_matched_transitions','adjacent_ID_changes','GT_with_multiple_matched_IDs','IDs_matched_to_multiple_GT','GT_identities','predicted_identities']
 pooled=[]
 for label in ['A','A_physical','C','P1','T2','T3']:
  for scope in ['union','front']:
   rows=[r for r in scenes if r['label']==label and r['scope']==scope]
   assert len(rows)==6
   row=dict(label=label,scope=scope,scenes=6,logs=5,groups=120,**{k:sum(int(r[k]) for r in rows) for k in ints})
   row.update(precision=row['TP']/row['predictions'],recall=row['TP']/row['GT'],F1=2*row['TP']/(row['GT']+row['predictions']),IDF1=2*row['IDTP']/(row['identity_GT']+row['identity_predictions']),duplicates_per_prediction=row['duplicates']/row['predictions'],duplicates_per_GT=row['duplicates']/row['GT'])
   assert row['GT']==(687 if scope=='union' else 385)
   pooled.append(row)
 write_csv(R/'最终120_分批主表.csv',batches);write_csv(R/'最终120_逐scene_log.csv',scenes);write_csv(R/'最终120_聚合主表.csv',pooled)
 by={(r['label'],r['scope']):r for r in pooled}
 assert (by['A','union']['TP'],by['A','union']['predictions'])==(139,224)
 assert (by['C','union']['TP'],by['C','union']['predictions'],by['C','union']['duplicates'])==(308,566,22)
 assert (by['T3','union']['TP'],by['T3','union']['predictions'],by['T3','union']['duplicates'],by['T3','union']['IDTP'])==(309,537,11,281)
 reg=json.loads((R/'registration.json').read_text()); protect={p:digest(p)==sha for p,sha in reg['protected_sha256'].items()};assert all(protect.values())
 run=experiment_directory('nuscenes','validation','mvrepair_validation_final_delivery_20261123_v1',create=True)
 snap={}
 for p in sorted((PROJECT_ROOT/'scripts').glob('multiview_identity_repair_20261123_*.py')):
  d=run/'source_snapshot'/p.relative_to(PROJECT_ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);snap[str(p)]=dict(sha256=digest(p),snapshot=str(d))
 manifests=[PROJECT_ROOT/'dataset/processed'/n/'manifest.json' for n in ['multiview_20261123_pilot_v1','multiview_identity_repair_20261123_validation_v1','multiview_identity_repair_20261123_nextlogs_v1']]
 check=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_snapshots=snap,registered_protected_unchanged=protect,manifest_sha256={str(p):digest(p) for p in manifests},rows=dict(batch=len(batches),scene=len(scenes),pooled=len(pooled)),scope='6 scenes / 5 logs /120 groups, original development/debug; all now seen validation/regression',original_threshold_xy_m=3,training=0,new_test_opened=False,actual_frontend_new_views=240,actual_frontend_GPU_s=61.3659,final_wave_CPU_wall_s=json.loads((PROJECT_ROOT/'outputs/results/nuscenes/validation/iterations/mvrepair_final_nextlogs_wave_20261123_v1/complete.json').read_text()).get('elapsed_s'),free_GiB=shutil.disk_usage(PROJECT_ROOT).free/2**30)
 (R/'最终交付核验.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
 lines=['# 多相机身份修复最终独立复核','','原同类别XY≤3m，六类、visibility≥2、中心视域与1–65m深度门不变；全120时刻/6scene/5log，union687、front385、side-only302对象帧。按scene独立计算身份再累加，不跨scene匹配ID，不重复计入不同run复制的A/C。原38开发/调试来源，当前均为已查看验证/回归；pilot两个scene同一个log，不是两份独立泛化证据。','','|方法|口径|TP/GT|预测|重复|P %|R %|F1 %|IDF1 %|IDTP|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in pooled:lines.append(f'|{r["label"]}|{r["scope"]}|{r["TP"]}/{r["GT"]}|{r["predictions"]}|{r["duplicates"]}|{100*r["precision"]:.2f}|{100*r["recall"]:.2f}|{100*r["F1"]:.2f}|{100*r["IDF1"]:.2f}|{r["IDTP"]}|')
 lines+=['','T3v2相对C保留原匹配并净增1个对象帧、重复22→11、IDTP277→281；相对P1则TP/IDTP完全相同，仅少2预测/1重复。T3前视P59.41/ID42.63仍低公平A的61.09/43.23；本轮未解决全部前视精度/身份护栏，不能只写总体提升。','','第三批0042/0047固定后一次开启，完整40时刻：P1/T2/T3 union94/225、137预测、7重复、IDTP94；严格front48/137、57预测、1重复、IDTP48。A为front49/137；负项保留。P1/T2/T3完整预测40/40逐字段相同，时序接受、原型上下文/绑定均0；只能支持既有链路在此批的结果，不能宣称验证新增T3时序机制。封存文件是评价前历史，未覆盖；后续一次开启见final_nextlogs_wave_v1及run complete。','','共同配对位置：跨log P1→T3同177 union对象帧/115连续对，均值中心误差0.9224437922m、误差变化0.5250261744m完全相同；同70 front对象帧/43连续对0.7258397962/0.4902700695m完全相同。第三批同94 union与48 front对象帧也完全相同。误差变化使用(pred−GT)差，移除GT真实位移；GT仍在官方sample时刻，和曝光不同步，未运动补偿，故不宣称定位更准。','','共同前视身份：第三批P1/T3同48对象帧，链IDTP47、IDFN/IDFP各1、29连续对中1次ID变化，两者完全相同。此固定匹配子集诊断不是全分母IDF1，也不是人工身份真值。早期39da链的来源已用深拷贝当帧状态和真实关联矩阵复核，registry历史见证不能替代当帧精确来源。','','T3v2实现已独立核：从PartialFragmentMapper取得未预删部件；原生canonical分数0.80888的15点厢体优于0.52825的21点车头，仅一个代表、每组一次hits。6.07381m接管边在原最大基数分配前阻断。T3v1的继承预删实现无效，历史保留且不纳入有效表。自动几何同实体假设仍非人审身份标签。','','负结果全部保留：U1、L1/L2/L3/L4、L5、O1/O2/O3、T1/PT1/T2。O1双pass存在潜在输入改写风险，但本批69碰撞view独立审计0索引错位/0二次mask变化；O2恢复输入/RNG后全80逐字段等O1，不能把其负项归因已证实索引bug。O3可靠归属无收益，不采用；不再新建O4。','','证据：','']
 for label,p in [('分批完整主表',R/'最终120_分批主表.csv'),('逐scene/log表',R/'最终120_逐scene_log.csv'),('聚合主表',R/'最终120_聚合主表.csv'),('保护与源码指纹',R/'最终交付核验.json'),('第三批逐帧得失',R/'final_nextlogs_v1_changes.csv'),('P1/T3共同位置',R/'T3_vs_P1_crosslog_v1_paired_position.csv'),('原点索引与真实7点互投',R/'T1_actual_points_bidirectional_audit.json'),('O1输入风险反证',R/'O1_two_pass_input_audit.json'),('新批原型触发核验',R/'final_nextlogs_mechanism_activation.json')]:lines.append(f'- [{label}]({p})')
 lines+=['','本线两批240次真实新前端GPU约61.37秒；最终确认波次纯CPU，无新模型/训练。真实图像/传感器/点索引、未来点0与缺深度保留见各输入审计；离线公开数据回放不冒充车辆闭环。下一阶段1/3/6对照单独登记，不回写本轮方法或成绩。']
 (R/'最终独立复核与跨log结果.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(passed=True,pooled=pooled,report=str(R/'最终独立复核与跨log结果.md')),ensure_ascii=False))
if __name__=='__main__':main()
