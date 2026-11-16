"""从真实完成的CSV生成可编辑Word/Excel交付表；不写论文正文，不编造未完成分数。"""
import csv,html,json,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,Alignment,PatternFill
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'tables/completed_office_002';sources={}
def read(name):
 p=R/name;sources[str(p)]=digest(p)
 return list(csv.DictReader(p.open()))
def f(value,percentage=False):
 if value is None or value=='':return '未测'
 x=float(value);return f'{x*100:.2f}%' if percentage else f'{x:.3f}'
def pivot(rows,key):
 out={}
 for row in rows:out.setdefault(row[key],{})[row['metric']]=row
 return out
def xmltext(s):return html.escape(str(s),quote=True)
def p(s):return '<w:p><w:r><w:t xml:space="preserve">'+xmltext(s)+'</w:t></w:r></w:p>'
def table(headers,rows):
 prop='<w:tblPr><w:tblW w:w="14500" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="12"/><w:bottom w:val="single" w:sz="12"/></w:tblBorders></w:tblPr>'
 out=['<w:tbl>'+prop]
 for i,row in enumerate([headers]+rows):
  out.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>')
  for v in row:out.append('<w:tc><w:tcPr><w:tcW w:w="1800" w:type="dxa"/></w:tcPr>'+p(v)+'</w:tc>')
  out.append('</w:tr>')
 return ''.join(out)+'</w:tbl>'
def main():
 O.mkdir(exist_ok=False);sections=[]
 a=pivot(read('nuscenes_ablation/statistics_001/metrics.csv'),'condition');keys=['object_recall','matched_prediction_fraction','center_error_m','IDF1','relation_F1','mean_mapping_relation_s'];sections.append(('nuScenes六组对象/关系模块', ['配置','找全率','匹配预测比例','XY误差/m','IDF1','关系F1','建图关系秒/帧'],[[k]+[f(v[m]['value'],m not in ['center_error_m','mean_mapping_relation_s']) for m in keys] for k,v in a.items()],'38场景、11日志，每条件1508帧；仅对象/关系，复杂Qwen未接入此六组。CI和配对差异见nuscenes_ablation/statistics_001/metrics.csv；匹配预测比例不是官方AP。'))
 a=read('method_comparison/ab3dmot_statistics_002/comparison.csv');group={}
 for row in a:group.setdefault((row['dataset'],row['method']),{})[row['metric']]=row
 keys=['recall','matched_prediction_fraction','center_error_m','IDF1','duplicates'];sections.append(('公开AB3DMOT核心的共享前端适配',['数据集','方法','找全率','匹配比例','XY误差/m','IDF1','重复节点'],[[d,k]+[f(v[m]['value'],m in ['recall','matched_prediction_fraction','IDF1']) for m in keys] for (d,k),v in group.items()],'nuScenes1508帧，KITTI10序列1000帧。输入是部分可见点云轴对齐框，非作者原检测器或排行榜复现。本方法KITTI的IDF1和重复节点差于AB3DMOT，完整保留。'))
 a=read('method_comparison/recorded_retrieval_audit_001/counts.csv');sections.append(('旧真实记录目标检索的复核',['方法','选对/733','找对率','无匹配误选/382','误选率'],[[v['method'],v['positive_selected_correct'],f(int(v['positive_selected_correct'])/733,True),v['no_match_false_selection'],f(int(v['no_match_false_selection'])/382,True)] for v in a],'共享1115条旧开发查询，733有目标、382当前范围无匹配。本轮新增GPU调用0。稀疏CLIP网格是受控简化，不是完整VLMaps复现。'))
 a=read('method_comparison/planning_official_snapshots_001/comparison.csv');sections.append(('官方传感器快照规划诊断',['内部实现','同输入快照','返回路线','整车完整审计通过','规划诊断均秒'],[[v['method'],v['cases'],v['routes'],v['full_vehicle_audit_pass'],f(v['mean_planner_and_diagnostic_s'])] for v in a],'24份快照来自4次开发episode。三种规划器为内部受控实现；这里没有新增物理执行或实际到达。'))
 a=read('tables/semantic_summary_001/qwen_six_models.csv');sections.append(('给定官方几何图的Qwen六模型诊断',['模型','严格正确/135','格式有效/135','编号辅助正确/135','真实GPU均秒'],[[v['model'],v['strict_correct'],v['format_valid'],v['target_only_correct'],f(v['mean_GPU_latency_s'])] for v in a],'540训练、180验证、135开发留出；9种受控任务，每条共享相同输入。此表使用给定官方几何和受控证据字段，不是真实感知结果。编号辅助为统一后验口径，严格主指标不删除。'))
 a=[v for v in read('tables/semantic_summary_001/semantic_ablation_three_seeds.csv') if v['stratum']=='all'];sections.append(('给定图合作模块三种子消融',['配置','任务准确率均值','种子样本标准差','日志CI下限','日志CI上限'],[[v['condition'],f(v['accuracy_mean_3seeds'],True),f(v['accuracy_sample_std_3seeds']),f(v['log_cluster_ci95_low'],True),f(v['log_cluster_ci95_high'],True)] for v in a],'每种子135条、仅2开发日志。固定顺序消融优于完整模块，受限句法强规则100%，负结果保留。离线合作重放不冒充新GPU调用或在线延迟。'))
 a=pivot(read('nuscenes_ablation/category_semantics_001/comparison.csv'),'condition');keys=['positive_proposal_accuracy','no_match_false_proposal_rate','candidate_coverage_positive','current_evidence_supported_correct_proposal_rate'];sections.append(('六组建图接相同清晰类别合作分支',['配置','提议选对/1882','无匹配误选/2642','正目标候选覆盖','证据支持且提议正确'],[[k]+[f(v[m]['value'],True) for m in keys] for k,v in a.items()],'各4524次固定类别查询，其中1882有目标。仅离线编号提议，没有Qwen复杂分支/真实锁定/到达；去深度或去动态后授权为0受证据字段/静止门闸影响，不能独立解释为整个导航模块增益。'))
 a=read('carla_official/ablation_statistics_004/episodes.csv');sections.append(('官方CARLA逐任务独立审计',['任务/条件','状态','首选正确','锁定正确','几何到达','严格到达','加速度峰值'],[[v['episode']+'/'+v['condition'],v['final_status'],v['first_proposal_correct_ever'],v['correct_stable_lock_ever'],v['geometric_identity_stop_arrival'],v['strict_safe_arrival'],format(float(v['maximum_acceleration_mps2']),'.6f')] for v in a],'完成14/15回合，最后无匹配/去证据检查回合因10GiB空间守卫停止，原始前缀保留。2正任务+1负任务，5条件，未激活复杂Qwen。低速25Nm Model3物理配置单列，不是原厂Tesla。负任务不进入正任务到达率分母。'))
 wb=Workbook();wb.remove(wb.active);body=[p('已完成实验交付表（开发验证，不代表整个课题验收）'),p('本文件只汇总已完成结果，不是论文正文。所有比例、分母、限制和负结果保留；原始CSV包含CI和证据路径。')]
 for i,(title,headers,rows,note) in enumerate(sections,1):
  ws=wb.create_sheet(f'{i:02d}');ws.append([title]);ws.append(headers)
  for row in rows:ws.append(row)
  ws.append([note]);ws.freeze_panes='B3';ws.auto_filter.ref=f'A2:{ws.cell(2,len(headers)).column_letter}{2+len(rows)}'
  for col in ws.columns:
   ws.column_dimensions[col[0].column_letter].width=25
   for cell in col:cell.font=Font(name='Noto Serif CJK SC',size=10);cell.alignment=Alignment(vertical='top',wrap_text=True)
  for cell in ws[2]:cell.font=Font(name='Noto Serif CJK SC',size=10,bold=True)
  body.extend((['<w:p><w:r><w:br w:type="page"/></w:r></w:p>'] if i>1 else [])+[p(f'表{i}  '+title),table(headers,rows),p(note)])
 wb.save(O/'已完成实验交付表.xlsx')
 sect='<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700"/></w:sectPr>'
 doc='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+sect+'</w:body></w:document>'
 styles='<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="Noto Serif CJK SC"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults></w:styles>'
 types='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
 rels='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
 docrels='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
 with zipfile.ZipFile(O/'已完成实验交付表.docx','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name,data in {'[Content_Types].xml':types,'_rels/.rels':rels,'word/document.xml':doc,'word/styles.xml':styles,'word/_rels/document.xml.rels':docrels}.items():ET.fromstring(data);z.writestr(name,data)
 assert len(load_workbook(O/'已完成实验交付表.xlsx').sheetnames)==len(sections)
 dump(O/'receipt.json',dict(tables=len(sections),source_sha256=sources,word_sha256=digest(O/'已完成实验交付表.docx'),excel_sha256=digest(O/'已完成实验交付表.xlsx'),generator_sha256=digest(__file__),word_generated_by='standards-compliant OOXML with stdlib; workspace dependency loader tool unavailable in this session',office_xml_validated=True,excel_reopened=True,PDF_generated=False,thesis_prose_written=False,goal_complete=False));print(json.dumps(dict(tables=len(sections)),ensure_ascii=False))
if __name__=='__main__':main()
