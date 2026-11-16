"""完整38场景语义消融的可编辑结果表，不撰写论文正文。"""
import csv,json,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,Alignment
from conceptgraph.completion.stage9 import dump,digest
from export_completed_tables_002 import p,table
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/full_semantics_near6_002';O=R/'tables/full_semantics_near6_office_002'
LABELS=dict(full='完整',without_depth_quality='去深度质量',without_one_to_one='去一对一关联',without_dynamic_maintenance='去动态维护',without_short_term_expiry='去短期过期',without_relations='去关系边')

def read(path):return list(csv.DictReader(path.open()))

def main():
 report=json.loads((D/'report.json').read_text());assert report['actual_requests']==9048
 data=read(D/'metrics.csv');index={(r['condition'],r['role'],r['scope'],r['metric']):r for r in data};routes=read(D/'detail_statistics_001/routing_and_latency.csv');ri={(r['condition'],r['role']):r for r in routes}
 O.mkdir(exist_ok=False);wb=Workbook();wb.remove(wb.active);docs=[];exported=[]
 headers=['建图条件','合作正确/全部','选对/普通正目标','选对率及日志95%CI','完整几何规则选对','候选包含目标','Qwen调用数']
 for role,title in [('all','38开发场景全帧：含训练和验证日志'),('development_holdout','开发留出日志：与训练/验证分开')]:
  rows=[]
  for condition,label in LABELS.items():
   a=index[condition,role,'all','correct'];b=index[condition,role,'ordinary_positive','correct'];g=index[condition,role,'ordinary_positive','strong_rule_correct'];c=index[condition,role,'ordinary_positive','covered'];u=ri[condition,role]
   rows.append([label,f"{a['numerator']}/{a['denominator']}",f"{b['numerator']}/{b['denominator']}",f"{float(b['value'])*100:.2f}% [{float(b['ci95_low'])*100:.2f}, {float(b['ci95_high'])*100:.2f}]",f"{g['numerator']}/{g['denominator']}",f"{c['numerator']}/{c['denominator']}",u['Qwen_calls']])
  note=f"维护图实际邻近距离6米，旧8米结果单独保留。每条件{a['denominator']}条，每帧固定循环九层任务中的一条，六条件共享查询/官方目标；普通正目标{b['denominator']}条，按{b['logs']}日志进行4000次bootstrap，seed17。全体38场景11日志含训练/验证；开发留出仅2日志且已用于开发诊断，均不是最终封存测试。缺失正确候选保留为失败。Qwen仅在冻结合作逻辑需要时实际调用，规则分支未调用模型；此表不是Qwen单独正确率，也不是物理锁定/实际到达率。"
  ws=wb.create_sheet('全体开发场景' if role=='all' else '开发留出2日志');ws.append([title]);ws.append(headers)
  for row in rows:ws.append(row)
  ws.append([note]);ws.freeze_panes='B3';docs.extend([p(title),table(headers,rows),p(note)])
  if role=='all':docs.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  exported.append(dict(role=role,rows=rows))
 for name,path in [('原始分层统计和CI',D/'metrics.csv'),('模块配对差异',D/'paired_differences.csv'),('路由格式延迟',D/'detail_statistics_001/routing_and_latency.csv'),('同输入规则比较',D/'detail_statistics_001/paired_rule_comparison.csv')]:
  zz=read(path);ws=wb.create_sheet(name);ws.append(list(zz[0]))
  for row in zz:ws.append(list(row.values()))
  ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 for ws in wb:
  for col in ws.columns:
   ws.column_dimensions[col[0].column_letter].width=25
   for cell in col:cell.font=Font(name='Noto Serif CJK SC',size=10);cell.alignment=Alignment(wrap_text=True,vertical='top')
 wb.save(O/'修正邻近距离后全量语义消融.xlsx');assert len(load_workbook(O/'修正邻近距离后全量语义消融.xlsx').sheetnames)==6
 sect='<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700"/></w:sectPr>'
 doc='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(docs)+sect+'</w:body></w:document>';root=ET.fromstring(doc);assert len(root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'))==2
 with zipfile.ZipFile(R/'tables/completed_office_002/已完成实验交付表.docx') as source,zipfile.ZipFile(O/'修正邻近距离后全量语义消融.docx','x',compression=zipfile.ZIP_DEFLATED) as target:
  for name in source.namelist():target.writestr(name,doc.encode() if name=='word/document.xml' else source.read(name))
 dump(O/'receipt.json',dict(tables=2,table_content=exported,source_sha256={str(p):digest(p) for p in [D/'report.json',D/'metrics.csv',D/'detail_statistics_001/routing_and_latency.csv',Path(__file__)]},outputs_sha256={p.name:digest(p) for p in O.iterdir()},word_XML_table_count=2,excel_sheets=6,thesis_prose_written=False,whole_project_sealed_test=False,goal_complete=False));print('two editable full semantic tables exported',flush=True)

if __name__=='__main__':main()
