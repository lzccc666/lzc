"""实际维护关系边接语义的六条件开发诊断表；只从完成的真实结果导出。"""
import csv,json,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,Alignment
from conceptgraph.completion.stage9 import dump,digest
from export_completed_tables_002 import p,table
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/complex_semantics_development_001';O=R/'tables/complex_semantics_office_001'
LABELS=dict(full='完整',without_depth_quality='去深度质量',without_one_to_one='去一对一关联',without_dynamic_maintenance='去动态维护',without_short_term_expiry='去短期过期',without_relations='去关系边')

def main():
 report=json.loads((D/'report.json').read_text());assert report['actual_new_GPU_requests']==648
 O.mkdir(exist_ok=False);data=list(csv.DictReader((D/'metrics.csv').open()));index={(r['condition'],r['scope'],r['metric']):r for r in data};rows=[]
 for condition,label in LABELS.items():
  rows.append([label]+[index[condition,scope,metric]['numerator'] for scope,metric in [('all','raw_correct'),('all','cooperation_correct'),('ordinary_positive','raw_correct'),('ordinary_positive','cooperation_correct'),('ordinary_positive','candidate_covered')]])
 headers=['建图条件','Qwen正确/108','合作正确/108','Qwen选对/60','合作选对/60','候选含目标/60']
 title='六组实际场景图接同一冻结Qwen：开发子集诊断'
 note='每条件108条相同原始查询/官方目标，来自12帧、2场景、2已开发日志；实际新增648次GPU推理和648次CPU合作。关系边直接来自图维护器；去关系条件与完整条件的候选/标签逐条相同。候选随其他建图模块真实变化，不能把漏检后的拒答算找到目标。仅固定种子17、不是38场景1508帧复杂语义全量验收、不是新封存测试，也没有物理锁定/到达信用。与旧108条的候选ID顺序和关系来源不同，不能把两套结果相减称单模块因果增益。分层/配对日志CI见原metrics.csv和paired_differences.csv。'
 wb=Workbook();ws=wb.active;ws.title='六条件';ws.append([title]);ws.append(headers)
 for row in rows:ws.append(row)
 ws.append([note]);ws.freeze_panes='B3'
 details=wb.create_sheet('原始统计和CI');details.append(list(data[0]));
 for row in data:details.append(list(row.values()))
 for sheet in wb:
  for col in sheet.columns:
   sheet.column_dimensions[col[0].column_letter].width=25
   for cell in col:cell.font=Font(name='Noto Serif CJK SC',size=10);cell.alignment=Alignment(wrap_text=True,vertical='top')
 wb.save(O/'实际场景图语义诊断.xlsx');assert len(load_workbook(O/'实际场景图语义诊断.xlsx').sheetnames)==2
 sect='<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700"/></w:sectPr>'
 doc='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+p(title)+table(headers,rows)+p(note)+sect+'</w:body></w:document>';ET.fromstring(doc)
 with zipfile.ZipFile(R/'tables/completed_office_002/已完成实验交付表.docx') as source,zipfile.ZipFile(O/'实际场景图语义诊断.docx','x',compression=zipfile.ZIP_DEFLATED) as target:
  for name in source.namelist():target.writestr(name,doc.encode() if name=='word/document.xml' else source.read(name))
 dump(O/'receipt.json',dict(tables=1,source_report_sha256=digest(D/'report.json'),source_metrics_sha256=digest(D/'metrics.csv'),generator_sha256=digest(__file__),word_sha256=digest(O/'实际场景图语义诊断.docx'),excel_sha256=digest(O/'实际场景图语义诊断.xlsx'),word_XML_parsed=True,excel_reopened=True,thesis_prose_written=False,goal_complete=False));print(json.dumps(dict(tables=1,rows=rows),ensure_ascii=False))

if __name__=='__main__':main()
