from pathlib import Path
import json,re,csv
O=Path(__file__).resolve().parents[1];D=O/'references'
refs=json.loads((D/'verified_references.json').read_text(encoding='utf-8'))
s=(O/'sources/论文正文.md').read_text(encoding='utf-8')
topics={
'几何定位与度量语义建图':'1502.00956 1610.06475 2007.11898 1609.05130 1903.01177 1910.02490 2106.14386 1808.08378 1804.09194 1806.05620',
'车载三维感知与多模态融合':'1812.05784 2006.11275 2205.13542 2203.17270',
'视觉语言基础模型与开放词汇前端':'2103.00020 2201.03546 2304.07193 2112.03857 2201.02605 2303.05499 2401.17270 2304.02643 2408.00714',
'开放词汇三维表示':'2211.15654 2302.07241 2303.09553 2306.13631 2310.03923',
'多对象关联与身份评价':'1602.00763 1703.07402 2110.06864 2008.08063 1609.01775 2009.07736',
'三维场景图与关系':'2002.06289 2004.03967 2201.13360 2402.12259 2403.17846 2410.11989 2503.08474',
'语言目标与机器人规划':'1711.07280 2207.04429 2210.05714 2312.03275 2307.06135 2307.05973 2410.08189',
'道路数据与指代':'1909.10838 1903.11027 KITTI2013',
'多模态模型与参数高效适配':'2409.12191 2502.13923 2106.09685 2305.14314',
'图搜索采样规划与车辆规划':'Dijkstra1959 Hart1968 1105.1186 1404.2334 Dolgov2010 1604.07446 2212.10156',
}
topicmap={key:topic for topic,ids in topics.items() for key in ids.split()}
sections={};current=''
for line in s.splitlines():
 if line.startswith('## '):current=line[3:]
 for group in re.findall(r'\[@([^\]]+)\]',line):
  for key in group.split(';'):sections.setdefault(key,[]);sections[key]+=([current] if current not in sections[key] else [])
rows=[]
for r in refs:
 key=r['id'];p=D/key;full=list(p.glob('*fulltext.txt'));read=p/'阅读核对.md'
 reading='作者页面完整摘要；本轮按摘要中的方法范围引用'
 if read.exists():reading='作者摘要及方法相关内容，详见'+key+'/阅读核对.md'
 elif full:reading='既有作者摘要及全文缓存；前次相关内容阅读记录见交接资料'
 rows.append({'引用序号':r['number'],'ID':key,'主题':topicmap[key],'工作':r['title'],'作者':'; '.join(r['authors']),'引用年份':r['date'][:4],'稳定原始来源':r['url'],'DOI':r.get('doi',''),'方法要点及支撑范围':r['claim'],'正文位置':'；'.join(sections[key]),'本轮阅读依据':reading,'全文缓存':'; '.join(x.name for x in full),'获取日期':r.get('access_date','2026-11-15')})
with (D/'文献调研矩阵.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
firsttwo=s[s.index('# 第1章 '):s.index('# 第3章 ')]
cited=set(';'.join(re.findall(r'\[@([^\]]+)\]',firsttwo)).split(';'))
report='''# 前两章文献调研与引用说明

本轮围绕三维对象建图、语言目标选择及道路离线规划，完成代表工作检索、原作者页面核对、论点对应与正文整合。调研按问题组织，服务于硕士论文前两章的技术定位；检索记录用于说明覆盖范围。

检索日期：2026年11月20日。检索工具为网页检索与原始页面直接获取。来源包括arXiv作者提交页面及既有CVF、作者机构、数据集和出版者资料。已有来源缓存继续保留。新增工作逐项提取完整作者表、题名、首次提交日期、DOI字段及摘要。正式书目采用预印本的条目明确标为arXiv在线文献，其年份为首次提交年；会议正式发表年可能不同，未以会议名称推断出版字段。

检索词组包括：object-level semantic mapping、dynamic semantic SLAM、open-vocabulary 3D mapping、3D scene graph construction、language-grounded robot navigation、multi-object data association、BEV camera LiDAR fusion、vehicle motion planning、Qwen vision language LoRA。以主题检索定位代表工作，再核对其原始来源及与论文问题的关系。

纳入依据：至少直接解释观测组织、视觉语义、对象身份、关系表达、语言目标或几何规划中的一项机制。采用同一工作的预印本作为稳定引用入口，不把预印本与正式版本重复计数。不同代际系统按其独立方法工作列入。排除仅有二手描述、编号与题名不一致、超出本文任务且没有直接方法联系的条目，以及用户指定排除的文献。

阅读深度分别记录：原始摘要核对、已获取全文、相关方法内容核对。获取全文本身不等于通读；仅按摘要核对的工作用于研究路线和方法概要，不承担具体实验分数或未在摘要说明的实现细节。新增论文的外部实验数字没有写入本项目结果。原有实验数据与评价口径继续以冻结报告为依据。

第一章按研究路线比较“解决何种问题、使用何种表示、对本文有何启发”；第二章解释这些思想如何对应到坐标变换、特征归属、一对一分配、图关系及规划状态。前两章各处引用与来源逐项登记在《文献调研矩阵.csv》中。

'''
report+=f'正式参考文献共{len(refs)}项，前两章独立引用{len(cited)}项；其中已有文献26项，新增{len(refs)-26}项。\n\n'
for topic,ids in topics.items():
 relevant=[r for r in refs if r['id'] in ids.split()]
 report+='## '+topic+'\n\n'
 for r in relevant:report+=f"[{r['number']}] {r['title']}（{r['date'][:4]}）。{r['claim']} [原始来源]({r['url']})\n\n"
(D/'前两章文献调研报告.md').write_text(report,encoding='utf-8')
(O/'qa/文献引用检查.json').write_text(json.dumps({'正式参考文献':len(refs),'前两章独立引用':len(cited),'全部正文引用':all(r['id'] in sections for r in refs),'重复ID':len(refs)-len({r['id'] for r in refs}),'排除文献出现在正文或正式书目':bool(re.search('conceptgraphs?',s+json.dumps(refs),re.I)),'条目完整':all(r['title'] and r['authors'] and r['date'] and r['url'] for r in refs),'主题分布':{t:len(v.split()) for t,v in topics.items()}},ensure_ascii=False,indent=2),encoding='utf-8')
print('literature report',len(refs),'first two chapters',len(cited))
