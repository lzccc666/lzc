"""按当前文件生成交接与结构记录，不创建压缩包。"""
from pathlib import Path
import json,csv,hashlib,shutil,re,subprocess,os
O=Path(__file__).resolve().parents[1]
def read(name):return json.loads((O/name).read_text(encoding='utf-8'))
q=read('qa/最终质量检查.json');counts=read('qa/字数与结构检查.json');pages=read('qa/page_inspection.json');mp=read('qa/page_map.json');refs=read('references/verified_references.json');lit=read('qa/文献引用检查.json')
catalog=O.parent if os.name=='nt' else Path('/root/autodl-tmp/lizichong/paper')
backup=catalog/'历史版本/阶段快照_文献扩充前_20261120/旧说明备份';backup.mkdir(parents=True,exist_ok=True)
for name in ['修订交接说明.md','质量检查记录.md','论文目录与图表结构说明.md']:
 if (O/name).exists() and not (backup/name).exists():shutil.copy2(O/name,backup/name)
review=read('qa/视觉巡检确认.json') if (O/'qa/视觉巡检确认.json').exists() else {}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pdfhash=sha(O/'硕士论文初稿.pdf')
figure_audit=read('qa/图表生成审计.json')
figure_audit['当前版核查日期']='2026-11-20'
for fig in figure_audit['图表']:
 fig['PNG_SHA256']=sha(O/'figures'/fig['文件名']);fig['PDF_SHA256']=sha(O/'figures'/fig['矢量文件名'])
(O/'qa/图表生成审计.json').write_text(json.dumps(figure_audit,ensure_ascii=False,indent=2),encoding='utf-8')
fonts=subprocess.run(['pdffonts',str(O/'硕士论文初稿.pdf')],check=True,stdout=subprocess.PIPE,text=True).stdout
(O/'qa/PDF字体检查.txt').write_text(fonts,encoding='utf-8')
visual_ok=review.get('pdf_sha256')==pdfhash and review.get('reviewed_pages')==q['pages'] and review.get('passed') is True
q['visual_qa_passed']=visual_ok;q['visual_review']='本版逐页渲染与布局检查完成；详见质量检查记录。' if visual_ok else '当前PDF逐页视觉确认待完成。'
(O/'qa/最终质量检查.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
main_count=q['word_main_prose_cjk_characters'];npages=q['pages']
full_count=sum(any((O/'references'/r['id']).glob('*fulltext.txt')) for r in refs)
method_count=sum((O/'references'/r['id']/'阅读核对.md').exists() for r in refs)
review_text=f'全部{npages}页已经渲染并逐页检查布局；'+review.get('scope','') if visual_ok else f'全部{npages}页机器检查已生成，逐页视觉确认待完成。'
handoff=f'''# 毕业论文文献调研修订交接

修订日期：2026年11月20日。正式题目为《面向智能车辆的三维语义场景图构建与导航方法研究》。

当前修订号为v4。本机当前版入口为 `D:/study/Python.Project/lzc/paper/当前版_v4`，远端入口为 `seeta-nmb1:/root/autodl-tmp/lizichong/paper/当前版_v4`。`硕士论文初稿.docx`为可编辑Word，`硕士论文初稿.pdf`为对应转换PDF。旧稿统一从paper/历史版本的v1、v2、v3及阶段快照查阅，版本关系见paper/00_稿件版本索引.md。旧paper/毕业论文为指向当前v4的兼容目录联接。文件按目录保存，不创建ZIP交付包。

## 本轮改动

第一章重写研究背景与研究内容，按几何建图、视觉语义、对象关联、场景图、语言目标和道路规划组织研究现状。第二章补充坐标与表示分工、区域特征归属、一对一分配、动态关系维护和规划状态机制。新增文献与本文方法之间的联系直接写入相关段落。

正式引用{len(refs)}项工作，前两章实际独立引用{lit['前两章独立引用']}项，较上一版26项增加{len(refs)-26}项。作者、题名、日期和稳定链接逐项登记，研究路线覆盖基础方法及2025年相关进展。按作者摘要与方法相关内容分层记录阅读范围；已有{full_count}项全文缓存，其中本轮重点方法阅读记录{method_count}项。全文缓存数量表示获取状态，详细阅读位置见对应记录。预印本按首次提交年引用，书目明确标注arXiv；正式发表年份可能不同。

用户指定排除的文献未进入正文和正式书目。第三方检测、分割、特征及模型能力归属于原工作；代码和实验索引保存实际依赖位置。

## 当前文件状态

Word第1—7章普通正文纯段落为{main_count:,}汉字，排除摘要、目录、标题、表格、图注、参考文献、附录、数字和英文。PDF共{npages}页，正文从物理第{pages['body_start_physical_page']}页开始。结构为7章、27个二级小节、无三级小节；Word含20张原生表格、7幅图、11个OMML数学对象，目录42项。

`qa/最终质量检查.json`为本版机器检查，`qa/文献引用检查.json`对应本轮引用。{review_text} 目录按当前PDF页码核对，采用可编辑静态页码。

## 实验依据

当前研究使用nuScenes 38开发场景、11个log、1508帧，保留38个过去雷达缺测首帧。预留30场景、14log、1196帧处于后续封存测试阶段。KITTI Object 64图来自33drive；Tracking 90帧中74次受支持GT集中于0005的4个实例，465条不支持类别/DontCare记录保留。

Qwen训练2398条、验证603条，真实图像均为nuScenes开发数据子集。解析含构造表达，排序含受控符号候选，属性验证主要评价类别。Talk2Car准备439条原指令与图像，正式模型成绩待完成。

导航研究包括语言目标选择、安全接近位置及道路和车辆几何下的离线规划。地图诊断使用给定正确参考对象，完整条件生成94/212条路线，2/94通过静态车身检查。关系负结果、低覆盖、未知、缺深度、拒绝、失败与预算耗尽均保留。本轮为文献与文稿修订，实验报告和模型不变。

## 再生成与后续使用

修改`sources/论文正文.md`，表图声明见`tables.json`、`figures.json`。在原实验资料可访问的Linux环境运行`bash scripts/rebuild.sh --extract-evidence`，只读报告提取并以Python/matplotlib重绘7图；普通文字修订可运行不带参数的`rebuild.sh`复用图表。之后渲染并复核新PDF，运行`record_current_delivery.py`更新本版说明。

`assemble_revision.py`、`apply_literature_revision.py`等属于历史组装步骤，日常编辑直接修改主源稿。旧`record_revision_delivery.py`已停用。文献调研报告与59条阶段性材料的最终状态，以当前62条正式书目、主源稿及引用检查为准。

学校模板、作者身份信息及Microsoft Word客户端复排仍待作者定稿阶段处理。逐字人工校对、自然指令正式验证、细属性人审、新封存测试和车辆执行按实际进度继续完善。
'''
(O/'修订交接说明.md').write_text(handoff,encoding='utf-8')
quality=f'''# 当前稿质量检查记录

对象：`硕士论文初稿.docx`及对应PDF。日期：2026年11月20日。历史85页检查保存在历史版本中，本记录使用当前{npages}页文件。

## 内容与结构

实际Word七章纯正文{main_count:,}汉字，普通正文非空段落{counts['Word正文段落数']}个。7章、27个二级小节、无三级标题；20张可编辑表格、7图、11个可编辑数学对象；目录42项。正文、参考文献和附录标题在PDF中全部定位，页码收敛。引文和模板标记全部解析。

正式参考文献{len(refs)}项，前两章独立引用{lit['前两章独立引用']}项，重复ID为0。新增作者页面元数据和摘要已保存，引用核验清单、调研矩阵、BibTeX及RIS同步更新。每条文献的阅读范围单独记录。正文与正式书目排除指定文献。

## 数据与图表

`qa/evidence_audit.json`的98项分层计数核验通过，保留冻结原始报告及历史结果。7幅图由`build_evidence_figures.py`使用Python/matplotlib导出同名PNG/PDF，图4-1拼合真实RGB、深度、掩膜及保存预测，其余为流程、几何或统计图。图表编号、图名及来源登记在图表来源索引和图表生成审计中。

## 排版检查

{review_text}

机器检查替换字符{sum(x['replacement_chars'] for x in pages['pages_detail'])}、外边界溢出{sum(bool(x['outer_margin_overflow']) for x in pages['pages_detail'])}、缺失标题{len(pages['missing_headings'])}。页面图用于布局检查，重点核对目录、公式、表格、图注及文献换行；该检查与全文逐字人工校对分别记录。实际检查页与当前PDF指纹见`qa/视觉巡检确认.json`。

学校专用模板和Microsoft Word客户端复排待定稿阶段完成。现有排版为统一A4中文硕士论文样式。
'''
(O/'质量检查记录.md').write_text(quality,encoding='utf-8')
structure='# 论文目录与图表结构说明\n\n正式题目：《面向智能车辆的三维语义场景图构建与导航方法研究》。本版保留7章、27个二级小节、无三级小节；目录页码来自当前实际PDF。\n\n| 项目 | 正文印刷页码 |\n| --- | --- |\n'
for heading,num in mp.items():
 if heading.startswith('第') or heading=='参考文献' or heading.startswith('附录'):structure+=f'| {heading} | {num} |\n'
structure+=f'\nPDF物理页数{npages}；正文起始物理页{pages["body_start_physical_page"]}。Word原生表格20张、图7幅、数学对象11个，正式参考文献{len(refs)}项。\n\n'
for key,f in read('sources/figures.json').items():structure+='- '+Path(f['path']).stem+'（PNG及PDF）\n'
structure+='\n表图原始数值来源、生成脚本及缓存数据见图表来源索引、实验索引和qa/图表生成审计.json。\n'
(O/'论文目录与图表结构说明.md').write_text(structure,encoding='utf-8')
files=[O/name for name in ['硕士论文初稿.docx','硕士论文初稿.pdf','sources/论文正文.md','sources/论文整合源稿.md','references/verified_references.json','references/参考文献.bib','references/参考文献.ris','references/文献调研矩阵.csv','references/前两章文献调研报告.md','引用核验清单.csv','图表来源索引.csv','qa/最终质量检查.json','qa/字数与结构检查.json','qa/文献引用检查.json','修订交接说明.md','质量检查记录.md','论文目录与图表结构说明.md']]
for extra in ['references/参考文献真实性与年份核验报告.md','references/论文身份与发表状态核验.csv','references/论文身份与发表状态核验.json','00_当前版本说明.md','版本信息.json']:
 p=O/extra
 if p.exists() and p not in files:files.append(p)
with (O/'交付文件校验清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
 rows=[{'文件':p.name,'相对路径':p.relative_to(O).as_posix(),'字节数':p.stat().st_size,'SHA256':sha(p)} for p in files];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'pages':npages,'references':len(refs),'body_cjk':main_count,'fulltext_cache':full_count,'method_reading_records':method_count,'machine_checks':q['all_passed'],'visual_qa':visual_ok},ensure_ascii=False))
