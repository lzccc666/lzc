"""汇总本次修订交付。视觉检查记录须先由审阅者填写，并与当前PDF哈希一致。"""
raise SystemExit('本文件为旧版交付脚本，已停用。当前版运行 rebuild.sh 及 record_current_delivery.py；不创建压缩包。')
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import subprocess
import zipfile

O = Path(__file__).resolve().parents[1]
B = O.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def csvsave(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


docx = O / '硕士论文初稿_公开数据版_v2.docx'
pdf = O / '硕士论文初稿_公开数据版_v2.pdf'
review = json.loads((O / 'qa/视觉巡检确认.json').read_text())
assert review['PDF_SHA256'] == sha(pdf), 'PDF变动后须重新完成视觉巡检'
assert review['Word_SHA256'] == sha(docx), 'Word变动后须重新完成视觉巡检'
pages = json.loads((O / 'qa/page_inspection.json').read_text())
qa = json.loads((O / 'qa/最终质量检查.json').read_text())
counts = json.loads((O / 'qa/字数与结构检查.json').read_text())
evidence = json.loads((O / 'qa/evidence_audit.json').read_text())
assert qa['all_passed'] and counts['Word正文纯段落达到五万汉字']
assert review['已巡检物理页'] == list(range(1, pages['pages'] + 1))
N = counts['实际Word七章正文纯段落汉字数']

# 图表生成审计：所有最终图文件由同一个Python/matplotlib脚本导出；
# 图4-1另使用真实RGB、深度和检测掩膜作为输入，不把输入图片称作程序生成。
figmeta = json.loads((O / 'sources/figures.json').read_text())
figure_audit = []
for key, v in figmeta.items():
    png = Path(v['path'])
    figure_audit.append({
        '键': key, '文件名': png.name, '矢量文件名': png.stem + '.pdf',
        '图号与图名命名': png.name.startswith('Figure') and png.stem == v.get('filename'),
        '绘图脚本': v.get('generator', 'scripts/build_evidence_figures.py'),
        '绘图后端': v.get('backend', 'Python matplotlib'),
        '绘图语言': 'Python', '输出': v.get('export', '350 dpi PNG + vector PDF'),
        '输入类型': '真实RGB/深度/检测掩膜与保存预测的Python拼合图' if key == 'case' else
                   ('保存预测几何的Python派生图' if key == 'relations_case' else
                    ('方法流程的Python示意图' if key == 'pipeline' else '冻结报告数据的Python统计图')),
        '原始输入是否被误称为生成图': False,
    })
assert all(x['图号与图名命名'] and x['绘图语言'] == 'Python' for x in figure_audit)
dump(O / 'qa/图表生成审计.json', {'统一绘图脚本': 'scripts/build_evidence_figures.py',
                                  '统一后端': 'Python matplotlib', '图表': figure_audit,
                                  '说明': '图4-1含真实保存RGB/深度/检测掩膜输入，由Python完成叠加、标注和导出；其原始图片像素不是Python重新生成的实验画面。'})

# 扩写中新增的数值解释仍读取冻结报告，不新增统计实验。
mapping_path = B / 'statistics/causal_mapping_full_001/report.json'
mapping = json.loads(mapping_path.read_text())
kitti_path = B / 'statistics/kitti_object_cpu_full_001/report.json'
kitti = json.loads(kitti_path.read_text())
query_path = B / 'main_experiments/causal_retrieval_full_001/queries.jsonl'
targets = {('scene-0020', 16, 0), ('scene-0012', 0, 0), ('scene-0043', 28, 1)}
cases = []
for line_number, line in enumerate(query_path.open(), 1):
    r = json.loads(line)
    if (r['scene'], r['frame_idx'], r['query_index']) in targets and r['method'] == 'existing_object_joint_rule':
        cases.append({'line': line_number, 'record': r})
assert len(cases) == 3
dump(O / 'evidence/新增论述数值与案例核验.json', {
    '核验性质': '读取既有报告区间及原始查询；无新增模型推理或实验',
    'mapping_source': str(mapping_path), 'mapping_sha256': sha(mapping_path),
    'mapping_paired': [r for r in mapping['paired_differences'] if r['method'] == 'one_to_one_depth_quality' and r['group'] == 'all'],
    'kitti_source': str(kitti_path), 'kitti_sha256': sha(kitti_path),
    'kitti_common_supported_pairs': kitti['common_supported_pairs'],
    'queries_source': str(query_path), 'queries_sha256': sha(query_path), 'cases': cases,
    '正文简单算术': {'深度质量未配对GT观测': 5723-1723, '稳健运动未关联预测观测_不全等于真实误检': 3905-1720,
                '地图车身通过占全部查询百分比': 100*2/212, '地图车身通过占生成路线百分比': 100*2/94}})

fonts = subprocess.run(['pdffonts', str(pdf)], text=True, capture_output=True, check=True).stdout
(O / 'qa/PDF字体检查.txt').write_text(fonts)
font_lines = fonts.splitlines()[2:]
assert font_lines and all(re.search(r'\byes\s+yes\s+yes\b', r) for r in font_lines)
page_rows = []
for p in pages['pages_detail']:
    n = p['physical_page']
    page_rows.append({'物理页': n, '页图': f'qa/pages/{n:03d}.png',
                      '整页巡检拼图': f'qa/contact_{((n-1)//12)*12+1:03d}.jpg',
                      '版面巡检': '已逐页查看；未见溢出、异常空白或明显缺字',
                      '放大检查': review['放大检查物理页及重点'].get(str(n), '整页布局巡检'),
                      '机器文本字符数': p['characters'], '机器替换字符数': p['replacement_chars'],
                      '机器外边界溢出数': len(p['outer_margin_overflow']),
                      'PDF_SHA256': sha(pdf), '逐字校对': '未完成全面逐字人工校对'})
csvsave(O / 'qa/逐页检查记录.csv', page_rows)
qa['visual_review'] = f"{pages['pages']}页全部完成页图布局巡检；物理页4、5、31、34、57、62、73、81另作放大检查。未做全面逐字人工校对。"
qa['visual_review_pdf_sha256'] = sha(pdf)
qa['pdf_all_fonts_embedded'] = True
qa['source_bundle_excludes'] = ['大型原始数据集', '模型权重', '运行环境', '全部历史逐帧记录']
dump(O / 'qa/最终质量检查.json', qa)

old_source = (O.parent / 'sources/论文正文.md').read_text()
old_body = old_source[old_source.index('# 第1章'):old_source.index('# 参考文献')]
old_prose = re.sub(r'\{\{.*?\}\}|^#+ .*$', '', old_body, flags=re.M)
comparison = {'v1正文源纯段落汉字数': len(re.findall('[\u4e00-\u9fff]', old_prose)),
              'v1主文二级小节': len(re.findall('^## ', old_body, re.M)),
              'v1主文三级小节': len(re.findall('^### ', old_body, re.M)),
              'v2实际Word正文纯段落汉字数': N, 'v2主文二级小节': counts['主文二级小节数'],
              'v2主文三级小节': counts['三级标题数']}
dump(O / 'qa/修订前后结构比较.json', comparison)

(O / '修订交接说明.md').write_text(f'''# 公开数据版扩充修订稿 v2 交接说明

本版针对“子小节过多、每节内容不充足”和“全文至少5万字”重构。正式题目仍为《面向智能车辆的三维语义场景图构建与导航方法研究》。本次交付是可供导师审阅的完整初稿，不等于科研课题的全部验收。

## 实际文件与修订效果

`硕士论文初稿_公开数据版_v2.docx`为可编辑Word，`硕士论文初稿_公开数据版_v2.pdf`为实际转换并检查的{pages['pages']}页PDF。旧版v1文件在上级目录保留，未覆盖。

实际Word第1—7章纯段落共有**{N:,}个汉字**。只计U+4E00—U+9FFF汉字，排除标题、目录、中英文摘要、表格、图注、参考文献、附录、英文和数字。计数比“全文五万字”更严格；源稿对应统计为{counts['七章正文纯段落汉字数_不含标题图注表格']:,}汉字，差异来自公式中的中文在Word中转为数学对象，不计入普通段落。详见`qa/字数与结构检查.json`及逐节CSV。

原版正文{comparison['v1主文二级小节']}个二级小节、{comparison['v1主文三级小节']}个三级小节，现合并为{counts['主文二级小节数']}个二级小节、取消全部三级标题。各小节围绕一项完整问题展开；论文结构说明为短节，其余正文小节约1000—3600汉字。七章分别承担问题提出、理论基础、任务协议、建图方法、目标与规划方法、实验论证、总结展望。当前Word含20张原生可编辑表格、7张图、11个可编辑数学对象及27条参考文献。

第1—3章补充按问题组织的研究现状、现有方法的适用条件、模块接口、监督性质和评价信息隔离。第4章扩展深度污染机制、一对一分配规则、动态几何更新、时效清理以及关系增量维护的条件与开销。第5章补充专项数据结构、LoRA任务契约、候选支持与拒绝、接近位置和车辆道路约束之间的衔接。第6章扩展区间解释、分母变化、模块和参数的不同作用、可追溯查询案例以及错误沿流程传播的分析。第7章结论与证据对应，并保留缺口。新增讨论没有被写成新增实验结果。

## 数据与结论边界

正文只使用实际完成的nuScenes、KITTI及衍生专项验证。nuScenes主体为38个开发场景、11个log、1508帧，其中1470帧有过去雷达，38个缺测首帧保留；30个预留场景、14个log、1196帧没有新的封存模型测试成绩。KITTI Object的64张图片来自33个drive，仅作单帧诊断。Tracking三个30帧窗口共90帧，74次受支持GT集中于0005的4个实例，465条不支持类别/DontCare记录保留。

Qwen训练2398条、验证603条；真实图片训练25场景/7log、验证11场景/2log均为开发数据子集，不能增算场景。解析含构造表达，排序含受控符号bank，类别验证不能解释为颜色和细属性准确率。Talk2Car仅完成439条原始指令及对应图像准备，没有正式模型成绩。专项任务改善没有扩展为整车导航或独立道路泛化结论。

“导航”限于语言目标选择、安全接近位置选择和道路/车辆约束下的离线路线规划。给定正确参考对象的地图诊断已经明确声明。完整地图条件生成94/212条路线，只有2/94通过静态车身检查；生成路线不是实际到达。结构化关系没有优于直接几何，联合选择覆盖低，部分消融和参数变化无收益，这些负结果继续保留。失败、unknown、缺深度和预算耗尽不删除。

未纳入自建仿真主对比、语义消融、组合表、故障实验及到达成绩；未安装CARLA，未运行新训练或大规模模型实验，未修改基线、标准答案，未发布或推送仓库。缺少公开数据Qwen三功能分别关闭的下游消融，正文说明缺口。

## 追溯、引用与排版

`实验_图表_代码_原始结果索引.csv`含130条历史证据入口；`图表来源索引.csv`记录真实画面、掩膜、图表数据及代码；`evidence/新增论述数值与案例核验.json`保留本次新增区间讨论与三条查询原记录。数字依实际报告核对，不以早期README代替当前结果。

27条正式参考文献的作者、题名、年份和DOI或稳定链接已核对，阅读范围见`引用核验清单.csv`。其中24条获得全文并阅读相关内容；Dijkstra、Hart、Dolgov三条规划基础文献全文读取受限，未冒称已通读。原稿82条参考文献均登记处理状态，未采用的59条尚未全部完成外部元数据核验，不能直接转入正式文献表。BibTeX、RIS及阅读缓存随源包提供。本次修订复用已核验文献，没有因扩写虚构新引用。

未检出学校专用模板，采用A4、正文小四、1.4倍行距的统一中文硕士论文格式。封面不编号，前置页罗马i—iv，正文至附录阿拉伯1—80。正文、图表标题和目录均可编辑；目录是依据实际PDF页码形成的静态对照，未启用Word自动TOC域。修改内容或更换字体后必须重新生成并检查页码。PDF字体全部嵌入，部分西文由LibreOffice替换为Liberation Serif。

需要作者确认姓名、学号、学院、专业、导师、学校、日期，以及正式封面、声明、双面/章起奇数页、页眉和参考文献细则。未编造作者成果、致谢经历或学校声明。未进行Microsoft Word客户端复排、全面逐字人工校对或学校格式终审。

## 文件位置与再生成

所有修订产物位于本`revision_v2`目录。当前环境是Linux远程主机，虽已获本机访问授权，但没有可用的Windows目录挂载或本机文件工具，因而**本次未读取或写入`D:\\study\\Python.Project\\lzc\\paper`**。本版基于远程保存的原稿、上一版完整初稿及冻结实验材料完成，不能声称已同步作者Windows目录中的其他版本。

编辑`sources/论文正文.md`，表格数据编辑`sources/tables.json`；`sources/论文整合源稿.md`提供展开后的可读版本。运行`bash scripts/rebuild.sh`重新生成Word、PDF、目录和机器检查。可选`--extract-evidence`只读历史数据重绘，不运行模型。`assemble_revision.py`仅用于重建本次合并过程，默认再生成不调用它，以免覆盖作者对主源稿的修改。

依赖见`scripts/requirements.txt`；另需LibreOffice、Noto CJK字体及poppler。`.venv`不打入源包。交付包包括绘图源数据、PNG/矢量PDF、引用文件、索引和检查记录，不复制大型数据、模型权重与全部历史逐帧文件。只编辑文稿可使用缓存图表；重新提取数据仍依赖原项目路径。重建后必须重新巡视PDF，不能沿用旧文件的视觉确认。

保护检查中原始Word、原Markdown、原Git索引和历史验证报告SHA-256均与既有记录一致。历史科研状态不变，本次没有恢复其他科研Goal。

## 题目建议，尚未修改

现有正式题目中的“导航”容易被理解为车辆执行。建议作者与导师考虑《面向智能车辆的三维语义场景图构建与语言目标离线规划方法研究》，也可增加“公开录制数据”的限定。本稿继续保留原正式题目，未经作者确认不修改。
''')

(O / '质量检查记录.md').write_text(f'''# 扩充修订稿 v2 质量检查记录

检查对象为本目录v2 Word与PDF，交付指纹见`交付文件校验清单.csv`。检查时间见`qa/视觉巡检确认.json`。本记录针对当前文件，修改后必须重新检查。

## 字数和结构

实际Word第1—7章普通正文纯段落为**{N:,}汉字**，共{counts['Word正文段落数']}个非空段落。排除中英文摘要、目录、标题、表格、图注、参考文献、附录、数字和英文，未用全文件字符数冒充正文。逐节计数见`qa/各小节正文汉字数.csv`。源稿纯段落{counts['七章正文纯段落汉字数_不含标题图注表格']:,}汉字；公式中的中文转为数学对象后不计普通段落，形成7字差异。

正文7章、27个二级小节、无三级小节；目录含摘要、章、小节、参考文献和附录共42项。Word包含20个原生表格、7个图、11个OMML数学对象、51个原生下标、27条按引用顺序排列的文献。正文标题、公式号、表号、图号与引文标识均进行了结构核对，未见未解析模板标记。当前目录页码核对到PDF，目录和图表号为可编辑静态文本，并非自动交叉引用域。

## 数据和方法核对

`qa/evidence_audit.json`中98项分层计数检查通过，索引提供130条原始报告、代码、逐帧文件和图源入口；保护的4个历史文件哈希保持一致。本次扩写的建图配对差区间、KITTI共同支持误差和三条检索案例另存原始字段，未重新抽样产生新实验结论。

检查保留nuScenes缺测首帧、KITTI不支持类别、关系未解边、选择拒绝和关联未知、规划预算与失败计数。参数记录、消融记录和规划条件数没有用作独立场景数。GT监督、规则关系、弱标签、构造选择答案与规划参考分别说明。给定正确目标的诊断单列，未报告车辆实际到达。

27条正式引用全部在正文出现；24条取得全文并阅读相关内容，3条规划文献仅完成可访问的元数据或作者摘要核验。旧稿未采用文献的未完成外部核验状态继续保留，未隐去。没有宣称已完成全部旧稿文献全文核对。

## 转换和逐页布局巡检

LibreOffice实际将v2 Word转换为{pages['pages']}页PDF。修复过再生成脚本仍指向旧文件名的问题，增加转换成功日志检查；最终转换日志见`qa/rebuild_conversion_1.log`。42个标题在PDF定位，实际页码收敛，Word/PDF目录均无“—”待定页码。封面无页码，前置页i—iv，正文与附录1—80。

全部{pages['pages']}页均生成完整页图，并通过8张全页拼图逐页查看布局。物理页4、5的目录，31页真实检测/分割/深度案例，34页公式，57页建图表与图，62页Qwen表与图，73页参数表，81页参考文献另作放大检查。当前未见明显文本/图像溢出、缺字、意外空白页或模糊主图。章末和目录末页的自然留白保留。`qa/逐页检查记录.csv`逐页注明检查层级，不能解释成全面逐字校对。

机器检查：替换字符0，外边界文本溢出0，缺失标题0；14个PDF字体条目均嵌入并有Unicode映射。表格均可编辑，表行不跨页拆分，图与图注保持相邻。未进行Microsoft Word客户端复排或其他办公软件兼容性检查。

## 未完成和有效性限制

未取得学校正式模板，不能视为学校格式终审；未做全面逐字人工校对、查重或学术委员会审查。未读到Windows本地paper目录，不能保证与该目录所有版本合并。未完成3条受限规划文献全文阅读及59条未采用旧稿文献的外部元数据核验。

自然指令人审、颜色及细属性标签、独立新测试、公开数据Qwen三功能分别关闭的下游消融、外部充分正例序列及车辆执行仍未完成。当前检查证明本稿可追溯和可编辑，不将上述科研缺口视作已经完成。
''')

# 源包只打包交付与缓存证据；不把原始大型数据或环境装入。
root_files = [O / n for n in ['修订交接说明.md', '质量检查记录.md', '论文目录与图表结构说明.md', '图表来源索引.csv',
                              '硕士论文初稿_公开数据版_v2.docx', '硕士论文初稿_公开数据版_v2.pdf',
                              '实验_图表_代码_原始结果索引.csv', '引用核验清单.csv']]
bundle_files = root_files[:]
for dirname in ['sources', 'scripts', 'figures', 'figure_data', 'evidence', 'references']:
    bundle_files += [p for p in (O / dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
bundle_files += [p for p in (O / 'qa').iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.txt', '.log', '.jpg')]
bundle_files = sorted(set(bundle_files))
csvsave(O / 'qa/源包内容校验清单.csv', [{'相对路径': str(p.relative_to(O)), '字节数': p.stat().st_size, 'SHA256': sha(p)} for p in bundle_files])
bundle_files.append(O / 'qa/源包内容校验清单.csv')
archive = O / '论文源稿与图表_公开数据版_v2.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in bundle_files:
        z.write(p, arcname=str(p.relative_to(O)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(bundle_files)
files = [docx, pdf, archive, O / '修订交接说明.md', O / '质量检查记录.md',
         O / 'qa/字数与结构检查.json', O / 'qa/最终质量检查.json']
csvsave(O / '交付文件校验清单.csv', [{'文件': p.name, '相对路径': str(p.relative_to(O)),
                                  '字节数': p.stat().st_size, 'SHA256': sha(p)} for p in files])
print(json.dumps({'actual_word_cjk': N, 'pages': pages['pages'], 'reviewed_pages': len(page_rows),
                  'source_bundle_entries': len(bundle_files), 'source_bundle_bytes': archive.stat().st_size,
                  'all_passed': qa['all_passed']}, ensure_ascii=False))
