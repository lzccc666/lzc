"""生成交接材料与只读质量审计；不训练、不改写历史实验。"""
from pathlib import Path
import csv, json, re, hashlib, difflib, collections, importlib.metadata, zipfile
from lxml import etree
import pymupdf
from docx import Document

O = Path(__file__).resolve().parents[1]
S = (O / 'sources/论文正文.md').read_text()
T = json.loads((O / 'sources/tables.json').read_text())
F = json.loads((O / 'sources/figures.json').read_text())
R = json.loads((O / 'references/verified_references.json').read_text())
by = {r['id']: r for r in R}

def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def resolve(text):
    return re.sub(r'\[@([^\]]+)\]', lambda m: '[' + ','.join(str(by[k]['number']) for k in m[1].split(';')) + ']', text)

# 独立可读的整合源稿：模板源仍单独保留。
lines = []
for line in S.splitlines():
    if line.startswith('{{table:'):
        v = T[line[8:-2]]; lines.extend(['', v['title'], '', '| ' + ' | '.join(v['headers']) + ' |', '| ' + ' | '.join('---' for _ in v['headers']) + ' |'])
        lines.extend('| ' + ' | '.join(str(x).replace('|', '／') for x in row) + ' |' for row in v['rows'])
        lines.extend(['', '注：' + v['note'] + ' 数据来源：' + v['source'], ''])
    elif line.startswith('{{figure:'):
        key, caption = line[9:-2].split('|', 1)
        lines.extend(['', f"![{caption}](../figures/{key}.png)", ''])
    elif line == '{{references}}':
        lines.extend(f"[{r['number']}] {r['formatted']}\n" for r in R)
    else:
        lines.append(resolve(line))
(O / 'sources/论文整合源稿.md').write_text('\n'.join(lines))

# 原稿条目逐条结构与来源对照；没有核验的条目不冒充已核验。
oldpath = O / 'references/原稿参考文献检查.csv'
old = list(csv.DictReader(oldpath.open(encoding='utf-8-sig')))
normal = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
extra_ids = {24: '2306.14289', 56: '2306.12156', 74: '2011.13588', 81: '2310.01412', 82: '2310.03026'}
rows = []
for row in old:
    original = row['原条目']; number = int(re.match(r'\[(\d+)\]', original)[1])
    # 原稿作者中的“et al.”不作为题名终止点。
    body = re.sub(r'^\[\d+\]\s*', '', original)
    title = body.split('. ', 1)[1].split('[')[0].strip() if '. ' in body else ''
    scores = [(difflib.SequenceMatcher(None, normal(title), normal(r['title'])).ratio(), r) for r in R]
    score, match = max(scores, key=lambda x: x[0])
    adopted = score > .96
    status = '结构与题名检查完成；本稿未采用，原条目元数据未完成外部核验'
    url = ''; note = 'Crossref限流；不能将结构检查解释为真实性或原文支撑核验。'
    if adopted:
        status = '与本稿独立核验文献同题；本稿使用重新核验条目'
        url = match['url']; note = '原稿的出版年份/页码不直接继承；本稿预印本按首次提交年引用。'
        if number == 29:
            note += ' 原条目第三作者顺序与作者提交版本不同，采用作者页面完整作者表。'
    elif number in [24, 74, 81, 82]:
        status = 'arXiv作者页面题名、前三作者和首次提交年已核验；本稿未采用'
        url = 'https://arxiv.org/abs/' + extra_ids[number]
        note = '仅补充原稿检查；没有据此引入新的正文结论或实验。'
    rows.append({'原序号': number, '原条目': original, '提取题名': title, '检查状态': status,
                 '重新核验链接': url, '本稿对应序号': match['number'] if adopted else '', '备注': note})
save_csv(O / 'references/原稿参考文献逐条处理.csv', rows)

figrows = []
for line in S.splitlines():
    if not line.startswith('{{figure:'): continue
    key, caption = line[9:-2].split('|', 1)
    v = F[key]; sources = v['source'] if isinstance(v['source'], list) else [v['source']]
    figrows.append({'图号': caption.split(' ')[0], '键': key, '图注': caption,
                    '成图PNG': str(O / 'figures' / (Path(v['path']).name)), '矢量PDF': str(O / 'figures' / (Path(v['path']).stem + '.pdf')),
                    '原始来源': '; '.join(x if x.startswith('本论文') else str(Path(x) if Path(x).is_absolute() else O.parents[1] / x) for x in sources),
                    '绘图代码': str(O / 'scripts/build_evidence_figures.py'),
                    '数据源': str(O / 'figure_data'), '类型': '流程示意' if key == 'pipeline' else ('真实保存结果派生可视化' if key in ['case','relations_case'] else '冻结报告统计图')})
save_csv(O / '图表来源索引.csv', figrows)

packages = ['python-docx', 'PyMuPDF', 'matplotlib', 'numpy', 'Pillow', 'lxml', 'requests', 'beautifulsoup4']
versions = {x: importlib.metadata.version(x) for x in packages}
(O / 'scripts/requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items()) + '\n')

with zipfile.ZipFile(O / '硕士论文初稿.docx') as z:
    root = etree.fromstring(z.read('word/document.xml'))
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main', 'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
wordtext = ''.join(root.xpath('//w:t/text()|//m:t/text()', namespaces=ns))
pages = json.loads((O / 'qa/page_inspection.json').read_text())
evidence = json.loads((O / 'qa/evidence_audit.json').read_text())
doc = pymupdf.open(O / '硕士论文初稿.pdf')
actual_word = Document(O / '硕士论文初稿.docx')
in_main = False
main_prose = []
main_subsections = []
for paragraph in actual_word.paragraphs:
    text = paragraph.text
    if text == '第1章 绪论' and paragraph.style.name == 'Heading 1':
        in_main = True
    if text == '参考文献' and paragraph.style.name == 'Heading 1':
        in_main = False
    if not in_main:
        continue
    if paragraph.style.name == 'Heading 2':
        main_subsections.append({'小节': text, 'Word正文纯段落汉字数': 0})
    if paragraph.style.name in ('Normal', 'Body Text'):
        count = len(re.findall('[\u4e00-\u9fff]', text))
        main_prose.append(text)
        if main_subsections:
            main_subsections[-1]['Word正文纯段落汉字数'] += count
word_cjk = len(re.findall('[\u4e00-\u9fff]', ''.join(main_prose)))
source_main=S[S.index('# 第1章 '):S.index('# 参考文献')]
source_plain='\n'.join(line for line in source_main.splitlines() if line and not line.startswith(('#','{{','$$')))
count_report = {'全文源汉字数':len(re.findall('[\u4e00-\u9fff]', S)),
                '七章正文汉字数_含标题图注':len(re.findall('[\u4e00-\u9fff]',source_main)),
                '七章正文纯段落汉字数_不含标题图注表格':len(re.findall('[\u4e00-\u9fff]',source_plain)),
                '主文二级小节数':len(re.findall(r'^## [1-7]\.\d+ ',S,re.M)),
                '三级标题数':len(re.findall(r'^### ',S,re.M)),
                '源字符数':len(S), '达到五万汉字':word_cjk>=50000}
count_report.update({'实际Word七章正文纯段落汉字数': word_cjk,
                     'Word正文段落数': len([p for p in main_prose if p.strip()]),
                     'Word正文纯段落达到五万汉字': word_cjk >= 50000,
                     '计数规则': '仅统计第1至7章Normal/Body Text段落的U+4E00–U+9FFF汉字；不含标题、目录、中英文摘要、表格、图注、参考文献、附录、数字与英文。'})
(O / 'qa/字数与结构检查.json').write_text(json.dumps(count_report, ensure_ascii=False, indent=2))
save_csv(O / 'qa/各小节正文汉字数.csv', main_subsections)
toc_paras = [p.text for p in actual_word.paragraphs if '\t' in p.text and p.text.split('\t')[0] in json.loads((O/'qa/page_map.json').read_text())]
toc_pdf = ''.join(doc[i].get_text() for i in range(3, pages['body_start_physical_page']-1))
captions = [v['title'] for v in T.values()]
tables = len(root.xpath('//w:tbl', namespaces=ns)); formulas = len(root.xpath('//m:oMath', namespaces=ns))
checks = {
    '七章及中英文摘要': all(t in wordtext for t in ['第1章 绪论','第2章 相关理论与关键技术','第3章 任务定义与总体方法','第4章 三维语义场景图构建方法','第5章 基于场景图的语义目标选择与路径规划','第6章 实验与结果分析','第7章 总结与展望','中文摘要','ABSTRACT']),
    '20张Word原生可编辑表格': tables == 20,
    '7张真实来源或流程图': len(F) == 7 and len(root.xpath('//w:drawing', namespaces=ns)) == 7,
    '11个Word可编辑数学对象': formulas == 11,
    '没有遗留模板引用占位符': '{{' not in wordtext and '[@' not in wordtext,
    '实际Word正文纯段落汉字不少于五万': word_cjk >= 50000,
    '无三级标题': '\n### ' not in S,
    '目录实际页码收敛': not pages['page_map_changed'] and not pages['missing_headings'],
    'Word和PDF目录均无待定页码': len(toc_paras) == 42 and all(p.split('\t')[-1] != '—' for p in toc_paras) and '—' not in toc_pdf,
    'PDF无替换字与外边界溢出': not any(p['replacement_chars'] or p['outer_margin_overflow'] for p in pages['pages_detail']),
    '所有表题都在Word中': all(t in wordtext for t in captions),
    '参考文献不少于50条且全部正文引用': len(R) >= 50 and all(r['id'] in re.findall(r'[^;]+', ';'.join(re.findall(r'\[@([^\]]+)\]', S))) for r in R),
    '前两章独立引用不少于50项工作': len(set(';'.join(re.findall(r'\[@([^\]]+)\]', S[S.index('# 第1章 '):S.index('# 第3章 ')])).split(';'))) >= 50,
    '正文和正式参考文献无指定排除文献': not re.search(r'conceptgraphs?', S + json.dumps(R), re.I),
    f'分层原始计数{len(evidence["checks"])}项核验通过': evidence['passed'] and len(evidence['checks']) >= 97,
    '正式题目未擅改': root.xpath('string(//w:body)', namespaces=ns).find('面向智能车辆的三维语义场景图') >= 0,
}
pdf_sha=hashlib.sha256((O/'硕士论文初稿.pdf').read_bytes()).hexdigest()
review_file=O/'qa/视觉巡检确认.json'
review=json.loads(review_file.read_text()) if review_file.exists() else {}
visual_current=review.get('pdf_sha256')==pdf_sha and review.get('reviewed_pages')==len(doc) and review.get('passed') is True
qa = {'pages':len(doc), 'editable_tables':tables, 'figures':len(F), 'editable_equations':formulas,
      'native_subscripts':len(root.xpath('//m:sSub', namespaces=ns)), 'references':len(R), 'toc_entries':len(json.loads((O/'qa/headings.json').read_text())),
      'manuscript_source_characters':len(S), 'word_main_prose_cjk_characters':word_cjk, 'checks':checks, 'all_passed':all(checks.values()),
      'pdf_sha256':pdf_sha,'visual_qa_passed':visual_current,
      'visual_review':'当前PDF逐页布局检查完成，范围见质量检查记录。' if visual_current else '当前PDF机器检查已执行，待对本文件进行逐页视觉确认。',
      'not_completed':['未使用作者学校正式模板；学校信息未补','未在Microsoft Word客户端重新排版检查','Dijkstra/Hart/Dolgov全文读取受限','未采用的原稿文献并非全部完成外部元数据核验','未执行新独立模型测试、训练或车辆执行']}
(O / 'qa/最终质量检查.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2))
assert qa['all_passed'], qa

print(json.dumps(qa, ensure_ascii=False, indent=2))
