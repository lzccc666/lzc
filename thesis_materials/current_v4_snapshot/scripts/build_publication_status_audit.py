"""汇总一手来源核验，分别统计书目引用年、首次公开年和正式出版年。"""
from pathlib import Path
import collections
import csv
import hashlib
import json

O = Path(__file__).resolve().parents[1]
R = O / 'references'
A = R / 'status_audit_20261120'
refs = json.loads((R / 'verified_references.json').read_text(encoding='utf-8'))
groups = ['group_01_16.json', 'group_17_32.json', 'group_33_48.json', 'group_49_62.json']
rows = []
for name in groups:
    rows.extend(json.loads((A / name).read_text(encoding='utf-8')))
rows.sort(key=lambda r: r['number'])
assert [r['number'] for r in rows] == list(range(1, 63))
assert all(r['exists'] and r['evidence_urls'] for r in rows)
by = {r['number']: r for r in refs}
for r in rows:
    assert r['id'] == by[r['number']]['id']
    r['cited_version_year'] = int(by[r['number']]['date'][:4])
    r['formal_or_preprint_year'] = r['published_year'] or r['arxiv_year'] or r['cited_version_year']
    r['first_public_year'] = min(y for y in [r['arxiv_year'], r['published_year'], r['cited_version_year']] if y)
    r['has_confirmed_formal_publication'] = r['published_year'] is not None
    r['within_2022_2026_cited_version'] = 2022 <= r['cited_version_year'] <= 2026
    r['within_2022_2026_formal_priority'] = 2022 <= r['formal_or_preprint_year'] <= 2026

summary = {
    'checked_on': '2026-11-20',
    'recent_year_window': [2022, 2026],
    'total': len(rows),
    'verified_existing_papers': sum(r['exists'] for r in rows),
    'confirmed_formal_publications': sum(r['has_confirmed_formal_publication'] for r in rows),
    'preprints_confirmed_formal_publication_not_established': [r['number'] for r in rows if not r['has_confirmed_formal_publication']],
    'early_access_only_confirmed': 0,
    'recent_cited_version_year_count': sum(r['within_2022_2026_cited_version'] for r in rows),
    'recent_formal_priority_year_count': sum(r['within_2022_2026_formal_priority'] for r in rows),
    'recent_first_public_year_count': sum(2022 <= r['first_public_year'] <= 2026 for r in rows),
    'cited_version_year_histogram': dict(sorted(collections.Counter(r['cited_version_year'] for r in rows).items())),
    'formal_priority_year_histogram': dict(sorted(collections.Counter(r['formal_or_preprint_year'] for r in rows).items())),
    'word_sha256': hashlib.sha256((O / '硕士论文初稿.docx').read_bytes()).hexdigest(),
    'pdf_sha256': hashlib.sha256((O / '硕士论文初稿.pdf').read_bytes()).hexdigest(),
    'scope': '论文存在性、出版状态及年份核验；不等于全文通读、逐项论点支撑验证或所有作者卷期页逐字段无误。',
}
for field in ['recent_cited_version_year_count', 'recent_formal_priority_year_count', 'recent_first_public_year_count']:
    summary[field.replace('_count', '_percentage')] = round(summary[field] / len(rows) * 100, 2)
(R / '论文身份与发表状态核验.json').write_text(json.dumps({'summary': summary, 'references': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
csvrows = []
for r in rows:
    notes = r.get('notes', '')
    if isinstance(notes, list): notes = '；'.join(notes)
    csvrows.append({
        '序号': r['number'], '文献ID': r['id'], '题名': r['title'], '论文真实存在': r['exists'],
        '当前引用版本年': r['cited_version_year'], 'arXiv首发年': r['arxiv_year'] or '',
        '已核实正式发表年': r['published_year'] or '', '正式发表或预印本年': r['formal_or_preprint_year'],
        '状态': '正式发表已确认' if r['has_confirmed_formal_publication'] else '预印本已确认；正式发表未核实',
        '会议或期刊': r['venue'] or '', 'DOI': r['doi'] or '',
        '当前引用版本年在2022至2026': r['within_2022_2026_cited_version'],
        '正式优先口径在2022至2026': r['within_2022_2026_formal_priority'],
        '一手来源': ' ; '.join(r['evidence_urls']), '说明': notes,
    })
with (R / '论文身份与发表状态核验.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(csvrows[0]))
    writer.writeheader(); writer.writerows(csvrows)

n = len(rows)
formal = summary['confirmed_formal_publications']
pre = summary['preprints_confirmed_formal_publication_not_established']
lines = [
    '# 参考文献真实性与年份核验报告', '',
    '核验日期：2026年11月20日。对象：当前毕业论文的62项正式书目。', '',
    f'62项均能对应真实论文。{formal}项取得正式会议或期刊发表证据；其余{len(pre)}项已确认arXiv预印本，正式出版状态本轮未建立。预印本身份符合本轮允许范围；没有将未核实的录用或正式发表状态补写为事实。', '',
    '## 近五年统计', '',
    '本轮“近五年”按2022—2026五个自然年计算。三种口径分别计算，不以更新日期、下载日期或预印本最新版本年替代论文年份。', '',
    '| 口径 | 近五年项数 | 占62项比例 |', '| --- | ---: | ---: |',
    f"| 当前论文所引版本年份 | {summary['recent_cited_version_year_count']} | {summary['recent_cited_version_year_percentage']:.1f}% |",
    f"| 已核实正式发表年优先；其余使用预印本首发年 | {summary['recent_formal_priority_year_count']} | {summary['recent_formal_priority_year_percentage']:.1f}% |",
    f"| 工作首次公开年份 | {summary['recent_first_public_year_count']} | {summary['recent_first_public_year_percentage']:.1f}% |", '',
    '真实性条件已满足；“近五年占多数”取决于统计口径。当前书目所引版本年尚未过半，正式发表年优先口径仅略过半，现有结构包含较多经典基础工作。后续补充优先选择2022—2026年与道路对象建图、动态关系维护及语言目标选择直接相关的论文，同时保留确有必要的经典算法和数据集依据。', '',
    '## 引用版本与年份处理', '',
    'Kimera-Multi、GLIP、ByteTrack及LoRA的预印本均首发于2021年，其正式论文发表于2022年，因此两种年份口径相差4项。当前稿引用arXiv时，其首发年份本身成立；若切换为正式版，应同时替换题名、作者顺序、会议或期刊、年份、页码和DOI，不能只把年份改晚。', '',
    '预印本与正式版的页码、作者顺序可能不同。ConceptFusion存在所引版本作者顺序差异；nuScenes、PointPillars、CenterPoint及GLIP的CVF开放版页码与IEEE注册页码需按各自版本记录。当前arXiv书目未填写这些正式出版页码，本轮不将不同版本元数据混拼。ConceptFusion是独立论文，用户排除的ConceptGraphs仍不在正式书目中。', '',
    '## 本轮仅确认预印本的条目', '',
]
for k in pre:
    r = rows[k - 1]
    note = r.get('notes', '')
    if isinstance(note, list): note = '；'.join(note)
    lines.append(f"- [{k}] {r['title']}：{r['arxiv_year']}，[{r['id']}]({by[k]['url']})。{note}")
lines += ['', '此分类表示本轮未取得对应正式出版证据，并不推断论文从未发表。AB3DMOT作者提及ECCV工作坊extended abstract，当前准确题名仍按已核实预印本处理。', '',
          '## 逐条记录', '',
          '| 序号 | 工作 | 当前引用年 | 正式年 | 状态 | 证据 |', '| --- | --- | ---: | ---: | --- | --- |']
for r in rows:
    links = '；'.join(f'[来源{i+1}]({url})' for i, url in enumerate(r['evidence_urls']))
    lines.append(f"| {r['number']} | {r['title'].replace('|', '／')} | {r['cited_version_year']} | {r['published_year'] or '未核实'} | {'正式发表' if r['has_confirmed_formal_publication'] else '已确认预印本'} | {links} |")
lines += ['', '## 核验范围与文件对应', '',
          '逐条检查arXiv原始记录、会议或出版社论文页面及可用的Crossref注册元数据；完整证据URL与状态写入同目录CSV/JSON，原始注册响应和分组核验文件保存在status_audit_20261120。检索结果只使用对应一手来源，不以聚合网站标签认定录用。', '',
          '本轮交付核验报告，正文、正式书目、Word及PDF内容保持原版。报告中的正式发表年统计与当前稿采用的预印本书目年份分别标注。该核验确认论文存在、状态和年份；文献全文获取/重点阅读深度仍沿用原有逐篇记录。', '',
          f"Word SHA-256：{summary['word_sha256']}。", '', f"PDF SHA-256：{summary['pdf_sha256']}。", '']
(R / '参考文献真实性与年份核验报告.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
