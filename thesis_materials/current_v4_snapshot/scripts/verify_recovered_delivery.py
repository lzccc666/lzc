"""恢复中断交付：只读核验引用、源稿、文档与封面修订前后页面。"""
from pathlib import Path
import hashlib
import json
import re
import zipfile

import pymupdf
from lxml import etree

O = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = (O / 'sources/论文正文.md').read_text(encoding='utf-8')
refs = json.loads((O / 'references/verified_references.json').read_text(encoding='utf-8'))
ids = {r['id'] for r in refs}
# 本稿章标题采用单个#；兼容带空格的编号。
chapter3 = re.search(r'^#\s+第\s*3\s*章', source, re.M)
assert chapter3, '未找到第三章边界'
first_two = source[:chapter3.start()]
cited = {k for m in re.findall(r'\[@([^\]]+)\]', first_two) for k in m.split(';')}
with zipfile.ZipFile(O / '硕士论文初稿.docx') as z:
    root = etree.fromstring(z.read('word/document.xml'))
    styles = etree.fromstring(z.read('word/styles.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
word_text = ''.join(root.xpath('//w:t/text()', namespaces=ns))
checks = {
    '62_unique_ids': len(refs) == len(ids) == 62,
    '62_unique_titles': len({re.sub(r'\W', '', r['title']).lower() for r in refs}) == 62,
    'all_62_cited_in_first_two_chapters': cited == ids,
    'complete_numbering_1_to_62': {r['number'] for r in refs} == set(range(1, 63)),
    'excluded_work_absent': not re.search('conceptgraph', source + json.dumps(refs), re.I),
    'no_title_style_border': not styles.xpath('//w:style[@w:styleId="Title"]/w:pPr/w:pBdr', namespaces=ns),
    'no_unresolved_word_citations': '[@' not in word_text and '{{' not in word_text,
}
assert all(checks.values()), checks
result = {
    'date': '2026-11-20',
    'origin_thread': '01a1e336-6294-7a92-9ab2-5b73154d0c49',
    'checks': checks,
    'docx_sha256': sha(O / '硕士论文初稿.docx'),
    'pdf_sha256': sha(O / '硕士论文初稿.pdf'),
    'source_sha256': sha(O / 'sources/论文正文.md'),
    'cached_fulltexts': sum(any((O / 'references' / r['id']).glob('*fulltext.txt')) for r in refs),
    'method_reading_records': sum((O / 'references' / r['id'] / '阅读核对.md').exists() for r in refs),
}
before_path = Path('/tmp/thesis_before_cover_fix.pdf')
if before_path.exists():
    before = pymupdf.open(before_path)
    current = pymupdf.open(O / '硕士论文初稿.pdf')
    assert len(before) == len(current) == 91
    identical = []
    for i in range(1, len(current)):
        a = before[i].get_pixmap(dpi=96)
        b = current[i].get_pixmap(dpi=96)
        if (a.width, a.height, a.samples) == (b.width, b.height, b.samples):
            identical.append(i + 1)
    assert identical == list(range(2, 92)), identical
    result['raster_comparison'] = {
        'before_pdf_sha256': sha(before_path),
        'dpi': 96,
        'pixel_identical_body_pages': identical,
        'all_pages_2_to_91_pixel_identical': True,
        'visual_review_inheritance': '旧聊天实际打开pair_001至pair_087共44张原尺寸双页图；本聊天复查改动封面并补查89—91页。',
    }
    protected = json.loads((O / 'qa/protected_sources.json').read_text())
    result['protected_non_git_sources'] = [
        {'path': r['path'], 'unchanged': sha(Path(r['path'])) == r['sha256']}
        for r in protected if '.git' not in Path(r['path']).parts
    ]
    assert all(r['unchanged'] for r in result['protected_non_git_sources'])
(O / 'qa/中断恢复独立核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'raster_comparison'}, ensure_ascii=False))
if 'raster_comparison' in result:
    print('封面调整后物理第2—91页逐像素一致；可继承旧聊天第2—88页布局检查。')
