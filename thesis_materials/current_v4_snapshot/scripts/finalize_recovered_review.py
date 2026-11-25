"""将跨聊天连续完成的逐页布局检查登记到当前PDF指纹。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

O = Path(__file__).resolve().parents[1]
Q = O / 'qa'
verify = json.loads((Q / '中断恢复独立核验.json').read_text(encoding='utf-8'))
assert verify['raster_comparison']['all_pages_2_to_91_pixel_identical']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(O / '硕士论文初稿.pdf') == verify['pdf_sha256']
assert sha(O / '硕士论文初稿.docx') == verify['docx_sha256']
backup = O / '历史版本/文献扩充前'
backup.mkdir(parents=True, exist_ok=True)
for name in ['视觉巡检确认.json', '逐页检查记录.csv']:
    if (Q / name).exists() and not (backup / name).exists():
        shutil.copy2(Q / name, backup / name)
review = {
    'date': '2026-11-20',
    'pdf_sha256': verify['pdf_sha256'],
    'docx_sha256': verify['docx_sha256'],
    'reviewed_pages': 91,
    'passed': True,
    'scope': '布局检查跨两个聊天连续完成。旧聊天打开pair_001—pair_087共44张原尺寸双页图（物理1—88页），发现封面蓝线后已修正；本聊天查看修订后pair_001、pair_089及pair_091，完成新封面及89—91页检查。封面调整前后2—91页在96 DPI下逐像素一致，继承旧聊天2—88页的检查。检查范围为版面、表图、公式、图注及文献换行，不等于全文逐字校对或每篇文献通读。',
    'prior_thread_id': verify['origin_thread'],
    'inherited_review_pages': list(range(2, 89)),
    'current_thread_opened_pages': [1, 2, 89, 90, 91],
    'pixel_comparison_record': 'qa/中断恢复独立核验.json',
    'independent_render_record': 'qa/独立渲染一致性检查.json',
}
(Q / '视觉巡检确认.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
pages = json.loads((Q / 'page_inspection.json').read_text(encoding='utf-8'))
assert pages['pages'] == 91
rows = []
for p in pages['pages_detail']:
    n = p['physical_page']
    rows.append({
        '物理页': n,
        '检查来源': '本聊天查看原尺寸页面图' if n in review['current_thread_opened_pages'] else '继承旧聊天逐页检查；本轮逐像素一致性核验',
        '替换字符数': p['replacement_chars'],
        '外边界溢出': bool(p['outer_margin_overflow']),
        '页首': ' / '.join(p['first_lines']),
        '页尾': ' / '.join(p['last_lines']),
        'PDF_SHA256': review['pdf_sha256'],
    })
with (Q / '逐页检查记录.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('当前91页布局检查登记完成；PDF指纹与独立核验一致。')
