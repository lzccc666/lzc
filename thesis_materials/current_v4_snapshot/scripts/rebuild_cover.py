from pathlib import Path
import subprocess,shutil,json
O=Path(__file__).resolve().parents[1]
shutil.copy2(O/'硕士论文初稿.pdf','/tmp/thesis_before_cover_fix.pdf')
for script in ['build_word.py']:
 subprocess.run([str(O/'.venv/bin/python'),str(O/'scripts'/script)],check=True)
subprocess.run(['libreoffice','-env:UserInstallation=file://'+str(O/'qa/lo_cover_profile'),'--headless','--convert-to','pdf','--outdir',str(O),str(O/'硕士论文初稿.docx')],check=True)
for script in ['qa_document.py','finish_delivery.py']:
 subprocess.run([str(O/'.venv/bin/python'),str(O/'scripts'/script)],check=True)
import pymupdf
a=pymupdf.open('/tmp/thesis_before_cover_fix.pdf');b=pymupdf.open(O/'硕士论文初稿.pdf')
unchanged=[]
for i,(x,y) in enumerate(zip(a,b)):
 aw=[(tuple(round(v,2) for v in w[:4]),w[4]) for w in x.get_text('words')];bw=[(tuple(round(v,2) for v in w[:4]),w[4]) for w in y.get_text('words')]
 if aw==bw:unchanged.append(i+1)
out={'before_pages':len(a),'after_pages':len(b),'unchanged_word_layout_pages':unchanged,'all_body_pages_unchanged':len(a)==len(b) and all(i in unchanged for i in range(2,len(a)+1)),'change':'封面Title样式移除模板蓝色段落下边线'}
(O/'qa/封面样式调整检查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
assert out['all_body_pages_unchanged'],out
print(json.dumps(out,ensure_ascii=False))
