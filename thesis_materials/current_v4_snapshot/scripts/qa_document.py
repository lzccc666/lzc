from pathlib import Path
import pymupdf,json,re,hashlib,csv
from PIL import Image,ImageOps,ImageDraw,ImageFont
O=Path(__file__).resolve().parents[1];P=O/'硕士论文初稿.pdf';doc=pymupdf.open(P);entries=json.loads((O/'qa/headings.json').read_text());normal=lambda s:re.sub(r'\s+','',s);emap={normal(t):t for t,l in entries};found={};qa=[];(O/'qa/pages').mkdir(exist_ok=True)
for i,p in enumerate(doc):
 d=p.get_text('dict');texts=[];bad=[];overflow=[];sizes=[]
 for b in d['blocks']:
  big=''.join(s['text'] for l in b.get('lines',[]) for s in l['spans'] if s['size']>=14)
  if normal(big) in emap:found[emap[normal(big)]]=i
  for line in b.get('lines',[]):
   txt=''.join(s['text'] for s in line['spans']);texts.append(txt);size=max([s['size'] for s in line['spans']],default=0)
   if size>=14 and normal(txt) in emap:found[emap[normal(txt)]]=i
   for s in line['spans']:
    if '\ufffd' in s['text']:bad.append(s['text'])
    x0,y0,x1,y1=s['bbox']
    if x0<20 or x1>p.rect.width-20 or y0<12 or y1>p.rect.height-12:overflow.append(s['text'])
 text=p.get_text();qa.append({'physical_page':i+1,'characters':len(text),'images':len(p.get_image_info()),'replacement_chars':len(bad),'outer_margin_overflow':overflow,'text_y_min':min([b['bbox'][1] for b in d['blocks'] if 'lines'in b],default=0),'text_y_max':max([b['bbox'][3] for b in d['blocks'] if 'lines'in b],default=0),'first_lines':texts[:4],'last_lines':texts[-4:]})
 p.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(O/'qa/pages'/f'{i+1:03d}.png')
for stale in (O/'qa/pages').glob('*.png'):
 if int(stale.stem)>len(doc):stale.unlink()
base=found['第1章 绪论'];roman=lambda n:('x'*(n//10))+['','i','ii','iii','iv','v','vi','vii','viii','ix'][n%10]
mp={t:(i-base+1 if i>=base else roman(i)) for t,i in found.items()};old=json.loads((O/'qa/page_map.json').read_text()) if (O/'qa/page_map.json').exists() else {}; (O/'qa/page_map.json').write_text(json.dumps(mp,ensure_ascii=False,indent=2));(O/'qa/page_inspection.json').write_text(json.dumps({'pages':len(doc),'body_start_physical_page':base+1,'missing_headings':[t for t,l in entries if t not in found],'page_map_changed':mp!=old,'pages_detail':qa},ensure_ascii=False,indent=2));(O/'qa/pdf_fulltext.txt').write_text('\n\n'.join(p.get_text() for p in doc))
# 全页缩略审查；每张12页，所有物理页均有完整页图。
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for start in range(0,len(doc),12):
 sheet=Image.new('RGB',(1440,4*720),'#d6d6d6');draw=ImageDraw.Draw(sheet)
 for n in range(start,min(start+12,len(doc))):
  im=Image.open(O/'qa/pages'/f'{n+1:03d}.png').convert('RGB');im.thumbnail((460,670));x=(n-start)%3*480+(480-im.width)//2;y=(n-start)//3*720+30;sheet.paste(im,(x,y));draw.text((x,y-23),f'PAGE {n+1}',fill='black',font=font)
 sheet.save(O/'qa'/f'contact_{start+1:03d}.jpg',quality=92)
print(json.dumps({'pages':len(doc),'body_start':base+1,'headings_found':len(found),'expected':len(entries),'map_changed':mp!=old,'replacement_chars':sum(x['replacement_chars'] for x in qa),'outer_margin_overflow':sum(bool(x['outer_margin_overflow']) for x in qa)},ensure_ascii=False))
