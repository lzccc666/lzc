"""核对独立渲染与交付PDF一致，并制作不缩放的双页检查图。"""
from pathlib import Path
import json,re
import pymupdf
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parents[1];Q=O/'qa';folder=Q/'rendered';pairs=Q/'review_pairs';pairs.mkdir(exist_ok=True)
original=pymupdf.open(O/'硕士论文初稿.pdf');rendered=pymupdf.open(folder/'硕士论文初稿.pdf')
assert len(original)==len(rendered)
same_text=[];same_positions=[]
for a,b in zip(original,rendered):
 same_text.append(a.get_text()==b.get_text())
 aw=[(tuple(round(z,2) for z in w[:4]),w[4]) for w in a.get_text('words')];bw=[(tuple(round(z,2) for z in w[:4]),w[4]) for w in b.get_text('words')]
 same_positions.append(aw==bw)
result={'pages':len(original),'same_page_text':all(same_text),'same_word_positions':all(same_positions),'different_pages':[i+1 for i,(a,b) in enumerate(zip(same_text,same_positions)) if not (a and b)],'render_dpi':96}
(Q/'独立渲染一致性检查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
assert result['same_page_text'] and result['same_word_positions'],result
images=sorted(folder.glob('page-*.png'),key=lambda p:int(re.search(r'(\d+)\.png$',p.name)[1]));assert len(images)==len(original),(len(images),len(original))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for start in range(0,len(images),2):
 im=Image.open(images[start]).convert('RGB');w,h=im.size;sheet=Image.new('RGB',(w*2+12,h+26),'#d0d0d0');draw=ImageDraw.Draw(sheet)
 for offset in range(2):
  if start+offset>=len(images):continue
  page=Image.open(images[start+offset]).convert('RGB');assert page.size==(w,h)
  x=offset*(w+12);sheet.paste(page,(x,26));draw.text((x+8,3),f'PAGE {start+offset+1}',font=font,fill='black')
 sheet.save(pairs/f'pair_{start+1:03d}.png')
print(json.dumps(result,ensure_ascii=False));print('native page pairs',len(list(pairs.glob('*.png'))))
