from pathlib import Path
import re,json
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK,WD_TAB_ALIGNMENT,WD_TAB_LEADER
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
O=Path(__file__).resolve().parents[1];source=(O/'sources/论文正文.md').read_text();T=json.loads((O/'sources/tables.json').read_text());F=json.loads((O/'sources/figures.json').read_text());refs=json.loads((O/'references/verified_references.json').read_text());refmap={r['id']:r for r in refs};pages=json.loads((O/'qa/page_map.json').read_text()) if (O/'qa/page_map.json').exists() else {}
D=Document();sec=D.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(2.6);sec.bottom_margin=Cm(2.5);sec.left_margin=Cm(2.8);sec.right_margin=Cm(2.5);sec.header_distance=Cm(1.3);sec.footer_distance=Cm(1.3)
styles=D.styles
for style in styles:
 # 清除默认模板的标题装饰线，保持学位论文黑色标题样式。
 if style.name in ('Title','Subtitle'):
  for border in list(style._element.iter(qn('w:pBdr'))):border.getparent().remove(border)
for name in ['Normal','Body Text']:
 s=styles[name];s.font.name='Times New Roman';s.font.size=Pt(12);s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC');s.paragraph_format.line_spacing=1.4;s.paragraph_format.first_line_indent=Pt(24);s.paragraph_format.space_after=Pt(0)
for name,size in [('Title',24),('Heading 1',18),('Heading 2',15),('Heading 3',12)]:
 s=styles[name];s.font.name='Times New Roman';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor(0,0,0);s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC');s.paragraph_format.first_line_indent=Pt(0);s.paragraph_format.space_before=Pt(14);s.paragraph_format.space_after=Pt(9);s.paragraph_format.keep_with_next=True
styles['Heading 1'].paragraph_format.page_break_before=True;styles['Heading 1'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
for name in ['Caption','TOC Heading']:
 s=styles[name];s.font.name='Times New Roman';s.font.size=Pt(10.5);s.font.color.rgb=RGBColor(0,0,0);s.font.italic=False;s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC');s.paragraph_format.first_line_indent=Pt(0);s.paragraph_format.line_spacing=1.15
settings=D.settings.element;upd=OxmlElement('w:updateFields');upd.set(qn('w:val'),'false');settings.append(upd)
D.core_properties.title='面向智能车辆的三维语义场景图构建与导航方法研究';D.core_properties.subject='三维语义场景图硕士论文修订稿';D.core_properties.author='';D.core_properties.keywords='nuScenes;KITTI;场景图;离线规划'
def font(run,size=12,bold=False):
 run.font.name='Times New Roman';run.font.size=Pt(size);run.font.bold=bold;run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
def textresolve(t):
 def repl(m):return '['+','.join(str(refmap[k]['number']) for k in m.group(1).split(';'))+']'
 return re.sub(r'\[@([^\]]+)\]',repl,t)
def para(t='',style=None):
 p=D.add_paragraph(style=style);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY if style is None else p.alignment;p.add_run(textresolve(t))
 if globals().get('front_mode',False) and style is None:
  p.paragraph_format.line_spacing=1.25
  for r in p.runs:font(r,11)
 return p
def field(p,code):
 r=p.add_run();a=OxmlElement('w:fldChar');a.set(qn('w:fldCharType'),'begin');b=OxmlElement('w:instrText');b.set(qn('xml:space'),'preserve');b.text=' '+code+' ';c=OxmlElement('w:fldChar');c.set(qn('w:fldCharType'),'end');r._r.extend([a,b,c])
def section(fmt=None,start=None):
 s=D.add_section(WD_SECTION_START.NEW_PAGE);s.header.is_linked_to_previous=False;s.footer.is_linked_to_previous=False
 if fmt:
  pg=OxmlElement('w:pgNumType');pg.set(qn('w:fmt'),fmt);pg.set(qn('w:start'),str(start or 1));s._sectPr.append(pg);p=s.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;field(p,'PAGE')
 return s
# 封面不写作者身份与虚构学校声明。
for _ in range(3):para()
p=para('硕 士 学 位 论 文','Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
for _ in range(2):para()
p=para('面向智能车辆的三维语义场景图\n构建与导航方法研究','Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=para('文献调研修订稿');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
for _ in range(3):para()
for t in ['研究范围：nuScenes、KITTI及其衍生专项验证','作者与学校信息：待作者按学校要求补入','整理日期：2026年11月20日']:
 p=para(t);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
section('lowerRoman',1)
bookmark_id=0
heading_names=[]
def heading(t,level):
 global bookmark_id
 p=para(t,'Heading '+str(level));bookmark_id+=1;a=OxmlElement('w:bookmarkStart');a.set(qn('w:id'),str(bookmark_id));a.set(qn('w:name'),'heading_'+str(bookmark_id));b=OxmlElement('w:bookmarkEnd');b.set(qn('w:id'),str(bookmark_id));p._p.insert(0,a);p._p.append(b);heading_names.append((t,level,bookmark_id));return p
# 图表标题和表格均可编辑；不把表格嵌入为图片。
def table(key):
 v=T[key];p=para(v['title'],'Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
 tb=D.add_table(rows=1,cols=len(v['headers']));tb.autofit=False;tb.alignment=WD_ALIGN_PARAGRAPH.CENTER;width=15.7/len(v['headers'])
 for c in tb.columns:c.width=Cm(width)
 for i,h in enumerate(v['headers']):tb.rows[0].cells[i].text=str(h)
 for row in v['rows']:
  for c,t in zip(tb.add_row().cells,row):c.text=str(t)
 for ri,row in enumerate(tb.rows):
  pr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');pr.append(cant)
  if ri==0:
   rep=OxmlElement('w:tblHeader');pr.append(rep)
  for cell in row.cells:
   tcPr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single' if (edge=='top' and ri==0) or (edge=='bottom' and ri in [0,len(tb.rows)-1]) else 'nil');e.set(qn('w:sz'),'8' if ri in [0,len(tb.rows)-1] else '4');borders.append(e)
   tcPr.append(borders)
   for p in cell.paragraphs:
    p.paragraph_format.keep_with_next=(ri<len(tb.rows)-1);p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(2 if key=='supervision' else 4);p.paragraph_format.space_before=Pt(2 if key=='supervision' else 4);p.alignment=WD_ALIGN_PARAGRAPH.CENTER if len(v['headers'])>3 else WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs:font(r,8.5 if len(v['headers'])>=8 else 9.5,ri==0)
 note=para('注：'+v['note']+' 数据来源：'+v['source'],'Caption');note.paragraph_format.space_after=Pt(8)
 # 长路径允许在斜杠处换行。
 for r in note.runs:r.text=r.text.replace('/','/\u200b')
def figure(key,caption):
 p=para();p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(O/'figures'/Path(F[key]['path']).name),width=Cm(15.6))
 p=para(caption,'Caption');p.paragraph_format.keep_together=True;p.paragraph_format.space_after=Pt(8)
def math_run(text):
 r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r

def math_expression(text):
 root=OxmlElement('m:oMath');cursor=0
 pattern=r'([A-Za-z]+|Σ|Π)_(\{[^}]+\}|\([^)]*\)|[A-Za-z0-9]+)'
 for match in re.finditer(pattern,text):
  if match.start()>cursor:root.append(math_run(text[cursor:match.start()]))
  sub=OxmlElement('m:sSub');base=OxmlElement('m:e');base.append(math_run(match.group(1)));idx=OxmlElement('m:sub');val=match.group(2)
  if val.startswith(('{','(')):val=val[1:-1]
  nested=math_expression(val)
  for child in list(nested):idx.append(child)
  sub.extend([base,idx]);root.append(sub);cursor=match.end()
 if cursor<len(text):root.append(math_run(text[cursor:]))
 return root

def render(lines):
 for line in lines:
  if not line.strip():continue
  if line.startswith('#'):
   lev=len(line)-len(line.lstrip('#'));heading(line[lev:].strip(),lev)
  elif line.startswith('{{table:'):table(line[8:-2])
  elif line.startswith('{{figure:'):
   key,cap=line[9:-2].split('|',1);figure(key,cap)
  elif line=='{{references}}':
   for r in refs:
    p=para(f"[{r['number']}] {r['formatted']}");p.paragraph_format.first_line_indent=Pt(-18);p.paragraph_format.left_indent=Pt(18);p.paragraph_format.line_spacing=1.2;p.paragraph_format.space_after=Pt(6)
    for run in p.runs:font(run,10.5)
  elif line.startswith('$$'):
   p=para();p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(6);p.paragraph_format.space_after=Pt(6);p._p.append(math_expression(line[2:-2]))
  else:para(line)
front,body=source.split('# 第1章',1);front_mode=True;render(front.splitlines());front_mode=False
# 目录页码来自转换后的PDF；重复构建直到稳定。
p=para('目 录','Heading 1')
entries=[(x[2:].strip(),1) if x.startswith('# ') else (x[3:].strip(),2) for x in source.splitlines() if x.startswith('# ') or x.startswith('## ')]
for title,level in entries:
 p=para();p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.left_indent=Pt((level-1)*14);p.paragraph_format.line_spacing=1.1;p.paragraph_format.space_after=Pt(3);p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5-(level-1)*.5),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS);font(p.add_run(title+'\t'+str(pages.get(title,'—'))),10.5)
section('decimal',1);s=D.sections[-1];h=s.header.paragraphs[0];h.alignment=WD_ALIGN_PARAGRAPH.CENTER;font(h.add_run('面向智能车辆的三维语义场景图构建与导航方法研究'),9)
render(('# 第1章'+body).splitlines())
D.save(O/'硕士论文初稿.docx');(O/'qa/headings.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2));print('docx generated',len(D.paragraphs),'paragraphs',len(D.tables),'editable tables',len(D.inline_shapes),'images')
