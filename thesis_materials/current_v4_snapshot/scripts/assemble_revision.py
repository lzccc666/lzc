from pathlib import Path
import re,json
O=Path(__file__).resolve().parents[1];parts=json.loads((O/'sources/原版分段.json').read_text());expand={}
for p in (O/'sources/扩写段落').glob('*.md'):
 chunks=re.split(r'^@@([\d.]+)\s*$',p.read_text(),flags=re.M)
 for i in range(1,len(chunks),2):expand[chunks[i]]=chunks[i+1].strip()
s=(O/'sources/合并正文模板.md').read_text()
def combine(m):
 old=m[2]
 for key,v in parts.items():
  if '本章小结' in key and v:old=old.replace(v,'')
 new=expand[m[4]].split('\n\n');lead=new[0];rest='\n\n'.join(new[1:])
 return m[1]+lead+'\n\n'+old.strip()+'\n\n'+rest+'\n\n'
s=re.sub(r'(## [^\n]+\n\n)(.*?)(\{\{expand:([\d.]+)\}\})',combine,s,flags=re.S)
s=s.replace('算法4-1的顺序为：','对象图构建算法的执行顺序为：')
s=s.replace('本文未证明可交互驾驶、完整实时控制闭环或车辆实际到达，也不宣称整个科研课题已经验收完成。','本文结论限于公开录制数据的建图、目标选择与离线几何评价。')
s=s.replace('公开数据版 · 完整初稿 v1','公开数据版 · 扩充修订稿 v2')
(O/'sources/论文正文.md').write_text(s)
body=s[s.index('# 第1章'):s.index('# 参考文献')];plain=re.sub(r'\{\{.*?\}\}|^#+ .*$', '',body,flags=re.M);count=lambda t:len(re.findall('[\u4e00-\u9fff]',t))
report={'全文源汉字数':count(s),'七章正文汉字数_含标题图注':count(body),'七章正文纯段落汉字数_不含标题图注表格':count(plain),'主文二级小节数':len(re.findall('^## ',body,re.M)),'三级标题数':len(re.findall('^### ',s,re.M)),'源字符数':len(s)}
report['达到五万汉字']=report['七章正文纯段落汉字数_不含标题图注表格']>=50000
(O/'qa/字数与结构检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
assert report['达到五万汉字']
