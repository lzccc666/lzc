import runpy,sys,requests,json
from pathlib import Path
import pymupdf
from bs4 import BeautifulSoup
O=Path(__file__).resolve().parents[1];D=O/'references'
# 重用只读下载函数，不执行原脚本的列表循环。
s=(O/'scripts/collect_literature.py').read_text().split('records=[]')[0];ns={'__file__':str(O/'scripts/collect_literature.py')};exec(s,ns)
r=ns['fetch']('1711.07280');rs=json.loads((D/'metadata.json').read_text());rs=[x for x in rs if x['id'] not in ['1802.02179','1711.07280','KITTI2013','Dijkstra1959','Hart1968','Dolgov2010']];rs.append(r)
for key,doi in [('KITTI2013','10.1177/0278364913491297'),('Dijkstra1959','10.1007/BF01386390'),('Hart1968','10.1109/TSSC.1968.300136'),('Dolgov2010','10.1177/0278364909359210')]:
 p=D/key;p.mkdir(exist_ok=True)
 if (p/'crossref.json').exists():j=json.loads((p/'crossref.json').read_text())
 else:
  try:
   q=requests.get('https://api.crossref.org/works/'+doi,timeout=45);q.raise_for_status();j=q.json()['message'];(p/'crossref.json').write_text(json.dumps(j,ensure_ascii=False,indent=2))
  except Exception:
   if key!='Dolgov2010':raise
   j={'title':['Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments'],'author':[{'given':a,'family':b} for a,b in [('Dmitri','Dolgov'),('Sebastian','Thrun'),('Michael','Montemerlo'),('James','Diebel')]],'published':{'date-parts':[[2010]]},'container-title':['The International Journal of Robotics Research'],'volume':'29','issue':'5','page':'485-501'}
 rs.append(dict(id=key,doi=doi,url='https://doi.org/'+doi,title=j['title'][0],authors=[x.get('given','')+' '+x['family'] for x in j['author']],date=str(j['published']['date-parts'][0][0]),venue=j.get('container-title',[''])[0],volume=j.get('volume',''),issue=j.get('issue',''),pages=j.get('page',''),status=('出版社元数据已核验' if key=='Dolgov2010' else 'Crossref元数据已核验')))
# 作者/出版机构公开全文。
urls={'KITTI2013':'https://www.cvlibs.net/publications/Geiger2013IJRR.pdf','Dolgov2010':'https://ai.stanford.edu/~ddolgov/papers/dolgov_ijrr10.pdf','Hart1968':'https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf','2304.02643':'https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf'}
for key,url in urls.items():
 try:
  q=requests.get(url,timeout=70);q.raise_for_status();p=D/key;p.mkdir(exist_ok=True);(p/'primary.pdf').write_bytes(q.content);doc=pymupdf.open(p/'primary.pdf');t='\n'.join(x.get_text() for x in doc);(p/'primary_fulltext.txt').write_text(t)
  for r in rs:
   if r['id']==key:r.update(fulltext_url=url,fulltext_chars=len(t),status='元数据及一手全文已获取',pdf_pages=len(doc))
  doc.close()
 except Exception as e:print(key,str(e),flush=True)
(D/'metadata.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2))
print('DONE',len(rs),flush=True)
