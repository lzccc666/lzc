from pathlib import Path
import concurrent.futures as cf,requests,re,json,time,xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pymupdf
O=Path(__file__).resolve().parents[1]; D=O/'references';D.mkdir(exist_ok=True)
ids=['1609.05130','1804.09194','1808.08378','1910.02490','2002.06289','2004.03967','2309.16650','2211.15654','2303.09553','2210.05714','2207.04429','2103.00020','2304.02643','2409.12191','2106.09685','2305.14314','1903.11027','1909.10838','1604.07446','1609.01775','2009.07736','2401.17270','1711.07280']
def fetch(i):
 p=D/i;p.mkdir(exist_ok=True);r={'id':i,'url':'https://arxiv.org/abs/'+i,'access_date':'2026-11-15'}
 try:
  q=requests.get(r['url'],timeout=50) if not (p/'landing.html').exists() else None
  if q is not None:q.raise_for_status();(p/'landing.html').write_text(q.text)
  s=BeautifulSoup((p/'landing.html').read_text(),'html.parser');meta=lambda n:[x.get('content') for x in s.select(f'meta[name="{n}"]')]
  for k,n in [('title','citation_title'),('authors','citation_author'),('date','citation_date'),('doi','citation_doi')]:
   a=meta(n);r[k]=a if k=='authors' else (a[0] if a else '')
  a=s.select_one('blockquote.abstract');r['abstract']=a.get_text(' ',strip=True) if a else ''
  if not (p/'paper.pdf').exists():
   q=requests.get('https://arxiv.org/pdf/'+i,timeout=100);q.raise_for_status();(p/'paper.pdf').write_bytes(q.content)
  doc=pymupdf.open(p/'paper.pdf');t='\n'.join(x.get_text() for x in doc);(p/'fulltext.txt').write_text(t);r['pages']=len(doc);r['fulltext_chars']=len(t);r['status']='原始作者页面及全文已获取';doc.close()
 except Exception as e:r['status']=str(e)
 return r
records=[]
for i in ids:
 r=fetch(i);records.append(r);(D/'metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(i,r.get('title'),r['status'],flush=True)
