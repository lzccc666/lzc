from pathlib import Path
import concurrent.futures as cf, subprocess,sys,json
from pypdf import PdfReader
O=Path(__file__).resolve().parents[1];D=O/'references'
ids=['2201.13360','2402.12259','2403.17846','2307.06135','2310.03923','2306.13631','2205.13542','2006.11275','1602.00763','2110.06864','1105.1186','1404.2334']
def fetch(key):
 p=D/key;pdf=p/'paper.pdf'
 try:
  if not pdf.exists():subprocess.run(['curl.exe' if sys.platform=='win32' else 'curl','-L','--fail','--retry','1','--max-time','70','-sS','https://arxiv.org/pdf/'+key,'-o',str(pdf)],check=True)
  doc=PdfReader(pdf);text='\n'.join(x.extract_text() for x in doc.pages);(p/'fulltext.txt').write_text(text,encoding='utf-8');return {'id':key,'pages':len(doc.pages),'chars':len(text),'status':'全文已获取；阅读范围见文献调研矩阵'}
 except Exception as e:return {'id':key,'status':str(e)}
with cf.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,ids))
(D/'全文获取记录.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:print(r,flush=True)
