from pathlib import Path
import requests,re,json,csv,difflib,time
O=Path(__file__).resolve().parents[1];P=O.parents[4]/'论文1-3章初稿.md'
text=P.read_text();lines=[s.strip() for s in text.splitlines() if re.match(r'^\[\d+\]',s)]
rows=[]
for line in lines:
 m=re.search(r'\]\s+.*?\.\s+(.+?)\[(?:J|C|EB|R)',line);title=m.group(1) if m else ''
 row={'原条目':line,'提取题名':title,'状态':'未纳入本稿；仅原稿检查','匹配题名':'','DOI':'','相似度':''}
 try:
  q=requests.get('https://api.crossref.org/works',params={'query.title':title,'rows':1},timeout=15);q.raise_for_status();j=q.json()['message']['items'][0];got=j['title'][0];norm=lambda x:re.sub('[^a-z0-9]','',x.lower());ratio=difflib.SequenceMatcher(None,norm(title),norm(got)).ratio();row.update({'匹配题名':got,'DOI':j.get('DOI',''),'相似度':round(ratio,3),'状态':'题名元数据匹配；未据此继承原文支撑' if ratio>.85 else '匹配不足，未采用此原条目'})
 except Exception as e:row['状态']='外部查询未完成/限流；未据此采用：'+str(e)[:90]
 rows.append(row)
 # 保存逐条进度，限流时不反复请求。
 with (O/'references/原稿参考文献检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
 if '429' in row['状态']:
  for rest in lines[len(rows):]:rows.append({'原条目':rest,'提取题名':'','状态':'Crossref已限流；未自动验证，未直接沿用','匹配题名':'','DOI':'','相似度':''})
  break
 time.sleep(.4)
with (O/'references/原稿参考文献检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('原稿条目',len(rows))
