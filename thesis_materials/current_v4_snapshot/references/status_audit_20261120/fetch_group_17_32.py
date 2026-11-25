from pathlib import Path
import json,re,concurrent.futures,urllib.request,html

O=Path(__file__).resolve().parent
refs=json.loads((O.parent/'verified_references.json').read_text(encoding='utf-8'))
folder=O/'group_17_32_cache';folder.mkdir(exist_ok=True)
def fetch(r):
    try:
        with urllib.request.urlopen(r['url'],timeout=35) as q:
            text=q.read().decode('utf-8');status=q.status
        (folder/(r['id']+'.html')).write_text(text,encoding='utf-8')
        tags=re.findall(r'<meta\s+([^>]+)>',text)
        data={}
        for tag in tags:
            fields=dict(re.findall(r'(\w+)="([^"]*)"',tag))
            if fields.get('name','').startswith('citation_'):
                data.setdefault(fields['name'],[]).append(html.unescape(fields.get('content','')))
        history=re.search(r'<div class="submission-history"[^>]*>(.*?)</div>',text,re.S)
        history=html.unescape(re.sub('<[^>]+>',' ',history[1])) if history else None
        return {'number':r['number'],'id':r['id'],'title':data.get('citation_title',[None])[0],'dates':data.get('citation_date',[]),'history':re.sub(r'\s+',' ',history).strip() if history else None,'http_status':status}
    except Exception as e:
        return {'number':r['number'],'id':r['id'],'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    out=list(ex.map(fetch,[r for r in refs if 17<=r['number']<=32]))
(O/'group_17_32_live_arxiv.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
