from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,urllib.request
out=Path(__file__).resolve().parents[1];root=out/'source_bundle/identity_reference_001';root.mkdir(exist_ok=False)
assert shutil.disk_usage(root).free>6000000000
(root/'budget.json').write_text(json.dumps(dict(maximum_download_bytes=1048576,maximum_files=8,source='https://github.com/JonathonLuiten/TrackEval',global_environment_changed=False,goal_complete=False)))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def read(url):
 with opener.open(urllib.request.Request(url,headers={'User-Agent':'thesis-identity-audit'}),timeout=40) as r:
  data=r.read(1048577);assert len(data)<=1048576;return data
metadata=read('https://api.github.com/repos/JonathonLuiten/TrackEval/commits/master')
commit=json.loads(metadata)['sha'];(root/'commit_metadata.json').write_bytes(metadata)
files=[];total=len(metadata)
for name in ['trackeval/metrics/identity.py','trackeval/metrics/_base_metric.py','trackeval/_timing.py','trackeval/utils.py','LICENSE']:
 url=f'https://raw.githubusercontent.com/JonathonLuiten/TrackEval/{commit}/{name}'
 data=read(url);total+=len(data);assert total<=1048576
 p=root/'original'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 files.append(dict(path=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),commit=commit,files=files,total_download_bytes=total,global_environment_changed=False,goal_complete=False)
(root/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
