"""以SSH传输当前论文目录中的差异文件，不保存传输压缩包。"""
from pathlib import Path
import subprocess,hashlib,json,tarfile,shlex,os,sys
O=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/lizichong/paper/当前版_v4'
HOST='seeta-nmb1';SSH=['ssh','-T','-o','ConnectTimeout=20',HOST]
command=shlex.quote(REMOTE+'/.venv/bin/python')+' '+shlex.quote(REMOTE+'/scripts/remote_sync_manifest.py')
q=subprocess.run(SSH+[command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
manifest=json.loads(q.stdout.decode('utf-8'));changed=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in manifest['files']:
 p=O/row['path']
 if not p.exists() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:changed.append(row['path'])
print('差异文件',len(changed),'字节',sum(x['bytes'] for x in manifest['files'] if x['path'] in changed),flush=True)
if changed:
 command='tar -C '+shlex.quote(REMOTE)+' -czf - -T -'
 proc=subprocess.Popen(SSH+[command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 proc.stdin.write(('\n'.join(changed)+'\n').encode('utf-8'));proc.stdin.close()
 received=0
 with tarfile.open(fileobj=proc.stdout,mode='r|gz') as stream:
  for member in stream:
   if not member.isfile():raise ValueError('non-regular tar member '+member.name)
   target=(O/member.name).resolve()
   if not target.is_relative_to(O.resolve()):raise ValueError('outside output folder')
   target.parent.mkdir(parents=True,exist_ok=True)
   data=stream.extractfile(member).read();part=target.with_name(target.name+'.part');part.write_bytes(data);os.replace(part,target)
   received+=1
   if received%40==0:print('已同步',received,flush=True)
 stderr=proc.stderr.read().decode('utf-8',errors='replace');result=proc.wait()
 if result:raise RuntimeError(stderr)
 if received!=len(changed):raise ValueError((received,len(changed)))
errors=[]
for row in manifest['files']:
 p=O/row['path']
 if not p.exists() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(row['path'])
manifest.update({'local':str(O.resolve()),'verified_files':len(manifest['files'])-len(errors),'errors':errors,'transport':'SSH差异文件流；未创建ZIP文件'})
(O/'本机同步校验清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('逐文件同步核验',manifest['verified_files'],'/',len(manifest['files']),flush=True)
if errors:raise SystemExit(str(errors))
