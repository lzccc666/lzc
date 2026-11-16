"""只读校验本轮关闭产物，压缩逐项SHA256；活动日志/运行缓存明确排除。"""
import datetime,gzip,hashlib,json,os,shutil,time
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];O=R/'storage_audit'
MANIFEST=O/'artifact_manifest_001.jsonl.gz';RECEIPT=O/'artifact_manifest_receipt_001.json'
EXCLUDE=['annotation/server_001.log','tables/completed_office_001/runtime/','storage_audit/artifact_manifest_001.jsonl.gz','storage_audit/artifact_manifest_receipt_001.json']
def all_bytes():return sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
def main():
 started=time.monotonic();before_total=all_bytes()
 if before_total+3*2**20>10*2**30:raise OSError('逐项清单预留3MiB不足；不突破10GiB')
 assert shutil.disk_usage(R).free>=20*2**30
 files=[];links=[];excluded=[]
 for root,dirs,names in os.walk(R,followlinks=False):
  for name in dirs+names:
   p=Path(root)/name;relative=str(p.relative_to(R))
   if p.is_symlink():links.append(dict(path=relative,target=os.readlink(p)))
  for name in names:
   p=Path(root)/name;relative=str(p.relative_to(R))
   if p.is_symlink():continue
   if any(relative==x or relative.startswith(x) for x in EXCLUDE):excluded.append(relative)
   else:files.append(p)
 source_bytes=0
 with MANIFEST.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as out:
  for i,p in enumerate(sorted(files)):
   first=p.stat();h=digest(p);last=p.stat()
   assert (first.st_size,first.st_mtime_ns)==(last.st_size,last.st_mtime_ns),'活动文件未关闭 '+str(p)
   out.write((json.dumps(dict(path=str(p.relative_to(R)),bytes=first.st_size,sha256=h),ensure_ascii=False,separators=(',',':'))+'\n').encode());source_bytes+=first.st_size
   if i%500==0:
    out.flush()
    if before_total+raw.tell()+65536>10*2**30:raise OSError('清单输出接近预算；保留前缀，不标验证成功')
 verified=0
 with gzip.open(MANIFEST,'rt') as f:
  for line in f:
   row=json.loads(line);assert (R/row['path']).stat().st_size==row['bytes'];verified+=1
 assert verified==len(files)
 receipt=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_hashed=len(files),source_bytes=source_bytes,manifest_bytes=MANIFEST.stat().st_size,manifest_sha256=digest(MANIFEST),manifest_records_reopened=verified,symlinks=links,excluded_files=excluded,excluded_prefixes=EXCLUDE,exclusion_reason='人工服务仍运行，其访问日志可继续增长；LibreOffice运行配置/锁/缓存可变。清单及自身回执不能自包含，清单哈希在本回执。',external_bulk_scope='符号链接仅记录入口，不追入原始数据或共享CARLA安装；外部对象按各release/安装哈希/源清单追溯，不能把此清单说成共享卷全部物理文件均已哈希',all_included_files_stable_during_hash=True,elapsed_s=time.monotonic()-started,project_budget_bytes=10*2**30,project_bytes_before_manifest=before_total,project_bytes_after_receipt=None,physical_free_bytes=shutil.disk_usage(R).free,deleted_files=[],source_sha256=digest(__file__),goal_complete=False)
 dump(RECEIPT,receipt);receipt['project_bytes_after_receipt']=all_bytes();receipt['within_project_directory_budget']=receipt['project_bytes_after_receipt']<10*2**30;dump(RECEIPT,receipt)
 assert all_bytes()<10*2**30
 print(json.dumps({k:receipt[k] for k in ['files_hashed','manifest_bytes','project_bytes_after_receipt','within_project_directory_budget','elapsed_s']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
