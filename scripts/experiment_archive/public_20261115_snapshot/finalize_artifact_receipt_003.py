"""复用原清单，重新逐项校验旧条目及收尾文件；另写003回执，不覆写清单。"""
import datetime,gzip,json,shutil,time
from pathlib import Path
from conceptgraph.completion.stage9 import digest
R=Path(__file__).resolve().parents[1];S=R/'storage_audit';M=S/'artifact_manifest_001.jsonl.gz';P=S/'artifact_manifest_receipt_002.json';FINAL=S/'artifact_manifest_receipt_003.json'

def safe_path(relative):
 p=Path(relative)
 if p.is_absolute() or '..' in p.parts:raise ValueError('清单不是安全相对路径')
 return R/p

def checked(relative,size,sha):
 p=safe_path(relative);assert not p.is_symlink() and p.is_file();before=p.stat();assert before.st_size==size and digest(p)==sha;after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)

def main():
 if FINAL.exists():raise FileExistsError('003回执已存在，不覆写')
 for pid,script in [(56357,'nuscenes_full_semantics_001.py'),(57128,'finish_full_semantics_pipeline_001.py'),(57772,'run_official_actuation_pilot_001.py')]:
  cmd=Path(f'/proc/{pid}/cmdline')
  if cmd.exists() and script.encode() in cmd.read_bytes():raise RuntimeError(f'自己的作业{pid}仍运行，不能收尾冻结')
 for cmd in Path('/proc').glob('[0-9]*/cmdline'):
  try:raw=cmd.read_bytes()
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  if any(s in raw for s in [b'run_official_actuation_pilot_002.py',b'run_official_actuation_pilot_003.py',b'carla_acceleration_feedback_pilot_003.py']):raise RuntimeError('新的官方标定仍运行，不能收尾冻结')
 assert (R/'json/full_semantics_followup_001/report.json').exists(),'全量语义后续未完成，不能写完成后回执'
 pilot=R/'json/official_actuation_pilot_execution_003';assert any((pilot/name).exists() for name in ['report.json','failure.json','blocked_storage.json']),'标定尚无终态'
 if (pilot/'server_started.json').exists():assert (pilot/'server_shutdown.json').exists(),'自己的官方服务尚未关闭'
 started=time.monotonic();previous=json.loads(P.read_text());assert digest(M)==previous['manifest_sha256'];seen=set();total=0
 with gzip.open(M,'rt') as source:
  for line in source:
   row=json.loads(line);assert row['path'] not in seen;checked(row['path'],row['bytes'],row['sha256']);seen.add(row['path']);total+=row['bytes']
 assert len(seen)==previous['manifest_entries_rehashed'] and total==previous['manifest_bytes_covered']
 for row in previous['closing_supplemental_files']:
  assert row['path'] not in seen;checked(row['path'],row['bytes'],row['sha256']);seen.add(row['path'])
 anchor_hash=digest(P);seen.add(str(P.relative_to(R)));excluded=previous['live_mutable_exclusions'];extra=[]
 for p in sorted(R.rglob('*')):
  relative=str(p.relative_to(R))
  if not p.is_file() or p.is_symlink() or p==FINAL or relative in seen or any(relative==x or relative.startswith(x) for x in excluded):continue
  before=p.stat();sha=digest(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'新文件仍在变化：'+relative;extra.append(dict(path=relative,bytes=before.st_size,sha256=sha))
 assert digest(P)==anchor_hash
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(R).free;assert free>=20*2**30
 receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_finalizer_sha256=digest(__file__),previous_receipt_sha256=anchor_hash,manifest_sha256=digest(M),previous_manifest_entries_rehashed=previous['manifest_entries_rehashed'],previous_manifest_bytes=total,previous_closing_supplemental_entries_rehashed=len(previous['closing_supplemental_files']),previous_all_checked_now=True,new_closing_supplemental_files=extra,mutable_exclusions_unchanged=excluded,receipt_self_excluded=True,shared_input_scope='原共享KITTI/CARLA安装/模型仍按其已有release哈希；本清单不冒称对共享卷全量重新校验。新官方运行日志按server_shutdown.json逐项哈希。',project_budget_bytes=10*2**30,project_bytes_before_receipt=size,physical_free_bytes_before_receipt=free,large_tasks_allowed=False,whole_project_accepted=False,goal_complete=False,completion_limitations=['真实人工审核、新封存测试、完整复杂语义官方CARLA与干净环境复现仍需另行验收；本回执只证明文件完整性','旧历史/失败/负结果/原错误回执均保留；不因哈希通过宣称实验或整课题完成','人审服务保持可用，其活动日志排除；新提交超出当前时点快照，之后需单独验收'],new_sources='source_bundle/closing_supplement_003（如已生成，见新条目）',verification_wall_s=time.monotonic()-started)
 text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n';length=len(text.encode());assert size+length+4096<10*2**30,'不足以安全写回执，未写文件'
 with FINAL.open('x') as f:f.write(text)
 print(json.dumps(dict(old_entries_verified=len(seen)-1,new_entries=len(extra),receipt_bytes=length,remaining_KiB=(10*2**30-size-length)/1024,goal_complete=False),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
