"""保留001清单与写回执失败，另核验全部清单哈希并一次性新建最终回执。"""
import datetime,gzip,json,shutil,time
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];O=R/'storage_audit';M=O/'artifact_manifest_001.jsonl.gz';INITIAL=O/'artifact_manifest_receipt_001.json';FINAL=O/'artifact_manifest_receipt_002.json'

def main():
 if FINAL.exists():raise FileExistsError('不覆盖最终回执')
 started=time.monotonic();initial=json.loads(INITIAL.read_text());assert digest(M)==initial['manifest_sha256']
 seen=set();total=0
 with gzip.open(M,'rt') as f:
  for line in f:
   row=json.loads(line);relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts and row['path'] not in seen
   p=R/relative;assert not p.is_symlink() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'];seen.add(row['path']);total+=row['bytes']
 assert len(seen)==initial['files_hashed'] and total==initial['source_bytes']
 excluded=initial['excluded_prefixes'];supplement=[]
 for p in sorted(R.rglob('*')):
  relative=str(p.relative_to(R))
  if not p.is_file() or p.is_symlink() or relative in seen or any(relative==x or relative.startswith(x) for x in excluded) or p==FINAL:continue
  supplement.append(dict(path=relative,bytes=p.stat().st_size,sha256=digest(p)))
 current=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(R).free;assert current+65536<10*2**30 and free>=20*2**30
 receipt=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=digest(M),initial_receipt_sha256=digest(INITIAL),manifest_entries_rehashed=len(seen),manifest_bytes_covered=total,all_included_file_sizes_and_hashes_match_now=True,closing_supplemental_files=supplement,original_command_returncode=1,original_failure='清单生成/读回成功；stage9.dump使用独占x模式，程序第二次对同名001回执dump触发FileExistsError。首回执和全部清单保留，未覆盖任何原文件。',repair='本002回执另路径一次性创建，重新逐项计算全部清单对象SHA256；未重跑实验、未修改模型或指标',initial_receipt_final_capacity_fields_incomplete=True,project_budget_bytes=10*2**30,project_bytes_before_this_receipt=current,remaining_bytes_before_this_receipt=10*2**30-current,receipt_reserved_bytes=65536,within_project_directory_budget=True,physical_free_bytes=free,large_tasks_allowed=False,large_task_reason='仅少量MiB剩余且提高预算询问待答；disk_status_006的large_project_writes_allowed只是未计具体作业预留的基本容量条件，不能授予大任务预算',shared_install_budget_scope_note=initial['external_bulk_scope'],live_mutable_exclusions=excluded,deleted_files=[],goal_complete=False,whole_project_accepted=False,verification_elapsed_s=time.monotonic()-started,finalizer_source_sha256=digest(__file__))
 dump(FINAL,receipt)
 actual=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());assert actual<10*2**30
 print(json.dumps(dict(files_rehashed=len(seen),closing_supplemental_files=len(supplement),project_bytes_after_receipt=actual,remaining_MiB=(10*2**30-actual)/2**20,goal_complete=False)),flush=True)

if __name__=='__main__':main()
