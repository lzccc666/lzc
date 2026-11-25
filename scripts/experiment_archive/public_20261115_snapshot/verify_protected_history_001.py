"""只读核验原Goal受保护清单；不删除、不修复、不改历史文件。"""
import datetime,json,time
from pathlib import Path
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,dump,digest
R=Path(__file__).resolve().parents[1];O=R/'json/protected_history_audit_001'
def main():
 O.mkdir(exist_ok=False);manifest=OLD/'source_bundle/protected_manifest.jsonl';baseline=OLD/'baseline_before.json';b=json.loads(baseline.read_text());assert digest(manifest)==b['protection_manifest_sha256'];checked=0;total=0;changed=[];begun=time.monotonic()
 with (O/'mismatches.jsonl').open('x') as out:
  for row in map(json.loads,manifest.open()):
   p=PROJECT/row['path'];reason=None
   if not p.is_file():reason='missing_or_not_file'
   elif p.stat().st_size!=row['size']:reason='size_changed'
   elif digest(p)!=row['sha256']:reason='sha256_changed'
   if reason:
    diff=dict(path=str(p),reason=reason,expected_size=row['size'],expected_sha256=row['sha256'],current_size=p.stat().st_size if p.exists() else None);changed.append(diff);out.write(json.dumps(diff,ensure_ascii=False)+'\n');out.flush()
   checked+=1;total+=row['size']
   if checked%25000==0:print(json.dumps(dict(files=checked,bytes=total,mismatches=len(changed),elapsed_s=time.monotonic()-begun)),flush=True)
 assert checked==b['protected_files'];dump(O/'report.json',dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_baseline_created_utc=b['created_utc'],protected_files_checked=checked,protected_bytes=total,mismatches=changed,all_protected_paths_match_baseline=not changed,manifest_sha256=digest(manifest),baseline_sha256=digest(baseline),source_sha256=digest(__file__),elapsed_s=time.monotonic()-begun,writes_to_protected_files=False,repairs_or_deletions=False,concurrent_GPU_evaluation=True,performance_scope_note='GPU仍为单模型进程，主机后台存在CPU/I/O只读审计；不宣称独占整机延迟',goal_complete=False));print(json.dumps(dict(checked=checked,mismatches=len(changed)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
