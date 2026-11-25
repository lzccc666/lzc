"""核查已关闭结果的JSON/JSONL/CSV与新源码语法；主动排除在写模型输出和故意损坏测试夹具。"""
import ast,csv,gzip,json,math,time
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'json/artifact_validation_002'
EXCLUDE=['json/software_validation_001/tmp/','source_bundle/reproduction_smoke_001/checkout/','tables/completed_office_001/runtime/']
def finite(v):
 if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite JSON number')
 if isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def main():
 O.mkdir(exist_ok=False);counts=dict(json=0,jsonl=0,jsonl_gz=0,csv=0,python=0,records=0);errors=[];skipped=0
 for p in sorted(R.rglob('*')):
  if p.is_symlink() or not p.is_file():continue
  relative=str(p.relative_to(R))
  if any(relative.startswith(x) for x in EXCLUDE) or p.is_relative_to(O):skipped+=1;continue
  suffix='jsonl_gz' if p.name.endswith('.jsonl.gz') else p.suffix.lstrip('.')
  if suffix not in ['json','jsonl','jsonl_gz','csv'] and not (suffix=='py' and relative.startswith('source/')):continue
  before=(p.stat().st_size,p.stat().st_mtime_ns)
  try:
   if suffix=='json':finite(json.loads(p.read_text()));counts['json']+=1
   elif suffix in ['jsonl','jsonl_gz']:
    handle=gzip.open(p,'rt') if suffix=='jsonl_gz' else p.open()
    with handle:
     for line in handle:
      if not line.endswith('\n'):raise ValueError('unfinished JSONL line')
      finite(json.loads(line));counts['records']+=1
    counts[suffix]+=1
   elif suffix=='csv':
    with p.open(newline='') as handle:
     rows=csv.reader(handle);header=next(rows)
     if len(header)!=len(set(header)):raise ValueError('duplicate CSV columns')
     for row in rows:
      if len(row)!=len(header):raise ValueError('CSV column count differs')
    counts['csv']+=1
   else:ast.parse(p.read_text());counts['python']+=1
   if before!=(p.stat().st_size,p.stat().st_mtime_ns):raise ValueError('file changed during read, not a closed artifact')
  except Exception as exc:errors.append(dict(path=str(p),error=repr(exc)))
 dump(O/'report.json',dict(counts=counts,errors=errors,excluded_prefixes=EXCLUDE,skipped_files=skipped,source_sha256=digest(__file__),scope='已关闭文件结构校验，不代替语义正确性、人工核验或新GPU推理',all_checked_passed=not errors,completed_real_graph_eval_validated=True,goal_complete=False));print(json.dumps(dict(counts=counts,errors=errors),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
