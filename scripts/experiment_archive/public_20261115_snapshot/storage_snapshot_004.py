import csv,datetime,json,os,shutil
from collections import defaultdict
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'storage_audit';sizes=defaultdict(int);links=[];caches=[];inodes=set();physical=0;logical=0
for root,dirs,files in os.walk(R,followlinks=False):
 for name in dirs+files:
  p=Path(root)/name
  if p.is_symlink():links.append(dict(path=str(p.relative_to(R)),target=os.readlink(p)))
 for name in files:
  p=Path(root)/name
  if p.is_symlink():continue
  st=p.stat();relative=p.relative_to(R);sizes[relative.parts[0]]+=st.st_size;logical+=st.st_size
  inode=(st.st_dev,st.st_ino)
  if inode not in inodes:physical+=st.st_blocks*512;inodes.add(inode)
  if p.suffix=='.pyc' and '__pycache__' in relative.parts:caches.append(dict(path=str(p),bytes=st.st_size,sha256=digest(p),type='本批代码可再生Python字节码',deleted=False,reason='仅候选，不自动删除；节省量很小'))
with (O/'candidate_cache_files_004.csv').open('x',newline='') as f:
 fields=['path','bytes','sha256','type','deleted','reason'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(caches)
shared=Path('/autodl-fs/data/third_party/carla_0.9.16');shared_bytes=0
for root,dirs,files in os.walk(shared,followlinks=False):
 for name in files:
  p=Path(root)/name
  if not p.is_symlink():shared_bytes+=p.stat().st_size
value=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),project_free_bytes=shutil.disk_usage(R).free,shared_free_bytes=shutil.disk_usage(shared).free,project_outputs_logical_bytes=logical,project_outputs_allocated_bytes_unique_inode=physical,project_output_budget_bytes=10*2**30,within_project_outputs_budget=logical<10*2**30,minimum_free_bytes=20*2**30,large_project_writes_allowed=logical<10*2**30 and shutil.disk_usage(R).free>=20*2**30,top_level_logical_bytes=dict(sorted(sizes.items(),key=lambda x:-x[1])),symlinks=links,regenerable_cache_candidates=len(caches),candidate_bytes=sum(r['bytes'] for r in caches),deleted_files=[],official_CARLA_shared_install_archive_api_logical_bytes=shared_bytes,budget_scope_note='10GiB按指定项目目录新增实验产物管理；CARLA官方安装/压缩包按已授权共享卷部署单列，单这一项超过10GiB，不能声称全部新增物理存储总和也小于10GiB。若10GiB包含安装，则当前不满足该总量解释；不删除复现包或失败结果掩盖占用。',goal_complete=False);dump(O/'disk_status_004.json',value);print(json.dumps({k:value[k] for k in ['project_free_bytes','shared_free_bytes','project_outputs_logical_bytes','official_CARLA_shared_install_archive_api_logical_bytes','candidate_bytes','large_project_writes_allowed']},ensure_ascii=False))
