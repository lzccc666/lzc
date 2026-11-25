"""记录真实依赖安装来源和模块入口；不装包、不加载权重、不泄露URL凭据。"""
import importlib.metadata as metadata
import importlib.util
import json,sys
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'source_bundle/dependency_origins_001'

def safe_url(value):
 parsed=urlsplit(value)
 if parsed.scheme=='file':return urlunsplit(('file','',parsed.path,'',''))
 return urlunsplit((parsed.scheme,parsed.hostname or '',parsed.path,'',''))

def main():
 O.mkdir(exist_ok=False);distributions=[]
 for dist in sorted(metadata.distributions(),key=lambda d:(d.metadata.get('Name') or '').lower()):
  row=dict(name=dist.metadata.get('Name'),version=dist.version)
  origin=dist.read_text('direct_url.json')
  if origin:
   value=json.loads(origin);row['direct_origin']=dict(url=safe_url(value.get('url','')),vcs_info=value.get('vcs_info'),editable=value.get('dir_info',{}).get('editable',False))
  distributions.append(row)
 sys.path.append('/autodl-fs/data/third_party/carla_0.9.16/python_site');modules=[]
 for name in ['conceptgraph','torch','torchvision','transformers','peft','ultralytics','mobile_sam','segment_anything','open_clip','open3d','cv2','nuscenes','carla','filterpy','numba','lap','sklearn','shapely','numpy','scipy']:
  try:spec=importlib.util.find_spec(name)
  except (ImportError,ValueError,ModuleNotFoundError):spec=None
  origin=spec.origin if spec else None;p=Path(origin) if origin and origin not in ['built-in','frozen'] else None
  modules.append(dict(module=name,found=spec is not None,origin=origin,entry_sha256=digest(p) if p is not None and p.is_file() else None,namespace_locations=list(spec.submodule_search_locations or []) if spec else []))
 dump(O/'report.json',dict(python=sys.version,executable=sys.executable,distributions=distributions,module_entries=modules,module_entry_hash_is_not_whole_package_hash=True,install_performed=False,clean_environment_install_passed=False,weights_loaded=False,source_sha256=digest(__file__),limitations=['仅记录本机真实依赖与入口，不能声称从零安装成功','direct_url去掉用户名密码、查询参数和fragment；本地路径只是复现来源线索','缺失模块未由审计安装；运行时可能通过各runner临时sys.path或固定兼容副本加载','模块入口哈希不代替整个第三方包/轮子哈希；官方CARLA包哈希另有独立记录'],goal_complete=False))
 (O/'依赖说明.md').write_text('本报告列实际Python环境的安装版本、经去凭据处理的本地/Git来源和关键模块入口。它是复现准备资料，不是pip安装成功回执。MobileSAM等本地来源需按direct_origin检查源码是否可恢复；模型权重与数据按各实验release恢复。尚未创建新Conda环境，clean_environment_install_passed=false。\n',encoding='utf-8')
 print(json.dumps(dict(distributions=len(distributions),modules=len(modules),missing=[m['module'] for m in modules if not m['found']])),flush=True)

if __name__=='__main__':main()
