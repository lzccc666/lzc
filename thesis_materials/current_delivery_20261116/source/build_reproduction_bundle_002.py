"""只读打包当前实际源码与依赖版本，不复制权重/原始数据、不修改Git或历史交付。"""
import datetime,hashlib,json,os,platform,subprocess,tarfile
from pathlib import Path
from conceptgraph.completion.stage9 import PROJECT,dump,digest
R=Path(__file__).resolve().parents[1];O=R/'source_bundle/reproduction_002'
def main():
 O.mkdir(parents=True,exist_ok=False);files=[]
 for directory in [PROJECT/'conceptgraph',PROJECT/'scripts',R/'source']:
  for p in directory.rglob('*'):
   if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in {'.py','.yaml','.yml','.txt','.md','.json'}:files.append(p)
 for n in ['setup.py','environment.yml','README.md','README_nuscenes_minimal.md','README_streamlined_detection.md']:
  if (PROJECT/n).exists():files.append(PROJECT/n)
 # AB3DMOT许可证没有后缀，也必须随官方源代码及兼容副本保留。
 files.extend((R/'source/third_party').rglob('LICENSE'));files=sorted(set(files));manifest=[dict(path=str(p),archive_path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
 before={str(p):digest(p) for p in files}
 with tarfile.open(O/'actual_source_snapshot.tar.gz','x:gz') as archive:
  for p in files:archive.add(p,arcname=str(p.relative_to(PROJECT)),recursive=False)
 assert all(digest(p)==before[str(p)] for p in files),'source changed while bundling'
 dump(O/'source_manifest.json',manifest)
 githead=subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT,capture_output=True,text=True).stdout.strip();dirty=bool(subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=PROJECT,capture_output=True,text=True).stdout.strip())
 import importlib.metadata as metadata
 packages={}
 for name in ['torch','torchvision','transformers','peft','accelerate','ultralytics','open-clip-torch','open3d','opencv-python','numpy','scipy','numba','filterpy','lap','nuscenes-devkit','omegaconf','shapely','scikit-learn','supervision','Pillow']:
  try:packages[name]=metadata.version(name)
  except metadata.PackageNotFoundError:packages[name]=None
 record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),platform=platform.platform(),actual_packages=packages,git_head=githead,tracked_working_tree_dirty=dirty,git_commit_performed=False,data_or_weights_copied=False,new_source_count=len(files),archive_bytes=(O/'actual_source_snapshot.tar.gz').stat().st_size,archive_sha256=digest(O/'actual_source_snapshot.tar.gz'),source_manifest_sha256=digest(O/'source_manifest.json'),limitations=['当前实际源码快照而非干净Git commit；不能只用Git HEAD代替源码hash','源数据与基础模型通过各release定位；原始数据/权重未包含在代码压缩包','尚未在全新系统完整安装并重跑，不能授予clean_environment_passed','官方CARLA0.9.16、Python API及图形动态库在共享卷单独登记','原environment.yml与实际包版本不完全一致，actual_packages和runtime_pip_packages_001是本轮实际版本'],clean_environment_passed=False,goal_complete=False);dump(O/'receipt.json',record)
 (O/'复现说明.md').write_text('''# 当前软件复现快照

压缩包按项目根目录展开，包含当前实际`conceptgraph/`、相关`scripts/`、本批`source/`和配置/许可证。所有文件可按source_manifest.json逐一校验。没有包含原始数据、基础模型、历史checkpoint或历史实验结果；这些须按各release记录恢复到同样路径。本包不修改Git状态。

当前实际包版本见receipt.json以及本批json/runtime_pip_packages_001.json。旧environment.yml仅供历史参考，不能声称按它安装就复现了当前新实验。推荐在新Conda环境中按实际Python/PyTorch/CUDA与关键包版本安装，再将本源码放入项目根。尚未完整验证从全新环境安装，因此clean_environment_passed=false。

为避免过多BLAS线程以及向旧目录写Python缓存，先设置：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/autodl-tmp/lizichong
R=/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115
PY=/root/miniconda3/envs/conceptgraph/bin/python
```

CPU建图、统计与审计使用CUDA_VISIBLE_DEVICES=''；实际前端和Qwen推理使用0。Qwen训练与CARLA/前端不得同时占用GPU，已有真实OOM记录保留。Qwen推理runner已在import torch之前显式加载当前Conda libstdc++，这是本机ABI兼容措施，不是修改系统依赖。

KITTI输入选择清单及release在本批kitti_tracking_expanded/与mapping_evaluation/；nuScenes原始帧来自历史mapping_evaluation/causal_mapping_full_001/，新六组结果分别在nuscenes_ablation/strict_*。重跑数据迭代器不应访问测试标签作推理输入。所有脚本拒绝覆盖旧run目录；在干净输出目录或修改为新的run-id进行重跑，不能删除旧结果后占用原ID。

可先执行各runner的--help查看参数：kitti_tracking_gpu_001.py、nuscenes_strict_ablation_002.py、ab3dmot_public_frontend_002.py、train_official_graph_adapter_002.py、train_real_graph_adapter_004.py。新真实图训练的处理文件通过processor_reference.json引用固定基础模型，Adapter、optimizer/scheduler/RNG全量保留。AB3DMOT需NUMBA_DISABLE_JIT=1，固定官方核心及单行NumPy兼容补丁已经随包保留。不同数据集的时间频率及max_age=2帧语义见其release。

CARLA API路径为/autodl-fs/data/third_party/carla_0.9.16/python_site；官方服务器必须离屏渲染并以非root用户运行。启动命令及物理配置见本批carla_official/各release和共享运行日志。闭环运行期间不能让另一个客户端推进世界时钟。新批次按ablation_registration_003.json核对源码，执行run_carla_ablation_batch_003.py；它不能覆盖已经存在的日志/episode目录。本批14回合完整、末回合因10GiB守卫停止并保留前缀；必须用新run-id做补充，不覆盖该负记录。完整复杂Qwen语义闭环仍未验收。

若空间不足20GiB、项目新增产物接近10GiB或GPU OOM，停止大任务并保留原始前缀、failure.json与恢复记录。缓存只列候选，无授权自动删除正式结果。所有新交付均未因此标记整个Goal完成。
''',encoding='utf-8')
 print(json.dumps(dict(files=len(files),bytes=record['archive_bytes'],clean_environment_passed=False,goal_complete=False)))
if __name__=='__main__':main()
