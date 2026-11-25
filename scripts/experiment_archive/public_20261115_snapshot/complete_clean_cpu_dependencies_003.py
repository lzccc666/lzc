"""监测实际安装进程结束后，从冻结纯源码构建本地依赖并核验新site-packages。"""
import argparse,datetime,json,os,subprocess,time,traceback
from pathlib import Path
import install_clean_cpu_environment_001 as base
R=base.R;O=base.O

def tick(pid):
 try:
  x=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if x[0]=='Z' else x[19]
 except FileNotFoundError:return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args();initial=tick(a.wait_pid);frozen={str(p):base.sha(p) for p in [Path(__file__),Path(base.__file__),O/'vendor_source_manifest_001.json']}
 for p in (O/'vendor_sources').rglob('*'):
  if p.is_file():frozen[str(p)]=base.sha(p)
 base.save('local_dependencies_registration_003.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wait_PID=a.wait_pid,wait_start_tick=initial,source_sha256=frozen,source_rebuild_instead_of_binary_site_packages_copy=True,old_dependencies_source_reused=True,goal_complete=False))
 begun=time.monotonic()
 while initial is not None and tick(a.wait_pid)==initial:
  if time.monotonic()-begun>3600:raise TimeoutError('等待核心安装1小时，未结束他进程')
  time.sleep(15)
 assert json.loads((O/'installation_report_002.json').read_text())['clean_venv_core_installation_passed'];assert all(base.sha(p)==h for p,h in frozen.items())
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CONFIG_FILE='/dev/null',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(O/'tmp'),XDG_CACHE_HOME=str(O/'cache'),YOLO_CONFIG_DIR=str(O/'yolo_config'));py=str(O/'venv/bin/python');receipts=[]
 receipts.append(base.step('build_tools_003',[py,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','--report',str(O/'build_tools_pip_report_003.json'),'wheel==0.45.1','setuptools==75.8.0','h5py==3.14.0','tyro==0.9.35','PyMuPDF==1.26.6'],env))
 wheel=Path('/autodl-fs/data/third_party/carla_0.9.16/install/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl')
 receipts.append(base.step('local_modules_003',[py,'-m','pip','install','--no-cache-dir','--no-build-isolation','--index-url','https://pypi.org/simple','--report',str(O/'local_pip_report_003.json'),'-c',str(O/'constraints.txt'),*[str(O/'vendor_sources'/x) for x in ['gradslam','clip','segment_anything']],str(wheel)],env))
 receipts.append(base.step('pip_check_003',[py,'-m','pip','check'],env))
 code="""import importlib,json,sys
from pathlib import Path
prefix=Path(sys.prefix).resolve();rows=[]
assert sys.prefix!=sys.base_prefix and not any('/envs/conceptgraph/lib/python3.10/site-packages' in p for p in sys.path)
for name in ['numpy','scipy','torch','torchvision','open3d','cv2','hydra','omegaconf','ultralytics','supervision','open_clip','transformers','peft','gradslam','clip','segment_anything','carla','fitz']:
 module=importlib.import_module(name);p=Path(module.__file__).resolve();assert p.is_relative_to(prefix),(name,str(p),str(prefix));rows.append(dict(module=name,path=str(p),version=str(getattr(module,'__version__','unknown'))))
import torch
assert torch.__version__=='2.0.1+cpu' and not torch.cuda.is_available()
print(json.dumps(dict(prefix=str(prefix),sys_path=sys.path,modules=rows,CPU_only=True)))
"""
 receipts.append(base.step('import_isolation_003',[py,'-c',code],env))
 base.save('dependencies_report_003.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=receipts,clean_venv_dependency_install_and_imports_passed=True,whole_chain_runtime_passed=False,GPU_reproduction_passed=False,source_reused_for_local_modules=True,old_binary_site_packages_reused=False,official_CARLA_API_wheel_sha256=base.sha(wheel),source_sha256=frozen,resources=base.guard(),goal_complete=False));print('fresh CPU dependencies and imports passed; actual chain still required',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:base.save('failure_003.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
