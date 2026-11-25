"""保留首次pip失败并在同一新venv补装，升级pip修复名称规范化和补全官方索引。"""
import datetime,json,os,traceback
from pathlib import Path
import install_clean_cpu_environment_001 as base
R=base.R;O=base.O

def main():
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CONFIG_FILE='/dev/null',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(O/'tmp'),XDG_CACHE_HOME=str(O/'cache'),YOLO_CONFIG_DIR=str(O/'yolo_config'))
 frozen={str(p):base.sha(p) for p in [Path(__file__),Path(base.__file__),O/'failure_001.json',O/'requirements.txt',O/'constraints.txt']};base.save('resume_registration_002.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_sha256=frozen,reason='pip23名称规范化问题及仅PyTorch索引缺少flit_core；升级pip25.3，PyPI主索引+PyTorch额外索引',same_venv=True,failed_logs_preserved=True,existing_environment_dependencies_reused=False,goal_complete=False));py=str(O/'venv/bin/python');receipts=[]
 receipts.append(base.step('bootstrap_pip_002',[py,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','--report',str(O/'pip_upgrade_report_002.json'),'pip==25.3'],env))
 receipts.append(base.step('install_cpu_torch_002',[py,'-m','pip','install','--no-cache-dir','--timeout','30','--retries','2','--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cpu','--report',str(O/'torch_pip_report_002.json'),'-c',str(O/'constraints.txt'),'torch==2.0.1+cpu','torchvision==0.15.2+cpu'],env))
 receipts.append(base.step('install_project_packages_002',[py,'-m','pip','install','--no-cache-dir','--timeout','30','--retries','2','--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cpu','--prefer-binary','--report',str(O/'project_pip_report_002.json'),'-c',str(O/'constraints.txt'),'-r',str(O/'requirements.txt')],env))
 base.save('installation_report_002.json',dict(steps=receipts,clean_venv_core_installation_passed=True,local_gradslam_and_SAM_still_required=True,whole_chain_runtime_passed=False,GPU_reproduction_passed=False,sources_sha256=frozen,resources=base.guard(),goal_complete=False));print('fresh CPU package installation complete',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:base.save('failure_002.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
