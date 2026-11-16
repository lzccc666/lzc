"""补充新CPU环境缺失的几何库；安装到新目录，保留旧venv和历史冻结清单。"""
import json
import os
from pathlib import Path
import subprocess

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    out=R/'reproduction/clean_cpu_navigation_dependencies_002'
    out.mkdir(exist_ok=False)
    py=R/'reproduction/clean_cpu_install_001/venv/bin/python'
    env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','PIP_CONFIG_FILE':'/dev/null',
        'PIP_INDEX_URL':'https://pypi.org/simple','PIP_DISABLE_PIP_VERSION_CHECK':'1'}
    freeze=lambda: subprocess.run([str(py),'-m','pip','freeze','--all'],env=env,check=True,capture_output=True,text=True).stdout
    before=freeze()
    (out/'old_venv_freeze_before.txt').open('x').write(before)
    command=[str(py),'-m','pip','install','--no-deps','--no-compile','--index-url','https://pypi.org/simple',
        '--cache-dir',str(out/'cache'),'--target',str(out/'site-packages'),
        '--report',str(out/'pip_report.json'),'shapely==2.1.2']
    with (out/'install.log').open('x') as stream:
        result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    after=freeze()
    (out/'old_venv_freeze_after.txt').open('x').write(after)
    assert before==after
    report=dict(exit_code=result.returncode,command=command,old_venv_dependencies_unchanged=True,
        new_dependency_overlay=str(out/'site-packages'),source_sha256=digest(__file__),
        reason='首次完整导航导入缺少shapely；此前三帧感知复现未覆盖车辆几何模块',goal_complete=False)
    dump(out/'receipt.json',report)
    assert result.returncode==0
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
