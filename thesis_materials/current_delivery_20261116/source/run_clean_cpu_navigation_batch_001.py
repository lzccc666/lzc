"""在独立CPU venv与恢复源码中运行一个完整官方导航任务，前后核对依赖。"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]
O = R/'json/clean_cpu_navigation_jobs_001'


def main():
    bankfile=R/'carla_official/clean_cpu_navigation_registration_001.json'
    bank=json.loads(bankfile.read_text())
    assert all(digest(p)==h for p,h in bank['source_sha256'].items())
    assert all(digest(x['checkout'])==x['sha256'] for x in bank['source_overlays'])
    O.mkdir(exist_ok=False)
    py=bank['runtime_python']
    checkout=bank['checkout']
    env={**os.environ, 'PYTHONPATH':checkout, 'PYTHONDONTWRITEBYTECODE':'1',
        'THESIS_ISOLATED_CHECKOUT':checkout,'THESIS_ISOLATED_SOURCE':bank['embedded_source'],
        'YOLO_AUTOINSTALL':'false','PIP_CONFIG_FILE':'/dev/null','PIP_INDEX_URL':'https://pypi.org/simple',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
        'YOLO_CONFIG_DIR':str(O/'yolo_config'),'XDG_CACHE_HOME':str(O/'cache'),
        'HF_HOME':str(O/'hf_cache')}
    env.pop('PYTHONHOME',None)
    before=subprocess.run([py,'-m','pip','freeze','--all'],env=env,cwd=checkout,
        check=True,capture_output=True,text=True).stdout
    (O/'installed_freeze_before.txt').open('x').write(before)
    log=O/'car20.log'
    started=time.monotonic()
    with log.open('x') as stream:
        result=subprocess.run([py,str(R/'source/carla_clean_cpu_navigation_episode_001.py'),
            '--episode','car20','--condition','full_rule_geometry'],env=env,cwd=checkout,
            stdout=stream,stderr=subprocess.STDOUT,check=False)
    after=subprocess.run([py,'-m','pip','freeze','--all'],env=env,cwd=checkout,
        check=True,capture_output=True,text=True).stdout
    (O/'installed_freeze_after.txt').open('x').write(after)
    assert before==after, '运行中依赖发生变化，不授予隔离复现信用'
    row=dict(exit_code=result.returncode,wall_s=time.monotonic()-started,log_sha256=digest(log),audits=[])
    dest=R/'carla_official/clean_cpu_navigation_001/car20/full_rule_geometry'
    if (dest/'summary.json').exists():
        for source in ['audit_carla_arrival_003.py','audit_carla_choices_005.py']:
            auditlog=O/(source+'.log')
            with auditlog.open('x') as stream:
                audit=subprocess.run([py,str(R/'source'/source),'--episode',
                    'clean_cpu_navigation_001/car20/full_rule_geometry'],env=env,cwd=checkout,
                    stdout=stream,stderr=subprocess.STDOUT,check=False)
            row['audits'].append(dict(source=source,exit_code=audit.returncode,log_sha256=digest(auditlog)))
    modules=dest/'fresh_environment_modules.json'
    isolated=json.loads(modules.read_text())['third_party_modules_isolated'] if modules.exists() else False
    complete=result.returncode==0 and isolated and len(row['audits'])==2 and all(x['exit_code']==0 for x in row['audits'])
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[row],
        actual_started_navigation_episodes=1,planned_episodes=1,subprocesses_and_audits_complete=complete,
        dependencies_unchanged_during_execution=True,third_party_dependencies_isolated=isolated,
        rule_only=True,actual_CPU_Qwen_generations=0,actual_GPU_inference_calls=0,
        official_server_graphics_GPU=True,registration_sha256=digest(bankfile),
        strict_arrival_must_read_independent_reports=True,whole_project_sealed_test=False,goal_complete=False)
    dump(O/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not complete:
        raise RuntimeError('独立CPU环境导航运行或隔离审计未完成；保留原始记录')


if __name__=='__main__':
    main()
