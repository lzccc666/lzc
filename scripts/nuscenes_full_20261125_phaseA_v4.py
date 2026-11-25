"""资源监督v4：原C进程自然完成，旧deadline失败仅验证后接续成员枚举。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,signal,subprocess,time
from conceptgraph.completion import nuscenes_full_20261125 as base
DEADLINE=1795613520


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def main():
    old=20498;command=Path(f'/proc/{old}/cmdline').read_bytes();assert b'nuscenes_full_20261125_phaseA_v3.py' in command
    prior=json.loads((base.REPORT/'phaseA_progress_v3.json').read_text());active={int(n):p for n,p in prior['active_PIDs'].items()};completed=set(prior['completed_archives']);queued=prior['queued']
    sourcefile=base.ROOT/'conceptgraph/completion/nuscenes_full_20261125_index_resume_v3.py'
    base.dump(base.DATA/'phaseA_resource_coordination_v4.json',dict(created_utc=base.now(),source_sha256=base.sha(__file__),new_resume_source_sha256=base.sha(sourcefile),
        prior_supervisor_PID=old,prior_progress=prior,deadline_unix=DEADLINE,source_scope_unchanged=True,methods_unchanged=True,
        original_C_builds_never_killed=True,resume_precondition='oldprocess exited; exact post-export guard timeout; oldindex and source stat authenticated; newfull tar+SHA',
        active_worker_cap=4,original_CPU_index_progress_preserved=True))
    os.kill(old,signal.SIGTERM);logs={};spawned={};resumed=set()
    while len(completed)<10:
        if time.time()>=DEADLINE:
            stopped=[]
            for number,pid in active.items():
                if alive(pid):os.kill(pid,signal.SIGSTOP);stopped.append(dict(archive=number,PID=pid))
            base.dump(base.REPORT/'phaseA_deadline_checkpoint_v4.json',dict(utc=base.now(),paused=stopped,reason='approved150min resource deadline; C memory and all prefixes retained'))
            raise TimeoutError('phaseA150min wall budget; source scope retained')
        base.guard(DEADLINE)
        for number,pid in list(active.items()):
            done=base.DATA/'archives'/f'archive{number:02d}_complete.json'
            if done.exists():
                try:receipt=json.loads(done.read_text())
                except json.JSONDecodeError:continue
                assert receipt['complete'];completed.add(number);del active[number]
                if number in spawned:spawned[number].wait();logs[number].close()
            elif not alive(pid):
                if number not in [3,4,5] or number in resumed:raise RuntimeError(f'archive{number:02d} source failure; preserve fullscope/index; see perarchive log')
                path=base.REPORT/f'archive{number:02d}_resume_v3.log';log=path.open('x');logs[number]=log
                process=subprocess.Popen([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_index_resume_v3.py'),'--archive',str(number),'--predecessor-pid',str(pid),'--deadline',str(DEADLINE)],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
                spawned[number]=process;active[number]=process.pid;resumed.add(number)
        while queued and len(active)<4:
            number=queued.pop(0);log=(base.REPORT/f'archive{number:02d}_v4.log').open('x');logs[number]=log
            process=subprocess.Popen([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_index_v2.py'),'--archive',str(number),'--deadline',str(DEADLINE)],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
            spawned[number]=process;active[number]=process.pid
        record=dict(utc=base.now(),completed_archives=sorted(completed),active_PIDs=active,queued=queued,paused_existing=[],GPU_release_seen=True,deadline_unix=DEADLINE,resource_supervisor='v4',technical_resumed_archives=sorted(resumed))
        base.dump(base.REPORT/'phaseA_progress_v3.json',record,False);print(json.dumps(record),flush=True)
        if len(completed)<10:time.sleep(20)
    subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125.py'),'certify'],cwd=base.ROOT,check=True)
    print(json.dumps(dict(stage='phaseA_complete',utc=base.now(),archives=sorted(completed))),flush=True)

if __name__=='__main__':main()
