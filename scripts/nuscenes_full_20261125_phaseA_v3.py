"""共享IO资源调度：保现有C索引进度，GPU释放后恢复四worker；源与方法不变。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,signal,subprocess,time
from conceptgraph.completion import nuscenes_full_20261125 as base
DEADLINE=1795613520
EXISTING={2:19280,3:19281,4:19283,5:19282}
NOTICE=base.DATA/'KITTI_GPU_released_notice.json'


def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();return stat.split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    old=19066
    cmd=Path(f'/proc/{old}/cmdline').read_bytes()
    assert b'nuscenes_full_20261125_phaseA_v2.py' in cmd
    freeze=dict(created_utc=base.now(),source_sha256=base.sha(__file__),resource_only=True,deadline_unix=DEADLINE,
        previous_deadline_unix=1795609920,authority='root explicit150min phaseA extension to2026-11-25T13:32UTC',
        existing_PIDs=EXISTING,paused_PIDs=[19281,19283,19282],active_archive_before_GPU_release=2,
        post_release_parallel_workers=4,trigger_notice=str(NOTICE),source_scope_unchanged=True,methods_unchanged=True,
        prior_coordinator_terminated_PID=old,termination='SIGTERM coordinator only; existing C-index child processes not killed; old sources/log retained',
        accounting='paused wall time distinguished from actual worker CPU; preserve all original source/index prefixes')
    base.dump(base.DATA/'phaseA_resource_coordination_v3.json',freeze)
    os.kill(old,signal.SIGTERM)
    active=dict(EXISTING);completed={1};queued=list(range(6,11));resumed=False;logs={};spawned={}
    while len(completed)<10:
        base.guard(DEADLINE)
        if NOTICE.exists() and not resumed:
            release=json.loads(NOTICE.read_text());assert release['explicit_release']
            resumed_pids=[]
            for number,pid in EXISTING.items():
                if number!=2 and alive(pid):os.kill(pid,signal.SIGCONT);resumed_pids.append(pid)
            base.dump(base.DATA/'index_workers_resumed_v3.json',dict(utc=base.now(),PIDs=resumed_pids,signal='SIGCONT',release_notice_sha256=base.sha(NOTICE)))
            resumed=True
        for number,pid in list(active.items()):
            done=base.DATA/'archives'/f'archive{number:02d}_complete.json'
            if done.exists():
                completed.add(number);del active[number]
                if number in spawned:spawned[number].wait();logs[number].close()
            elif not alive(pid):
                raise RuntimeError(f'archive{number:02d} exited without complete receipt; retained index/member prefix requires technical resume')
        while resumed and queued and len(active)<4:
            number=queued.pop(0);log=(base.REPORT/f'archive{number:02d}_v3.log').open('x');logs[number]=log
            process=subprocess.Popen([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_index_v2.py'),'--archive',str(number),'--deadline',str(DEADLINE)],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
            spawned[number]=process;active[number]=process.pid
        record=dict(utc=base.now(),completed_archives=sorted(completed),active_PIDs=active,queued=queued,
            paused_existing=[pid for n,pid in active.items() if not resumed and n!=2],GPU_release_seen=resumed,deadline_unix=DEADLINE)
        base.dump(base.REPORT/'phaseA_progress_v3.json',record,False);print(json.dumps(record),flush=True)
        if len(completed)<10:time.sleep(20)
    subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125.py'),'certify'],cwd=base.ROOT,check=True)
    print(json.dumps(dict(stage='phaseA_complete',utc=base.now(),archives=sorted(completed))),flush=True)

if __name__=='__main__':main()
