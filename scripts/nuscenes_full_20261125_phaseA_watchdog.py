"""阶段A硬预算看护：到期只暂停本轮索引进程，保内存进度与全部旧源。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,signal,time
from conceptgraph.completion import nuscenes_full_20261125 as source
DEADLINE=1795613520

def main():
    source.dump(source.DATA/'phaseA_watchdog_v1.json',dict(created_utc=source.now(),deadline_unix=DEADLINE,source_sha256=source.sha(__file__),action='SIGSTOP only matching current index workers if source notcertified; no kill/delete',GPU_calls=0))
    while time.time()<DEADLINE:
        if (source.DATA/'source_certification.json').exists():return
        time.sleep(min(20,max(.1,DEADLINE-time.time())))
    if (source.DATA/'source_certification.json').exists():return
    progress=json.loads((source.REPORT/'phaseA_progress_v3.json').read_text());paused=[]
    for number,pid in progress['active_PIDs'].items():
        p=Path(f'/proc/{pid}')
        if not p.exists():continue
        cmd=(p/'cmdline').read_bytes()
        if b'nuscenes_full_20261125_index_v2.py' not in cmd:continue
        stat=(p/'stat').read_text().split();os.kill(pid,signal.SIGSTOP)
        paused.append(dict(archive=number,PID=pid,CPU_s=(int(stat[13])+int(stat[14]))/os.sysconf('SC_CLK_TCK'),previous_state=stat[2]))
    source.dump(source.REPORT/'phaseA_deadline_checkpoint.json',dict(utc=source.now(),paused=paused,source_scope_850_retained=True,reason='approved150min wall deadline; source index memory/prefixes preserved for resource amendment',GPU_calls=0))

if __name__=='__main__':main()
