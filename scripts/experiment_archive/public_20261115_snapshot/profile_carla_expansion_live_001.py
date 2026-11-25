"""独立只读采样本任务进程CPU高水位和GPU占用；不把采样峰值冒充CUDA精确峰值。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from conceptgraph.completion.stage9 import digest,dump

R=Path(__file__).resolve().parents[1]


def stat(pid):
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return p,fields
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launcher-pid',type=int,required=True);args=parser.parse_args()
    watched=stat(args.launcher_pid);assert watched is not None;start=watched[1][19]
    out=R/'performance/carla_expansion_live_profile_001';out.mkdir(parents=True,exist_ok=False)
    ticks=os.sysconf('SC_CLK_TCK');episodes={};samples=0;failures=0
    dump(out/'registration.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launcher_PID=args.launcher_pid,launcher_start_tick=start,interval_s=3.,CPU_peak='Linux VmHWM for individual actual process',GPU_peak='3s nvidia-smi used_gpu_memory sampled maximum, not exact allocator peak',source_sha256=digest(__file__),first_episode_already_closed_not_profiled=True,actual_new_model_or_physics_calls=0,goal_complete=False))
    with (out/'samples.jsonl').open('x') as stream:
        while True:
            watched=stat(args.launcher_pid)
            if watched is None or watched[1][19]!=start:break
            begun=time.monotonic();gpu={}
            try:
                result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
                for line in result.stdout.splitlines():
                    a,b=[s.strip() for s in line.split(',')]
                    if b.isdigit():gpu[int(a)]=int(b)
            except (subprocess.SubprocessError,ValueError):failures+=1
            uptime=float(Path('/proc/uptime').read_text().split()[0])
            for p in Path('/proc').glob('[0-9]*'):
                try:
                    argv=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
                    if len(argv)<2 or not argv[1].endswith('/source/carla_expansion_episode_010.py'):continue
                    identity=int(p.name);state=stat(identity)
                    if state is None:continue
                    values={k[2:]:argv[i+1] for i,k in enumerate(argv[:-1]) if k in ['--episode','--seed','--mode','--condition']}
                    status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if ':' in line}
                    hwm=int(status.get('VmHWM','0 kB').split()[0])/1024
                    rss=int(status.get('VmRSS','0 kB').split()[0])/1024
                    elapsed=uptime-int(state[1][19])/ticks
                    row=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=identity,start_tick=state[1][19],elapsed_process_s=elapsed,VmHWM_MiB=hwm,VmRSS_MiB=rss,GPU_sampled_used_MiB=gpu.get(identity),**values)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();samples+=1
                    aggregate=episodes.setdefault(str(identity),dict(**values,PID=identity,start_tick=state[1][19],first_sample_elapsed_s=elapsed,samples=0,VmHWM_MiB=0.,sampled_GPU_peak_MiB=None))
                    aggregate['samples']+=1;aggregate['VmHWM_MiB']=max(aggregate['VmHWM_MiB'],hwm)
                    if identity in gpu:aggregate['sampled_GPU_peak_MiB']=max(aggregate['sampled_GPU_peak_MiB'] or 0,gpu[identity])
                except (OSError,UnicodeError,ValueError,IndexError):continue
            time.sleep(max(.1,3.-(time.monotonic()-begun)))
    dump(out/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=samples,gpu_query_failures=failures,episodes=list(episodes.values()),sampled_GPU_peaks_are_lower_bounds=True,CPU_VmHWM_is_process_lifetime_peak_until_last_sample=True,first_episode_not_observed=True,samples_sha256=digest(out/'samples.jsonl'),actual_new_model_or_physics_calls=0,goal_complete=False))


if __name__=='__main__':main()
