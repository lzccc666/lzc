"""只读采样本Goal CARLA进程资源，不推进世界、不控制车辆；采样峰值不冒充精确峰值。"""
import argparse,datetime,json,subprocess,time
from pathlib import Path
import psutil
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'carla_official/resources_observer_001'
def main():
 p=argparse.ArgumentParser();p.add_argument('--driver-pid',type=int,required=True);a=p.parse_args();driver=psutil.Process(a.driver_pid);assert any('run_carla_ablation_batch_003.py' in s for s in driver.cmdline());O.mkdir(exist_ok=False);start=time.monotonic();max_total_gpu=0;max_child_rss=0;rows=0
 with (O/'samples.jsonl').open('x') as f:
  while driver.is_running() and driver.status()!=psutil.STATUS_ZOMBIE:
   children=[]
   try:
    for child in driver.children(recursive=True):
     if child.is_running():children.append(dict(pid=child.pid,name=child.name(),rss_bytes=child.memory_info().rss,cmdline=child.cmdline()))
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   z=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True);gpu=z.stdout.strip();parts=gpu.split(',');memory=float(parts[0]) if z.returncode==0 and parts[0].strip() else None
   if memory is not None:max_total_gpu=max(max_total_gpu,memory)
   max_child_rss=max([max_child_rss]+[c['rss_bytes'] for c in children]);r=dict(wall_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observer_elapsed_s=time.monotonic()-start,gpu_total_device_used_mib=memory,gpu_raw=gpu,child_processes=children,observational_only=True);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();rows+=1;time.sleep(2)
 dump(O/'report.json',dict(samples=rows,interval_s=2,observed_total_GPU_peak_mib=max_total_gpu,observed_child_RSS_peak_bytes=max_child_rss,peak_kind='2秒采样下界；不是torch算子分配精确峰值；仅观察开始后当前批次，早期episode不可补推',gpu_scope='设备总体，含官方CARLA服务器与前端/其他进程；不能按此捏造逐方法显存优势',source_sha256=digest(__file__),goal_complete=False))
if __name__=='__main__':main()
