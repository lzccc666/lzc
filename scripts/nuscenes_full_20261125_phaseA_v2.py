"""阶段A资源调度修订：保持源/索引算法，首试后4包并行守90分钟。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from conceptgraph.completion import nuscenes_full_20261125 as base
from scripts.nuscenes_full_20261125_phaseA import launch


def main():
    base.dump(base.DATA/'phaseA_parallelism_v2.json',dict(source_sha256=base.sha(__file__),created_utc=base.now(),
        resource_only=True,parallel_archive_workers=4,previous_workers=2,
        evidence='archive01 C-index589.93s,80.98MB; sequential member enumeration additional; shared source read throughput slower than historical estimate',
        order='archive01 complete; fixed8source-only resource pilot; archives02..10 workers4; certify all850',
        deadline_unix=1795609920,GPU_calls=0,parameter_search=False,
        old_coordinator_preserved=str(base.ROOT/'scripts/nuscenes_full_20261125_phaseA.py')))
    first=base.DATA/'archives/archive01_complete.json';started=time.monotonic()
    while not first.exists():
        base.guard(1795609920)
        print(json.dumps(dict(stage='waiting_first_archive',elapsed_s=time.monotonic()-started,utc=base.now())),flush=True);time.sleep(30)
    with (base.REPORT/'resource_pilot.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_pilot.py')],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    pilot=json.loads((base.REPORT/'resource_pilot_summary.json').read_text());assert pilot['passed']
    print(json.dumps(dict(stage='pilot_passed',utc=base.now(),peak_scene_temporary_bytes=pilot['peak_scene_temporary_bytes'])),flush=True)
    completed=[1]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(launch,n) for n in range(2,11)]
        for future in as_completed(futures):
            completed.append(future.result())
            base.dump(base.REPORT/'phaseA_progress.json',dict(completed_archives=sorted(completed),expected=10,utc=base.now()),False)
    subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125.py'),'certify'],cwd=base.ROOT,check=True)
    print(json.dumps(dict(stage='phaseA_complete',completed=completed,utc=base.now())),flush=True)

if __name__=='__main__':main()
