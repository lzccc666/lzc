"""有界CPU阶段A：首包/8场景资源试通过后，两路源包认证并全范围收口。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from conceptgraph.completion import nuscenes_full_20261125 as base


def launch(number):
    path=base.REPORT/f'archive{number:02d}_v2.log'
    if path.exists():raise FileExistsError(path)
    with path.open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_index_v2.py'),
            '--archive',str(number),'--deadline','1795609920'],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'archive{number:02d} failed; prefix retained; see {path}')
    return number


def main():
    base.dump(base.DATA/'phaseA_coordinator_freeze.json',dict(source_sha256=base.sha(__file__),created_utc=base.now(),
        first_archive_already_running=True,order='archive01 complete; first8source-only resource pilot; archives02..10 workers2; certify all850',
        deadline_unix=1795609920,GPU_calls=0,parameter_search=False))
    started=time.monotonic();first=base.DATA/'archives/archive01_complete.json'
    while not first.exists():
        base.guard(1795609920)
        print(json.dumps(dict(stage='waiting_first_archive',elapsed_s=time.monotonic()-started,utc=base.now())),flush=True)
        time.sleep(30)
    with (base.REPORT/'resource_pilot.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125_pilot.py')],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    pilot=json.loads((base.REPORT/'resource_pilot_summary.json').read_text());assert pilot['passed']
    print(json.dumps(dict(stage='pilot_passed',pilot=pilot,utc=base.now())),flush=True)
    completed=[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,n) for n in range(2,11)]
        for future in as_completed(futures):
            completed.append(future.result())
            base.dump(base.REPORT/'phaseA_progress.json',dict(completed_archives=sorted(completed),expected=10,utc=base.now()),False)
    subprocess.run([sys.executable,'-u',str(base.ROOT/'scripts/nuscenes_full_20261125.py'),'certify'],cwd=base.ROOT,check=True)
    print(json.dumps(dict(stage='phaseA_complete',completed=completed,utc=base.now())),flush=True)

if __name__=='__main__':main()
