"""Bounded CPU consumer of durably accepted official-run scene streams.

Idle waiting is not model computation.  Methods, labels and scores are the
frozen fair-v2 implementation; this controller only schedules and losslessly
compresses this run's intermediate text after its checksums are verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from conceptgraph.completion import nu_task_replay_20261125 as base
from conceptgraph.completion import nu_task_summary_20261125 as summary
from conceptgraph.completion.nu_task_replay_fair_20261125_v2 import register

PRODUCER=base.ROOT/'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_main_v1'
REPORT=base.REPORT/'watch_v2'
RUNS=[base.experiment_directory('nuscenes','validation','nu_task_replay_20261125_v2_'+suffix) for suffix in ('retrieval','confirmation')]
WALL_LIMIT=36*3600
PROCESS_LIMIT=1800.
RESULT_LIMIT=128*1024**2
RAW_ROWS=base.rows
RAW_SHA=base.sha


def effective_rows(path):
    path=Path(path)
    if not path.exists() and Path(str(path)+'.gz').exists():path=Path(str(path)+'.gz')
    yield from RAW_ROWS(path)


def effective_sha(path):
    path=Path(path)
    if path.exists():return RAW_SHA(path)
    archive=Path(str(path)+'.gz')
    if not archive.exists():raise FileNotFoundError(path)
    h=hashlib.sha256()
    with gzip.open(archive,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def size():
    return sum(p.stat().st_size for folder in [base.DATA,base.REPORT,*RUNS] if folder.exists()
        for p in folder.rglob('*') if p.is_file())


def archive_scene(scene):
    receipts=[]
    for folder in RUNS:
        for name in ['method_predictions.jsonl','evaluation.jsonl']:
            path=folder/scene/name;digest=RAW_SHA(path)
            target=Path(str(path)+'.gz');temp=Path(str(target)+'.part')
            if target.exists():raise FileExistsError(target)
            with path.open('rb') as source,temp.open('xb') as output:
                with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,filename='') as compressed:
                    for chunk in iter(lambda:source.read(1024**2),b''):compressed.write(chunk)
                output.flush();os.fsync(output.fileno())
            h=hashlib.sha256()
            with gzip.open(temp,'rb') as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise AssertionError('lossless_archive_mismatch')
            os.rename(temp,target)
            receipts.append(dict(original_logical_path=str(path),uncompressed_SHA256=digest,
                gzip_path=str(target),gzip_SHA256=RAW_SHA(target),uncompressed_bytes=path.stat().st_size,
                compressed_bytes=target.stat().st_size))
    receipt=REPORT/scene/'archive_receipt.json';base.write(receipt,dict(scene=scene,files=receipts,
        lossless_verified=True,original_method_receipts_preserved=True,
        action='this consumer run own intermediate plaintext removed only after durable gzip+uncompressedSHA verification'))
    for item in receipts:Path(item['original_logical_path']).unlink()
    return receipt


def producer_terminal():
    for name in ['terminal_status.json','inference_partial.json','interrupted.json','stopped.json']:
        path=PRODUCER/name
        if path.exists():return dict(path=str(path),record=json.loads(path.read_text()))
    fallback=base.ROOT/'outputs/reports/nuscenes_full_20261125/execution_terminal_status_v6.json'
    if fallback.exists():return dict(path=str(fallback),record=json.loads(fallback.read_text()))
    for path in sorted((PRODUCER/'runtime_attempts').glob('*/inference_partial.json')) if (PRODUCER/'runtime_attempts').exists() else []:
        return dict(path=str(path),record=json.loads(path.read_text()))
    return None


def watch():
    register();base.verify(method_only=True);REPORT.mkdir(parents=True,exist_ok=True)
    tasks=list(RAW_ROWS(base.DATA/'method_tasks.jsonl'));scenes=sorted({t['scene'] for t in tasks})
    predecessor_path=base.REPORT/'watch_v1/progress.json'
    predecessor=json.loads(predecessor_path.read_text())
    frozen=base.DATA/'watch_registration_v2.json'
    own_progress=REPORT/'progress.json'
    previous=json.loads(own_progress.read_text()) if own_progress.exists() else predecessor
    if frozen.exists():
        fixed_start=json.loads(frozen.read_text())['fixed_start_UTC']
    else:
        fixed_start=(datetime.fromisoformat(predecessor['utc'])-timedelta(seconds=predecessor['wait_elapsed_s'])).isoformat()
    prior_active=float(previous['active_child_wall_s'])
    elapsed_before_start=max(0.,(datetime.now(timezone.utc)-datetime.fromisoformat(fixed_start)).total_seconds())
    registration=dict(fixed_start_UTC=fixed_start,predecessor_registration_SHA=RAW_SHA(base.DATA/'watch_registration_v1.json'),
        technical_amendment='v2 preserves cumulative36h/active1800s across restart, charges failed child elapsed in finally, reads v6fallback and runtime_attempts; methods/scoring unchanged',policy=dict(wait_wall_limit_s=WALL_LIMIT,active_child_wall_limit_s=PROCESS_LIMIT,
        persistent_results_limit_bytes=RESULT_LIMIT,poll_s=20,GPU_calls=0,
        producer=str(PRODUCER),required_scenes=scenes,source_tasks=2315,pending_source_labels=60,
        only_closed_complete_SHA=True,partial_or_pilot_consumption=False,
        method_and_evaluation='unchanged frozen fair-v2; strict-motion confirmation separate from geometric retrieval'),
        source_sha256={str(p):RAW_SHA(p) for p in [Path(__file__),base.ROOT/'scripts/nu_task_watch_20261125_v2.py']},
        fair_amendment_sha256=RAW_SHA(base.DATA/'fair_comparison_amendment_v2.json'),
        source_manifest_sha256=RAW_SHA(base.DATA/'manifest.json'),
        compression='gzip of own intermediate raw JSONL, decompressionSHA exactly matches original saved method/eval receipts; both raw and gzipSHA preserved',
        terminal_policy='explicit producer terminal marker or root incomplete result stops with exact unprocessed taskIDs; all2315 consumed exits immediately')
    if frozen.exists():
        if json.loads(frozen.read_text())!=registration:raise AssertionError('watch_registration_changed')
    else:base.write(frozen,registration)
    with (REPORT/'consumer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();active=prior_active;done=set();status='waiting_for_formal_closed_scenes'
        for scene in scenes:
            if (REPORT/scene/'complete.json').exists():done.add(scene)
        def progress(reason=None):
            record=dict(utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),status=status,
                completed_scenes=sorted(done),pending_scenes=[s for s in scenes if s not in done],
                pending_task_ids=[t['task_id'] for t in tasks if t['scene'] not in done],
                fixed_start_UTC=fixed_start,wait_elapsed_s=elapsed_before_start+time.monotonic()-started,active_child_wall_s=active,controller_process_CPU_s=time.process_time(),
                persistent_bytes=size(),source_registration_SHA=RAW_SHA(frozen),reason=reason)
            path=REPORT/'progress.json'
            with path.open('w') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
            return record
        try:
            while len(done)<len(scenes):
                base.guard()
                if size()>RESULT_LIMIT:status='resource_stop_pending';progress('128MiB result cap');return
                if elapsed_before_start+time.monotonic()-started>WALL_LIMIT:status='wait_budget_stop_pending';progress('36h waiting cap');return
                if active>PROCESS_LIMIT:status='CPU_budget_stop_pending';progress('1800s cumulative active child wall');return
                for scene in scenes:
                    if scene in done:continue
                    marker=PRODUCER/scene/'complete.json'
                    if not marker.exists():continue
                    closed=json.loads(marker.read_text())
                    if closed.get('complete') is not True:continue
                    frame_file=PRODUCER/scene/'frames.jsonl.gz'
                    if RAW_SHA(frame_file)!=closed['frames_sha256']:raise AssertionError('formal_scene_SHA_mismatch')
                    status='processing_closed_scene';progress(scene)
                    for action in ['infer','evaluate']:
                        if active>=PROCESS_LIMIT:raise TimeoutError('active CPU cap before next command')
                        logpath=REPORT/scene/(action+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
                        command=[sys.executable,'-m','scripts.nu_task_replay_fair_20261125_v2',action,'--scene',scene]
                        if action=='infer':command+=['--frame-file',str(frame_file)]
                        tick=time.monotonic()
                        try:
                            with logpath.open('x') as log:
                                result=subprocess.run(command,cwd=base.ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,timeout=max(1,PROCESS_LIMIT-active))
                        finally:
                            active+=time.monotonic()-tick
                        if result.returncode:raise RuntimeError(f'fixed_{action}_failed:{scene}; source denominator remains pending')
                    archived=archive_scene(scene)
                    base.rows=effective_rows;base.sha=effective_sha
                    summary.main()
                    base.write(REPORT/scene/'complete.json',dict(scene=scene,producer_complete_SHA=RAW_SHA(marker),
                        archive_receipt_SHA=RAW_SHA(archived),completed_utc=datetime.now(timezone.utc).isoformat()))
                    done.add(scene);status='waiting_for_formal_closed_scenes';progress()
                if len(done)==len(scenes):break
                terminal=producer_terminal()
                full=PRODUCER/'inference_complete.json'
                if terminal or full.exists():
                    status='producer_terminal_pending';progress(terminal or 'producer complete but required frozen scenes missing');return
                progress();time.sleep(20)
            status='all2315_tasks_complete';progress()
            base.rows=effective_rows;base.sha=effective_sha;summary.main()
        except BaseException as error:
            status='technical_stop_pending';record=progress(repr(error))
            base.write(REPORT/'technical_stop.json',record)
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('watch','summary'))
    args=parser.parse_args()
    if args.action=='watch':watch()
    else:base.rows=effective_rows;base.sha=effective_sha;summary.main()


if __name__=='__main__':main()
