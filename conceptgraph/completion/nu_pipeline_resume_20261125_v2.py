"""交接技术草稿：生产执行被硬阻断，不能作为新single8运行入口。

待对齐single8 DATA/main/runtime、32h/48h/5.25GiB有效资源、明确关系release。
已实现的缓存/锁/发布恢复代码保留供下一电脑审读，尚未真实长跑验证。
用户已暂停完整实验；除纯CPU selftest外，所有CLI动作立即拒绝。

The CPU-only check and selftest never start inference.  Execute is an explicit
continuation of the existing full850 run, not the one-shot bootstrap controller.
Old attempts and partial postprocessing remain in place.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/processed/nuscenes_full_20261125_v1'
REG = ROOT / 'dataset/processed/nu_pipeline_resume_20261125_v2/registration.json'
RUN = ROOT / 'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_main_v1'
PILOT_DATA = ROOT / 'dataset/processed/nuscenes_full_20261125_resource_v1'
PILOT_RUN = RUN.parent / 'nuscenes_full_20261125_resource_v1'
TMP = ROOT / 'dataset/processed/nuscenes_full_20261125_tmp'
REPORT = ROOT / 'outputs/reports/nu_pipeline_resume_20261125/v2'
GPU_LOCK = Path('/root/autodl-tmp/lizichong/outputs/reports/shared_gpu_20261125.lock')
RUNTIME = ROOT / 'scripts/nuscenes_full_20261125_resource_runtime_v6.py'
PYTHON = '/root/miniconda3/envs/conceptgraph/bin/python'


class AuxiliaryPending(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def verify_hashes(table):
    for path, expected in table.items():
        if sha(path) != expected:
            raise AssertionError('frozen_SHA_mismatch:' + str(path))


def process_identity(pid):
    try:
        tail = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if tail[0] == 'Z':
            return None
        return {'PID': int(pid), 'start_ticks': int(tail[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def lease_alive(record):
    identity = process_identity(record['PID'])
    return identity is not None and identity['start_ticks'] == record['start_ticks']


def active_pipeline_processes():
    result = []
    needles = (b'nuscenes_full_20261125_execute_', b'nuscenes_full_20261125_resource_runtime_',
               b'nuscenes_full_20261125_cpu_cache_repair_', b'nu_pipeline_resume_20261125',
               b'nuscenes_full_20261125_evaluate.py',b'nuscenes_full_20261125_statistics.py',
               b'nuscenes_full_20261125_sharedscan_evaluate.py')
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = (directory / 'cmdline').read_bytes().split(b'\0')
            # Compare arguments, not shell command text or arbitrary embedded code.
            if not arguments or Path(os.fsdecode(arguments[0])).name not in ('python', 'python3', 'python3.10', 'python3.11'):
                continue
            named = any(any(needle in argument for needle in needles) for argument in arguments[1:4])
            owned_descriptor = False
            if not named and any(b'multiprocessing.spawn' in a for a in arguments[1:4]):
                for descriptor in (directory/'fd').iterdir():
                    try:
                        opened = Path(os.readlink(descriptor).removesuffix(' (deleted)'))
                        if opened.is_absolute() and any(owner.resolve() in opened.parents for owner in [RUN,PILOT_RUN,TMP]):
                            owned_descriptor = True; break
                    except (FileNotFoundError,PermissionError):
                        continue
            if named or owned_descriptor:
                identity = process_identity(int(directory.name))
                if identity:
                    result.append(identity)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


@contextmanager
def exclusive_driver(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'resume_driver.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('active_resume_driver_lock') from error
        lease = folder / 'active_lease.json'
        if lease.exists() and lease_alive(load(lease)):
            raise RuntimeError('active_resume_driver_PID')
        previous = load(lease) if lease.exists() else None
        if previous is not None:
            archive = folder / ('stale_lease_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
            write(archive, previous)
        record = {**process_identity(os.getpid()), 'started_utc': now(), 'source_sha256': sha(__file__)}
        temporary = lease.with_suffix('.new')
        with temporary.open('w') as out:
            json.dump(record, out); out.flush(); os.fsync(out.fileno())
        os.replace(temporary, lease)
        try:
            yield record
        finally:
            # The history receipt, not a deleted lease, records a clean release.
            record['released_utc'] = now()
            write(folder / ('released_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), record)
            lease.unlink()


def verify_closed(folder, scene):
    folder = Path(folder); record = load(folder / 'complete.json')
    assert record['complete'] is True and record['scene'] == scene['scene']
    assert record['counts']['frames'] == scene['frames']
    assert record['request_sha256'] == scene['request_sha256']
    verify_hashes({folder / 'frames.jsonl.gz': record['frames_sha256'],
                   folder / 'source_members.json': record['source_members_sha256']})
    if not (folder / 'sharedscan_complete.json').exists():
        raise AuxiliaryPending(str(folder))
    control = load(folder / 'sharedscan_complete.json')
    if control.get('complete') is not True:
        raise AuxiliaryPending(str(folder))
    assert control['counts']['frames'] == scene['frames']
    verify_hashes({folder / 'sharedscan_frames.jsonl.gz': control['frames_sha256']})
    if 'auxiliary_sharedscan' in record:
        if record['auxiliary_sharedscan'].get('complete') is not True:
            raise AuxiliaryPending(str(folder))
        assert record['auxiliary_sharedscan']['frames_sha256'] == control['frames_sha256']
    return dict(record, auxiliary_sharedscan=control)


def verify_output_aliases(folder):
    allowed = [RUN.resolve(), PILOT_RUN.resolve(),
               (RUN.parent / 'nuscenes_full_20261125_cached_cpu_preflight_v1').resolve()]
    for name in ['frames.jsonl.gz','source_members.json','frontend_receipt.json','sharedscan_frames.jsonl.gz','sharedscan_complete.json']:
        path = (Path(folder) / name).resolve()
        assert any(root in path.parents for root in allowed), 'unregistered_output_alias:' + str(path)


def register():
    pilot = load(PILOT_DATA / 'method_manifest.json')
    sources = [Path(__file__).resolve(), ROOT / 'scripts/nu_pipeline_resume_20261125_v2.py', RUNTIME,
               ROOT / 'conceptgraph/completion/nuscenes_full_20261125_resource_runtime_v6.py',
               ROOT / 'conceptgraph/completion/nuscenes_full_20261125_cpu_runtime_v6.py']
    record = dict(schema_version=1, source_sha256={str(p): sha(p) for p in sources},
                  manifest_sha256=sha(DATA / 'manifest.json'), pilot_method_sha256=sha(PILOT_DATA / 'method_manifest.json'),
                  original_source_sha256=pilot['source_sha256'], weights_sha256=pilot['weights_sha256'],
                  configs=pilot['configs'], expected_scenes=850, expected_frames=34149,
                  model_math_changed=False, GT_in_recovery=False, full_GPU_replay_of_cached_images=False,
                  policy='closed outputs reused; complete frontend cache CPU-only repair; incomplete cache blocks with exact missing members; old attempts retained',
                  postprocessing='original frozen functions under a new input-alias overlay, preserving partial evaluation folders',
                  GPU_lock=str(GPU_LOCK), runtime_script=str(RUNTIME))
    if REG.exists():
        assert load(REG) == record, 'resume_registration_changed'
    else:
        write(REG, record)
    return record


def cache_candidate(scene):
    """Identify a complete frontend cache without reading labels or model weights."""
    request = next(rows(scene['request'])); tokens = {v['camera'] for g in request['groups'] for v in g['views'].values()}
    attempted = sorted((RUN / scene['scene'] / 'attempts').glob('*'), reverse=True)
    failures = []
    for folder in attempted:
        receipt_path = folder / 'frontend_receipt.json'
        if not receipt_path.exists():
            continue
        receipt = load(receipt_path); members = load(folder / 'source_members.json')
        missing = []; temporary_roots = set()
        assert members['request_sha256'] == scene['request_sha256']
        for token, member in members['members'].items():
            assert token in request['sensors'] and member['filename'] == request['sensors'][token]['filename']
            candidate = Path(member['path'])
            for _ in Path(member['filename']).parts:
                candidate = candidate.parent
            assert candidate.parent.resolve() == TMP.resolve() and candidate.name.startswith(scene['scene'] + '__')
            temporary_roots.add(candidate)
            if not Path(member['path']).exists():
                missing.append(str(member['path']))
            elif sha(member['path']) != member['sha256']:
                raise AssertionError('cached_source_SHA_mismatch:' + str(member['path']))
        if set(members['members']) != set(request['sensors']):
            raise AssertionError('cached_source_sensor_token_set_mismatch')
        assert len(temporary_roots) == 1, 'mixed_temporary_cache_owners'
        temp = temporary_roots.pop()
        assert not (set(receipt['detections']) - tokens), 'cached_frontend_unknown_sensor_token'
        if set(receipt['detections']) != tokens:
            missing.extend(sorted(tokens - set(receipt['detections'])))
        for token, detection in receipt['detections'].items():
            path = temp / '_this_run_detections' / (token + '.pkl.gz')
            if not path.exists():
                missing.append(str(path))
            elif sha(path) != detection['sha256']:
                raise AssertionError('cached_frontend_SHA_mismatch:' + str(path))
        geom = temp / '_this_run_geometry.json'
        if not geom.exists():
            missing.append(str(geom))
        if not missing:
            return dict(scene=scene['scene'],attempt=str(folder),temporary=str(temp),geometry_sha256=sha(geom))
        failures.append(dict(attempt=str(folder),missing=missing))
    leftovers = sorted(str(p) for p in TMP.glob(scene['scene'] + '__*') if p.is_dir()) if TMP.exists() else []
    if attempted or leftovers:
        raise RuntimeError('incomplete_cache_requires_explicit_recovery:' + json.dumps(dict(scene=scene['scene'],attempts=failures,temporary=leftovers)))
    return None


def preflight():
    frozen = register()
    active = active_pipeline_processes()
    if active:
        raise RuntimeError('active_pipeline_PID:' + json.dumps(active))
    with GPU_LOCK.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('active_shared_GPU_lock') from error
    assert shutil.disk_usage(RUN.resolve()).free >= 20 * 2**30, 'disk20GiB guard'
    clock = load(DATA / 'overall_execution_clock_v1.json')
    assert time.time() < clock['absolute_deadline_unix'], 'absolute overall36h budget exhausted'
    assert clock['source_manifest_sha256'] == frozen['manifest_sha256']
    verify_hashes(frozen['source_sha256']); verify_hashes(frozen['original_source_sha256'])
    verify_hashes(frozen['weights_sha256'])
    assert sha(DATA / 'manifest.json') == frozen['manifest_sha256']
    certificate = load(DATA / 'source_certification.json')
    assert certificate['complete'] and certificate['fully_member_available_scenes'] == 850
    assert certificate['source_manifest_sha256'] == frozen['manifest_sha256']
    verify_hashes({DATA/'member_binding.jsonl.gz':certificate['member_binding_sha256']})
    # Recheck the small index receipts and index files, never re-scan/hash the
    # multi-hundred-GB original sensor archives during a resume.
    archive_receipt_sha256 = {}
    for number in range(1,11):
        receipt = DATA/'archives'/f'archive{number:02d}_complete.json'
        archive = load(receipt); assert archive['complete'] is True
        verify_hashes({archive['members']:archive['members_sha256'],archive['seek_index']:archive['seek_index_sha256']})
        stat = Path(archive['archive']).stat()
        assert stat.st_size == archive['archive_bytes'] and stat.st_mtime_ns == archive['archive_mtime_ns']
        archive_receipt_sha256[str(receipt)] = sha(receipt)
    method = load(DATA / 'method_manifest.json'); manifest = load(DATA / 'manifest.json')
    assert load(RUN / 'manifest.json') == method
    assert method['source_manifest_sha256'] == frozen['manifest_sha256']
    assert method['source_certification_sha256'] == sha(DATA / 'source_certification.json')
    assert method['weights_sha256'] == frozen['weights_sha256'] and method['configs'] == frozen['configs']
    for p, digest in method['source_sha256'].items():
        assert frozen['original_source_sha256'][p] == digest
    scenes = manifest['scenes']
    assert len(scenes) == 850 and len({s['scene'] for s in scenes}) == 850
    assert sum(s['frames'] for s in scenes) == 34149
    prior_GPU_s = 0.
    for attempt in (RUN / 'runtime_attempts').glob('*'):
        if (attempt / 'GPU_acquired.json').exists():
            assert (attempt / 'GPU_release.json').exists(), 'unclosed_GPU_occupancy:' + str(attempt)
            acquired = load(attempt / 'GPU_acquired.json')
            identity = process_identity(acquired['PID'])
            if identity and identity['PID'] in {p['PID'] for p in active_pipeline_processes()}:
                raise RuntimeError('old_GPU_PID_active:' + str(acquired['PID']))
            prior_GPU_s += float(load(attempt / 'GPU_release.json')['GPU_occupied_wall_s'])
    closed = []; sealable = []; cached = []; new = []
    for scene in scenes:
        verify_hashes({scene['request']: scene['request_sha256']})
        folder = RUN / scene['scene']
        if folder.is_symlink():
            assert folder.resolve().parent == PILOT_RUN.resolve(), 'unexpected_scene_alias'
        if (folder / 'complete.json').exists():
            try:
                verify_closed(folder, scene); verify_output_aliases(folder)
                closed.append(scene['scene']); continue
            except AuxiliaryPending:
                # A primary-only close can be completed from its retained full cache.
                pass
        complete_attempts = []
        candidates = list((folder / 'attempts').glob('*'))
        candidates.extend(RUN.glob('resume_attempts/*/repair_work/' + scene['scene'] + '/attempts/*'))
        for attempt in sorted(candidates, reverse=True):
            if (attempt / 'complete.json').exists() and (attempt / 'sharedscan_complete.json').exists():
                try:
                    verify_closed(attempt, scene); verify_output_aliases(attempt)
                    complete_attempts.append(attempt)
                except AuxiliaryPending:
                    pass
        if complete_attempts:
            sealable.append(dict(scene=scene['scene'],attempt=str(complete_attempts[0]))); continue
        candidate = cache_candidate(scene)
        if candidate:
            cached.append(candidate)
        else:
            new.append(scene['scene'])
    from conceptgraph.completion.nuscenes_full_20261125_resource_runtime_v6 import physical_bytes
    assert physical_bytes() <= 4.5 * 2**30, 'nu4.5GiB guard'
    if new:
        assert prior_GPU_s < 12 * 3600, 'cumulative12h GPU budget exhausted; CPU-only sealed/cached recovery remains separate'
    return dict(checked_utc=now(),closed=closed,sealable=sealable,cached=cached,new=new,
                original850_retained=True,prior_GPU_occupied_s=prior_GPU_s,remaining_GPU_budget_s=max(0.,12*3600-prior_GPU_s),registration_sha256=sha(REG),
                source_certification_sha256=sha(DATA / 'source_certification.json'),archive_receipt_sha256=archive_receipt_sha256)


def canonical_publish(scene, output, record, attempt):
    root = RUN / scene; root.mkdir(exist_ok=True)
    complete = root / 'complete.json'
    if complete.exists():
        history = attempt / (scene + '_previous_complete.json')
        assert not history.exists()
        # No reader may observe old complete=true with a partially updated set
        # of aliases. The original marker entity is retained, not overwritten.
        os.rename(complete,history)
    for name in ['frames.jsonl.gz','source_members.json','frontend_receipt.json','sharedscan_frames.jsonl.gz','sharedscan_complete.json']:
        target = root / name; origin = Path(output) / name
        previous = None
        assert origin.is_file()
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == origin.resolve():
                continue
            assert target.is_symlink(), 'refuse_overwrite_actual_canonical_entity'
            previous = dict(path=str(target),old_target=os.readlink(target),old_SHA256=sha(target))
        temporary = target.with_name(name + '.resume_new')
        temporary.symlink_to(origin.resolve())
        if previous is not None:
            write(attempt / ('superseded_alias_' + scene + '_' + name.replace('.', '_') + '.json'), previous)
        os.replace(temporary, target)
    temporary = root / 'complete.resume_new.json'
    write(temporary, dict(record, accepted_attempt=str(output),resume_source_sha256=sha(__file__)))
    os.replace(temporary, complete)


def cpu_repair(item, scene, work):
    """Spawn-only worker; all detections are frozen real cache, never GT."""
    import pickle
    from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as pilot
    base = pilot.base; source = pilot.source
    temp = Path(item['temporary']); old = Path(item['attempt']); work = Path(work)
    request = next(rows(scene['request'])); gpu = load(old / 'frontend_receipt.json'); gpu.pop('GT_in_method', None)
    members = load(old / 'source_members.json')['members']
    for token, detection in gpu['detections'].items():
        path = temp / '_this_run_detections' / (token + '.pkl.gz')
        assert sha(path) == detection['sha256']; detection['path'] = str(path)
        with gzip.open(path,'rb') as f:
            obj = pickle.load(f)
        assert base.prediction_digest(obj) == detection['prediction_sha256']
    geometry = temp / '_this_run_geometry.json'; assert sha(geometry) == item['geometry_sha256']
    with gzip.open(DATA / 'runtime_geometry_v5.json.gz','rt') as f:
        tables = json.load(f)
    wanted_poses = {s['ego_pose_token'] for s in request['sensors'].values()}
    wanted_cals = {s['calibrated_sensor_token'] for s in request['sensors'].values()}
    assert load(geometry) == dict(poses={t:tables['poses'][t] for t in wanted_poses},calibrations={t:tables['calibrations'][t] for t in wanted_cals})
    verify_hashes({r['path']:r['sha256'] for r in members.values()})
    base.DATA = DATA; base.RUN = work
    output = work / scene['scene'] / 'attempts' / 'CPU_cache_repair'; output.mkdir(parents=True)
    for name in ['frontend_receipt.json','source_members.json']:
        shutil.copy2(old / name, output / name)
    original_IO = RUN / 'IO_leases' / (temp.name.replace('__','_') + '.json')
    recovered_IO = load(original_IO)['source_IO_lease_wall_s'] if original_IO.exists() else None
    result = pilot.aux.cpu_scene_with_control(scene,request,str(temp),members,str(geometry),gpu,recovered_IO or 0.,str(output))
    assert result['auxiliary_sharedscan']['complete']
    return dict(scene=scene['scene'],output=str(output),record=result,new_GPU_calls=0,
                source_IO_time_recovery_unavailable=recovered_IO is None,original_receipts_preserved=True)


def step(attempt, name, arguments, *, cuda):
    path = attempt / (name + '.log'); started = time.monotonic()
    with path.open('x') as log:
        process = subprocess.Popen([PYTHON,'-u',*arguments],cwd=ROOT,env={**os.environ,
            'CUDA_VISIBLE_DEVICES':'0' if cuda else '', 'LD_LIBRARY_PATH':'/root/miniconda3/envs/conceptgraph/lib',
            'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
        write(attempt / (name + '_PID.json'), {**process_identity(process.pid),'started_utc':now()})
        code = process.wait()
    write(attempt / (name + '_exit.json'),dict(returncode=code,elapsed_s=time.monotonic()-started))
    if code:
        raise RuntimeError(name + '_failed_preserved')


def terminal_generation(attempt, status, **fields):
    """Archive previous bridge generation before publishing a new one."""
    record = dict(status=status,utc=now(),resume_generation=str(attempt),
                  source_sha256=sha(__file__),all850_scope_retained=True,**fields)
    for index,path in enumerate([RUN/'terminal_status.json',ROOT/'outputs/reports/nuscenes_full_20261125/execution_terminal_status_v6.json']):
        if path.exists():
            previous = attempt / ('prior_terminal_' + status + '_' + str(index) + '.json')
            write(previous,load(path))
        temporary = path.with_name(path.name + '.resume_new')
        write(temporary,record);os.replace(temporary,path)


def evaluation_stage(attempt, stage):
    """Run the original evaluation function with only its output RUN redirected."""
    overlay = Path(attempt) / 'postprocess'; overlay.mkdir(exist_ok=True)
    manifest = load(DATA / 'manifest.json')
    for scene in manifest['scenes']:
        target = overlay / scene['scene']
        if not target.exists():target.symlink_to((RUN / scene['scene']).resolve(),target_is_directory=True)
    complete = overlay / 'inference_complete.json'
    if not complete.exists():complete.symlink_to((RUN / 'inference_complete.json').resolve())
    from conceptgraph.completion import nuscenes_full_20261125_evaluate as ev
    from conceptgraph.completion import nuscenes_full_20261125_run as base
    ev.RUN = overlay; base.RUN = overlay
    if stage == 'evaluation':
        ev.evaluate()
    elif stage == 'statistics':
        from conceptgraph.completion import nuscenes_full_20261125_statistics as statistics
        statistics.RUN = overlay; statistics.run()
    elif stage == 'sharedscan_evaluation':
        from conceptgraph.completion import nuscenes_full_20261125_sharedscan_evaluate as shared
        shared.main()
    else:
        raise ValueError(stage)


def execute():
    with exclusive_driver(REPORT):
        checked = preflight()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        attempt = RUN / 'resume_attempts' / stamp; attempt.mkdir(parents=True)
        write(attempt / 'preflight.json',checked)
        manifest = load(DATA / 'manifest.json'); scenes = {s['scene']:s for s in manifest['scenes']}
        try:
            terminal_generation(attempt,'running')
            for item in checked['sealable']:
                record = verify_closed(item['attempt'],scenes[item['scene']])
                canonical_publish(item['scene'],item['attempt'],record,attempt)
            if checked['cached']:
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing
                from conceptgraph.completion.nuscenes_full_20261125_cpu_runtime_v6 import initialize_cpu_worker
                with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu_worker) as pool:
                    for item in checked['cached']:
                        repaired = pool.submit(cpu_repair,item,scenes[item['scene']],str(attempt / 'repair_work')).result()
                        write(attempt / (item['scene'] + '_CPU_repair.json'),repaired)
                        canonical_publish(item['scene'],repaired['output'],repaired['record'],attempt)
            if checked['new']:
                step(attempt,'frozen_fullrun',[str(RUNTIME),'fullrun'],cuda=True)
            elif not (RUN / 'inference_complete.json').exists():
                records = [verify_closed(RUN / s['scene'],s) for s in scenes.values()]
                counts = Counter()
                for record in records:counts.update(record['counts'])
                write(RUN / 'inference_complete.json',dict(complete=True,scenes=850,expected_scenes=850,counts=dict(counts),
                    all_existing_closed_outputs_reused=True,new_GPU_calls=0,source_manifest_sha256=sha(DATA / 'manifest.json')))
            assert load(RUN / 'inference_complete.json')['complete']
            for stage in ['evaluation','statistics','sharedscan_evaluation']:
                step(attempt,stage,['-m','scripts.nu_pipeline_resume_20261125_v2','stage','--resume-attempt',str(attempt),'--stage',stage],cuda=False)
            write(attempt / 'complete.json',dict(complete=True,utc=now(),new_method_configuration=False,
                postprocess_root=str(attempt / 'postprocess'),all850_retained=True,source_registration_sha256=sha(REG)))
            terminal_generation(attempt,'complete')
        except BaseException as error:
            write(attempt / 'stopped.json',dict(status='stopped',utc=now(),error_type=type(error).__name__,error=str(error),all850_retained=True))
            terminal_generation(attempt,'stopped',error_type=type(error).__name__,error=str(error))
            raise


def selftest():
    """Bounded real locks/PID/filesystem probes, without importing torch/CUDA."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='nu_resume_contract_') as d:
        root = Path(d); target=root/'source';target.write_text('frozen')
        digest=sha(target);verify_hashes({target:digest})
        target.write_text('corrupt')
        try:verify_hashes({target:digest})
        except AssertionError:pass
        else:raise AssertionError('bad_SHA_not_rejected')
        with exclusive_driver(root/'driver'):
            try:
                with exclusive_driver(root/'driver'):pass
            except RuntimeError:pass
            else:raise AssertionError('duplicate_driver_not_rejected')
        write(root/'driver/active_lease.json',dict(PID=2147483647,start_ticks=1))
        with exclusive_driver(root/'driver'):
            assert list((root/'driver').glob('stale_lease_*.json'))
        write(root/'driver/active_lease.json',{**process_identity(os.getpid())})
        try:
            with exclusive_driver(root/'driver'):pass
        except RuntimeError:pass
        else:raise AssertionError('active_PID_not_rejected')
    REPORT.mkdir(parents=True,exist_ok=True)
    out=REPORT/('selftest_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    write(out,dict(passed=True,corrupt_SHA_rejected=True,duplicate_lock_rejected=True,
        active_PID_rejected=True,stale_PID_recovered=True,GPU_calls=0,model_imports=0,source_sha256=sha(__file__)))
    print(json.dumps(dict(report=str(out),passed=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['register','check','execute','selftest','stage'])
    parser.add_argument('--run-directory',default=str(RUN));parser.add_argument('--runtime-script',default=str(RUNTIME))
    parser.add_argument('--resume-attempt');parser.add_argument('--stage',choices=['evaluation','statistics','sharedscan_evaluation'])
    args=parser.parse_args()
    if args.action!='selftest':
        tracked=ROOT/'doc/交接/20261125/执行状态.json'
        if not tracked.exists() or load(tracked).get('user_handoff_hold',{}).get('active') is not False:
            raise RuntimeError('TRACKED_USER_HANDOFF_HOLD: explicit human continue is required; missing or active state forbids execution')
        handoff=ROOT/'dataset/processed/nuscenes_full_20261125_v1/user_handoff_hold_v1.json'
        if handoff.exists() and load(handoff).get('active') is True:
            raise RuntimeError('USER_HANDOFF_HOLD: all full/watch/resume execution paused; an old relation release cannot override this hold')
        raise RuntimeError('HANDOFF_ONLY_DRAFT_EXECUTION_BLOCKED: user paused; single8 namespace/resource/release integration and independent audit are still required')
    assert Path(args.run_directory).resolve()==RUN.resolve() and Path(args.runtime_script).resolve()==RUNTIME.resolve(), 'only frozen official run/runtime allowed'
    if args.action=='register':print(json.dumps(register()))
    elif args.action=='check':print(json.dumps(preflight()))
    elif args.action=='execute':execute()
    elif args.action=='selftest':selftest()
    else:
        path=Path(args.resume_attempt).resolve();assert RUN.resolve()/'resume_attempts' in path.parents
        assert (path/'preflight.json').exists();evaluation_stage(path,args.stage)


if __name__=='__main__':main()
