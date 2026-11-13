"""从独立源码根执行同一真实CPU导航worker；显式资产依赖和导入隔离。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest['isolated_root']).resolve()
    original = Path(manifest['original_project']).resolve()
    environment_prefix = Path(manifest['environment_prefix']).resolve()
    assert Path(sys.prefix).resolve() == environment_prefix
    assert sys.prefix != sys.base_prefix
    old_site = Path(manifest['original_environment_site_packages']).resolve()
    assert not any(Path(p).resolve().is_relative_to(old_site) for p in sys.path if p)
    sys.path[:] = [str(root)]+[p for p in sys.path if p and Path(p).resolve() not in {original, root}]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('交付复演必须隐藏CUDA')
    import conceptgraph
    import torch
    assert torch.version.cuda is None
    from conceptgraph.completion.stage9 import OUTPUT, PROJECT
    from conceptgraph.completion import interactive_environment
    from conceptgraph.completion.live_navigation_worker import run
    if Path(conceptgraph.__file__).resolve().parent != root/'conceptgraph' or PROJECT != root:
        raise ValueError('未从隔离源码导入')
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    ultralytics.utils.NUM_THREADS = 1
    ultralytics.utils.torch_utils.NUM_THREADS = 1
    torch.set_num_threads(1)
    extension_path = Path(manifest['extension_path'])
    if sha(extension_path) != manifest['extension_sha256']:
        raise ValueError('停车保持策略指纹不同')
    specification = importlib.util.spec_from_file_location('delivery_stop_commitment', extension_path)
    extension = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(extension)
    from conceptgraph.completion import interactive_navigation
    interactive_navigation.PerceptionVehicleAgent = extension.PerceivedStopCommitmentAgent
    for name, expected in manifest['source_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError('独立源码指纹不一致：'+name)
    for name, expected in manifest['model_sha256'].items():
        if sha(name) != expected:
            raise ValueError('模型指纹不一致')
    asset = Path(manifest['asset_path'])
    if sha(asset) != interactive_environment.ASSET_SHA:
        raise ValueError('交付资产内容不同')
    # 只重绑定声明的外部文件位置；加载内容仍受原环境SHA常量约束。
    interactive_environment.ASSET = asset
    task_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    if sha(task_path) != manifest['task_release_sha256']:
        raise ValueError('任务清单不同')
    destination = OUTPUT/'interactive_simulation/live_navigation_sessions_001'/manifest['session']
    destination.mkdir(parents=True, exist_ok=False)
    release = dict(task_id=manifest['task_id'], seed=manifest['seed'], wall_budget_s=900,
        max_sim_s=30, max_frontend_calls=80, storage_budget_bytes=500_000_000,
        minimum_free_bytes=6_000_000_000, source_sha256=manifest['source_sha256'],
        task_release_sha256=manifest['task_release_sha256'], model_sha256=manifest['model_sha256'],
        explicit_asset_path=str(asset), asset_sha256=sha(asset),
        independent_test=False, posthoc_stop_commitment_diagnostic=True,
        extension_sha256=manifest['extension_sha256'], manuscript_text_written=False, goal_complete=False)
    save(destination/'release.json', release)
    command = dict(command_id='frozen-automatic-run-001', command='run',
        submitted_utc=datetime.now(timezone.utc).isoformat(), automatic_operation=True)
    with (destination/'commands.jsonl').open('x') as handle:
        handle.write(json.dumps(command, ensure_ascii=False)+'\n')
    proc = Path('/proc')/str(os.getpid())
    save(destination/'worker_process.json', dict(pid=os.getpid(),
        start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
        environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS', 'PYOPENGL_PLATFORM', 'PYTHONPATH']},
        cwd=str(Path.cwd()), actual_source_root=str(PROJECT)))
    from conceptgraph.scripts import streamlined_detections
    original_detect = streamlined_detections.detect_rgb_file
    thread_records = []

    def bounded_detect(*call_args, **call_kwargs):
        before = torch.get_num_threads()
        if before != 1:
            raise RuntimeError('实际前端调用前线程预算不同')
        value = original_detect(*call_args, **call_kwargs)
        after = torch.get_num_threads()
        thread_records.append(dict(call_index=len(thread_records), torch_threads_before=before,
            torch_threads_after=after, CUDA_initialized=torch.cuda.is_initialized()))
        with (destination/'thread_observations.jsonl').open('a') as handle:
            handle.write(json.dumps(thread_records[-1])+'\n')
        if after != 1 or torch.cuda.is_initialized():
            raise RuntimeError('实际前端调用突破CPU资源约束')
        return value

    streamlined_detections.detect_rgb_file = bounded_detect
    started = time.monotonic()
    try:
        run(destination)
    finally:
        modules = {}
        for name, module in list(sys.modules.items()):
            if name == 'conceptgraph' or name.startswith('conceptgraph.'):
                path = getattr(module, '__file__', None)
                if path:
                    location = Path(path).resolve()
                    if not location.is_relative_to(root):
                        raise ValueError('发现原工作区模块回落：'+str(location))
                    modules[name] = dict(path=str(location), sha256=sha(location))
        dependencies = {}
        for dependency in ['torch', 'torchvision', 'segment_anything', 'ultralytics',
                           'open_clip', 'cv2', 'open3d', 'gradslam', 'numpy', 'scipy',
                           'pyrender', 'OpenGL', 'shapely', 'faiss']:
            imported = sys.modules[dependency]
            path = Path(imported.__file__).resolve()
            assert path.is_relative_to(environment_prefix), (dependency, str(path))
            dependencies[dependency] = dict(path=str(path), sha256=sha(path))
        assert not any(name.startswith('__editable__') for name in sys.modules)
        assert not any(Path(p).resolve().is_relative_to(old_site) for p in sys.path if p)
        save(destination/'import_audit.json', dict(passed=True, modules=modules,
            verified_dependency_modules=dependencies, environment_prefix=str(environment_prefix),
            base_prefix=sys.base_prefix, interpreter=str(Path(sys.executable).resolve()),
            original_site_packages_absent=True, editable_finders_absent=True,
            CPU_torch_version=torch.__version__, CUDA_build=torch.version.cuda,
            new_Python_environment=True, clean_OS=False,
            all_conceptgraph_modules_from_isolated_root=True,
            original_project_removed_from_search_path=str(original) not in sys.path,
            CUDA_initialized=torch.cuda.is_initialized(), torch_threads=torch.get_num_threads(),
            source_root=str(PROJECT), asset_path=str(asset), asset_sha256=sha(asset),
            extension_sha256=sha(extension_path), actual_frontend_thread_observations=thread_records,
            runner_source_sha256=sha(__file__), wall_s=time.monotonic()-started,
            Python_environment_reused=False, clean_OS_or_new_environment=True, goal_complete=False))
    result = json.loads((destination/'result.json').read_text())
    print(json.dumps(dict(case=str(destination), status=result['status'],
        actual_frontend_calls=result['actual_frontend_calls'], controls=result['saved_control_intervals'],
        four_outcomes={k:result[k] for k in ['route_generated', 'selected_object_evaluated_correct',
            'stable_correct_lock', 'actual_full_perception_arrival']}, new_independent_scene_credit=0,
        frozen_main_table_credit=False, goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
