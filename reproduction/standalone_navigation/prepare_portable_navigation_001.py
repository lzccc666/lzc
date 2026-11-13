"""从发布清单准备独立完整导航，不依赖原项目的历史输出或数据集。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--extra-library-path', type=Path, action='append', default=[])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    release_path = args.release.resolve()
    release = json.loads(release_path.read_text())
    package, source = release_path.parent, args.source_root.resolve()
    destination = args.destination.resolve()
    assert not destination.exists(), '为保护失败记录，不覆盖已有目录'
    assert shutil.disk_usage(destination.parent).free >= 6_000_000_000
    assert len(release['source_sha256']) == 390
    # 写入前一次性校验全部小输入和依赖，不复制旧结果。
    for name, expected in release['source_sha256'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(source/path) == expected, name
        assert (source/path).stat().st_mode & 0o777 == release['source_modes'][name], name
    for name, expected in release['package_sha256'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(package/path) == expected, name
    model_root = args.model_dir.resolve()
    for name, expected in release['model_relative_sha256'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(model_root/path) == expected, name
    assert sha(args.asset) == release['asset_sha256']
    runtime = args.runtime_root.resolve()
    assert (runtime/'python_deps/OpenGL/__init__.py').is_file()
    assert (runtime/'extracted/usr/lib/x86_64-linux-gnu/libOSMesa.so.8').is_file()
    python = args.python.resolve()
    assert python.is_file()
    for library in args.extra_library_path:
        assert library.is_dir()
    destination.mkdir()
    root = destination/'checkout'
    root.mkdir()
    for name in release['source_sha256']:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    (root/'checkpoint').symlink_to(model_root, target_is_directory=True)
    dependencies = destination/'dependency_inputs'
    dependencies.mkdir()
    asset = dependencies/'ToyCar_002.glb'
    shutil.copy2(args.asset, asset)
    scripts = root/'reproduction'
    scripts.mkdir()
    for name in ['standalone_cpu_runner_002.py', 'perceived_stop_commitment_001.py']:
        shutil.copy2(package/name, scripts/name)
    task = root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/synthetic_tasks_v1.json'
    task.parent.mkdir(parents=True)
    shutil.copy2(package/'synthetic_tasks_v1.json', task)
    index = source/'.git/index'
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), original_project=str(source),
        isolated_root=str(root), source_sha256=release['source_sha256'], business_source_files=390,
        source_modes=release['source_modes'], task_release_sha256=sha(task),
        task_id='config00-positive0', seed=17, session='standalone_config00_positive0_s17_portable_001',
        model_sha256={str(root/'checkpoint'/name): value for name, value in release['model_relative_sha256'].items()},
        asset_path=str(asset), asset_sha256=sha(asset),
        extension_path=str(scripts/'perceived_stop_commitment_001.py'),
        extension_sha256=sha(scripts/'perceived_stop_commitment_001.py'),
        runner_sha256=sha(scripts/'standalone_cpu_runner_002.py'),
        original_index_sha256=sha(index) if index.is_file() else None,
        portable_release_sha256=sha(release_path), posthoc_stop_commitment_diagnostic=True,
        historical_frames_or_results_copied=False, clean_OS_or_new_environment=False,
        original_historical_outputs_required=False, manuscript_text_written=False, goal_complete=False)
    save(destination/'manifest.json', manifest)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', LP_NUM_THREADS='1', PYOPENGL_PLATFORM='osmesa',
        PYGLET_SHADOW_WINDOW='false', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(runtime/'python_deps'),
        LD_LIBRARY_PATH=':'.join([str(runtime/'extracted/usr/lib/x86_64-linux-gnu')]
            + [str(path.resolve()) for path in args.extra_library_path]
            + [str(python.parent.parent/'lib')]))
    command = [str(python), 'reproduction/standalone_cpu_runner_002.py', '--manifest', '../manifest.json']
    save(destination/'prepared.json', dict(passed=True, source_files=390, command=command, cwd=str(root),
        declared_environment={key: environment[key] for key in ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH',
            'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS']},
        actual_model_calls=0, execution_requested=args.execute, goal_complete=False))
    if args.execute:
        with (destination/'worker.log').open('x') as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=960, check=False)
        save(destination/'launcher_exit.json', dict(returncode=result.returncode,
            task_result_must_be_checked_separately=True, goal_complete=False))
        raise SystemExit(result.returncode)
    print(json.dumps(dict(prepared=True, actual_model_calls=0, destination=str(destination))))


if __name__ == '__main__':
    main()
