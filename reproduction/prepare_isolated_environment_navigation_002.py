"""在已实际安装的新Python环境中准备并执行同一源码完整导航小任务。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    package = OUTPUT / 'reproduction/standalone_portable_release_001'
    environment_root = args.environment.resolve()
    installed_path = environment_root / 'installed.json'
    installed = json.loads(installed_path.read_text())
    assert installed['installed']
    assert sha(environment_root / 'plan.json') == installed['plan_sha256']
    plan = json.loads((environment_root / 'plan.json').read_text())
    python = Path(installed['python'])
    prefix = Path(installed['environment'])
    assert python.resolve().is_relative_to(prefix)
    destination = args.destination.resolve()
    assert destination.is_relative_to(OUTPUT / 'reproduction') and not destination.exists()
    runtime = OUTPUT / 'source_bundle/software_render_runtime_001'
    runner = OUTPUT / 'source_bundle/standalone_cpu_runner_004.py'
    base = package / 'prepare_portable_navigation_001.py'
    source = OUTPUT / 'reproduction/portable_input_source_001'
    # 复用原发布包的390源文件、模型/资产校验；不启动旧环境worker。
    command = [sys.executable, str(base), '--release', str(package / 'release.json'),
               '--source-root', str(source), '--model-dir', '/root/autodl-tmp/lizichong/checkpoint',
               '--asset', '/root/autodl-fs/third_party/urban_scene_assets/ToyCar_002.glb',
               '--runtime-root', str(runtime), '--python', str(python),
               '--extra-library-path', '/root/autodl-fs/third_party/graphics_runtime/lib',
               '--extra-library-path', '/root/miniconda3/envs/conceptgraph/lib',
               '--destination', str(destination)]
    subprocess.run(command, check=True, timeout=90)
    manifest_path = destination / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_path.rename(destination / 'manifest_before_environment.json')
    (destination / 'prepared.json').rename(destination / 'prepared_before_environment.json')
    root = Path(manifest['isolated_root'])
    target_runner = root / 'reproduction/standalone_cpu_runner_004.py'
    shutil.copy2(runner, target_runner)
    assert sha(target_runner) == sha(runner)
    manifest.update(
        environment_prefix=str(prefix), original_environment_site_packages=plan['original_site_packages'],
        environment_installed_manifest=str(installed_path),
        environment_installed_manifest_sha256=sha(installed_path),
        runner_sha256=sha(target_runner), new_Python_environment=True,
        clean_OS=False, clean_OS_or_new_environment=True,
        original_conda_standard_library_and_native_system_libraries_shared=True,
        original_python_site_packages_reused=False,
    )
    save(manifest_path, manifest)
    temporary = destination / 'tmp'
    temporary.mkdir()
    # 图形原生库共享并明确声明；Python OpenGL绑定已复制到新环境，不设原站点PYTHONPATH。
    process_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', LP_NUM_THREADS='1', PYOPENGL_PLATFORM='osmesa',
                       PYGLET_SHADOW_WINDOW='false', PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TMPDIR=str(temporary), MPLCONFIGDIR=str(destination / 'matplotlib'),
                       YOLO_CONFIG_DIR=str(destination / 'ultralytics'),
                       XDG_CACHE_HOME=str(destination / 'cache'),
                       LD_LIBRARY_PATH=str(runtime / 'extracted/usr/lib/x86_64-linux-gnu')
                       + ':/root/autodl-fs/third_party/graphics_runtime/lib'
                       + ':/root/miniconda3/envs/conceptgraph/lib')
    process_env.pop('PYTHONPATH', None)
    worker = [str(python), 'reproduction/standalone_cpu_runner_004.py', '--manifest', '../manifest.json']
    save(destination / 'prepared.json', dict(
        prepared=True, prepared_utc=datetime.now(timezone.utc).isoformat(),
        command=worker, cwd=str(root), environment_prefix=str(prefix),
        environment_installed_sha256=sha(installed_path), source_files=390,
        runner_sha256=sha(target_runner), wrapper_sha256=sha(__file__),
        declared_environment={key: process_env[key] for key in [
            'CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'LP_NUM_THREADS', 'LD_LIBRARY_PATH', 'TMPDIR', 'MPLCONFIGDIR', 'YOLO_CONFIG_DIR']},
        Python_path_override_absent=True, execution_requested=args.execute,
        actual_frontend_calls=0, complete_navigation_reproduction_executed=False,
        goal_complete=False,
    ))
    if args.execute:
        with (destination / 'worker.log').open('x') as log:
            result = subprocess.run(worker, cwd=root, env=process_env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=960, check=False)
        save(destination / 'launcher_exit.json', dict(returncode=result.returncode,
             task_result_must_be_checked_separately=True, goal_complete=False))
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
