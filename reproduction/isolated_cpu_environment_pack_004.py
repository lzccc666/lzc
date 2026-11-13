"""锁定并复制CPU复演依赖；安装仅在原GPU队列终止且存储门槛满足后执行。"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
ORIGINALS = {
    268029: '1379768112', 38569: '1375457897', 288085: '1380200568',
    199838: '1378262585', 211077: '1378476126',
}
WHEELS = [
    dict(name='torch', version='2.0.1+cpu',
         url='https://download-r2.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp310-cp310-linux_x86_64.whl',
         sha256='fec257249ba014c68629a1994b0c6e7356e20e1afc77a87b9941a40e5095285d'),
    dict(name='torchvision', version='0.15.2+cpu',
         url='https://download-r2.pytorch.org/whl/cpu/torchvision-0.15.2%2Bcpu-cp310-cp310-linux_x86_64.whl',
         sha256='aae0be6883d2cd5a23cb544ee0928288a27df0455430ef9dd6e631c5464095f5'),
]


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


def original_live(pid, tick):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == tick
    except FileNotFoundError:
        return False


def allowed_relative(path):
    return (not path.is_absolute() and '..' not in path.parts
            and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pth'}
            and not path.parts[0].startswith('__editable__'))


def prepare(destination):
    assert not destination.exists()
    original = OUTPUT / 'reproduction/isolated_cpu_environment_002/plan.json'
    plan = json.loads(original.read_text())
    site = Path(plan['original_site_packages'])
    files = {r['destination']: r for r in plan['files']}
    for name in ['torch', 'torchvision']:
        version = metadata.version(name)
        plan['versions'][name] = version
        for top in [site / name, site / (name + '.libs'), site / (name + '-' + version + '.dist-info')]:
            if not top.exists():
                continue
            for item in sorted(top.rglob('*')):
                relative = item.relative_to(site)
                if item.is_file() and allowed_relative(relative):
                    files[str(relative)] = dict(source=str(item), destination=str(relative), bytes=item.stat().st_size)
    assert plan['versions']['torch'] == '2.0.1+cu118'
    assert plan['versions']['torchvision'] == '0.15.2+cu118'
    plan.update(prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha(__file__), inherited_frozen_plan=str(original),
        inherited_frozen_plan_sha256=sha(original), wheels=[],
        files=sorted(files.values(), key=lambda row: row['destination']),
        file_count=len(files), copy_bytes=sum(row['bytes'] for row in files.values()),
        maximum_package_bytes=7_000_000_000,
        dependency_policy='从本机已验证依赖复制到新venv；不读取原site-packages；保留CUDA构建但仅CPU执行，CUDA_VISIBLE_DEVICES为空且初始化状态必须为false',
        CPU_only_binary=False, network_downloads_required=False,
        previous_failed_environments_preserved=True)
    assert plan['copy_bytes'] < 6_500_000_000
    destination.mkdir()
    save(destination / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, files=len(files), bytes=plan['copy_bytes'], goal_complete=False)), flush=True)


def install(destination):
    plan_path = destination / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert sha(__file__) == plan['source_script_sha256']
    assert not any(original_live(int(pid), tick) for pid, tick in plan['original_process_start_ticks'].items()), '原GPU队列仍活跃，不安装'
    assert shutil.disk_usage(destination).free >= plan['minimum_free_bytes_before_install']
    assert not (destination / 'environment').exists(), '失败环境保留，不能覆盖重试'
    for name, version in plan['versions'].items():
        assert metadata.version(name) == version, name
    begun = time.monotonic()
    environment = destination / 'environment'
    temporary = destination / 'tmp'
    temporary.mkdir()
    process_env = dict(os.environ, TMPDIR=str(temporary), PYTHONDONTWRITEBYTECODE='1',
                       PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONNOUSERSITE='1')
    process_env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-m', 'venv', '--copies', '--without-pip', str(environment)],
                   env=process_env, check=True, timeout=60)
    python = environment / 'bin/python'
    assert python.resolve().is_relative_to(environment)
    site = environment / 'lib/python3.10/site-packages'
    assert site.is_dir()
    copied = 0
    with (destination / 'copied_files.jsonl').open('x') as ledger:
        for index, record in enumerate(plan['files']):
            relative = Path(record['destination'])
            assert allowed_relative(relative)
            source, target = Path(record['source']), site / relative
            assert source.stat().st_size == record['bytes']
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            source_sha = sha(source)
            shutil.copy2(source, target)
            assert sha(target) == source_sha
            ledger.write(json.dumps(dict(**record, sha256=source_sha)) + '\n')
            copied += record['bytes']
            if index % 100 == 0:
                ledger.flush()
                assert shutil.disk_usage(destination).free >= plan['minimum_free_bytes_during_install']
                assert time.monotonic() - begun < plan['maximum_install_wall_s']
                assert copied < plan['maximum_package_bytes']
    assert not list(site.glob('*.pth')), '环境不得通过pth加载原站点或editable finder'
    check = (
        'import json,sys,torch,torchvision,segment_anything;'
        'from pathlib import Path;'
        'assert sys.prefix != sys.base_prefix;'
        "assert torch.version.cuda == '11.8';"
        'assert not torch.cuda.is_initialized();'
        'assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix)) '
        'for m in [torch,torchvision,segment_anything]);'
        'print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,'
        'torch=torch.__version__,torchvision=torchvision.__version__,'
        'segment_anything=segment_anything.__file__,CUDA_initialized=False)))'
    )
    probe_env = dict(process_env, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                     OMP_NUM_THREADS='1', LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib')
    result = subprocess.run([str(python), '-I', '-c', check], cwd=destination,
                            env=probe_env, capture_output=True, text=True, timeout=90)
    save(destination / 'interpreter_probe.json', dict(returncode=result.returncode,
         stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0
    installed_bytes = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    assert installed_bytes < plan['maximum_package_bytes']
    save(destination / 'installed.json', dict(
        installed=True, plan_sha256=sha(plan_path), copied_files=plan['file_count'],
        copied_bytes=copied, environment=str(environment), python=str(python),
        interpreter_probe=json.loads(result.stdout), saved_bytes=installed_bytes,
        elapsed_s=time.monotonic() - begun, original_environment_modified=False,
        original_site_packages_in_sys_path=False, editable_finders_copied=False,
        original_conda_standard_library_and_native_system_libraries_shared=True,
        complete_navigation_reproduction_executed=False, actual_frontend_calls=0,
        actual_model_calls=0, goal_complete=False,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    destination = args.destination.resolve()
    assert destination.is_relative_to(OUTPUT / 'reproduction')
    if args.install:
        try:
            install(destination)
        except BaseException as error:
            if destination.is_dir() and not (destination / 'install_failed.json').exists():
                save(destination / 'install_failed.json', dict(error=repr(error), goal_complete=False))
            raise
    else:
        prepare(destination)


if __name__ == '__main__':
    main()
