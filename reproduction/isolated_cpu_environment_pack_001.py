"""锁定并复制CPU复演依赖；安装仅在原GPU队列终止且存储门槛满足后执行。"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urldefrag, urljoin
from urllib.request import urlopen


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
    return (
        not path.is_absolute() and '..' not in path.parts
        and '__pycache__' not in path.parts and path.suffix != '.pyc'
        and 'cuda' not in path.parts and path.parts[0] not in {'nvfuser', 'nvidia'}
        and not path.parts[0].startswith(('__editable__', 'torch', 'nvidia'))
        and path.suffix != '.pth'
    )


def prepare(destination):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    assert not destination.exists()
    probe = OUTPUT / 'reproduction/isolated_environment_import_preflight_001/report.json'
    report = json.loads(probe.read_text())
    assert report['passed'] and not report['CUDA_initialized']
    site = Path(report['current_environment_site_packages'])
    seeds = list(report['distributions']) + [
        'open-clip-torch', 'supervision', 'hydra-core', 'omegaconf', 'gradslam',
        'faiss-cpu', 'pyrender', 'trimesh', 'shapely', 'safetensors',
        'ftfy', 'regex', 'tqdm', 'matplotlib', 'scikit-learn', 'scipy',
        'pyquaternion', 'kornia', 'numpy', 'polars', 'ultralytics-thop',
    ]
    selected, missing, violations = {}, [], []
    excluded = {'torch', 'torchvision', 'triton', 'conceptgraph', 'pytorch3d'}
    pending = seeds[:]
    while pending:
        name = canonicalize_name(pending.pop())
        if name in selected or name in excluded or name.startswith('nvidia-'):
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        selected[name] = dist
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency in excluded or dependency.startswith('nvidia-'):
                continue
            try:
                version = metadata.version(dependency)
                if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                    violations.append(dict(package=name, requirement=str(requirement), installed=version))
            except metadata.PackageNotFoundError:
                pass
            pending.append(dependency)
    files = {}

    def add(path, relative):
        if allowed_relative(relative) and path.is_file():
            target = str(relative)
            record = dict(source=str(path), destination=target, bytes=path.stat().st_size)
            if target in files:
                assert files[target] == record
            files[target] = record

    for dist in selected.values():
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry))
            try:
                relative = path.relative_to(site)
            except ValueError:
                continue
            add(path, relative)
    # 包拥有者元数据可能不完整；补齐实际导入过的整个顶层包，保留其数据文件。
    tops = set()
    for record in report['module_files'].values():
        path = Path(record['path'])
        if path.is_relative_to(site):
            relative = path.relative_to(site)
            if allowed_relative(relative):
                tops.add(relative.parts[0])
    for name in sorted(tops):
        path = site / name
        if path.is_dir():
            for item in sorted(path.rglob('*')):
                add(item, item.relative_to(site))
        else:
            add(path, path.relative_to(site))
    vendor = Path(report['module_files']['segment_anything']['path']).parent
    assert not vendor.is_relative_to(site)
    for item in sorted(vendor.rglob('*')):
        add(item, Path('segment_anything') / item.relative_to(vendor))
    assert 'segment_anything/__init__.py' in files
    assert all(not record['destination'].startswith(('torch/', 'torchvision/')) for record in files.values())
    total = sum(record['bytes'] for record in files.values())
    assert total < 3_000_000_000, total
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attributes):
            if tag == 'a':
                self.links.extend(value for key, value in attributes if key == 'href')

    wheel_indexes = []
    for wheel in WHEELS:
        url = 'https://download.pytorch.org/whl/cpu/' + wheel['name'] + '/'
        with urlopen(url, timeout=30) as response:
            content = response.read(8_000_001)
        assert len(content) <= 8_000_000
        parser = Links()
        parser.feed(content.decode())
        expected = wheel['url'] + '#sha256=' + wheel['sha256']
        assert expected in [urljoin(url, link) for link in parser.links]
        wheel_indexes.append(dict(url=url, sha256=hashlib.sha256(content).hexdigest(),
                                  bytes=len(content), checked_utc=datetime.now(timezone.utc).isoformat()))
    destination.mkdir()
    plan = dict(
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha(__file__), probe_sha256=sha(probe),
        original_environment=str(Path(sys.prefix).resolve()), original_site_packages=str(site),
        versions={name: dist.version for name, dist in sorted(selected.items())},
        missing_metadata_requirements=sorted(set(missing)),
        existing_requirement_violations=violations,
        requirement_violations_are_not_silently_repaired=True,
        files=sorted(files.values(), key=lambda row: row['destination']),
        copy_bytes=total, file_count=len(files), wheels=WHEELS,
        wheel_source_indexes=wheel_indexes,
        original_process_start_ticks=ORIGINALS,
        minimum_free_bytes_before_install=11_000_000_000,
        minimum_free_bytes_during_install=6_000_000_000,
        maximum_package_bytes=5_000_000_000, maximum_install_wall_s=2400,
        vendor_source=str(vendor), source_model_weights_copied=False,
        new_environment_installed=False, actual_navigation_executions=0, goal_complete=False,
    )
    save(destination / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, files=len(files), copy_bytes=total,
                          missing=plan['missing_metadata_requirements'],
                          existing_requirement_violations=violations,
                          new_environment_installed=False)), flush=True)


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
    wheels = destination / 'wheels'
    wheels.mkdir()
    wheel_paths = []
    for wheel in plan['wheels']:
        name = wheel['url'].rsplit('/', 1)[1].replace('%2B', '+')
        path = wheels / name
        assert urldefrag(wheel['url'])[0].startswith('https://download-r2.pytorch.org/')
        received = 0
        with urlopen(wheel['url'], timeout=45) as response, path.open('xb') as stream:
            while block := response.read(1024 * 1024):
                received += len(block)
                assert received < 240_000_000
                assert time.monotonic() - begun < plan['maximum_install_wall_s']
                stream.write(block)
        assert sha(path) == wheel['sha256'], path
        wheel_paths.append(str(path))
    with (destination / 'pip_install.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                        '--no-compile', '--target', str(site)] + wheel_paths,
                       env=process_env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=600)
    assert not list(site.glob('*.pth')), '环境不得通过pth加载原站点或editable finder'
    check = (
        'import json,sys,torch,torchvision,segment_anything;'
        'from pathlib import Path;'
        'assert sys.prefix != sys.base_prefix;'
        'assert torch.version.cuda is None;'
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
