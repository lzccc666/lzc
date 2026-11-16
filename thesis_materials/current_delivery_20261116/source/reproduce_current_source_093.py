"""当前源码打包、独立展开并运行全部软件测试；复用依赖，不启动模型或CARLA。"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

PROJECT = Path('/root/autodl-tmp/lizichong')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'source_bundle/reproduction_093'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    begun = time.monotonic()
    spec = importlib.util.spec_from_file_location('delivery_preflight_093', ROOT/'source/prepare_git_delivery_077.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT/'preflight'
    module.main()
    report = json.loads((module.OUT/'report.json').read_text())
    assert report['ready_for_later_code_only_snapshot']
    hashes = json.loads((module.OUT/'source_files.json').read_text())
    archive = OUT/'当前项目源码_093.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name, expected in sorted(hashes.items()):
            assert sha(PROJECT/name) == expected
            tar.add(PROJECT/name, arcname=name, recursive=False)
    checkout = OUT/'checkout'
    checkout.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == len(hashes)
        for item in members:
            relative = Path(item.name)
            assert item.isfile() and not relative.is_absolute() and '..' not in relative.parts
        tar.extractall(checkout, members=members)
    assert all(sha(checkout/name) == expected for name, expected in hashes.items())
    source = checkout/ROOT.relative_to(PROJECT)/'source'
    work = OUT/'software_tests'
    work.mkdir(exist_ok=False)
    extra = sorted(str(p) for p in source.glob('test_*.py'))
    args = ['tests', *extra, '-q', '--basetemp='+str(work/'tmp'), '-o',
            'cache_dir='+str(work/'pytest_cache'), '--junitxml='+str(work/'junit.xml')]
    # 在同一解释器内检查实际导入路径，并于测试后保留模块来源，不能只改cwd声称独立复现。
    runner = '''import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
proof=pathlib.Path(sys.argv[2])
assert pathlib.Path(importlib.util.find_spec('conceptgraph').origin).resolve().is_relative_to(root)
import pytest
result=pytest.main(sys.argv[3:])
origins={}
for name,module in list(sys.modules.items()):
 path=getattr(module,'__file__',None)
 if name=='conceptgraph' or name.startswith('conceptgraph.'):
  if path: origins[name]=str(pathlib.Path(path).resolve())
assert origins and all(pathlib.Path(p).is_relative_to(root) for p in origins.values())
proof.write_text(json.dumps(dict(returncode=result,project_module_origins=origins,all_project_modules_from_snapshot=True),ensure_ascii=False,indent=2))
raise SystemExit(result)
'''
    api = Path('/autodl-fs/data/third_party/carla_0.9.16/python_site')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, [checkout, source, api])),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               LD_PRELOAD='/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6')
    command = ['/root/miniconda3/envs/conceptgraph/bin/python', '-B', '-c', runner,
               str(checkout), str(work/'module_origins.json'), *args]
    write(OUT/'protocol.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
          command=command, source_files=len(hashes), source_manifest_sha256=sha(module.OUT/'source_files.json'),
          archive_sha256=sha(archive), budget_s=600, extracted_source_hashes_verified=True,
          existing_dependencies_reused=True, dependency_install_performed=False,
          GPU_hidden=True, CARLA_python_API_only_path=str(api), CARLA_server_started=False,
          source_dataset_or_sensor_copies=False, goal_complete=False))
    with (work/'pytest.log').open('x') as log:
        process = subprocess.run(command, cwd=checkout, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=600)
    source_unchanged = all(sha(PROJECT/name) == expected for name, expected in hashes.items())
    extracted_unchanged = all(sha(checkout/name) == expected for name, expected in hashes.items())
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), returncode=process.returncode,
                  source_count=len(hashes), archive_sha256=sha(archive),
                  source_files_unchanged=source_unchanged, extracted_sources_unchanged=extracted_unchanged,
                  module_origin_proof=str(work/'module_origins.json'), elapsed_s=time.monotonic()-begun,
                  actual_GPU_calls=0, actual_new_CARLA_episodes=0,
                  clean_environment_install_passed=False,
                  scope='最新源码独立展开及全部软件测试；复用既有依赖，不是另一操作系统完整安装或全传感器重跑',
                  goal_complete=False)
    write(OUT/'receipt.json', result)
    print(json.dumps(result), flush=True)
    assert process.returncode == 0 and source_unchanged and extracted_unchanged


if __name__ == '__main__':
    main()
