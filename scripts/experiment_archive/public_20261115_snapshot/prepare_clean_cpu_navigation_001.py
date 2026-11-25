"""恢复封存源码到新checkout，准备CPU依赖隔离的官方物理导航实跑。"""
import ast
import datetime
import difflib
import json
from pathlib import Path
import shutil
import tarfile

from conceptgraph.completion.stage9 import PROJECT, digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'reproduction/clean_cpu_navigation_001'
    out.mkdir(exist_ok=False)
    envroot = R/'reproduction/clean_cpu_install_001'
    assert json.loads((envroot/'dependencies_report_003.json').read_text())['clean_venv_dependency_install_and_imports_passed']
    archive = R/'source_bundle/reproduction_003/actual_source_snapshot.tar.gz'
    receipt = json.loads((archive.parent/'receipt.json').read_text())
    assert digest(archive) == receipt['archive_sha256']
    checkout = out/'checkout'
    checkout.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        tar.extractall(checkout)
    manifest = json.loads((archive.parent/'source_manifest.json').read_text())
    assert all(digest(checkout/x['archive_path'])==x['sha256'] for x in manifest)
    embedded = checkout/R.relative_to(PROJECT)/'source'
    embedded.mkdir(parents=True,exist_ok=True)
    supplements = []
    for source in sorted((R/'source').glob('*.py')):
        dest = embedded/source.name
        if dest.exists():
            assert digest(dest)==digest(source), '不得用当前文件覆盖已有归档差异：'+source.name
        else:
            shutil.copyfile(source,dest)
        supplements.append(dict(source=str(source), checkout=str(dest), sha256=digest(dest)))
    (checkout/'checkpoint').symlink_to(PROJECT/'checkpoint',target_is_directory=True)
    old = R/'source/carla_prius_navigation_episode_005.py'
    original = old.read_text()
    prefix = '''import os as _os, sys as _sys
_isolated = _os.environ['THESIS_ISOLATED_CHECKOUT']
_embedded = _os.environ['THESIS_ISOLATED_SOURCE']
_live_source = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [_isolated, _embedded] + [p for p in _sys.path if p != _live_source]
assert not any('/envs/conceptgraph/lib/python3.10/site-packages' in p for p in _sys.path)
assert _sys.prefix != _sys.base_prefix
'''
    derived = prefix+original.replace('prius_navigation_registration_005.json','clean_cpu_navigation_registration_001.json').replace('prius_controller_navigation_005','clean_cpu_navigation_001')
    derived = derived.replace("cfg=SimpleNamespace(device='cuda:0')","cfg=SimpleNamespace(device='cpu')")
    derived = derived.replace("model.to('cuda:0');model.overrides['device']='cuda:0'","model.to('cpu');model.overrides['device']='cpu'")
    derived = derived.replace('maximum_wall_s=600','maximum_wall_s=2400').replace('time.monotonic()-started>600','time.monotonic()-started>2400')
    key = "finally:\n  for a in reversed(actors):"
    assert key in derived
    derived = derived.replace(key,"""finally:
  import sys,importlib
  module_rows=[]
  for name in ['numpy','scipy','torch','torchvision','open3d','cv2','hydra','omegaconf','ultralytics','supervision','open_clip','transformers','peft','gradslam','clip','segment_anything','carla']:
   m=importlib.import_module(name);module_rows.append(dict(name=name,path=str(Path(m.__file__).resolve()),version=str(getattr(m,'__version__','unknown'))))
  isolation=all(Path(m['path']).is_relative_to(Path(sys.prefix).resolve()) for m in module_rows)
  dump(OUT/'fresh_environment_modules.json',dict(prefix=sys.prefix,base_prefix=sys.base_prefix,modules=module_rows,third_party_modules_isolated=isolation,original_core_module=str(Path(nav.__file__).resolve()),isolated_checkout=_isolated,actual_torch_CUDA=False,goal_complete=False))
  assert isolation
  for a in reversed(actors):""")
    ast.parse(derived)
    runtime = R/'source/carla_clean_cpu_navigation_episode_001.py'
    runtime.open('x').write(derived)
    (out/'source_delta.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),derived.splitlines(True),fromfile=str(old),tofile=str(runtime))))
    bank = json.loads((R/'carla_official/prius_navigation_registration_005.json').read_text())
    bank['episodes'] = [x for x in bank['episodes'] if x['episode']=='car20']
    assert len(bank['episodes'])==1
    paths = [runtime, Path(__file__), envroot/'installed_freeze_004.txt', envroot/'explicit_runtime_dependency_addendum_004.json']
    bank['source_sha256'].update({str(x):digest(x) for x in paths})
    bank.update(device='cpu', planned_episodes=1, checkout=str(checkout), embedded_source=str(embedded),
        runtime_python=str(envroot/'venv/bin/python'), maximum_wall_s=2400,
        reproduction_scope='新安装CPU依赖+新恢复源码的完整规则感知/建图/锁定/规划/控制/官方车辆物理导航；非GPU数值或实时性能复现，Qwen复杂闭环另报',
        maximum_reserve_per_episode_bytes=1*2**30, source_archive_sha256=digest(archive),
        base_source_files=len(manifest), source_overlays=supplements, whole_project_sealed_test=False,
        goal_complete=False)
    registration = R/'carla_official/clean_cpu_navigation_registration_001.json'
    dump(registration,bank)
    dump(out/'preparation_receipt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registration_sha256=digest(registration), original_archive_files_readback_passed=len(manifest),
        supplemental_source_files=len(supplements), new_checkout=str(checkout), runtime=str(runtime),
        existing_environment_or_results_modified=False, goal_complete=False))
    print(json.dumps(dict(registration=str(registration),original_source_files=len(manifest),
        overlay_sources=len(supplements)),ensure_ascii=False))


if __name__ == '__main__':
    main()
