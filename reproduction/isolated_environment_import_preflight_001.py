"""只读探测完整CPU导航的导入来源，供新环境封装；不计实际导航复现。"""
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    output = Path(sys.argv[1]).resolve()
    source = Path(sys.argv[2]).resolve()
    assert output.is_dir() and source.is_dir()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    original = Path('/root/autodl-tmp/lizichong')
    sys.path[:] = [str(source)] + [
        p for p in sys.path if p and Path(p).resolve() not in {original, source}
    ]
    imported = [
        'conceptgraph.completion.live_navigation_worker',
        'conceptgraph.completion.interactive_navigation',
        'conceptgraph.scripts.streamlined_detections',
        'segment_anything',
    ]
    for name in imported:
        importlib.import_module(name)
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    module_files = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            resolved = Path(path).resolve()
            if name == 'conceptgraph' or name.startswith('conceptgraph.'):
                assert resolved.is_relative_to(source), (name, str(resolved))
            module_files[name] = dict(path=str(resolved), sha256=sha(resolved))
    ownership = metadata.packages_distributions()
    distribution_names = sorted({
        owner
        for name in module_files
        for owner in ownership.get(name.split('.')[0], [])
    })
    distributions = {}
    for name in distribution_names:
        dist = metadata.distribution(name)
        distributions[name] = dict(
            version=dist.version,
            root=str(Path(dist.locate_file('')).resolve()),
            requires=dist.requires or [],
        )
    site_packages = Path(torch.__file__).resolve().parents[1]
    external = {
        name: record for name, record in module_files.items()
        if not Path(record['path']).is_relative_to(site_packages)
        and not Path(record['path']).is_relative_to(source)
        and 'site-packages' in Path(record['path']).parts
    }
    report = dict(
        passed=True, time_utc=datetime.now(timezone.utc).isoformat(),
        imported_entry_modules=imported, source_root=str(source),
        interpreter=str(Path(sys.executable).resolve()),
        current_environment_site_packages=str(site_packages),
        module_files=module_files, distributions=distributions,
        external_site_package_modules=external,
        CUDA_initialized=False, actual_frontend_calls=0,
        actual_navigation_executions=0,
        independent_new_Python_environment_executed=False,
        original_environment_reused_for_import_probe=True,
        original_environment_modified=False, goal_complete=False,
    )
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(
        passed=True, modules=len(module_files), distributions=len(distributions),
        external_modules=len(external), CUDA_initialized=False,
        independent_new_Python_environment_executed=False,
    )), flush=True)


if __name__ == '__main__':
    main()
