"""顺序启动真实CPU冷导航及零后端调用热导航；失败保留，不自动重试。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)


def main():
    base = OUTPUT/'reproduction/three_task_fixedlogical_001'
    manifest = json.loads((base/'manifest.json').read_text())
    release = json.loads((base/'experiment_release.json').read_text())
    root = Path(manifest['isolated_root'])
    proc = Path('/proc')/str(os.getpid())
    save(base/'launcher_process.json', dict(pid=os.getpid(), start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(__file__), created_utc=datetime.now(timezone.utc).isoformat()))
    runtime = OUTPUT/'source_bundle/software_render_runtime_001'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        LP_NUM_THREADS='1', PYOPENGL_PLATFORM='osmesa', PYGLET_SHADOW_WINDOW='false', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
        ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    try:
        results = {}
        for phase in ['cold', 'warm']:
            for name, expected in release['source_sha256'].items():
                assert sha(PROJECT/name) == sha(root/name) == expected
            command = [sys.executable, str(root/'reproduction/three_task_fixedlogical_worker_001.py'),
                '--manifest', str(base/'manifest.json'), '--experiment-release', str(base/'experiment_release.json'), '--phase', phase]
            with (base/f'{phase}_worker.log').open('x') as log:
                child = subprocess.run(command, cwd=root, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, timeout=3900 if phase == 'cold' else 1500, check=False)
            save(base/f'{phase}_exit.json', dict(returncode=child.returncode, goal_complete=False))
            assert child.returncode == 0, phase+'失败保留，不重试或伪造后续验收'
            path = root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/three_task_fixedlogical_001'/phase/'result.json'
            results[phase] = json.loads(path.read_text())
        assert results['cold']['actual_Qwen_backend_calls'] == 3
        assert results['warm']['actual_Qwen_backend_calls'] == 0 and results['warm']['cache_all_hits']
        save(base/'execution_finished.json', dict(results=results, raw_navigation_executions=2,
            independent_business_comparison_and_geometry_still_required=True, goal_complete=False))
        print(json.dumps(dict(executed=True, cold_calls=3, warm_calls=0,
            independent_audit_pending=True, goal_complete=False)), flush=True)
    except BaseException as exc:
        save(base/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
