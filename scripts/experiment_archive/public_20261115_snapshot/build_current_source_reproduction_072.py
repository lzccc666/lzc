"""冻结实际源码并独立展开，重放旧开发证据；不打开正在运行的新测试答案。"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

from conceptgraph.completion.quality import object_frame_metrics
from public_query_intake_055 import choose, strong_rule

PROJECT = Path('/root/autodl-tmp/lizichong')
OLD = PROJECT/'outputs/project_v2/thesis_completion_v1'
ROOT = OLD/'public_experiment_completion_20261115'
OUT = ROOT/'source_bundle/reproduction_072'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


SMOKE = '''"""只读取随包旧开发派生证据，无原始传感器、权重或GPU请求。"""
import argparse
import hashlib
import json
from pathlib import Path

from conceptgraph.completion.quality import object_frame_metrics
import conceptgraph.completion.quality as quality
from public_query_intake_055 import choose, strong_rule
import public_query_intake_055 as intake

parser = argparse.ArgumentParser()
parser.add_argument('--checkout', required=True)
parser.add_argument('--inputs', required=True)
args = parser.parse_args()
checkout = Path(args.checkout).resolve()
assert Path(quality.__file__).resolve().is_relative_to(checkout)
assert Path(intake.__file__).resolve().is_relative_to(checkout)
inputs = json.loads(Path(args.inputs).read_text())
results = dict(mapping=[], decisions=[])
for case in inputs['mapping']:
    results['mapping'].append(object_frame_metrics(case['predictions'], case['truth']))
for case in inputs['decisions']:
    results['decisions'].append(dict(
        cooperation=choose(case['query_text'], case['candidate_nodes'], case['prediction']),
        strong_rule=strong_rule(case['query_text'], case['candidate_nodes'], relation_radius_m=15.)))
assert results == inputs['expected'], '独立展开后的业务结果不同'
canonical = json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
print(json.dumps(dict(mapping_frames=len(results['mapping']),
    actual_CPU_cooperation_decisions=len(results['decisions']),
    actual_CPU_strong_rule_decisions=len(results['decisions']),
    business_results_equal=True, results_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    source_modules=[quality.__file__, intake.__file__],
    actual_GPU_calls=0, actual_CARLA_episodes=0, new_accuracy_credit=False)))
'''


def main():
    OUT.mkdir(exist_ok=False)
    selected = []
    for folder in [PROJECT/'conceptgraph', PROJECT/'scripts', PROJECT/'tests', ROOT/'source']:
        selected += [p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
            and '__pycache__' not in p.parts and (p.suffix in {'.py', '.yaml', '.yml', '.txt', '.md', '.json'}
            or p.name in {'LICENSE', 'NOTICE'})]
    selected += [PROJECT/name for name in ['README.md', 'README_nuscenes_minimal.md',
        'README_streamlined_detection.md', 'setup.py', 'environment.yml', 'LICENSE'] if (PROJECT/name).is_file()]
    selected = sorted(set(selected))
    fingerprints = {str(p.relative_to(PROJECT)): sha(p) for p in selected}
    release = ROOT/'main_experiments/sealed_public_058/release.json'
    protocol = dict(UTC=datetime.now(timezone.utc).isoformat(), purpose='当前实际代码交付快照和旧开发证据独立源码重放',
        file_count=len(selected), file_hashes=fingerprints, source_release_sha256=sha(release),
        old_inputs='3个旧开发建图帧、108条旧开发真实感知图请求及其已有真实模型输出',
        new_sealed_ground_truth_opened=False, actual_GPU_budget=0, actual_CARLA_budget=0,
        maximum_new_output_MiB=64, clean_environment_install=False, existing_dependencies_reused=True,
        no_original_sensor_or_weight_copies=True, goal_complete=False)
    write(OUT/'protocol.json', protocol)
    with tarfile.open(OUT/'actual_source_snapshot.tar.gz', 'x:gz') as archive:
        for path in selected:
            archive.add(path, arcname=str(path.relative_to(PROJECT)), recursive=False)
    assert all(sha(PROJECT/name) == expected for name, expected in fingerprints.items())
    checkout = OUT/'checkout'
    checkout.mkdir()
    with tarfile.open(OUT/'actual_source_snapshot.tar.gz') as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        archive.extractall(checkout)
    assert all(sha(checkout/name) == expected for name, expected in fingerprints.items())
    evidence = dict(mapping=[], decisions=[], expected=dict(mapping=[], decisions=[]))
    input_hashes = {}
    for scene in ['scene-0012', 'scene-0013', 'scene-0014']:
        path = OLD/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl'
        input_hashes[str(path)] = sha(path)
        with path.open() as f:
            frame = next(row for row in map(json.loads, f)
                if row['predictions']['one_to_one_depth_robust_motion'] and row['evaluation_only_official'])
        case = dict(predictions=frame['predictions']['one_to_one_depth_robust_motion'],
                    truth=frame['evaluation_only_official'])
        evidence['mapping'].append(case)
        evidence['expected']['mapping'].append(object_frame_metrics(case['predictions'], case['truth']))
    data_path = ROOT/'qwen_adapter_fair_comparison/real_graph_dataset_003/development_holdout.jsonl'
    model_path = ROOT/'qwen_adapter_fair_comparison/real_graph_fair_gpu_001/real_graph_seed17/raw_records.jsonl'
    for path in [data_path, model_path]:
        input_hashes[str(path)] = sha(path)
    predictions = {r['sample_id']: r for r in map(json.loads, model_path.open())}
    for row in map(json.loads, data_path.open()):
        raw = predictions[row['sample_id']]
        prediction = {key: raw[key] for key in ['schema_valid', 'normalized_output']}
        case = dict(query_text=row['query_text'], candidate_nodes=row['candidate_nodes'], prediction=prediction)
        evidence['decisions'].append(case)
        evidence['expected']['decisions'].append(dict(
            cooperation=choose(case['query_text'], case['candidate_nodes'], prediction),
            strong_rule=strong_rule(case['query_text'], case['candidate_nodes'], relation_radius_m=15.)))
    assert len(evidence['decisions']) == 108
    write(OUT/'saved_development_evidence.json', evidence)
    with (OUT/'run_saved_evidence_smoke.py').open('x') as f:
        f.write(SMOKE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONPATH=str(checkout)+':'+str(checkout/ROOT.relative_to(PROJECT)/'source'))
    command = [sys.executable, str(OUT/'run_saved_evidence_smoke.py'),
               '--checkout', str(checkout), '--inputs', str(OUT/'saved_development_evidence.json')]
    process = subprocess.run(command, cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    for name, content in [('stdout.log', process.stdout), ('stderr.log', process.stderr)]:
        with (OUT/name).open('x') as f:
            f.write(content)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['mapping_frames'] == 3 and result['actual_CPU_cooperation_decisions'] == 108
    packages = {}
    for name in ['torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'ultralytics',
                 'open-clip-torch', 'open3d', 'opencv-python', 'numpy', 'scipy', 'numba', 'filterpy',
                 'lap', 'nuscenes-devkit', 'omegaconf', 'shapely', 'scikit-learn', 'supervision',
                 'Pillow', 'pydantic', 'openpyxl', 'pytest', 'ruff']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    write(OUT/'runtime_environment.json', dict(python=platform.python_version(),
        python_executable=sys.executable, platform=platform.platform(), packages=packages,
        wheel_reinstallation_tested=False))
    write(OUT/'receipt.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        source_count=len(selected), archive_sha256=sha(OUT/'actual_source_snapshot.tar.gz'),
        source_protocol_sha256=sha(OUT/'protocol.json'), extracted_sources_verified=True,
        actual_source_import_paths_verified=True, smoke=result,
        saved_evidence_sha256=sha(OUT/'saved_development_evidence.json'),
        original_development_inputs_sha256=input_hashes,
        protected_development_inputs_unchanged=all(sha(p) == h for p, h in input_hashes.items()),
        source_files_unchanged=all(sha(PROJECT/name) == expected for name, expected in fingerprints.items()),
        actual_GPU_calls=0, actual_CARLA_episodes=0, clean_environment_passed=False,
        scope='源码独立展开及旧开发业务重放；复用现有依赖，不是完整传感器或新环境复现',
        goal_complete=False))
    with (OUT/'复现说明.md').open('x') as f:
        f.write('# 当前代码复现包\n\n源码包包含当前conceptgraph、scripts、tests以及本批source、配置和上游许可证，按项目根相对路径展开；protocol.json列全部指纹。它保留旧输出路径布局，不能只用Git HEAD代替本包。没有原始传感器或模型权重。\n\n')
        f.write('已在checkout独立展开并逐文件核验，从展开源码重算3个旧开发建图帧、108次合作决策及108次强规则；与当前源码的相同输入输出逐项一致。只说明源码与业务重放一致，不计新GPU、正确率、独立场景或到达。不是新环境安装；runtime_environment.json是当前实际版本清单。\n\n')
        f.write('在相同依赖环境中，可将项目源码展开到任意目录，以其为PYTHONPATH首项，并加上展开目录下outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/source。执行随包run_saved_evidence_smoke.py --checkout <展开目录> --inputs <saved_development_evidence.json>。本小试不依赖原项目数据或权重，但完整runner仍按各release声明的数据/权重绝对位置运行；不要把它描述成完整工程已可任意路径迁移。\n\n')
        f.write('当前主封存运行由现有76738会话监督，不在这里启动。运行过的固定路径会拒绝覆盖；不要重启旧批次。CARLA保持用户指定首10回合，不因复现包开启服务。当前项目产物上限25GiB、物理空闲下限20GiB；未来大传感器存共享卷的授权仍有效。旧10GiB文档作为历史保留。\n\n')
        f.write('本包尚未进行Git提交/推送，也未将整个Goal标为完成。后续以新封存结果及最终验收清单继续补齐。\n')
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 64*1024**2
    print(json.dumps(dict(source_count=len(selected), smoke=result, goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
