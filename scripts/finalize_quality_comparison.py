"""核验显著改善早停交付：原输入、冻结源码、确定性、测试和预算。"""
from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'outputs/reports/recall_config_goal_20261120_v1'
RESULTS = PROJECT / 'outputs/results/nuscenes/validation/iterations'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    comparison = read(REPORT / 'B1_comparison/comparison.json')
    if not comparison['numerical_early_stop_gate']:
        raise ValueError('尚未满足显著改善的数值门槛')
    paths = {b: RESULTS / f'config_quality_{b}_full_20261120_v1' for b in ['B0', 'B1']}
    runs = {b: read(p / 'report.json') for b, p in paths.items()}
    releases = {b: read(p / 'release.json') for b, p in paths.items()}
    checks, verification = {}, {}
    for b, path in paths.items():
        r = runs[b]
        if (r['scenes'], r['frames'], r['logs'], r['result']['counts']['truth']) != (38, 1508, 11, 5723):
            raise ValueError('范围或分母不完整')
        frozen = releases[b]['sources_sha256']
        for source, sha in frozen.items():
            if digest(path / 'source_snapshot' / Path(source).relative_to(PROJECT)) != sha:
                raise ValueError('冻结源码发生变化')
            if Path(source).name != 'run_mapping_quality_comparison.py' and digest(source) != sha:
                raise ValueError('推理或评价代码发生变化')
        for source, sha in read(path / 'input_fingerprints.json').items():
            if digest(source) != sha:
                raise ValueError('输入指纹变化')
        for source, sha in releases[b]['protected_sha256'].items():
            if digest(source) != sha:
                raise ValueError('关键保护文件变化')
        vp = RESULTS / f'config_quality_{b}_verification_20261120_v1'
        v = read(vp / 'report.json')
        if (v['frames'], v['scenes'], v['predictions_exactly_reproduced_every_frame']) != (1508, 38, True):
            raise ValueError('同配置全范围确定性核验未完成')
        for source, sha in v['output_sha256'].items():
            if digest(vp / source) != sha:
                raise ValueError('关联与确认核验输出变化')
        verification[b] = v
    common = releases['B0']['sources_sha256'].keys() & releases['B1']['sources_sha256'].keys()
    common = [p for p in common if Path(p).name != 'run_mapping_quality_comparison.py']
    checks['common_inference_evaluation_source_sha_equal'] = all(
        releases['B0']['sources_sha256'][p] == releases['B1']['sources_sha256'][p] for p in common)
    checks['all_input_sha_equal'] = read(paths['B0'] / 'input_fingerprints.json') == read(paths['B1'] / 'input_fingerprints.json')
    original = RESULTS / 'vehicle_frontend_full_20261118_v1'
    for name, sha in read(original / 'release.json')['weights_sha256'].items():
        if digest(PROJECT / 'checkpoint' / name) != sha:
            raise ValueError('原前端模型权重变化')
    checks['original_frontend_weights_unchanged'] = True
    checks['baseline_per_frame_and_identity_reproduced'] = (runs['B0']['baseline_counts_reproduced']
        and runs['B0']['result']['counts']['IDTP'] == 2040
        and runs['B0']['result']['counts']['predictions'] == 4713
        and runs['B0']['result']['counts']['true_positive'] == 2260)
    checks['full_scope_determinism_and_internal_audit'] = True
    checks['protected_key_fingerprints_unchanged'] = True
    tests = (REPORT / 'related_tests_verified.log').read_text()
    checks['related_tests_64_passed'] = '64 passed' in tests and 'FAILED' not in tests
    sources = [PROJECT / p for p in [
        'conceptgraph/slam/streamlined_mapping.py', 'conceptgraph/slam/utils.py',
        'conceptgraph/slam/audited_mapper.py', 'conceptgraph/slam/quality_mapper.py',
        'conceptgraph/slam/sparse_observations.py', 'conceptgraph/completion/config_comparison.py',
        'scripts/run_mapping_quality_comparison.py', 'scripts/summarize_quality_configuration.py',
        'scripts/verify_mapping_quality_evidence.py', 'scripts/finalize_quality_comparison.py',
        'tests/test_sparse_quality_mapping.py']]
    static = subprocess.run([sys.executable, '-m', 'ruff', 'check', *map(str, sources)],
                            capture_output=True, text=True, cwd=PROJECT)
    diff = subprocess.run(['git', 'diff', '--check'], capture_output=True, text=True, cwd=PROJECT)
    (REPORT / 'static_checks_final.log').write_text(static.stdout + static.stderr + diff.stdout + diff.stderr)
    checks['ruff_passed'] = static.returncode == 0
    checks['git_diff_check_passed'] = diff.returncode == 0
    goal = read(REPORT / 'goal_snapshot.json')['goal']
    elapsed = time.time() - goal['createdAt']
    artifacts = [RESULTS / f'config_quality_{b}_{mode}_20261120_v1'
                 for b in ['B0', 'B1'] for mode in ['pilot', 'full', 'verification']]
    artifact_bytes = sum(p.stat().st_size for root in [*artifacts, REPORT]
                         for p in root.rglob('*') if p.is_file())
    free = shutil.disk_usage(PROJECT).free
    checks['within_7200_second_budget'] = elapsed < 7200
    checks['within_2gib_artifact_budget'] = artifact_bytes + sum(p.stat().st_size for p in sources) < 2 * 1024**3
    checks['minimum_free_20gib'] = free >= 20 * 1024**3
    if not all(checks.values()):
        raise AssertionError(checks)
    snapshot = REPORT / 'source_snapshot'
    for p in sources:
        target = snapshot / p.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    origin_counts = {'B0': {}, 'B1': {}}
    cases = {'car_losses': [], 'pedestrian_gains': []}
    sparse_only_matches = 0
    for scene in releases['B0']['scenes']:
        name = scene['scene']
        with gzip.open(paths['B0'] / name / 'frames.jsonl.gz', 'rt') as a, gzip.open(paths['B1'] / name / 'frames.jsonl.gz', 'rt') as b:
            for old, new in zip(map(json.loads, a), map(json.loads, b)):
                old_matched = {m['truth_index'] for m in old['metrics']['B0']['matches']}
                new_matched = {m['truth_index'] for m in new['metrics']['B1']['matches']}
                for m in new['metrics']['B1']['matches']:
                    pred = new['predictions']['B1'][m['prediction_index']]
                    source = pred.get('position_source', 'original_pointcloud_geometry')
                    origin_counts['B1'][source] = origin_counts['B1'].get(source, 0) + 1
                    sparse_only_matches += source == 'confirmed_sparse_lidar' and m['truth_index'] not in old_matched
                for i, truth in enumerate(old['evaluation_only_official']):
                    kind = ('car_losses' if truth['category'] == 'car' and i in old_matched - new_matched
                            else 'pedestrian_gains' if truth['category'] == 'pedestrian' and i in new_matched - old_matched else None)
                    if kind and len(cases[kind]) < 8:
                        cases[kind].append(dict(scene=name, frame_idx=old['frame_idx'],
                            category=truth['category'], instance=truth['instance_token'],
                            rgb=old['input']['rgb'], official_center=truth['center'],
                            baseline_predictions=old['predictions']['B0'], candidate_predictions=new['predictions']['B1']))
    write(REPORT / 'representative_cases.json', cases)
    versions = {}
    for name in ['numpy', 'scipy', 'torch', 'open3d', 'omegaconf', 'Pillow']:
        versions[name] = metadata.version(name)
    write(REPORT / 'validation_report.json', dict(
        complete=True, stop_reason='significant_improvement', default_changed=False,
        configurations={'B0': '完整复现与核验', 'B1': '显著净收益且必要核验完成',
                        'B2': '提前停止后未实施/未评价', 'B3': '提前停止后未实施/未评价'},
        checks=checks, baseline=runs['B0'], candidate=runs['B1'],
        statistics=comparison['statistics'], delta_percentage_points=comparison['delta_percentage_points'],
        verification=verification, matched_position_sources=origin_counts['B1'],
        new_matches_from_sparse_positions=sparse_only_matches,
        source_attribution_caveat='匹配位置来源统计，不是各子机制独立因果贡献；模块A为整体包',
        experiment_artifacts_bytes=artifact_bytes, free_disk_gib=free / 1024**3,
        elapsed_wall_s=elapsed, total_budget_s=7200,
        finished_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        related_tests='64 passed', full_test_suite_run=False,
        source_sha256={str(p): digest(p) for p in sources},
        library_versions=versions, python=sys.version,
        new_frontend_calls=0, training_runs=0, test_accessed=False, actual_arrival_evaluated=False,
        limitations=['验证/调试集结果，不是独立测试泛化', '汽车、公交、卡车和近距离召回有回退',
                    '重复节点由92增至116', 'F1和IDF1的校正区间含0，主召回通过校正门槛',
                    '稀疏位置节点不代表完成点云形状重建，默认和导航锁定策略未改变',
                    '未比较B2/B3，不能声称B1是全局最优', '未重新核验全部历史文件的SHA']))
    print(json.dumps(dict(complete=True, checks=checks, elapsed_wall_s=elapsed,
                         artifact_mib=artifact_bytes / 1024**2)), flush=True)


if __name__ == '__main__':
    main()
