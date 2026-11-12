"""固定真实前端的动态维护/过期策略消融，复用已核验完整条件、实际重建其余条件。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.fault_robustness import mapping_config, run_case
from conceptgraph.completion.fault_statistics import METRICS, finite, ratios
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump


SETTINGS = {'full': {}, 'without_dynamic_maintenance': {'enable_dynamic_object_handling': False},
            'without_short_term_expiry': {'dynamic_ttl_s': 1e9}}


def statistics(source, release, rows):
    logs = sorted(set(release['scene_logs'].values()))
    conditions = list(SETTINGS)
    totals = np.zeros((len(logs), len(conditions), 7))
    for row in rows:
        p = Path(row['report_path'])
        if digest(p) != row['report_sha256']:
            raise ValueError('mapping module report changed')
        report = json.loads(p.read_text())
        if digest(p.parent/'frames.jsonl') != report['records_sha256']:
            raise ValueError('mapping module raw observations changed')
        m = report['metrics']
        tp = m['true_positive']
        totals[logs.index(row['log_token']), conditions.index(row['condition'])] += [
            tp, m['truth'], m['predictions'], tp*(m['center_error_m_mean'] or 0),
            m['identity_switches'], m['adjacent_matched_transitions'], m['duplicate_nodes']]
    rng = np.random.default_rng(17)
    points = ratios(totals.sum(axis=0))
    samples = ratios(totals[rng.integers(0, len(logs), (4000, len(logs)))].sum(axis=1))
    estimates, differences = {}, []
    for j, condition in enumerate(conditions):
        estimates[condition] = {}
        for k, metric in enumerate(METRICS):
            v = samples[:, j, k]
            v = v[np.isfinite(v)]
            estimates[condition][metric] = dict(estimate=finite(points[j, k]),
                ci95=np.quantile(v, [.025, .975]).tolist() if len(v) else None, valid_resamples=len(v))
            if j:
                delta = samples[:, j, k]-samples[:, 0, k]
                delta = delta[np.isfinite(delta)]
                differences.append(dict(condition=condition, baseline='full', metric=metric,
                    difference=finite(points[j, k]-points[0, k]),
                    ci95=np.quantile(delta, [.025, .975]).tolist() if len(delta) else None))
    destination = OUTPUT/'statistics'/source.name
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'true_positive', 'truth', 'predictions',
                         'center_error_sum_m', 'identity_switches', 'matched_transitions', 'duplicate_nodes'])
        for i, log in enumerate(logs):
            for j, condition in enumerate(conditions):
                writer.writerow([log, condition, *totals[i, j].tolist()])
    dump(destination/'report.json', dict(conditions=estimates, paired_differences=differences,
        source_release_sha256=digest(source/'release.json'), source_cases_sha256=digest(source/'cases.jsonl'),
        source_summary_sha256=digest(source/'summary.json'), statistics_source_sha256=digest(__file__),
        scenes=len(release['scene_logs']), logs=len(logs), bootstrap=dict(unit='采集log', paired=True, seed=17, repetitions=4000),
        limitations=['使用当前正式完整配置和相同前端；完整seed17记录复用，不增加独立样本',
            '关闭动态模块也关闭运动元数据更新，unknown不得自动当作static；单独关闭TTL只改变过期策略',
            '中心距离一对一诊断，不将全部未匹配预测叫作已确认误检或全部多身份关联叫作真实错合并',
            '仅开发建图；未执行车辆或开启新封存测试，未改变正在运行的导航源码与配置',
            'CPU与其他作业共享，无独占性能信用'], independent_test=False, actual_arrival_evaluated=False, goal_complete=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--scene', action='append')
    parser.add_argument('--wall-budget-s', type=float, default=1800.)
    args = parser.parse_args()
    begun = time.monotonic()
    known = {r['scene']: r for r in available_development()}
    scenes = args.scene or sorted(known)
    if not set(scenes) <= known.keys() or len(scenes) != len(set(scenes)):
        raise ValueError('fixed unique development scenes required')
    dest = OUTPUT/'ablations'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    sources = [OUTPUT/'mapping_evaluation/robust_motion_full_001'/s/'frames.jsonl' for s in scenes]
    modules = [Path(__file__), PROJECT/'conceptgraph/completion/fault_robustness.py',
        PROJECT/'conceptgraph/completion/fault_statistics.py', PROJECT/'conceptgraph/completion/stage9.py',
        PROJECT/'conceptgraph/completion/depth.py', PROJECT/'conceptgraph/completion/association.py',
        PROJECT/'conceptgraph/completion/motion.py', PROJECT/'conceptgraph/completion/quality.py',
        *sorted((PROJECT/'conceptgraph/slam').glob('*.py')), PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml']
    frozen = {str(p.relative_to(PROJECT)): digest(p) for p in modules}
    baseline = OUTPUT/'faults/recorded_fault_full_001'
    release = dict(conditions=SETTINGS, seed=17, scene_logs={s: known[s]['log_token'] for s in scenes},
        baseline_reports={s: digest(baseline/s/'seed17/unperturbed/report.json') for s in scenes},
        source_frames_sha256={str(p): digest(p) for p in sources}, source_sha256=frozen,
        complete_config=OmegaConf.to_container(mapping_config()), wall_budget_s=args.wall_budget_s,
        storage_budget_bytes=300000000, baseline_reused=True, expected_cases=len(scenes)*3,
        config_applied_to_running_navigation=False, independent_test=False, performance_credit=False, goal_complete=False)
    dump(dest/'release.json', release)
    with tarfile.open(dest/'source_snapshot.tar.gz', 'x:gz') as archive:
        for module in modules:
            archive.add(module, arcname=str(module.relative_to(PROJECT)))
    rows = []
    try:
        with (dest/'cases.jsonl').open('x') as handle:
            for path in sources:
                if digest(path) != release['source_frames_sha256'][str(path)]:
                    raise ValueError('source observations changed')
                frames = [json.loads(line) for line in path.open()]
                classes_path = Path(frames[0]['detections']).parents[2]/'gsa_classes_streamlined_yolo.json'
                frontend = json.loads((classes_path.parent/'frontend_complete.json').read_text())
                if any(digest(p) != sha for p, sha in frontend['output_sha256'].items()):
                    raise ValueError('frontend changed')
                classes = json.loads(classes_path.read_text())
                for condition, changes in SETTINGS.items():
                    config = mapping_config()
                    for key, value in changes.items():
                        setattr(config, key, value)
                    if condition == 'full':
                        report_path = baseline/path.parent.name/'seed17/unperturbed/report.json'
                        if digest(report_path) != release['baseline_reports'][path.parent.name]:
                            raise ValueError('complete baseline changed')
                    else:
                        case = dest/path.parent.name/condition
                        case.mkdir(parents=True)
                        run_case(frames, 'unperturbed', 17, case, classes,
                                 begun+args.wall_budget_s, inference_config=config)
                        report_path = case/'report.json'
                    row = dict(scene=path.parent.name, log_token=known[path.parent.name]['log_token'], condition=condition,
                        inference_config=OmegaConf.to_container(config), report_path=str(report_path),
                        report_sha256=digest(report_path), reused_baseline=condition == 'full')
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush()
                    print(json.dumps(dict(scene=path.parent.name, condition=condition), ensure_ascii=False), flush=True)
                    if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > release['storage_budget_bytes']:
                        raise RuntimeError('module ablation storage budget')
        if any(digest(PROJECT/p) != sha for p, sha in frozen.items()):
            raise ValueError('module ablation frozen source changed')
        dump(dest/'summary.json', dict(cases=len(rows), scenes=len(scenes), new_rebuilds=len(scenes)*2,
            reused_complete_baselines=len(scenes), unique_frames=sum(len(p.read_text().splitlines()) for p in sources),
            records_sha256=digest(dest/'cases.jsonl'), elapsed_s=time.monotonic()-begun,
            independent_test=False, goal_complete=False))
        statistics(dest, release, rows)
    except BaseException as error:
        dump(dest/'interrupted.json', dict(error=repr(error), completed_cases=len(rows), elapsed_s=time.monotonic()-begun))
        raise


if __name__ == '__main__':
    main()
