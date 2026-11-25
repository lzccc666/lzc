"""冻结B0/B1/R1的同条件阈值敏感性；先逐帧复现原3米协议。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from conceptgraph.completion.quality import object_frame_metrics  # noqa: E402

ROOT = PROJECT / 'outputs/results/nuscenes/validation/iterations'
REPORT_ROOT = PROJECT / 'outputs/reports/metric_optimization_20261123/threshold_sensitivity'
RUNS = {
    'B0': 'config_quality_B0_full_20261120_v1',
    'B1': 'config_quality_B1_full_20261120_v1',
    'R1': 'recall_refinement_R1_full_20261120_v2',
}
CLASSES = {'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}
COUNT_KEYS = ('true_positive', 'truth', 'predictions', 'unmatched_predictions',
              'false_negative', 'duplicate_nodes')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csv_write(path, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result(counts):
    c = dict(counts)
    tp, gt, pred = c['true_positive'], c['truth'], c['predictions']
    return dict(TP=tp, FP=pred-tp, FN=gt-tp, GT=gt, predictions=pred,
                precision=tp/pred if pred else None, recall=tp/gt if gt else None,
                F1=2*tp/(pred+gt) if pred+gt else None,
                duplicate_nodes=c['duplicate_nodes'],
                matched_center_error_mean_m=c['error_sum']/tp if tp else None,
                matched_pairs_over_3m=c['over_3m'])


def aggregate(counts, metrics):
    counts.update({key: metrics[key] for key in COUNT_KEYS})
    counts['error_sum'] += sum(m['distance_m'] for m in metrics['matches'])
    counts['over_3m'] += sum(m['distance_m'] > 3 for m in metrics['matches'])


def check_counts(counts, report):
    for key in ('true_positive', 'truth', 'predictions', 'duplicate_nodes'):
        if counts[key] != report['counts'][key]:
            raise ValueError(f'原3米计数未复现：{key}: {counts[key]} != {report["counts"][key]}')
    if not math.isclose(counts['error_sum'], report['counts']['center_error_sum_m'], abs_tol=1e-8):
        raise ValueError('原3米匹配中心误差和不一致')
    values = result(counts)
    for key in ('precision', 'recall', 'F1'):
        if not math.isclose(values[key], report['metrics'][key], abs_tol=1e-12):
            raise ValueError(f'原3米指标未复现：{key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-budget-s', type=float, default=1000)
    args = parser.parse_args()
    started = time.monotonic()
    output = REPORT_ROOT
    version = 1
    while output.exists():
        output = REPORT_ROOT / f'v{version:02d}'
        version += 1
    if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
        raise OSError('磁盘不足20GiB，未创建新输出')
    output.mkdir(parents=True, exist_ok=False)

    def guard():
        if time.monotonic()-started > args.wall_budget_s:
            raise TimeoutError('达到本轮墙钟预算，保留进度')
        if shutil.disk_usage(PROJECT).free < 20*1024**3:
            raise OSError('磁盘空闲低于20GiB，停止写入')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 127*1024**2:
            raise OSError('达到新增128MiB预算保护线')

    fingerprints = {}

    def freeze(path):
        fingerprints[str(path)] = sha(path)
        return fingerprints[str(path)]

    dump(output/'start_manifest.json', dict(
        started_utc=datetime.now(timezone.utc).isoformat(), wall_budget_s=args.wall_budget_s,
        total_task_budget_s=1200, storage_budget_bytes=128*1024**2,
        disk_free_bytes=shutil.disk_usage(PROJECT).free,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=PROJECT, text=True),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        source=str(Path(__file__).resolve()), source_sha256=freeze(Path(__file__).resolve()),
        thresholds_m=[1, 2, 3, 5], main_protocol_threshold_m=3,
        seed=None, seed_reason='无随机采样，复用确定性原匹配器',
        sensors_read=False, inference_executed=False, predictions_modified=False))
    try:
        manifests, reports, data = {}, {}, defaultdict(dict)
        reference_scenes = None
        for method, run in RUNS.items():
            guard()
            path = ROOT/run
            report, release = load(path/'report.json'), load(path/'release.json')
            if (report['condition'] != method or not report['full_validation_scope']
                    or (report['scenes'], report['logs'], report['frames']) != (38, 11, 1508)
                    or report['result']['counts']['truth'] != 5723 or release['threshold_m'] != 3):
                raise ValueError(f'{method}范围或原协议不一致')
            scope = [(s['scene'], s['log_token']) for s in release['scenes']]
            if reference_scenes is None:
                reference_scenes = scope
            if scope != reference_scenes:
                raise ValueError('三组场景或日志顺序不一致')
            for name in ('quality.py', 'association.py'):
                current = PROJECT/'conceptgraph/completion'/name
                frozen = path/'source_snapshot/conceptgraph/completion'/name
                if freeze(current) != freeze(frozen) or sha(frozen) != release['sources_sha256'][str(current)]:
                    raise ValueError('当前匹配器与冻结原实现不一致，停止统计')
            for filename in ('release.json', 'report.json', 'input_fingerprints.json'):
                freeze(path/filename)
            for scene, _ in scope:
                guard()
                scene_report = load(path/scene/'report.json')
                freeze(path/scene/'report.json')
                records_path = path/scene/'frames.jsonl.gz'
                if freeze(records_path) != scene_report['records_sha256']:
                    raise ValueError('冻结逐帧指纹发生变化')
                with gzip.open(records_path, 'rt') as stream:
                    rows = list(map(json.loads, stream))
                if len(rows) != scene_report['frames']:
                    raise ValueError('场景帧数与报告不同')
                if len({r['frame_idx'] for r in rows}) != len(rows):
                    raise ValueError('重复帧')
                data[method][scene] = rows
            reports[method] = report
            manifests[method] = dict(run_id=run, report_path=str(path/'report.json'),
                report_sha256=sha(path/'report.json'), release_path=str(path/'release.json'),
                release_sha256=sha(path/'release.json'), seed=release['seed'],
                config=release['config'], experimental_settings=release['experimental_settings'],
                algorithm_settings=release.get('algorithm_settings'),
                model_and_input_version='同一vehicle_frontend_full_20261118_v1冻结检测/分割/CLIP缓存，无参数更新',
                original_sources_sha256=release['sources_sha256'],
                original_inputs_sha256=release['inputs_sha256'],
                input_fingerprints_path=str(path/'input_fingerprints.json'),
                input_fingerprints_sha256=sha(path/'input_fingerprints.json'))
        # 保存的预测已经完成原类别、视野、缺测处理，不再次删帧或过滤对象。
        gt_classes = Counter()
        for scene, _ in reference_scenes:
            a = data['B0'][scene]
            for method in RUNS:
                b = data[method][scene]
                if len(a) != len(b):
                    raise ValueError('三组逐帧范围不同')
                for x, y in zip(a, b):
                    for key in ('scene', 'frame_idx', 'camera_timestamp_s', 'evaluation_only_official', 'input'):
                        if x[key] != y[key]:
                            raise ValueError(f'三组逐帧字段不同：{scene}/{key}')
                    for obj in y['evaluation_only_official']+y['predictions'][method]:
                        if obj['category'] not in CLASSES:
                            raise ValueError('原六类之外对象，停止而不擅自过滤')
            gt_classes.update(obj['category'] for row in a for obj in row['evaluation_only_official'])

        all_counts = defaultdict(Counter)
        scene_counts = defaultdict(Counter)
        log_counts = defaultdict(Counter)
        frame_evidence = []
        checks = []
        # 3米必须先全部通过；任何不一致直接退出，不能计算其他阈值。
        for threshold in (3, 1, 2, 5):
            for method in RUNS:
                for scene, log in reference_scenes:
                    guard()
                    for row in data[method][scene]:
                        metric = object_frame_metrics(row['predictions'][method],
                            row['evaluation_only_official'], threshold_m=threshold)
                        if threshold == 3 and metric != row['metrics'][method]:
                            raise ValueError(f'原3米逐帧指标/匹配未完全复现：{method}/{scene}/{row["frame_idx"]}')
                        for counter in (all_counts[threshold, method], scene_counts[threshold, method, scene],
                                        log_counts[threshold, method, log]):
                            aggregate(counter, metric)
                        frame_evidence.append(dict(threshold_m=threshold, method=method, scene=scene, log=log,
                            frame_idx=row['frame_idx'],
                            **{k: metric[k] for k in COUNT_KEYS},
                            matched_center_error_sum_m=sum(m['distance_m'] for m in metric['matches']),
                            matches_sha256=hashlib.sha256(json.dumps(metric['matches'], sort_keys=True).encode()).hexdigest()))
                    if threshold == 3:
                        check_counts(scene_counts[threshold, method, scene],
                            load(ROOT/RUNS[method]/scene/'report.json')['result'])
                c = all_counts[threshold, method]
                if c['truth'] != 5723:
                    raise ValueError('完整GT分母改变')
                if threshold == 3:
                    check_counts(c, reports[method]['result'])
                    checks.append(dict(method=method, frames=1508, per_frame_exact_match=True,
                                       scene_and_total_match=True, **result(c)))
            if threshold == 3:
                dump(output/'reproduction_3m.json', dict(passed=True, checks=checks))
                print('原3米三组逐帧、38场景及全量汇总全部复现；开始补充阈值。', flush=True)
        table = [dict(threshold_m=t, protocol='原主评价' if t == 3 else '补充敏感性评价',
                      method=m, scenes=38, logs=11, frames=1508, **result(all_counts[t, m]))
                 for t in (1, 2, 3, 5) for m in RUNS]
        csv_write(output/'threshold_metrics.csv', table)
        csv_write(output/'per_frame_counts.csv', frame_evidence)
        csv_write(output/'per_scene_metrics.csv', [dict(threshold_m=t, method=m, scene=s,
            **result(c)) for (t, m, s), c in sorted(scene_counts.items())])
        csv_write(output/'per_log_metrics.csv', [dict(threshold_m=t, method=m, log=log,
            **result(c)) for (t, m, log), c in sorted(log_counts.items())])
        dump(output/'threshold_metrics.json', dict(scope='验证／调试集完整范围，非独立测试',
            scene_log_manifest=reference_scenes, GT_class_counts=dict(gt_classes),
            matching='原同类水平XY中心距离<=阈值；最大合法一对一匹配数，再最小距离',
            frozen_visibility='直接使用冻结predictions与evaluation_only_official；不重新过滤，不排除空帧或缺测',
            source_versions=manifests, metrics=table,
            caveats=['1/2/5米仅补充评价，3米主协议保留；阈值不可跨方法混用',
                     '5米可以把3至5米的定位偏差计为TP，可能掩盖位置误差、邻近同类混淆',
                     '逐阈值重新一对一匹配，匹配身份可能变动，TP增加不代表预测位置变好',
                     '中心距离不检查三维高度、形状、身份连续或导航到达，不能冒充这些能力',
                     '不重算IDF1，不从阈值表推断身份稳定性或统计显著性']))
        fair = []
        inventories = [
            ('原方法及早期消融', 'confidence_full_20261118_v1', ['baseline', 'low_confidence', 'evidence_v2']),
            ('合理简单基线（已有负结果）', 'current_geometry_full_20261120_v2', ['current_geometry', 'single_frame_low_conf']),
            ('合理简单基线（观测级原型）', 'observation_tracking_full_20261120_v2', ['track_25', 'track_40']),
            ('真实前端替换对比，非完整强建图基线', 'ram_dino_full_20261118_v3', ['dino_fixed_native', 'ram_dino_native']),
        ]
        for role, run, conditions in inventories:
            p = ROOT/run/'report.json'
            d = load(p)
            freeze(p)
            if d['scenes'] != 38 or d.get('frames', d.get('frames_each')) != 1508:
                raise ValueError('清单证据范围不足')
            for condition in conditions:
                entry = d['conditions'][condition]
                if entry['counts']['truth'] != 5723:
                    raise ValueError('清单证据分母不同')
                fair.append(dict(role=role, run_id=run, condition=condition,
                    counts=entry['counts'], metrics=entry['metrics'],
                    report_path=str(p), report_sha256=sha(p),
                    evidence_level='仅核对冻结汇总；本轮未重新匹配该方法，不纳入本轮阈值表'))
        dump(output/'fair_comparison_inventory.json', dict(existing=fair,
            current_methods=list(RUNS), missing_strong_comparisons=[
                '同输入、同六类、同1508帧和3米协议下，成熟三维多目标跟踪/对象建图方法的真实完整复现',
                'B1稀疏估计内部子机制独立消融；已注册B2/B3不能当已实现或已运行',
                '固定方案后的未使用独立测试；本38场景仅验证／调试，历史已用测试只能称回归'],
            interpretation='B0为当前工程基线，并非公认最强方法；R1本轮只复现冻结指标，不重新宣称显著提升'))
        lines = ['# 冻结建图结果的同条件阈值对照', '',
            '范围：38验证／调试场景、11日志、1508帧，全部六类、5723次GT出现。',
            '三组均先逐帧完全复现3米存储匹配与计数，并核对38场景及总报告。', '',
            '| 阈值(m) | 方法 | TP | FP | FN | GT | Precision | Recall | F1 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in table:
            lines.append(f'| {row["threshold_m"]}{"（主）" if row["threshold_m"] == 3 else "（补）"} | '
                f'{row["method"]} | {row["TP"]} | {row["FP"]} | {row["FN"]} | {row["GT"]} | '
                f'{row["precision"]:.2%} | {row["recall"]:.2%} | {row["F1"]:.2%} |')
        lines += ['', '## 放宽门限不能称为算法改进', '',
            '所有方法的预测、GT、类别和视野／缺测处理均原样保留，只改变评价距离。1、2、5米均为补充表。',
            '匹配采用水平XY中心距离，不评价高度和形状；更宽门限可能掩盖定位偏差、近邻同类混淆。',
            '每个门限重新一对一指派，不能把不同阈值下TP差直接当作同一匹配对的变化。', '']
        for method in RUNS:
            c = result(all_counts[5, method])
            gain = c['TP']-all_counts[3, method]['true_positive']
            lines.append(f'- {method}：3→5米多计入{gain}个TP；5米匹配中{c["matched_pairs_over_3m"]}对实际水平误差超过3米。')
        lines += ['', '## 公平对比清单', '',
            '- 原方法：历史baseline与当前B0均有真实完整报告；不混称同一个版本。',
            '- 已运行改进：B1、R1在本表复现；提升及回退均保留，不用阈值表重新宣称统计显著。',
            '- 简单基线：单帧观测、当前帧几何替换、2.5/4米关联半径因果跟踪，均有真实全范围证据。',
            '- 前端替换：固定类别DINO及RAM+DINO真实运行，但不能当成成熟三维建图强基线。',
            '- 尚缺：统一协议下成熟三维跟踪/对象建图强方法的完整复现，B1内部独立消融及未使用独立测试。',
            '- B2/B3未实施；不得把削弱配置或注册配置包装成公认基线。', '',
            '逐项成绩、范围和局限见fair_comparison_inventory.json。原报告绝对路径、SHA-256、',
            '配置、源码与输入版本见threshold_metrics.json及input_manifest.json；逐帧/场景/日志CSV可复核。',
            '本轮未读取传感器大文件、未检测/建图/训练/Qwen/CARLA、未修改论文或默认算法。', '']
        (output/'阈值敏感性与公平对比.md').write_text('\n'.join(lines))
        guard()
        changed = [p for p, digest in fingerprints.items() if sha(Path(p)) != digest]
        dump(output/'input_manifest.json', fingerprints)
        if changed:
            raise ValueError(f'冻结输入/源码在运行期间改变：{changed}')
        shutil.copy2(Path(__file__).resolve(), output/'source_snapshot.py')
        parsed_json = sum(1 for p in output.glob('*.json') if load(p) is not None)
        csv_rows = {}
        for p in output.glob('*.csv'):
            with p.open() as stream:
                csv_rows[p.name] = sum(1 for _ in csv.DictReader(stream))
        if csv_rows['per_frame_counts.csv'] != 1508*3*4:
            raise ValueError('逐帧CSV范围错误')
        dump(output/'validation_report.json', dict(passed=True, original_3m_reproduced=True,
            frames_checked_3m=1508*3, threshold_rows=12, scenes=38, logs=11, GT=5723,
            parsed_json_files_before_validation=parsed_json, csv_rows=csv_rows,
            protected_read_inputs_unchanged=True, input_files_checked=len(fingerprints),
            elapsed_s=time.monotonic()-started, disk_free_bytes=shutil.disk_usage(PROJECT).free,
            output_bytes_before_validation=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            new_tests_written=False, verification='原3米逐帧匹配完全一致、逐场景/全汇总核对及JSON/CSV复读',
            limitations='本轮只对冻结输出再评价，不重跑建图、传感器或全量历史保护文件扫描'))
        load(output/'validation_report.json')
        print(json.dumps(dict(output=str(output), passed=True, rows=table), ensure_ascii=False), flush=True)
    except Exception as exc:
        if shutil.disk_usage(PROJECT).free >= 20*1024**3:
            dump(output/'failure_report.json', dict(passed=False, error=repr(exc),
                traceback=traceback.format_exc(), elapsed_s=time.monotonic()-started,
                supplementary_statistics_accepted=False))
        raise


if __name__ == '__main__':
    main()
