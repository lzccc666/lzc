"""导出完整故障与独立分阶段性能的冻结结果表；不撰写论文正文。"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
SOURCE = OUTPUT / 'source_bundle'
DEST = OUTPUT / 'experiment_tables/fault_performance_office_002'


def main():
    helper = SOURCE / 'development_result_tables_003.py'
    spec = importlib.util.spec_from_file_location('table_exports', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEST = DEST
    DEST.mkdir(exist_ok=False)
    (DEST / 'csv').mkdir()
    (DEST / 'tmp').mkdir()
    os.environ['TMPDIR'] = str(DEST / 'tmp')
    report_path = 'statistics/realtime_faults_gpu_full_001/report.json'
    report = module.load(report_path)
    assert report['trials'] == 180 and len(report['missing_complete_results']) == 1
    module.freeze('statistics/realtime_faults_gpu_full_001/trials.csv')
    module.freeze('statistics/realtime_three_model_full_001/trials.csv')
    exposure_path = 'interactive_simulation/realtime_faults_gpu_full_001/exposure_summary.json'
    exposure = module.load(exposure_path)
    event_path = 'faults/fault_event_full_independent_001/report.json'
    event = module.load(event_path)
    numeric_path = 'faults/fault_sensor_numeric_full_001/report.json'
    numeric = module.load(numeric_path)
    body_path = 'vehicle_execution/fault_execution_full_independent_001/report.json'
    body = module.load(body_path)
    assert event['passed'] and numeric['passed'] and body['audit_completed']
    assert event['audited_cases'] == numeric['cases'] == body['cases'] == 180
    assert body['reported_arrivals'] == body['conservative_hold_certificates'] == 10
    labels = dict(baseline='无注入基线', sensor_blackout_4s='断流4秒',
                  lidar_dropout_80pct='LiDAR丢失80%', depth_noise_02m='深度噪声0.2米',
                  pose_noise_025m_2deg='位姿噪声0.25米/2度',
                  detection_dropout_30pct='检测丢失30%')
    rows = []
    for method, label in labels.items():
        values = report['methods'][method]['by_kind']['reachable_positive']
        assert values['trials'] == 30
        rows.append(dict(condition=label, denominator=30,
                         **{k: int(values['four_outcomes'][k]['numerator']) for k in module.FIELDS}))
    module.add('T19', '完整GPU故障矩阵：四类结果计数',
        [module.col('condition', '条件'), module.col('denominator', '全体分母', 'int')]
        + [module.col(k, module.FIELD_LABELS[k], 'int') for k in module.FIELDS],
        rows, [report_path, body_path],
        '每条件10个冻结正任务×3种子；10配置共享3道路家族。180次均保留，179有完整结果，1次进程超时按未证实成功计入分母；其终态未知。10次到达通过保持区间审计；88例初态显式重建，不是原始观测。此批预算90秒、传感器TTL1.5秒，与主对比不同，不能直接拼接比较。')
    rows = [dict(condition=labels[k], denominator=v['frozen_trials'],
                 triggered=v['triggered'], unknown=v['unknown'],
                 blackout_stopped=v['blackout_stopped'] if k == 'sensor_blackout_4s' else None)
            for k, v in exposure['conditions'].items()]
    module.add('T20', '故障实际暴露与断流停车核验',
        [module.col('condition', '条件'), module.col('denominator', '分母', 'int'),
         module.col('triggered', '实际触发', 'int'), module.col('unknown', '暴露未知', 'int'),
         module.col('blackout_stopped', '断流后停车', 'int')], rows,
        [exposure_path, event_path, numeric_path],
        '基线不注入故障，0触发不表示缺失数据。断流29次触发、29次停车、29次新鲜观测恢复后运动；未触发的1次不获得断流安全信用。LiDAR超时1次暴露未知。数值审计核验19212对原始/有效帧；不宣称噪声服从已验证正态分布。非断流条件停车栏不适用，以空值保留。')
    rows = []
    for pair in report['paired_positive_differences']:
        if pair['a'] == 'baseline' and pair['metric'] == 'actual_full_perception_arrival':
            rows.append(dict(condition=labels[pair['b']], difference=100*pair['estimate'],
                             low=100*pair['ci95'][0], high=100*pair['ci95'][1], denominator=30))
    module.add('T21', '实际到达配对差值：无注入基线减故障',
        [module.col('condition', '故障条件'), module.col('difference', '差值/百分点', 'float'),
         module.col('low', '95%CI下限/百分点', 'float'), module.col('high', '95%CI上限/百分点', 'float'),
         module.col('denominator', '配对数', 'int')], rows, [report_path],
        '按10配置聚类配对bootstrap，4000次、seed17；负差表示故障条件更高。所有区间包括0，不宣称到达差异显著。未作多重比较校正。真实时钟下调度及轨迹差异保留。')
    perf_root = 'performance/runtime_profile_after_faults_002/complete.json'
    complete = module.load(perf_root)
    assert complete['actual_Qwen_calls'] == complete['Qwen_cache_requests'] == 216
    assert complete['cache_added_calls'] == 0
    names = [('base', '基础Qwen'), ('old', '旧Adapter'), ('new', '新Adapter'), ('frontend', '感知前端')]
    for number, (variant, label) in enumerate(names, 22):
        path = f'performance/runtime_profile_gpu_{variant}_001/report.json'
        data = module.load(path)
        assert data['unique_CUDA_process_verified'] and not data['end_to_end_throughput_credit']
        module.freeze(f'performance/runtime_profile_gpu_{variant}_001/timings.jsonl')
        rows = [dict(stage=k, calls=v['calls'], median_ms=v['median_s']*1000,
                     p95_ms=v['p95_s']*1000, throughput=v['successful_calls_per_measured_s'])
                for k, v in data['stages'].items()]
        module.add(f'T{number}', label+'：固定输入分阶段运行时间',
            [module.col('stage', '阶段/运行方式'), module.col('calls', '调用数', 'int'),
             module.col('median_ms', '中位数/毫秒', 'float'), module.col('p95_ms', 'P95/毫秒', 'float'),
             module.col('throughput', '成功调用/测量秒', 'float')], rows, [path],
            f'固定硬件、单CUDA进程抽样核验；峰值分配显存{data["peak_torch_GPU_allocated_MiB"]:.1f}MiB，保留显存{data["peak_torch_GPU_reserved_MiB"]:.1f}MiB。冷启动仅1次，不估计重复冷启动分布；warm_uncached为加载后无缓存，cache_hit为同输入复跑。吞吐仅本阶段调用数/测量总时间，不是闭环帧率。嵌套阶段时间不可相加；0.5秒抽样之间的外部GPU活动无法排除。')
    module.freeze('performance/runtime_profile_post_GPU_recorded_001/report.json')
    module.write_json(DEST / 'tables.json', module.TABLES)
    module.write_json(DEST / 'sources.json', module.SOURCES)
    module.exports()
    import openpyxl
    book = openpyxl.load_workbook(DEST / '开发实验结果表.xlsx')
    sheet = book.create_sheet('故障180逐任务')
    with (DEST / 'evidence/statistics/realtime_faults_gpu_full_001/trials.csv').open() as stream:
        for row in csv.reader(stream):
            sheet.append(row)
    assert sheet.max_row == 181
    sheet.freeze_panes = 'C2'
    sheet.auto_filter.ref = sheet.dimensions
    book.save(DEST / '开发实验结果表.xlsx')
    (DEST / 'README.md').write_text('本增量为已完成开发实验的T19—T25、冻结逐任务CSV和分阶段性能表，非论文正文。Word/Excel可编辑、PDF供核对。独立原始450表仅为背景证据，与本批不同预算，不能合并成功率。216真实Qwen调用、216缓存命中且新增调用0；格式通过不代表语义准确。性能不等于闭环实时性。人工审核及新封存测试未完成；goal_complete=false。\n')
    module.write_json(DEST / 'manifest.json', dict(
        generator_sha256=module.sha(__file__), export_helper_sha256=module.sha(helper),
        tables=len(module.TABLES), rows=sum(len(t['rows']) for t in module.TABLES),
        sources=module.SOURCES, output_sha256={p.name: module.sha(p) for p in DEST.iterdir() if p.is_file()},
        actual_new_model_calls=0, new_arrival_credit=0, manuscript_text_written=False, goal_complete=False))
    print(json.dumps(dict(destination=str(DEST), tables=len(module.TABLES), goal_complete=False)))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
