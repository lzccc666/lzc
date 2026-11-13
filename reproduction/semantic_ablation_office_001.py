"""将冻结消融表导出为可编辑Word/Excel与核对PDF，不起草正文。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
SOURCE = OUTPUT / 'source_bundle'
DEST = OUTPUT / 'experiment_tables/semantic_ablation_office_001'


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
    report_path = 'statistics/semantic_ablation_combined_004/report.json'
    report = module.load(report_path)
    assert report['trials'] == 750
    module.freeze('statistics/semantic_ablation_combined_004/trials.csv')
    module.freeze('statistics/realtime_three_model_full_001/trials.csv')
    audit = module.load('statistics/semantic_compatibility_evidence_check_004/report.json')
    assert audit['passed'] and audit['trials'] == 750
    chain = module.load('ablations/semantic_navigation_chain_004/complete.json')
    assert chain['statistics_report_sha256'] == module.sha(OUTPUT / report_path)
    evidence = [report_path, 'statistics/semantic_compatibility_evidence_check_004/report.json',
                'vehicle_execution/semantic_minus_execution_full_002/report.json']
    fields = module.FIELDS
    labels = {'rule': '规则', 'new_full': '新Adapter完整', 'new_minus_parse': '关闭解析',
              'new_minus_attributes': '关闭属性', 'new_minus_rerank': '关闭重排'}
    rows = []
    for method, label in labels.items():
        values = report['methods'][method]['by_kind']['reachable_positive']
        assert values['trials'] == 90
        rows.append(dict(method=label, denominator=90,
                         **{k: int(values['four_outcomes'][k]['numerator']) for k in fields}))
    module.add('T16', '语义消融：四类正任务结果（计数）',
               [module.col('method', '条件'), module.col('denominator', '正任务分母', 'int')]
               + [module.col(k, module.FIELD_LABELS[k], 'int') for k in fields], rows, evidence,
               '每条件50任务×3种子，正任务90、无匹配30、不可接近30。450新增+300复用记录，不增加独立任务数。1例不可接近中断保留为失败。59次新增到达有保持证据；59例初态显式重建。开发颜色任务，10配置共享3道路家族；非新封存测试。')
    for table_id, compared in [('T17', ['new_minus_parse', 'new_minus_attributes']),
                               ('T18', ['new_minus_rerank', 'rule'])]:
        rows = []
        for other in compared:
            for k in fields:
                pair = next(x for x in report['paired_positive_differences']
                            if x['a'] == 'new_full' and x['b'] == other and x['metric'] == k)
                rows.append(dict(comparison='完整减'+labels[other], metric=module.FIELD_LABELS[k],
                                 difference=pair['estimate']*100, low=pair['ci95'][0]*100,
                                 high=pair['ci95'][1]*100, denominator=90))
        module.add(table_id, '配对消融差异：完整新Adapter减对照（百分点）',
                   [module.col('comparison', '差值方向'), module.col('metric', '指标'),
                    module.col('difference', '差值/百分点', 'float'),
                    module.col('low', '95%CI下限/百分点', 'float'),
                    module.col('high', '95%CI上限/百分点', 'float'),
                    module.col('denominator', '配对正任务数', 'int')], rows, [report_path],
                   '配置聚类配对bootstrap，10配置、4000次、seed17；负值表示完整新Adapter较低。单位已转换为百分点，不是百分比相对变化。未作多重比较校正；共享道路家族限制泛化。实时资源/轨迹变化属于闭环条件差异，不视为单字段严格因果效应。')
    module.write_json(DEST / 'tables.json', module.TABLES)
    module.write_json(DEST / 'sources.json', module.SOURCES)
    module.exports()
    import openpyxl
    import csv
    book = openpyxl.load_workbook(DEST / '开发实验结果表.xlsx')
    sheet = book.create_sheet('消融750逐任务')
    with (DEST / 'evidence/statistics/semantic_ablation_combined_004/trials.csv').open() as stream:
        for row in csv.reader(stream):
            sheet.append(row)
    assert sheet.max_row == 751
    sheet.freeze_panes = 'C2'
    sheet.auto_filter.ref = sheet.dimensions
    book.save(DEST / '开发实验结果表.xlsx')
    (DEST / 'README.md').write_text('本增量仅含已冻结开发实验的T16—T18表、逐任务CSV和证据，非论文正文。Excel含原始主对比450和消融组合750两张逐任务表，二者有300条复用，禁止将1200视为独立实验。Word/Excel可编辑，PDF供核验。表号为开发表交付编号，最终论文编号留待写作统一。模型、提示和任务未修改；故障/性能/新环境及人工封存验收仍未完成。goal_complete=false。\n')
    module.write_json(DEST / 'manifest.json', dict(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        export_helper_sha256=module.sha(helper), tables=3, rows=21,
        sources=module.SOURCES, output_sha256={p.name: module.sha(p) for p in DEST.iterdir()
                                              if p.is_file()}, actual_model_calls=0,
        manuscript_text_written=False, goal_complete=False))
    print(json.dumps(dict(destination=str(DEST), tables=3, rows=21, goal_complete=False)))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
