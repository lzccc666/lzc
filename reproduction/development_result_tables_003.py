"""只导出已冻结的开发实验表和逐项来源；不生成论文正文，不运行模型。"""
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'
DEST = OUTPUT / 'experiment_tables/development_evidence_003'
sys.path.insert(0, str(OUTPUT / 'source_bundle/document_runtime'))
FIELDS = ['route_generated', 'selected_object_evaluated_correct',
          'stable_correct_lock', 'actual_full_perception_arrival']
FIELD_LABELS = dict(zip(FIELDS, ['生成路线', '选对对象', '正确锁定', '实际正确到达']))
NAV = {'rule': '规则', 'base_full': '基础Qwen', 'new_full': '新Adapter'}
METHODS = {'single_frame': '单帧', 'original_fusion': '原融合',
           'timestamp_only': '仅时间戳', 'one_to_one_no_expiry': '一对一无过期',
           'one_to_one_timestamp': '一对一与时间戳',
           'one_to_one_depth_quality': '一对一与深度质量',
           'one_to_one_depth_robust_motion': '深度稳健运动组合',
           'mask_points_mean': 'mask点均值',
           'mask_points_robust_mean': 'mask点稳健均值'}
MODELS = {'base_fullschema': '基础Qwen', 'old_adapter_fullschema': '旧Adapter',
          'new_adapter_seed17': '新Adapter/17', 'new_adapter_seed29': '新Adapter/29',
          'new_adapter_seed43': '新Adapter/43'}
SUPPORT = {'all_gated': '全部门控', 'mutually_unique_gated': '互唯一门控'}
SOURCES = {}
TABLES = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(relative):
    if relative not in SOURCES:
        original = OUTPUT / relative
        target = DEST / 'evidence' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(original)
        shutil.copy2(original, target)
        assert before == sha(target) == sha(original)
        SOURCES[relative] = {'sha256': before, 'bytes': target.stat().st_size,
                             'package_path': str(target.relative_to(DEST))}
    return DEST / SOURCES[relative]['package_path']


def load(relative):
    return json.loads(freeze(relative).read_text())


def col(key, label, kind='text'):
    return {'key': key, 'label': label, 'kind': kind}


def add(identifier, title, columns, rows, sources, note):
    assert rows and all(set(row) == {c['key'] for c in columns} for row in rows)
    for source in sources:
        freeze(source)
    table = dict(id=identifier, title=title, columns=columns, rows=rows,
                 sources=sources, note=note, role='开发证据，非新封存测试')
    TABLES.append(table)
    with (DEST / 'csv' / (identifier + '.csv')).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=[c['key'] for c in columns])
        writer.writeheader()
        writer.writerows(rows)


def display(value, kind):
    if value is None:
        return '未知/无分母'
    if kind == 'bool':
        return '是' if value else '否'
    if kind == 'rate':
        return f'{100 * value:.2f}%'
    if kind == 'pp':
        return f'{100 * value:+.2f}'
    if kind == 'float':
        return f'{value:.4f}'
    if kind == 'int':
        return str(int(value))
    return str(value)


def create_tables():
    main_path = 'statistics/realtime_three_model_full_001/report.json'
    main = load(main_path)
    csv_path = freeze('statistics/realtime_three_model_full_001/trials.csv')
    trials = list(csv.DictReader(csv_path.open()))
    assert len(trials) == 450 and main['trials'] == 450
    main_counts = []
    rates = []
    kind_counts = []
    for method in ['rule', 'base_full', 'new_full']:
        positive = main['methods'][method]['by_kind']['reachable_positive']
        matched = [r for r in trials if r['method'] == method and r['kind'] == 'reachable_positive']
        assert len(matched) == positive['trials'] == 90
        counts = {}
        for metric in FIELDS:
            # CSV业务真值只接纳显式True；缺失保留空值，同时仍留在90分母。
            counts[metric] = sum(r[metric] == 'True' for r in matched)
            metric_report = positive['four_outcomes'][metric]
            assert counts[metric] == metric_report['numerator']
            rates.append(dict(method=NAV[method], metric=FIELD_LABELS[metric],
                success=counts[metric], denominator=metric_report['denominator'],
                rate=metric_report['estimate'], lower=metric_report['ci95'][0],
                upper=metric_report['ci95'][1]))
        main_counts.append(dict(method=NAV[method], denominator=90, **counts,
            missing=sum(r['result_missing'] == 'True' for r in matched)))
        for kind, data in main['methods'][method]['by_kind'].items():
            subset = [r for r in trials if r['method'] == method and r['kind'] == kind]
            assert len(subset) == data['trials']
            values = {f: sum(r[f] == 'True' for r in subset) for f in FIELDS}
            assert all(values[f] == data['four_outcomes'][f]['numerator'] for f in FIELDS)
            kind_counts.append(dict(method=NAV[method], kind=kind, denominator=len(subset),
                                    **values, missing=sum(r['result_missing'] == 'True' for r in subset)))
    count_columns = [col('method', '方法'), col('denominator', '尝试数', 'int')]
    count_columns += [col(f, FIELD_LABELS[f]+'数', 'int') for f in FIELDS]
    count_columns += [col('missing', '完整结果缺失数', 'int')]
    main_note = ('每方法30个正任务×3种子=90次尝试；10配置共享3走廊家族。'
                 '450次总尝试中基础Qwen有1次进程超时无完整结果，空业务值不改写为False，分母保留。'
                 '对象正确性沿用固定自动GT关联；这些是开发合成颜色任务，不是人工自然指令或新封存。')
    add('T01', '真实感知主矩阵：正条件四项结果计数', count_columns, main_counts,
        [main_path, 'statistics/realtime_three_model_full_001/trials.csv'], main_note)
    add('T02', '主矩阵：正条件四项结果比例与聚类区间',
        [col('method', '方法'), col('metric', '指标'), col('success', '成功数', 'int'),
         col('denominator', '尝试数', 'int'), col('rate', '比例', 'rate'),
         col('lower', '95%下限', 'rate'), col('upper', '95%上限', 'rate')],
        rates, [main_path], '按配置配对聚类bootstrap，4000次，seed17；区间沿用冻结统计。'+main_note)
    differences = [dict(a=NAV[d['a']], b=NAV[d['b']], metric=FIELD_LABELS[d['metric']],
        difference=d['estimate'], lower=d['ci95'][0], upper=d['ci95'][1], denominator=d['denominator'])
        for d in main['paired_positive_differences']]
    add('T03', '主矩阵：正条件配对差值',
        [col('a', '方法A'), col('b', '方法B'), col('metric', '指标'),
         col('difference', 'A−B/百分点', 'pp'), col('lower', '95%下限/百分点', 'pp'),
         col('upper', '95%上限/百分点', 'pp'), col('denominator', '配对尝试', 'int')],
        differences, [main_path], '正负号均按A−B，包含新Adapter下降的全部结果；区间跨零不作显著性主张。'+main_note)
    add('T04', '主矩阵：全部任务类型及失败分母',
        [count_columns[0], col('kind', '任务类型')] + count_columns[1:], kind_counts,
        [main_path, 'statistics/realtime_three_model_full_001/trials.csv'],
        'reachable_positive=可达正条件；inaccessible=不可达；no_match=无匹配。'
        '负条件的无到达不是算法失败的通用定义，不能把拒绝合并成到达成功率。'+main_note)

    mapping_path = 'statistics/causal_mapping_full_001/report.json'
    mapping = load(mapping_path)
    rows = []
    for method, groups in mapping['methods'].items():
        d = groups['all']
        rows.append(dict(method=METHODS[method], truth=d['truth'], predictions=d['predictions'],
            tp=d['true_positive'], recall=d['recall']['value'],
            fraction=d['matched_prediction_precision']['value'],
            error=d['center_error_m_mean']['value'], duplicates=d['duplicate_nodes']))
    add('T05', '过去时刻LiDAR场景图：全帧对象评价',
        [col('method', '方法'), col('truth', 'GT观测', 'int'), col('predictions', '预测观测', 'int'),
         col('tp', '匹配数', 'int'), col('recall', '召回', 'rate'), col('fraction', '匹配预测比例', 'rate'),
         col('error', '匹配中心误差/m', 'float'), col('duplicates', '重复节点', 'int')],
        rows, [mapping_path], '38开发场景、11采集log、1508帧；1470帧有过去LiDAR，38缺失帧仍保留。'
        +mapping['evaluation_limit']+'；'+mapping['unmatched_prediction_limit']+'。不评价导航到达。')

    temporal_path = 'statistics/temporal_identity_full_001/report.json'
    temporal = load(temporal_path)
    rows = []
    for method, supports in temporal['methods'].items():
        for support, d in supports.items():
            c = d['counts']
            rows.append(dict(method=METHODS[method], support=SUPPORT[support],
                idtp=c['IDTP'], idfn=c['IDFN'], idfp=c['IDFP'], idf1=d['metrics']['IDF1']['value'],
                return_opportunities=c['official_return_opportunities'],
                return_unknown=c['official_return_unknown_id']))
    add('T06', '全连续序列身份：门控口径与未知返回',
        [col('method', '方法'), col('support', '支持口径'), col('idtp', 'IDTP', 'int'),
         col('idfn', 'IDFN', 'int'), col('idfp', 'IDFP', 'int'), col('idf1', 'IDF1', 'rate'),
         col('return_opportunities', '返回机会', 'int'), col('return_unknown', '返回未知', 'int')],
        rows, [temporal_path], '38开发场景、11log；3m同类别XY中心门控适配，非官方MOT成绩。'
        'IDFP是该门控身份定义的计数；不直接等同完整检测误检率。IDF1受检测覆盖影响；'
        '自动关联不是人工身份真值，未知返回不计作成功或错误。')

    fair_path = 'statistics/fair_three_task_full_validation_001/report.json'
    fair = load(fair_path)
    rows = []
    for task, data in fair['tasks'].items():
        for model, metric in data['models'].items():
            rows.append(dict(task=task, model=MODELS[model], correct=metric['correct'],
                requests=metric['requests'], accuracy=metric['accuracy'], lower=metric['ci95'][0],
                upper=metric['ci95'][1], clusters=data['clusters']))
    add('T07', '公平三任务验证：全部模型及三个新训练种子',
        [col('task', '任务'), col('model', '模型/种子'), col('correct', '正确数', 'int'),
         col('requests', '相同请求', 'int'), col('accuracy', '准确率', 'rate'),
         col('lower', '95%下限', 'rate'), col('upper', '95%上限', 'rate'), col('clusters', '聚类数', 'int')],
        rows, [fair_path], 'query_parse=指令解析；object_attributes此处只计crop粗类别；candidate_rerank=候选排序。'
        '每模型603相同请求；包含缓存复用，不等于603次新CUDA调用。解析为构造语法，crop和解析只有2验证log，'
        '排序为6受控符号bank，区间不稳定。无人工颜色细属性准确率信用，也不代表导航改善。')

    object_path = 'statistics/kitti_object_cpu_full_001/report.json'
    obj = load(object_path)
    rows = [dict(method=METHODS[m], truth=d['truth'], predictions=d['predictions'],
        matched=d['matches_3d'], recall=d['recall_3d'], pairs=d['conditional_localized'],
        conditional_error=d['conditional_center_error_m']) for m, d in obj['metrics'].items()]
    object_note = ('KITTI Object training均匀64帧、33个原始drive；CPU真实前端。'+obj['matching']+'。'
        '条件中心误差使用共同251对；检测匹配集合误差不能混为同一指标。预训练重叠未知，导航到达信用0。')
    add('T08', 'KITTI Object：对象定位诊断',
        [col('method', '方法'), col('truth', 'GT数', 'int'), col('predictions', '预测数', 'int'),
         col('matched', '3D匹配', 'int'), col('recall', '3D召回', 'rate'), col('pairs', '共同条件对', 'int'),
         col('conditional_error', '共同条件误差/m', 'float')], rows, [object_path], object_note)
    rows = [dict(metric=metric, unit='百分点' if metric != 'conditional_center_error_m' else 'm',
        difference=d['difference'], lower=d['ci95'][0], upper=d['ci95'][1],
        resamples=d['valid_resamples']) for metric, d in obj['paired_drive_bootstrap'].items()]
    # 差值原值始终保留比例单位；显示时由列独立单位说明，避免同列混合百分点和米。
    for row in rows:
        if row['unit'] == '百分点':
            for key in ['difference', 'lower', 'upper']:
                row[key] *= 100
    add('T09', 'KITTI Object：稳健均值减普通均值的配对差',
        [col('metric', '指标'), col('unit', '单位'), col('difference', '差值', 'float'),
         col('lower', '95%下限', 'float'), col('upper', '95%上限', 'float'), col('resamples', '有效重采样', 'int')],
        rows, [object_path], '按原始drive配对聚类bootstrap，4000次，seed17；召回/匹配预测比例差在本表已×100为百分点，'
        '误差差保留米，CSV同口径。负误差表示下降。'+object_note)
    rows = []
    for category, methods in obj['per_class'].items():
        for method, d in methods.items():
            rows.append(dict(category=category, method=METHODS[method], truth=d['truth'],
                matched=d['matches_3d'], recall=d['recall_3d'], pairs=d['conditional_localized'],
                error=d['conditional_center_error_m']))
    add('T10', 'KITTI Object：逐类结果及负结果',
        [col('category', '类别'), col('method', '方法'), col('truth', 'GT数', 'int'),
         col('matched', '匹配数', 'int'), col('recall', '召回', 'rate'), col('pairs', '条件对数', 'int'),
         col('error', '条件误差/m', 'float')], rows, [object_path], object_note+
        '保留Truck条件误差恶化；无有效分母用未知，不填0。bicycle行是Cyclist代理，20个GT，普通/稳健3D匹配分别6/7；2D共同条件对均为0，'
        '不纳入T08的319个主类别GT；代理不等同Cyclist完整检测能力。')

    track_path = 'statistics/kitti_tracking_full_001/report.json'
    track = load(track_path)
    rows = []
    for method, d in track['object']['methods'].items():
        c, m = d['counts'], d['metrics']
        rows.append(dict(method=METHODS[method], truth=c['truth'], predictions=c['predictions'],
            tp=c['true_positive'], unmatched=c['unmatched_predictions'], recall=m['recall'],
            fraction=m['matched_prediction_fraction'], error=m['center_error_m']))
    track_note = ('0000/0005/0013各中心30帧，共90帧；仅0005具有本主类别GT，合计74观测/4实例，'
        '其余原始Van/DontCare等保留，不能称三个有效正例序列。一个RGB与Object重合，不计独立样本。'
        '未匹配包含DontCare/未支持类别，匹配预测比例不是完整精确率；非官方KITTI AP/HOTA，无导航到达信用。')
    add('T11', 'KITTI Tracking：连续车载窗口对象评价',
        [col('method', '方法'), col('truth', '支持GT观测', 'int'), col('predictions', '预测观测', 'int'),
         col('tp', '匹配数', 'int'), col('unmatched', '未匹配预测', 'int'), col('recall', '召回', 'rate'),
         col('fraction', '匹配预测比例', 'rate'), col('error', '匹配误差/m', 'float')], rows, [track_path], track_note)
    rows = []
    for method, supports in track['identity']['methods'].items():
        for support, d in supports.items():
            c, m = d['counts'], d['metrics']
            rows.append(dict(method=METHODS[method], support=SUPPORT[support], idtp=c['IDTP'],
                idfn=c['IDFN'], idfp=c['IDFP'], idf1=m['IDF1']['value'],
                lower=m['IDF1']['ci95'][0], upper=m['IDF1']['ci95'][1]))
    add('T12', 'KITTI Tracking：身份计数及IDF1',
        [col('method', '方法'), col('support', '支持口径'), col('idtp', 'IDTP', 'int'),
         col('idfn', 'IDFN', 'int'), col('idfp', 'IDFP', 'int'), col('idf1', 'IDF1', 'rate'),
         col('lower', '95%下限', 'rate'), col('upper', '95%上限', 'rate')], rows, [track_path],
        '序列聚类bootstrap4000次，只有一个正例序列导致区间退化/不稳定，不能作为充分跨场景显著性证据。'+track_note)

    fault_path = 'faults/semantic_async_cpu_003/delivery_001/summary.json'
    fault = load(fault_path)
    conditions = [(('001' if d['condition'] in ['baseline', 'backend_exception'] else '003'), d)
                  for d in fault['reports']] + [('002', fault['previous_unexposed_negative'])]
    rows, outcomes = [], []
    for version, d in conditions:
        rows.append(dict(version=version, condition=d['condition'], exposed=d['fault_exposure_satisfied'],
            calls=d['actual_Qwen_calls'], frontend=d['actual_frontend_calls'],
            controls=d['actual_control_intervals'], expired_queue=d['expired_in_queue'],
            certified=d['independent_arrival_certified']))
        outcomes.append(dict(version=version, condition=d['condition'], **d['original_four_outcomes'],
                             hold=d['longest_whole_interval_hold_s']))
    assert sum(d['calls'] for d in rows) == fault['counted_new_CPU_Qwen_calls'] == 24
    fault_note = ('同一config00-positive0/seed17的CPU集成，共24次实际Qwen、113次真实前端、14675控制区间；'
        '001原终止协议，003固定180秒故障观察窗，002提前到达而故障未暴露的负结果保留。'
        '排队过期请求不算模型调用。共享CPU、TTL15秒，不替代正式GPU/TTL1.5秒故障；不增加独立任务或导航到达数。')
    add('T13', 'CPU语义异步故障：实际暴露与执行计数',
        [col('version', '协议版'), col('condition', '条件'), col('exposed', '暴露满足', 'bool'),
         col('calls', 'Qwen调用', 'int'), col('frontend', '真实前端', 'int'),
         col('controls', '控制区间', 'int'), col('expired_queue', '排队过期', 'int'),
         col('certified', '独立到达证据', 'bool')], rows, [fault_path], fault_note)
    add('T14', 'CPU语义异步故障：原四项结果与保持证据',
        [col('version', '协议版'), col('condition', '条件')] + [col(f, FIELD_LABELS[f], 'bool') for f in FIELDS]
        + [col('hold', '完整区间保持/s', 'float')], outcomes, [fault_path], fault_note+
        '保持来自全区间车体/目标身份/方向/速度独立复核。故障回退到达不能归为模型准确率或故障改善导航。')

    cache_base = 'reproduction/three_task_fixedlogical_001/independent_audit_001/'
    cache = load(cache_base+'report.json')
    assert cache['passed'] and all(d['equal'] for d in cache['comparisons'].values())
    rows = []
    cache_sources = [cache_base+'report.json']
    for name in ['cold', 'warm']:
        path = cache_base+name+'/report.json'
        d = load(path)
        cache_sources.append(path)
        rows.append(dict(mode=name, calls=d['actual_reported_Qwen_calls'], frontend=d['perception_frames'],
            controls=d['controls'], **d['four_outcomes'], hold=d['independent_hold_s']))
    add('T15', '固定逻辑三任务缓存：冷与热实际复现',
        [col('mode', '模式'), col('calls', '新增Qwen', 'int'), col('frontend', '真实前端', 'int'),
         col('controls', '控制区间', 'int')] + [col(f, FIELD_LABELS[f], 'bool') for f in FIELDS]
        + [col('hold', '保持/s', 'float')], rows, cache_sources,
        '同一小任务冷/热各61次新前端、301控制；冷3次CPU Qwen，热0且未加载Qwen。'
        '8类业务证据逐字一致，61个gzip容器解压业务字节一致，仅头部mtime有差。'
        '两次均预算结束未到达，保持仅1.2秒，不能按2秒阈值计到达。逻辑时钟同步协议，不是GPU真实异步性能。')

    # 额外保留独立核验和逐序列明细，避免只交付汇总表。
    for path in ['statistics/main_matrix_evidence_check_001/report.json',
                 'statistics/kitti_tracking_full_001/independent_audit_002/report.json',
                 'statistics/kitti_tracking_full_001/scope_audit_002/report.json',
                 'statistics/kitti_tracking_full_001/identity_by_sequence.json',
                 'statistics/kitti_tracking_full_001/identity_rows.json',
                 'faults/semantic_async_cpu_003/delivery_001/independent_call_and_execution_ledger_001.json']:
        freeze(path)


def exports():
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from playwright.sync_api import sync_playwright

    workbook = openpyxl.Workbook()
    index = workbook.active
    index.title = '索引与口径'
    for row in [['开发实验结果表与证据索引'], ['仅冻结的开发结果；非论文正文；goal_complete=false'],
                ['数值CSV/JSON保留原始精度；未知值留空；比例以0—1存储（T09已按列示单位换算）'],
                ['表号', '内容', '数据行', '范围']]:
        index.append(row)
    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Mm(297), Mm(210)
    section.left_margin = section.right_margin = Mm(15)
    section.top_margin = section.bottom_margin = Mm(13)
    for style in document.styles:
        if style.type == 1:
            style.font.name = 'Noto Sans CJK SC'
            style.font.size = Pt(9)
            style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK SC')
    document.styles['Normal'].paragraph_format.space_after = Pt(5)
    footer = section.footer.paragraphs[0]
    footer.alignment = 1
    footer.add_run('开发实验表 | ')
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    body = []
    for i, table in enumerate(TABLES):
        identifier = table['id']
        index.append([identifier, table['title'], len(table['rows']), table['role']])
        index.cell(index.max_row, 1).hyperlink = f"#'{identifier}'!A1"
        sheet = workbook.create_sheet(identifier)
        sheet.append([c['label'] for c in table['columns']])
        for row in table['rows']:
            sheet.append([row[c['key']] for c in table['columns']])
        last_data = sheet.max_row
        sheet.auto_filter.ref = sheet.dimensions
        sheet.freeze_panes = 'B2'
        for column, definition in zip(sheet.columns, table['columns']):
            sheet.column_dimensions[column[0].column_letter].width = 31 if definition['kind'] == 'text' else 19
            for cell in list(column)[1:]:
                cell.number_format = {'rate': '0.00%', 'pp': '+0.00;-0.00;0.00',
                                      'float': '0.0000', 'int': '0'}.get(definition['kind'], 'General')
        sheet.append([])
        sheet.append(['口径：'+table['note']])
        sheet.merge_cells(start_row=last_data+2, start_column=1,
                          end_row=last_data+2, end_column=len(table['columns']))
        sheet.row_dimensions[last_data+2].height = 85
        for source in table['sources']:
            sheet.append(['证据：'+source])
            sheet.cell(sheet.max_row, 1).hyperlink = SOURCES[source]['package_path']
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.print_options.horizontalCentered = True
        sheet.print_title_rows = '1:1'

        if i:
            document.add_page_break()
        document.add_heading(identifier+' '+table['title'], 1)
        document.add_paragraph(table['role']+'；计数不混同独立任务数量。')
        word_table = document.add_table(rows=1, cols=len(table['columns']))
        for cell, definition in zip(word_table.rows[0].cells, table['columns']):
            cell.text = definition['label']
        repeat = OxmlElement('w:tblHeader')
        word_table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in table['rows']:
            for cell, definition in zip(word_table.add_row().cells, table['columns']):
                cell.text = display(row[definition['key']], definition['kind'])
        for row in word_table.rows:
            cant_split = OxmlElement('w:cantSplit')
            row._tr.get_or_add_trPr().append(cant_split)
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(3)
                    paragraph.paragraph_format.space_before = Pt(3)
                    for run in paragraph.runs:
                        run.font.size = Pt(8)
        borders = OxmlElement('w:tblBorders')
        for edge in ['top', 'bottom', 'left', 'right', 'insideH', 'insideV']:
            border = OxmlElement('w:'+edge)
            border.set(qn('w:val'), 'single' if edge in {'top', 'bottom'} else 'nil')
            border.set(qn('w:sz'), '8')
            borders.append(border)
        word_table._tbl.tblPr.append(borders)
        document.add_paragraph('口径：'+table['note'])
        for source in table['sources']:
            p = document.add_paragraph('来源：'+source+' | SHA256 '+SOURCES[source]['sha256'])
            for run in p.runs:
                run.font.size = Pt(7)

        header = ''.join('<th>'+html.escape(c['label'])+'</th>' for c in table['columns'])
        rows = ''.join('<tr>'+''.join('<td>'+html.escape(display(row[c['key']], c['kind']))+'</td>'
                        for c in table['columns'])+'</tr>' for row in table['rows'])
        source_text = ''.join('<li><a href="'+html.escape(SOURCES[s]['package_path'])+'">'
            +html.escape(s)+'</a><br>SHA256 '+SOURCES[s]['sha256']+'</li>' for s in table['sources'])
        body.append('<section><h1>'+html.escape(identifier+' '+table['title'])+'</h1><p class="scope">'
            +html.escape(table['role'])+'；goal_complete=false</p><table><thead><tr>'+header
            +'</tr></thead><tbody>'+rows+'</tbody></table><p class="note">口径：'
            +html.escape(table['note'])+'</p><ul class="sources">'+source_text+'</ul></section>')

    raw_sheet = workbook.create_sheet('原始450逐任务')
    with (DEST/'evidence/statistics/realtime_three_model_full_001/trials.csv').open() as stream:
        for row in csv.reader(stream):
            raw_sheet.append(row)
    raw_sheet.freeze_panes = 'C2'
    raw_sheet.auto_filter.ref = raw_sheet.dimensions
    sources_sheet = workbook.create_sheet('证据哈希')
    sources_sheet.append(['原始相对路径', 'SHA256', '包内路径', '字节数'])
    for source, value in SOURCES.items():
        sources_sheet.append([source, value['sha256'], value['package_path'], value['bytes']])
        sources_sheet.cell(sources_sheet.max_row, 3).hyperlink = value['package_path']
    for sheet in workbook:
        for cell in sheet[1]:
            cell.font = Font(name='Noto Sans CJK SC', bold=True)
            cell.fill = PatternFill('solid', fgColor='E7EEF5')
            cell.border = Border(top=Side(style='medium'), bottom=Side(style='thin'))
        for row in sheet:
            for cell in row:
                cell.alignment = Alignment(vertical='center', wrap_text=True)
        if sheet.title not in {t['id'] for t in TABLES}:
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(18, max(len(str(c.value or '')) for c in column)))
    workbook.save(DEST/'开发实验结果表.xlsx')
    document.save(DEST/'开发实验结果表.docx')
    css = '''@page{size:A4 landscape;margin:12mm 14mm 15mm}*{box-sizing:border-box}
body{font-family:"Noto Sans CJK SC",sans-serif;color:#111;font-size:9pt;line-height:1.4}
section{break-before:page}section:first-child{break-before:auto}h1{font-size:16pt;margin:0 0 3mm}
.scope{color:#555;font-size:8pt;margin:0 0 3mm}table{width:100%;border-collapse:collapse;table-layout:auto;
border-top:1.3pt solid;border-bottom:1.3pt solid;font-size:8pt}thead{border-bottom:.6pt solid}
td,th{padding:2.2mm 1.5mm;text-align:left;overflow-wrap:anywhere}tr{break-inside:avoid}
.note{font-size:8pt;margin-top:4mm}.sources{font-size:6.5pt;overflow-wrap:anywhere;padding-left:4mm}
a{color:#234760;text-decoration:none}'''
    page_html = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>开发实验结果表</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>'
    (DEST/'开发实验结果表.html').write_text(page_html)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(PROJECT/'outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'), headless=True, args=['--no-sandbox', '--no-proxy-server'],
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        page = browser.new_page()
        page.goto((DEST/'开发实验结果表.html').as_uri())
        page.evaluate('document.fonts.ready')
        overflow = page.evaluate('''() => [...document.querySelectorAll('section')].map(s=>({
            title:s.querySelector('h1').textContent,horizontal:s.scrollWidth>s.clientWidth+1,
            height_px:s.getBoundingClientRect().height}))''')
        assert not any(d['horizontal'] for d in overflow)
        write_json(DEST/'browser_layout.json', overflow)
        page.pdf(path=str(DEST/'开发实验结果表.pdf'), print_background=True, display_header_footer=True,
            prefer_css_page_size=True, header_template='<span></span>',
            footer_template='<div style="font-size:8px;width:100%;text-align:center">开发证据 | <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    (DEST/'csv').mkdir()
    before_index = sha(PROJECT/'.git/index')
    try:
        create_tables()
        write_json(DEST/'tables.json', TABLES)
        write_json(DEST/'sources.json', SOURCES)
        exports()
        readme = ['# 开发实验结果表与证据索引', '',
            '本包只整理已完成开发实验，不含论文正文、章节或框架，不新增模型调用或实验到达信用。', '',
            'Excel和Word可编辑；PDF供核对；tables.json保留结构化值，csv逐表保留精度，evidence保存逐任务及原始统计快照。',
            'T09的比例差已经转为百分点，其余比例在JSON/CSV以0—1保存。null/空值为未知，不能补成零。', '',
            '原始完整传感器/模型输出仍在项目指定输出目录；本包是小体积结果索引，不自称完整原始数据包。', '',
            '待完成：450缺一消融、750公平组合、正式GPU故障及独占性能、人工审核/自然指令、至少20场景×20有效指令的新封存、最终保护核验及代码同步。',
            '当前整个课题goal_complete=false。论文写作按用户要求推迟，由后续GPT Work组织。', '',
            '表号 | 内容 | 数据行', '--- | --- | ---']
        readme += [f"{t['id']} | {t['title']} | {len(t['rows'])}" for t in TABLES]
        (DEST/'README.md').write_text('\n'.join(readme)+'\n')
        assert sha(PROJECT/'.git/index') == before_index
        assert all(sha(OUTPUT/source) == entry['sha256'] for source, entry in SOURCES.items())
        shutil.copy2(Path(__file__), DEST/'generator_source.py')
        write_json(DEST/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            tables=len(TABLES), data_rows=sum(len(t['rows']) for t in TABLES),
            frozen_sources=len(SOURCES), actual_new_model_calls=0, new_experiments=0,
            new_arrival_credit=0, new_human_labels=0, new_test_opened=False,
            manuscript_text_written=False, original_index_unchanged=True,
            original_index_sha256=before_index, source_sha256=sha(__file__),
            files={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
            independent_package_validation_pending=True, goal_complete=False))
        print(json.dumps(dict(exported=True, tables=len(TABLES), directory=str(DEST)), ensure_ascii=False))
    except BaseException as error:
        write_json(DEST/'interrupted.json', dict(type=type(error).__name__, message=str(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
