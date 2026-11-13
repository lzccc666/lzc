"""归档已结束的屏障实验，更新机器队列和交付进度；不新增模型执行。"""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import urllib.request
import zipfile

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
SOURCE = OUT/'source_bundle'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def reject_constant(value):
    raise ValueError('非法JSON常量：'+value)


def main():
    now = datetime.now(timezone.utc).isoformat()
    checkpoint = read(SOURCE/'source_checkpoint_037/checkpoint.json')
    assert len(checkpoint['sources']) == 390
    assert all(sha(PROJECT/p) == h for p, h in checkpoint['sources'].items())
    for name, expected in checkpoint['auxiliary_sources'].items():
        actual = SOURCE/name if (SOURCE/name).exists() else SOURCE/'github_sync_001'/name
        assert sha(actual) == expected
        assert sha(SOURCE/'source_checkpoint_037/reproduction_sources'/name) == expected
    assert all(sha(SOURCE/'github_sync_001/checkout'/p) == h
               for p, h in checkpoint['sources'].items())
    assert sha(PROJECT/'.git/index') == checkpoint['original_index_sha256']
    originals = {
        'MEMORY.md': 'f1db9a84f9b586fff70f61d09cd3bdda78d4b2cac953fbde8af45533ed034fd6',
        'PAPER_MEMORY.md': '2a978751be1efd888a130444bc31f146666c7f4fcb62e37c2f92aa2fe196f2b2',
        '初稿.docx': '3301e4fa163a7c0b5ddea37848b21f6e4c770a24e6b1ec54130b02549949095f',
        '论文1-3章初稿.md': '87b47a096a524e8e08b37d727285840e47e768da095ff123283b0fbc9a9a1ee7',
    }
    assert all(sha(PROJECT/p) == h for p, h in originals.items())
    roots = [OUT/'faults'/f'perceived_blockage_cpu_{i:03d}' for i in [1, 2]]
    releases = [r/'release.json' for r in roots]
    releases += [roots[1]/'recovery_001/release.json']
    frozen_count = 0
    for path in releases:
        release = read(path)
        for key in ['source_sha256', 'auxiliary_source_sha256']:
            for name, expected in release.get(key, {}).items():
                assert sha(PROJECT/name) == expected, name
                frozen_count += 1
    raw_roots = roots+[OUT/'visualizations/perceived_blockage_cpu_002',
        OUT/'visualizations/perceived_blockage_browser_qa_001',
        OUT/'visualizations/perceived_blockage_browser_qa_002']
    partial = 'faults/perceived_blockage_cpu_002/rule_blockage/independent_audit_001/barrier_sensor_and_path_details.json'
    failures, structured_files, structured_records = {}, 0, 0
    for root in raw_roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or not (path.suffix in {'.json', '.jsonl'} or path.name.endswith('.json.gz')):
                continue
            relative = str(path.relative_to(OUT))
            try:
                text = gzip.decompress(path.read_bytes()).decode() if path.name.endswith('.json.gz') else path.read_text()
                chunks = text.splitlines() if path.suffix == '.jsonl' else [text]
                for chunk in chunks:
                    json.loads(chunk, parse_constant=reject_constant)
                structured_files += 1
                structured_records += len(chunks)
            except (ValueError, UnicodeError) as error:
                failures[relative] = dict(error=str(error), sha256=sha(path),
                    expected_historical_failed_partial=True)
    assert set(failures) == {partial}, failures
    combined = read(roots[1]/'recovery_001/complete.json')
    assert combined['execution_complete'] and combined['all_required_exposures_passed']
    assert not combined['existing_rule_rerun'] and combined['actual_CPU_Qwen_calls'] == 1
    input_audit = read(roots[1]/'delayed_actual_rank/input_audit_003/report.json')
    assert input_audit['passed'] and input_audit['actual_new_model_calls_added_by_audit'] == 0
    for condition in ['rule_blockage', 'delayed_actual_rank']:
        assert read(roots[1]/condition/'geometry_witness_002/report.json')['passed']
    qa = read(OUT/'visualizations/perceived_blockage_browser_qa_002/report.json')
    assert qa['passed'] and len(qa['checks']) == 8 and qa['all_requests_GET']
    main_rows = [json.loads(line) for line in
        (OUT/'interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl').read_text().splitlines()]
    prefix = read(OUT/'vehicle_execution/semantic_minus_execution_full_001/progress.json')
    recent_mean = statistics.mean(r['process_wall_s'] for r in main_rows[-50:])
    remaining_h = (450-len(main_rows))*recent_mean/3600
    previous = read(SOURCE/'continuation_checkpoint_055.json')
    processes = []
    for entry in previous['process_snapshot']:
        pid = entry['pid']
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        actual = fields[19] if fields[0] != 'Z' else None
        assert actual == entry['expected_start_ticks'], pid
        processes.append(dict(entry, actual_start_ticks=actual, same_original_process_alive=True))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    human = {}
    for endpoint in ['review_status', 'natural_collection_status']:
        human[endpoint] = json.load(opener.open('http://127.0.0.1:8768/api/'+endpoint, timeout=10))
    protection = dict(updated_utc=now, business_files_checked=390,
        auxiliary_checkpoint_files_checked=len(checkpoint['auxiliary_sources']),
        Git_export_files_checked=390, original_index_unchanged=True, original_documents_sha256=originals,
        fault_release_source_references_checked=frozen_count,
        full_255170_entry_historical_audit_pending_final=True, manuscript_text_written=False, goal_complete=False)
    save(SOURCE/'perceived_blockage_protection_001.json', protection)
    structured = dict(passed_with_explicit_historical_partial=True, valid_structured_files=structured_files,
        valid_JSON_or_JSONL_records=structured_records, expected_failed_partials=failures,
        failed_partial_deleted_or_modified=False, actual_new_model_calls=0, goal_complete=False)
    save(SOURCE/'perceived_blockage_structured_validation_001.json', structured)
    handoff = '''# 真实屏障与迟到排序CPU开发实验交接

四次真实执行全部保留。第一版白色目标两次未形成故障前提，不能计通过；第二版红色目标规则与迟到排序两个条件暴露审核通过。红色条件合计48次真实感知前端、4931个控制区间、1次CPU基础Qwen调用。两者均为生成路线、选对对象、正确锁定为真，实际到达为假。

实际车辆先移动，再在观测间断时由原TTL刹停。停车后环境加入真实渲染屏障，代理通过实际深度/LiDAR估计障碍，清除受阻路线。迟到排序使用真实模型输出；该输出在新障碍发布后返回，因为候选/几何安全上下文改变被原处理逻辑拒绝。独立连续矩形车体和行驶方向审核通过，最终速度为零，没有新增到达信用。

规则条件的后轴中心线不穿屏障，但旧整车路径相撞，而且所有可接受终点姿态的车体均相撞；迟到排序条件的全部终点区域与起点分量被屏障隔断。独立解析证明和交叉采样见两个 geometry_witness_002/report.json，不能只用本规划器失败定义不可达。

限制：一个配置、固定种子的后验开发补充；先TTL停车再出现屏障，不代表运动中突发障碍安全。CPU感知TTL15秒、固定240秒观察窗，不混入正式GPU的1.5秒TTL实验、主450到达率、独占性能或新封存测试。

原审核NumPy计数导致的部分JSON、后轴点与整车判据修正、输入提示原文审核失败、手机菜单溢出均保留。最终输入审核003按原异步调度排序冻结JSON恢复提示键顺序，核对精确提示、候选安全证据与真实模型原文，未改原请求或重跑模型。最终查看器002采用短中文菜单名和宽度约束，桌面/手机共8个案例视口检查通过，114个请求全为GET，GT默认隐藏。

## 入口

- 实际完成汇总：faults/perceived_blockage_cpu_002/recovery_001/complete.json
- 两个原始独立审核：faults/perceived_blockage_cpu_002/{rule_blockage,delayed_actual_rank}/independent_audit_003/report.json
- 模型输入审核：faults/perceived_blockage_cpu_002/delayed_actual_rank/input_audit_003/report.json
- 完整回放：visualizations/perceived_blockage_cpu_002/index.html；8768只读服务下打开此路径。
- 浏览器验收：visualizations/perceived_blockage_browser_qa_002/report.json
- 结构化文件与历史失败清单：source_bundle/perceived_blockage_structured_validation_001.json
- 保护核验：source_bundle/perceived_blockage_protection_001.json

原worker、runner、release、观察图像和控制记录均随本增量包保留。按原协议可在新版本目录重新执行，不能覆盖原路径、把重跑当独立样本，或把CPU数据算成GPU故障验收。
'''
    handoff_path = OUT/'真实屏障与迟到排序_CPU交接_001.md'
    with handoff_path.open('x') as stream:
        stream.write(handoff)
    state = deepcopy(read(OUT/'实验完成度与交付清单_019.json'))
    assert all(sha(OUT/p) == h for p, h in state['evidence_sha256'].items())
    state.update(updated_utc=now, previous_checklist='实验完成度与交付清单_019.json',
        semantic_minus_recorded_snapshot=len(main_rows), semantic_minus_independently_verified_prefix_cases=prefix['audited_cases'],
        semantic_minus_current_incremental_audit=prefix, process_snapshot=processes,
        human_snapshot=human, available_disk_bytes=shutil.disk_usage(OUT).free,
        perceived_blockage_CPU_summary=combined, perceived_blockage_browser_QA_passed=True,
        scheduling_policy='不新增可选实验；原450消融→GPU故障→独占性能顺序继续，缺人工项单列',
        observed_recent50_case_mean_s=recent_mean, estimated_remaining_ablation_h=remaining_h,
        estimate_is_not_a_completion_promise=True, manuscript_text_written=False, goal_complete=False)
    for item in state['items']:
        if item['name'] == '语义消融':
            item['implemented'] = f'原GPU缺一完成{len(main_rows)}/450；原进程仍存活；独立审核前缀{prefix["audited_cases"]}例。'
        if item['name'] == '缺一450独立车身方向与保持审核':
            item['implemented'] = f'已审核{prefix["audited_cases"]}/450；到达{prefix["counts"]["reported_arrival"]}例均有保持证据；未替部分矩阵生成终态结果。'
    state['items'].append(dict(stage=12, name='真实屏障与迟到排序CPU开发补充', status='两次红目标暴露与四次完整回放通过',
        implemented='48真实前端、4931控制区间、1真实CPU排序；两例安全停车，未到达；白目标两次未暴露保留。',
        remaining='不替代正式GPU故障、新封存任务或更广动态交通安全。',
        evidence=['真实屏障与迟到排序_CPU交接_001.md', 'faults/perceived_blockage_cpu_002/recovery_001/complete.json']))
    evidence = [handoff_path, SOURCE/'perceived_blockage_protection_001.json',
        SOURCE/'perceived_blockage_structured_validation_001.json',
        roots[1]/'recovery_001/complete.json', roots[1]/'delayed_actual_rank/input_audit_003/report.json',
        OUT/'visualizations/perceived_blockage_browser_qa_002/report.json']
    state['evidence_sha256'].update({str(p.relative_to(OUT)):sha(p) for p in evidence})
    save(OUT/'实验完成度与交付清单_020.json', state)
    status = f'''# 当前Goal进度、风险与收尾安排（第20版）

更新：{now}。实验优先，不写论文正文/章节/框架。整体验收未完成，goal_complete=false。

## 已实际完成

|工作|证据范围|
|---|---|
|阶段9对象、关联与场景图|38开发场景、11log、1508帧；因果建图、关系、身份返回、故障/地图消融已评价|
|阶段10训练和公平对照|3种子Adapter真实CUDA训练；基础Qwen、旧Adapter、新Adapter公平三任务验证；主矩阵450条件结束，449有结果、1次超时计失败|
|阶段11车辆与交互|运动约束、实际控制、连续矩形车体/方向/终点保持核验；交互控制台；规则50次到达覆盖27不同正任务、10配置|
|KITTI|Object64帧、Tracking90连续帧真实车载开发诊断及独立审核；不宣称官方AP或语言导航到达|
|故障与复现|CPU语义故障、三任务缓存、隔离源码实际导航复演、屏障/迟到排序实际执行和回放验收|
|交付准备|390文件源码及Git导出、15张可编辑实验表、114行、15页PDF；主450和新增故障桌面/手机查看器|

主开发正任务每方法90例，生成路线/选对对象/正确锁定/实际到达分别为：规则83/84/82/50，基础Qwen66/72/66/39，新Adapter59/33/32/4。新Adapter未带来闭环改善；不把训练或结构化验证提升写成导航提升。此为开发合成任务，不是新自然指令封存结果。

## 当前主要耗时

原GPU消融完成{len(main_rows)}/450，独立审核{prefix['audited_cases']}/450；原8个进程身份一致，队列未重启。最近50例平均{recent_mean:.1f}秒/例，按相同速度剩余消融约{remaining_h:.1f}小时。每例包含真实感知/规划/控制，观察预算90秒，另有进程和模型加载开销；为保持正式计时条件，不并发正式模型。

之后仍有6例GPU故障试跑、180例正式故障及独占性能测量。若都顺利，剩余机器实验粗估12—16小时，含消融，尚不含新封存实验及人工采集审核；这是排程估计，故障试跑失败或门槛未过需单独处理，不能承诺完整课题同一时刻完成。

## 验收缺口和风险

- 人工审核{human['review_status']['human_submissions']}，自然表达提交{human['natural_collection_status']['human_authored_submissions']}，有效新测试指令{human['natural_collection_status']['qualified_test_queries']}。原任务最低20新场景×20有效指令=400条，当前采集计划30场景×20=600条；人工缺项不能由模板或自动标签替代。
- 新Adapter开发到达4/90，为实质负结果；不保证继续增加同类训练就能改善，当前先完成冻结消融以定位模块影响。
- 一个主矩阵进程超时、部分历史初态缺记录、近似GT时间对齐等限制保留；KITTI和CPU补充不充当独立封存、正式GPU或完整交通安全证据。
- 消融结束时存在已知旧源集合包含一个未使用文件的断言差异；保留原失败，兼容审核003及完整750行复核已预检、排队，不修改冻结业务或实验release。
- 目前余量约{shutil.disk_usage(OUT).free/1024**3:.1f}GiB，磁盘暂不阻塞；全255170项历史保护与最终回归仍待机器实验终态后执行。

## 收尾安排

不再增加可选实验。原队列按消融→正式故障→独占性能执行；同步完成原始证据、失败账目和增量交付。人工审核及新封存依赖单列，机器独立工作继续。当前源码390文件未推送；按已授权范围在代码实验核验收尾后同步到 https://github.com/lzccc666/lzc ，再提交完整工作交接。最近公开提交仍为f75725f8fd422ec619775bfa0407dc6733312488。

新增屏障实验见《真实屏障与迟到排序_CPU交接_001.md》；两次红目标合计48感知前端、4931控制区间和1次真实CPU Qwen，均安全停车、无到达，白目标两次未暴露保留。输入审核003和查看器002已核验；辅助脚本失败和一个部分JSON显式登记，未删除或伪装通过。
'''
    with (OUT/'实验完成度与交付清单_020.md').open('x') as stream:
        stream.write(status)
    continuation = deepcopy(previous)
    continuation.update(updated_utc=now, checklist='实验完成度与交付清单_020.md',
        semantic_minus_recorded=len(main_rows), semantic_minus_incremental_audit=prefix,
        process_snapshot=processes, human=human, perceived_blockage_completed=combined,
        optional_experiment_expansion_stopped=True, manuscript_text_written=False, goal_complete=False)
    save(SOURCE/'continuation_checkpoint_056.json', continuation)
    paths = {p for root in raw_roots for p in root.rglob('*') if p.is_file()}
    paths.update(p for p in SOURCE.glob('perceived_blockage*') if p.is_file()
                 and p.name != 'perceived_blockage_finalize_002.log')
    history = SOURCE/'perceived_blockage_format_history_001'
    paths.update(p for p in history.rglob('*') if p.is_file())
    paths.update(p for p in (OUT/'task_protocol').glob('perceived_blockage*.md'))
    paths.update(evidence+[OUT/'实验完成度与交付清单_020.md', OUT/'实验完成度与交付清单_020.json',
        SOURCE/'continuation_checkpoint_056.json'])
    hashes = {str(p.relative_to(OUT)):sha(p) for p in sorted(paths)}
    archive = OUT/'真实屏障与迟到排序_CPU交接增量包_001.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, str(path.relative_to(OUT)))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None and len(bundle.namelist()) == len(hashes)
        assert all(hashlib.sha256(bundle.read(p)).hexdigest() == h for p, h in hashes.items())
    receipt = dict(created_utc=now, archive=str(archive.relative_to(OUT)), sha256=sha(archive),
        bytes=archive.stat().st_size, files=len(hashes), file_sha256=hashes,
        explicitly_preserved_failed_partial=partial, active_finalizer_log_included=False,
        new_model_calls=0, new_navigation_runs=0, new_arrival_credit=0,
        manuscript_text_written=False, goal_complete=False)
    save(SOURCE/'perceived_blockage_bundle_001.json', receipt)
    print(json.dumps({k:v for k, v in receipt.items() if k != 'file_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
