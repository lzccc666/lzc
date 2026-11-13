"""归档旧交接入口，登记已完成机器实验及仍未满足的人工/封存条件。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT/name).read_text())


def main():
    validation = load('validation_report.json')
    for name, value in validation['evidence_sha256'].items():
        assert sha(ROOT/name) == value, name
    assert load('source_bundle/final_tests_003/complete.json')['passed']
    assert load('source_bundle/final_protection_003/report.json')['passed']
    assert load('reproduction/isolated_environment_after_GPU_005/complete.json')['complete']
    assert load('visualizations/thesis_portal_browser_qa_021/report.json')['passed']
    archive = ROOT/'source_bundle/status_aliases_before_024'
    archive.mkdir(exist_ok=False)
    names = ['validation_report.json', 'remaining_issues.md', 'experiment_report.md',
             'reproducible_commands.md', '工作交接_代码与实验.md', 'GPT_Work_写作交接.md',
             'visualizations/index.html']
    saved = {}
    for name in names:
        p = ROOT/name
        if p.exists():
            dest = archive/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            assert sha(p) == sha(dest)
            saved[name] = sha(dest)
    (archive/'manifest.json').write_text(json.dumps(dict(files=saved,
        old_evidence_verified=len(validation['evidence_sha256'])), ensure_ascii=False, indent=2)+'\n')
    evidence = [
        'interactive_simulation/realtime_faults_gpu_full_001/summary.json',
        'statistics/realtime_faults_gpu_full_001/report.json',
        'faults/fault_event_full_independent_001/report.json',
        'faults/fault_sensor_numeric_full_001/report.json',
        'vehicle_execution/fault_execution_full_independent_001/report.json',
        'performance/runtime_profile_after_faults_002/complete.json',
        'reproduction/isolated_environment_after_GPU_005/complete.json',
        'reproduction/isolated_cpu_executed_002/independent_audit_001/report.json',
        'source_bundle/final_tests_003/complete.json', 'source_bundle/final_protection_003/report.json',
        'source_bundle/source_checkpoint_038/checkpoint.json',
        'visualizations/gpu_fault_evidence_full_browser_002/report.json',
        'visualizations/thesis_portal_browser_qa_021/report.json',
        'experiment_tables/fault_performance_office_002/independent_audit_001/report.json',
        'source_bundle/github_sync_001/incremental_export_verification_029.json',
    ]
    def status(key, state, files, remaining):
        validation['statuses'][key] = dict(status=state, evidence=files, remaining=remaining)
    fault_files = evidence[:5]
    for key in ['故障安全', '真实时钟传感器故障']:
        status(key, 'GPU六条件180次完整记账及三类独立审计完成', fault_files,
            '179完整、1进程超时终态未知；10次原报告到达获保持证书，88例初态显式重建。29次实际断流均停车并恢复运动；未触发与未知不获故障安全信用。仅本仿真范围，不证明动态道路普遍安全。')
    status('性能', '固定输入单CUDA进程分阶段实测完成', [evidence[5]],
        '12前端、216真实Qwen、216缓存请求且新增调用0；冷启动各1次。报告中位/P95/阶段吞吐及显存，不等于闭环实时吞吐，抽样间隙外部GPU活动无法排除。')
    status('新的Python环境复现', '新venv真实CPU完整导航与主450重算通过', evidence[6:8],
        '38前端、187控制记录，正确到达保持2秒；四份450主表文件逐字节一致。复制依赖隔离，补齐YOLO惰性CLIP；CUDA构建但CPU执行、CUDA未初始化。共享基础解释器标准库和原生库，不是新操作系统；复演不新增主表到达信用。')
    status('测试', '452项全量测试、业务范围Ruff及diff检查通过', [evidence[8]],
        '原451通过/1失败保留，修正测试临时路径假设后452通过；只修改1个测试文件，业务运行源码不变。第三方全库Ruff不在此通过范围。')
    status('旧产物保护', '最终255170文件、56657631355字节SHA核验通过', [evidence[9]],
        '阶段1—8、模型、原论文稿和原Git索引未改；旧baseline_verification.json未覆盖。')
    status('统一可视化', '门户017与180故障完整查看器通过只读浏览器检查', evidence[11:13],
        '门户桌面/手机及79静态链接；故障180案例、19379图像与186项检查。交互控制另用8769，人工采集8768、审核8770；查看证据不增加到达或人工标签。')
    status('统一实验与交互入口', '门户017已连接完成实验与交互入口', [evidence[12]],
        '论文正文暂缓，人工及新封存门槛未完成。')
    status('开发实验表与来源核对', '原15表、3张消融表及7张故障/性能表已导出',
        ['experiment_tables/development_evidence_003/independent_audit_001/report.json',
         'experiment_tables/semantic_ablation_office_001/export_verification.json', evidence[13]],
        '新增7表57行、7页PDF、291单元格及40组原始计时重算通过；可编辑Word/Excel。非论文正文，非新封存主表。')
    status('统计', '主450、组合750及GPU故障180开发统计已完成',
        ['statistics/main_matrix_evidence_check_001/report.json',
         'statistics/semantic_compatibility_evidence_check_004/report.json', evidence[1], evidence[13]],
        '主450与组合750共享300，合计900唯一执行记录；故障批次另列预算与TTL。配置聚类区间及全部失败保留；新封存主表未完成。')
    status('结构化JSON校验', '全量扫描发现1份既有失败残片，已补注册待最终复核',
        ['source_bundle/structured_artifact_validation_final_002/report.json',
         'source_bundle/failed_json_payload_audit_002/report.json'],
        '242889文件、2499848 JSONL记录首轮扫描；旧70字节失败残片有历史TypeError记录，原字节保留。最终004扫描尚待执行，不以登记替代校验通过。')
    status('代码与实验交接', '390源码及结果增量029已准备，Git最终同步待完成',
        [evidence[10], evidence[14]], '代码提交与推送尚未执行；最终结构化检查后同步。人工与新封存仍缺，整个课题goal_complete=false。')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    human = {}
    for key, port in [('natural_collection_status', 8768), ('review_status', 8768), ('adjudication_status', 8770)]:
        with opener.open(f'http://127.0.0.1:{port}/api/{key}', timeout=10) as response:
            human[key] = json.load(response)
    validation.update(updated_utc=datetime.now(timezone.utc).isoformat(), goal_complete=False,
        new_test_opened=False, manuscript_work_deferred_by_user=True, manuscript_text_written=False,
        current_checklist='当前Goal进度与收尾安排_024.md', human_status_snapshot=human,
        current_machine_snapshot=dict(semantic_minus_attempted=450, semantic_minus_complete_results=449,
            semantic_combined_records=750, reused_main_records=300, GPU_fault_attempted=180,
            GPU_fault_complete_results=179, GPU_fault_missing_results=1, performance_formal_executed=True,
            actual_Qwen_calls=216, cache_requests=216, cache_added_calls=0,
            new_Python_environment_reproduction_passed=True, tests_passed=452,
            protection_passed=True, structured_final_passed=False, code_pushed=False, goal_complete=False),
        previous_entry_archive='source_bundle/status_aliases_before_024/manifest.json')
    validation['evidence_sha256'].update({p: sha(ROOT/p) for p in evidence})
    validation['evidence_files_checked'] = len(validation['evidence_sha256'])
    report = '''# 当前实验交付状态

整个课题仍为 `goal_complete=false`。机器实验已取得下列证据；人工标签和新封存测试没有被替代。论文正文、章节及整体框架按用户安排暂缓。

| 阶段 | 已实现、实际执行 | 未满足的验收 |
|---|---|---|
| 9：对象与图质量 | 38开发场景/11 log、1508帧的过去LiDAR重建；独立对象、关系、跨帧身份、错误证据与因果审计 | 颜色、细属性及歧义仍缺真实人工审核；跨帧关联为官方标签代理，不能当人工身份真值 |
| 10：语义训练 | 2398弱监督训练、603验证，3个种子各240步；5模型共3015次真实CUDA验证；450缺一消融及750组合统计 | 自然语言覆盖和人审标签不足；新Adapter下游没有改善；新封存主表未执行 |
| 11：控制与仿真 | 动作改变位姿与RGB/深度的完整感知闭环、运动约束与独立整车审核；主450全保留；规则和基础Qwen满足最低正确到达功能数量；GPU故障180全保留 | 结果限于合成任务/共享道路家族，不能宣称真实道路普遍安全或所有方法达标 |
| 12：评估与交付 | KITTI静态64帧、连续90帧；开发对比/消融；真实性能、缓存、浏览器、Word/Excel/PDF结果表；新Python环境完整导航与主表重算 | 至少20新场景×20有效指令的新封存测试未开启；最终结构校验和Git同步尚在收尾 |

主对比每方法90个正条件，四类结果为：规则83/84/82/50，基础Qwen66/72/66/39，新Adapter59/33/32/4，顺序是生成路线、选对对象、正确锁定、实际到达。主450含449完整结果与1超时。缺一新增450含449完整与1中断；组合750复用主对比300，不能把450+750称为1200次独立执行。完整新Adapter/减解析/减属性/减重排实际到达为4/48/5/6（各90分母）。负结果保留，不能归结为单字段严格因果效应。

GPU故障六条件各30次：无注入、断流、LiDAR丢失、深度噪声、位姿噪声、检测丢失，实际到达分别3/0/1/2/2/2。179完整、1超时未知；全部180留在分母。29次实际断流均停车并在新鲜观测后恢复运动。10次到达全部通过保持区间审核；88例初态显式重建，不冒充原始观测。预算/TTL与主对比不同，分别报告。

性能实际完成12次GPU前端、216次新Qwen推理和216缓存请求（新增调用0）。新Adapter无缓存解析中位1.875秒、P95 1.989秒；GPU暖态前端中位0.181秒、P95 0.252秒。这些是固定输入分阶段耗时，不能当完整闭环吞吐。

新venv复现完成38次真实CPU前端、187条控制记录和正确到达保持2秒；四份450主表输出逐字节一致。依赖复制隔离、GPU未初始化，标准库/原生系统库共享，PyTorch为cu118构建；不是纯CPU安装包或全新操作系统。先前网络下载中断及缺CLIP失败全部保留。

452项全量测试通过，业务范围Ruff、diff检查通过。仅修正一个测试的临时目录假设，业务运行代码不改。255170个受保护文件（56,657,631,355字节）最终哈希核验通过，原Git索引和历史保护报告未改变。

KITTI结果属于外部诊断：静态64帧/33 drive，3米中心匹配247/319→262/319；卡车误差未改善。连续固定3序列/90帧，74次受支持GT集中在0005的4个实例；其它465次不支持标签保留。不能声称官方AP、3个独立正序列、完整KITTI导航到达或预训练无重叠。

当前最重要的未完成项是**真实人工审核/自然指令及新封存**：截至本次核查，人工表达、审核记录、合格测试指令均为0。至少20个新场景、每场景20条有效指令（至少400条），方法冻结后才可开启；不能用模板、改写或自动标签补成已完成。采集服务8768，独立审核8770，实时交互8769。

统一入口：[门户017](visualizations/thesis_portal_017/index.html)。[主450回放](visualizations/realtime_main450_preview_003/index.html)、[故障180查看器](visualizations/gpu_fault_evidence_full_002/index.html)、[新增故障/性能结果表](experiment_tables/fault_performance_office_002/开发实验结果表.html)。完整原始传感器、模型、失败现场留在本输出目录；Git只同步代码与小型可审阅证据。

当前Git尚未推送；待最终结构化校验后同步用户指定仓库。以`validation_report.json`和后续Git回执为准。历史旧入口已按SHA归档到`source_bundle/status_aliases_before_024/`，其“等待中”描述仅代表当时状态。
'''
    (ROOT/'当前Goal进度与收尾安排_024.md').write_text(report)
    (ROOT/'experiment_report.md').write_text(report)
    (ROOT/'工作交接_代码与实验.md').write_text(report+'\n复现命令见[reproducible_commands.md](reproducible_commands.md)；原始证据与阶段状态见[validation_report.json](validation_report.json)。\n')
    (ROOT/'GPT_Work_写作交接.md').write_text('''# 后续写作的证据交接

本文件提供事实和证据边界，不起草论文正文、目录或框架。

请先阅读[代码与实验交接](工作交接_代码与实验.md)、[验收状态](validation_report.json)及[实验报告](experiment_report.md)。所有结论从冻结结果和逐任务表取数；禁止把生成路径、选对对象、正确锁定和实际到达混用。新Adapter下游负结果、失败分母、未知标签、显式重建初态、重复记录与共享道路家族均需保留。

Word/Excel/PDF表格分别在`experiment_tables/development_evidence_003/`（15表）、`semantic_ablation_office_001/`（3表）和`fault_performance_office_002/`（7表）。表号为开发交付编号；此处不决定论文最终章节编号。CSV、来源哈希与逐条件记录随表保存。

不能写成已完成：真实人工审核、合格自然指令覆盖、至少20新场景/400指令的新封存主表。合成开发任务、KITTI有限子集、官方GT时间近似与实例关联代理必须注明。规则/基础Qwen达到最低仿真功能数量，不代表所有方法或道路工况有效。独立新venv已实际复现，但共享基础标准库及原生系统库，不称全新操作系统。

原论文稿、MEMORY.md、PAPER_MEMORY.md和阶段1—8均保持原样。本轮不输出新正文或框架，后续由用户开展写作。整个课题仍为goal_complete=false。
''')
    (ROOT/'remaining_issues.md').write_text('''# 未完成条件

1. 真实人工颜色/细属性/歧义审核，以及独立审核的自然指令。当前正式提交与合格指令均0；自动证据包不能替代人工记录。
2. 至少20新场景×20有效指令、至少400条，按log/路段/表达家族隔离，模型及协议冻结后开启新测试。当前未开启，不能降门槛。
3. 新封存上的完整主对比/消融、连续回放、分母与统计验收，依赖前两项。
4. 最终结构化校验与Git同步正在收尾；查看validation_report.json及Git回执确认实际终态。
5. 论文正文与框架按用户明确要求暂缓，不能将本轮表格交付算成论文写作完成。

模型没有下游提升的负结果已保留，不通过改标签、删除失败或重复任务增加独立样本数。Goal保持false。
''')
    original_commands = (archive/'reproducible_commands.md').read_text()
    (ROOT/'reproducible_commands.md').write_text('''# 最新复现与交付入口

当前完整机器实验的结果以交接024为准；下方历史命令中的等待状态已过时，原run-id不可覆盖。重跑必须选新目录，并保留原SHA、随机种子及失败分母。原始权重/传感器数据不随Git上传。

- 当前源码：`source_bundle/source_checkpoint_038/source_current.tar.gz`，390文件；业务运行源码与037相同，仅一个测试夹具修正。
- 新环境实测回执：`reproduction/isolated_environment_after_GPU_005/complete.json`。准备/隔离/实际导航的完整argv在其jobs字段；安装004、追加CLIP的来源哈希见`reproduction/isolated_cpu_environment_004/`。固定绝对路径依赖和共享标准库/原生库明确保留。
- 便携参数入口：`reproduction/standalone_portable_release_001/prepare_portable_navigation_001.py --help`，传入release、源码根、模型、资产、图形运行时和新destination。不得把主表统计软链接暴露给导航推理；本次在导航结束后才添加统计只读输入。
- 全量测试：`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests`；452通过。业务检查：`python -m ruff check conceptgraph/completion conceptgraph/navigation tests`、`git diff --check`。
- GPU故障原180、语义缺一450和主450已有冻结release/原始记录，重跑入口沿用下列协议；不要挑选成功种子。
- 最新门户生成器：`source_bundle/thesis_portal_completed_017.py`；故障查看器`gpu_fault_evidence_viewer_004.py`；结果表`fault_performance_office_002.py`。这些程序有不覆盖已有目标的保护。

以下为归档版本的完整命令资料，保留参数来源；其中“进行中”不代表当前状态。历史原文件位于`source_bundle/status_aliases_before_024/reproducible_commands.md`。

---

'''+original_commands)
    (ROOT/'visualizations/index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=thesis_portal_017/index.html"><a href="thesis_portal_017/index.html">打开实验交付入口017</a></html>\n')
    for name in list(validation['evidence_sha256']):
        if name in saved:
            validation['evidence_sha256'][name] = sha(ROOT/name)
    (ROOT/'validation_report.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n')
    (archive/'update_receipt.json').write_text(json.dumps(dict(updated_utc=validation['updated_utc'],
        changed_aliases=names, new_evidence=evidence, old_aliases_preserved=True,
        source_sha256=sha(Path(__file__)), goal_complete=False), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
