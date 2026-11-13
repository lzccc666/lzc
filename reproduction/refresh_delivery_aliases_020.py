"""更新四个必需交付入口，保存完整旧版本，不改任何实验或论文稿。"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

OUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
SOURCE = OUT/'source_bundle'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    now = datetime.now(timezone.utc).isoformat()
    snapshot = SOURCE/'status_aliases_before_020'
    snapshot.mkdir(exist_ok=False)
    names = ['validation_report.json', 'experiment_report.md', 'remaining_issues.md', 'reproducible_commands.md']
    before = {}
    for name in names:
        before[name] = sha(OUT/name)
        shutil.copy2(OUT/name, snapshot/name)
        assert sha(snapshot/name) == before[name] == sha(OUT/name)
    state = read(OUT/'实验完成度与交付清单_020.json')
    assert all(sha(OUT/p) == h for p, h in state['evidence_sha256'].items())
    prefix_path = OUT/'vehicle_execution/semantic_minus_execution_full_001/progress.json'
    prefix = read(prefix_path)
    raw_path = OUT/'interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl'
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    save(snapshot/'semantic_minus_prefix_snapshot.json', prefix)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    human = {name:json.load(opener.open('http://127.0.0.1:8768/api/'+name, timeout=10))
             for name in ['review_status', 'natural_collection_status']}
    save(snapshot/'human_status_snapshot.json', human)
    for row in state['process_snapshot']:
        fields = (Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        assert fields[0] != 'Z' and fields[19] == row['expected_start_ticks']
    validation = deepcopy(read(snapshot/'validation_report.json'))
    validation.update(updated_utc=now, current_checklist='实验完成度与交付清单_020.json',
        current_machine_snapshot=dict(semantic_minus_completed=len(rows), semantic_minus_expected=450,
            independently_audited=prefix['audited_cases'], snapshot_sha256=sha(snapshot/'semantic_minus_prefix_snapshot.json'),
            original_processes_same=8, GPU_faults_and_performance_still_pending=True),
        human_status_snapshot=human, manuscript_text_written=False, goal_complete=False)
    statuses = validation['statuses']
    statuses['独立消融']['remaining'] = (f'当前实际{len(rows)}/450，独立审核{prefix["audited_cases"]}/450；'
        '完整750组合与独立统计待全部原始条件结束；原源集合断言差异保留，兼容003已预检并排队。')
    statuses['故障安全'].update(status='离线全量及CPU真实闭环补充完成，正式GPU仍待执行',
        remaining='CPU语义故障八条件、真实屏障与迟到排序已实际运行并独立审核；固定任务、CPU TTL15秒和开发补充不代替GPU1.5秒TTL、180条件或更广交通安全。')
    statuses['故障安全']['evidence'] += [
        'faults/semantic_async_cpu_003/delivery_001/summary.json',
        'faults/semantic_async_cpu_003/delivery_001/independent_call_and_execution_ledger_001.json',
        'faults/perceived_blockage_cpu_002/recovery_001/complete.json',
        'faults/perceived_blockage_cpu_002/delayed_actual_rank/input_audit_003/report.json',
        'faults/perceived_blockage_cpu_002/rule_blockage/geometry_witness_002/report.json',
        'faults/perceived_blockage_cpu_002/delayed_actual_rank/geometry_witness_002/report.json',
        'visualizations/perceived_blockage_browser_qa_002/report.json',
        'source_bundle/perceived_blockage_structured_validation_001.json']
    statuses['开发实验表与来源核对'] = dict(status='15表114行和15页PDF已实际导出并独立核验',
        evidence=['experiment_tables/development_evidence_003/complete.json',
            'experiment_tables/development_evidence_003/independent_audit_001/report.json',
            'source_bundle/development_result_tables_final_verification_002.json'],
        remaining='仅开发证据。尚无新的封存主表；Cyclist 3D关联6/20与7/20，2D共同条件对0，不写成零3D匹配。')
    if '交付复现环境' in statuses:
        statuses['交付复现环境'].update(status='独立源码实际完整任务通过，复用现有Python环境',
            remaining='便携入口、390源码、模型/资产SHA和472发行包版本清单可用；尚未在新的Python虚拟环境或新操作系统完成，不把源码隔离写成依赖环境隔离。')
    statuses['新的Python环境复现'] = dict(status='待实际执行',
        evidence=['reproduction/standalone_portable_release_001/README.md',
            'reproduction/portable_executed_001/independent_audit_001/report.json'],
        remaining='原便携实际执行使用既有Python/OSMesa。完整复制现有site-packages约11GiB，加上剩余主队列预计保存量会挤占正式GPU故障冻结16GB余量；暂不复制或安装，先完成原队列。后续选择有界隔离依赖准备并实际执行，不能只生成环境清单。')
    evidence = {}
    for row in statuses.values():
        for name in row.get('evidence', []):
            assert (OUT/name).is_file(), name
            evidence[name] = sha(OUT/name)
    validation['evidence_files_checked'] = len(evidence)
    validation['evidence_sha256'] = evidence
    staged = snapshot/'new_alias_contents'
    staged.mkdir()
    save(staged/'validation_report.json', validation)
    overview = (OUT/'实验完成度与交付清单_020.md').read_text()
    overview = overview.replace('# 当前Goal进度、风险与收尾安排（第20版）', '# 实验报告与交付入口（快照020）', 1)
    overview += ('\n\n本报告是冻结进度快照，不是整个Goal完成声明。完整旧报告已保留在'
        '`source_bundle/status_aliases_before_020/experiment_report.md`。本次只更新交付文档，不新增论文文字、模型调用或执行样本。\n')
    (staged/'experiment_report.md').write_text(overview)
    remaining = f'''# 待完成与风险（交付快照020）

更新时间：{now}。goal_complete=false；论文正文、章节和整体框架按用户要求暂缓。

|必需工作|当前证据与缺口|
|---|---|
|完整语义消融|{len(rows)}/450实际结束，独立审核{prefix['audited_cases']}/450；完整750组合与终态统计仍待原队列结束|
|正式GPU故障|6例smoke门槛及180例正式条件尚未执行；CPU试跑和屏障补充不替代|
|独占性能|12个前端、216个新Qwen和216个缓存请求主测尚未执行；共享CPU/主导航耗时不能充当该性能表|
|真实人工标签|审核{human['review_status']['human_submissions']}，自然表达{human['natural_collection_status']['human_authored_submissions']}，有效新指令{human['natural_collection_status']['qualified_test_queries']}；需真实审核及独立标签|
|新封存|最低20新场景、每场景20有效指令、总计至少400；方法/模型/指标和数据完整冻结后才开启；自然分布与均衡挑战分列|
|独立依赖环境|已在隔离390源码运行完整真实感知导航，但仍复用已有Python/OSMesa；新的Python环境或新系统尚无实际成功证据|
|最终交付|新封存主表、统一终态统计/可视化、全255170项保护、最终回归/结构校验、Git同步和完整交接待完成|

新Adapter开发正任务实际到达4/90，低于基础Qwen39/90和规则50/90；保留负结果。一个主450进程超时保留在分母。主450的54个初态重建案例与原始完整初态分列，所有安全主张限于已保存观测、车辆模型、方向与地图；不宣称未知连接100%正确。KITTI为对象和关联诊断，不是官方AP/HOTA或语言导航到达。

两个红目标屏障条件有真实LiDAR及模型响应证据，但均未到达；白目标两次未暴露仍保留。辅助审核一个部分JSON保存在原路径，已作为历史失败显式登记，不当有效JSON或成功结果。

为尽快收尾，不新增可选实验，不改原冻结任务、预算或成功标准；正式模型按原队列执行。新环境复制和安装暂不占用正式GPU故障所需存储余量。当前8个原进程身份一致；磁盘约31GiB，尚无即时空间阻塞。

旧完整待办记录保存在 `source_bundle/status_aliases_before_020/remaining_issues.md`。最新详细证据见 `实验完成度与交付清单_020.json`、`validation_report.json`，状态均不能由文件存在自动提升为完成。
'''
    (staged/'remaining_issues.md').write_text(remaining)
    commands = '''# 实验复现与交付入口（快照020）

当前Goal未完成，正文暂缓。以下示例使用已经核验的已有Python环境；不宣称新系统或新虚拟环境已复现。所有新增输出放在本Goal目录，已有失败和结果不得覆盖。完整历史训练、录制建图、KITTI和全量实验命令已原样保留在 `source_bundle/status_aliases_before_020/reproducible_commands.md`。

## 阅读已完成证据

- 当前进度及分项：`实验完成度与交付清单_020.md/json`、`validation_report.json`。
- 15张开发实验表：`experiment_tables/development_evidence_003/README.md`，同目录含Word、Excel、PDF、逐表CSV与来源。
- 主450全部尝试：`statistics/realtime_three_model_full_001/trials.csv`；449完整、1缺失，失败在分母。
- 主450查看器：8768服务下的 `/visualizations/realtime_main450_preview_003/index.html`。
- KITTI：`KITTI连续车载实验_交接_001.md`（若具体入口变化，以当前清单中KITTI条目的证据路径为准）。
- 屏障/迟到排序：`真实屏障与迟到排序_CPU交接_001.md`及 `/visualizations/perceived_blockage_cpu_002/index.html`。
- 已有便携小任务：`reproduction/standalone_portable_release_001/README.md`。

## 在同一机器准备一个全新的源码完整任务

这个入口从390源码发布包重建，不复制历史帧、检测或轨迹；三个前端权重和第三方资产从合法本地路径只读使用。先确认余量，原GPU正式队列活跃时不要另启GPU模型。不要复用示例输出目录；如果已存在，换新的独立版本号。

```bash
TASK_PROJECT=/root/autodl-tmp/lizichong
TASK_OUT="$TASK_PROJECT/outputs/project_v2/thesis_completion_v1"
TASK_RELEASE="$TASK_OUT/reproduction/standalone_portable_release_001"
TASK_PY=/root/miniconda3/envs/conceptgraph/bin/python
mkdir "$TASK_OUT/reproduction/portable_user_source_001"
tar -xzf "$TASK_RELEASE/source_current.tar.gz" -C "$TASK_OUT/reproduction/portable_user_source_001"
"$TASK_PY" "$TASK_RELEASE/prepare_portable_navigation_001.py" \
  --release "$TASK_RELEASE/release.json" \
  --source-root "$TASK_OUT/reproduction/portable_user_source_001" \
  --model-dir "$TASK_PROJECT/checkpoint" \
  --asset /root/autodl-fs/third_party/urban_scene_assets/ToyCar_002.glb \
  --runtime-root "$TASK_OUT/source_bundle/software_render_runtime_001" \
  --python "$TASK_PY" \
  --extra-library-path /root/autodl-fs/third_party/graphics_runtime/lib \
  --extra-library-path /root/miniconda3/envs/conceptgraph/lib \
  --destination "$TASK_OUT/reproduction/portable_user_run_001" \
  --execute
```

若只要预检，移除 `--execute` 并使用独立新目录；预检不算实际运行。检查 `launcher_exit.json` 与实际case的 `result.json`、`import_audit.json`、`execution.jsonl`、`perception.jsonl` 和 `thread_observations.jsonl`。Python退出0不等于正确到达。路线、对象、锁定和实际到达分开；安全和保持须独立复算。已完成参考在 `reproduction/portable_executed_001/independent_audit_001/report.json`，停车保持扩展为后验开发诊断，不回填主450。

## 从全部冻结450记录重算主统计

无需模型推理，但必须保留原450 release、全部尝试账目和449原case；摘要不能代替原始记录。输出run-id须尚未使用。

```bash
cd /root/autodl-tmp/lizichong
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_statistics \
  --input-run realtime_three_model_full_001 --run-id main_statistics_reproduce_001
```

结果为配置聚类开发统计；不能把种子、帧或模型调用当独立城市样本。尚在运行的缺一矩阵不能用此命令生成终态统计。

## 原活跃队列与人工入口

原队列自动按450语义消融→6例GPU故障smoke→180故障→独占性能执行。不要因观察超时、旧快照或等待文件而重启；先核对PID及/proc启动时刻。进度在 `vehicle_execution/semantic_minus_execution_full_001/progress.json`，原任务账目在 `interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl`；最终统计/故障/性能只有实际完整结束才计完成。

服务8768的人工核验为 `/annotation/natural_command_review_001/index.html`，独立自然表达采集为 `/annotation/human_natural_collection_001/index.html`，开发当前对象图审核为 `/visualizations/causal_development_review_002/index.html`。远程使用端口映射；网页具备提交功能不等于已收到真实人工记录。交互导航服务8769见 `交互仿真_操作与复现_001.md`。

## 工程核验

普通测试不加载真实大模型。最近原机452通过，纯代码导出450通过、2个真实数据跳过；最终队列结束后再进行所需终态检查。

```bash
cd /root/autodl-tmp/lizichong
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/conceptgraph/bin/python -m pytest -q tests
git diff --check
```

新的Python环境仍待实际准备/执行；472包版本清单不是自动安装成功证明。当前完整复制site-packages约11GiB，会挤占剩余正式实验冻结余量，因此不在原队列运行时做全量复制。模型、资产与系统图形依赖的SHA和外部路径见便携release；不自动下载或写回历史模型。
'''
    # KITTI交接用现有、可验证入口，避免凭印象写文件名。
    kitti = next(item for item in state['items'] if item['name'] == 'KITTI连续车载外部诊断')
    commands = commands.replace('`KITTI连续车载实验_交接_001.md`（若具体入口变化，以当前清单中KITTI条目的证据路径为准）',
        '当前清单的“KITTI连续车载外部诊断”条目；原始证据 `'+kitti['evidence'][0]+'`')
    (staged/'reproducible_commands.md').write_text(commands)
    for name in names:
        assert sha(OUT/name) == before[name], '源入口在准备期间变化'
        os.replace(staged/name, OUT/name)
    after = {name:sha(OUT/name) for name in names}
    assert all(sha(OUT/p) == h for p, h in state['evidence_sha256'].items())
    save(SOURCE/'delivery_alias_refresh_020.json', dict(updated_utc=now, before_sha256=before,
        after_sha256=after, old_contents_snapshot=str(snapshot.relative_to(OUT)),
        status_evidence_files_verified=len(evidence), all_previous167_evidence_references_unchanged=True,
        original_processes_same=8, actual_model_calls=0, new_navigation_runs=0,
        manuscript_text_written=False, goal_complete=False))
    print(json.dumps(dict(updated=names, completed_minus=len(rows), verified_evidence_files=len(evidence),
        original_processes_same=8, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
