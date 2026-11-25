"""逐项保留原Goal范围，核对已关闭证据；不提前给尚未完成的封存测试打分。"""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent
OUT = ROOT/'json/completion_audit_090'
EXECUTION = ROOT/'main_experiments/sealed_execution_075'
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(relative):
    path = ROOT/relative
    value = json.loads(path.read_text())
    SOURCES[relative] = sha(path)
    return value


def main():
    OUT.mkdir(exist_ok=False)
    # 实际读完整索引并逐项检查证据位置，不以首页或文件夹存在代替1000帧覆盖。
    index = ROOT/'kitti_tracking_expanded/graph_query_001/frame_index.csv'
    with index.open() as stream:
        kitti = list(csv.DictReader(stream))
    sequences = Counter(row['sequence'] for row in kitti)
    assert len(kitti) == 1000 and len(sequences) == 10 and set(sequences.values()) == {100}
    file_checks = Counter()
    for seq in sequences:
        rows = [r for r in kitti if r['sequence'] == seq]
        numbers = [int(r['original_frame_id']) for r in rows]
        assert numbers == list(range(numbers[0], numbers[0]+100))
        for row in rows:
            for key in ['rgb', 'depth', 'detection_pickle', 'detection_visualization',
                        'mapping_record', 'graph_query_record']:
                assert Path(row[key]).is_file(), (key, row['sequence'], row['frame_idx'])
                file_checks[key] += 1
    kreport = read('statistics/kitti_expanded_full_001/report.json')
    assert (kreport['frame_count'], kreport['sequence_count']) == (1000, 10)
    nu = read('nuscenes_ablation/statistics_001/report.json')
    assert (nu['scenes'], nu['logs'], nu['frames_per_condition']) == (38, 11, 1508)
    assert len(nu['conditions']) == 6 and nu['actual_new_mapper_calls'] == 9048
    training = []
    for seed in [17, 29, 43]:
        path = f'qwen_adapter_fair_comparison/training/real_graph_seed{seed}_001/training_report.json'
        report = read(path)
        assert report['status'] == 'completed' and report['completed_steps'] == 240
        assert not report['independent_test_used']
        checkpoint = Path(report['best_checkpoint']['path'])/'adapter_model.safetensors'
        assert sha(checkpoint) == report['best_checkpoint']['adapter_sha256']
        training.append(dict(seed=seed, steps=240, checkpoint=str(checkpoint),
                             checkpoint_sha256=sha(checkpoint), independent_test_used=False))
    dataset = read('qwen_adapter_fair_comparison/real_graph_dataset_003/manifest.json')
    for filename, expected in dataset['files'].items():
        assert sha(ROOT/'qwen_adapter_fair_comparison/real_graph_dataset_003'/filename) == expected
    split_logs = {k: set(v) for k, v in dataset['split_logs'].items()}
    names = list(split_logs)
    assert all(not (split_logs[a] & split_logs[b]) for i, a in enumerate(names) for b in names[i+1:])
    # 只检查完成回执中的业务输入/原文/调用数，不读取官方标签或计算准确率。
    with gzip.open(EXECUTION/'requests/inputs.jsonl.gz', 'rt') as stream:
        requests = [json.loads(line) for line in stream if line.strip()]
    request_hashes = {r['sample_id']: r['prompt_sha256'] for r in requests}
    assert len(request_hashes) == len(requests) == 878
    closed_models = []
    variants = ['base', 'old_adapter', 'real_graph_seed17', 'real_graph_seed29', 'real_graph_seed43']
    for index_number, variant in enumerate(variants, 1):
        receipt = EXECUTION/f'stage_{index_number:02d}_receipt.json'
        if not receipt.is_file():
            continue
        assert json.loads(receipt.read_text())['exit_code'] == 0
        relative = f'main_experiments/sealed_execution_075/models/{variant}/report.json'
        report = read(relative)
        output = EXECUTION/'models'/variant/'outputs.jsonl.gz'
        assert sha(output) == report['output_sha256']
        with gzip.open(output, 'rt') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        assert len(rows) == 878 and len({r['sample_id'] for r in rows}) == 878
        assert {r['sample_id']: r['prompt_sha256'] for r in rows} == request_hashes
        assert sum(r['actual_GPU_call'] for r in rows) == report['actual_GPU_calls'] == 878
        assert all(r['raw_output'] is not None and r['GPU_latency_s'] > 0 for r in rows)
        assert report['gt_received'] is False
        closed_models.append(dict(variant=variant, rows=878, actual_GPU_calls=878,
                                  same_request_ids_and_prompt_hashes=True, gt_received=False,
                                  output_sha256=report['output_sha256']))
    carla = read('carla_official/carla_ten_041/report.json')
    assert carla['closed_independently_audited'] == 10
    assert (carla['positive_task_denominator'], carla['strict_safe_arrival']) == (6, 1)
    assert carla['scope_stop']['official_CARLA_and_batch_stopped']
    ca = read('carla_official/identity_geometry_audit_046/report.json')
    assert ca['inference_receives_GT'] is False
    clean = read('reproduction/clean_cpu_navigation_readback_003/report.json')
    assert clean['strict_safe_arrival'] and clean['actual_new_official_navigation_episodes'] == 1
    humans = read('annotation/offline_imports/user_lzc_20261116_001/review_summary_001.json')
    natural_paths = list((ROOT/'annotation/human_generated_natural_submissions_071').glob('*.json'))
    natural = [json.loads(p.read_text()) for p in natural_paths]
    natural_count = len(natural)
    dev = read('qwen_adapter_fair_comparison/real_graph_completion_audit_001/report.json')
    software = read('json/software_validation_070/report.json')
    assert software['returncode'] == 0 and software['source_files_unchanged']
    history = read('json/protected_history_audit_001/report.json')
    assert history['all_protected_paths_match_baseline'] and history['protected_files_checked'] == 255170
    groups = []

    def gate(name, refs, status, supported, missing, evidence):
        groups.append(dict(name=name, original_requirement_refs=refs, status=status,
                           supported_by_evidence=supported, remaining_or_not_proven=missing,
                           evidence_paths=evidence))

    gate('任务定义与对象身份', ['阶段9.1—9.3', '阶段9.6—9.7'], 'partial',
         '版本化协议、独立身份与几何区域判定已用于真实CARLA；明确提议/锁定/路线/到达。',
         '全部自然指令的可接受对象集合、独立可达性及歧义尚未完成；不能以无路径自证不可达。',
         ['carla_official/identity_geometry_audit_046/report.json', 'carla_official/carla_ten_041/report.json'])
    gate('数据独立性与新封存测试', ['数据.1—数据.4', '数据.7—数据.8'], 'partial',
         '30新场景14日志1196帧；历史69场景26日志无log重合；精确/近似crop审计已做；878模板查询已冻结。',
         '模板主集不是人工自然分布；语言家族及地理近路段独立性证据仍不足。203条预选回放尚不能计为完成连续锁定评测。',
         ['main_experiments/sealed_public_058/release.json', 'annotation/historical_log_readback_056/report.json',
          'annotation/actual_sealed_crop_audit_062/report.json'])
    gate('标签覆盖与人工核验', ['数据.5—数据.6', '补充四'], 'partial',
         f'20张原人工颜色/类别审核记录已导入；真实自然表达原始记录当前{natural_count}条。新离线试用包已实际验证60张图片与导出。',
         '自然指令尚无已独立核验的有效标签；颜色样本小，细属性、三维身份、歧义及复核覆盖不足。人工原文不自动等于正确标签。',
         ['annotation/offline_imports/user_lzc_20261116_001/review_summary_001.json',
          'annotation/natural_offline_087/browser_qa_088/report.json'])
    gate('KITTI连续序列', ['补充三.1', '补充验收.KITTI'], 'completed_with_limits',
         '本次逐项复查10序列各100连续原始帧，1000帧的RGB/深度/检测/图像/建图/图查询路径均存在。三组建图有真实评价。',
         '这是10个100帧片段，使用声明的0.1s名义步长；非完整序列、非真实时钟导航、非实际到达；Van未入三类主表。',
         ['kitti_tracking_expanded/graph_query_001/frame_index.csv', 'statistics/kitti_expanded_full_001/report.json'])
    gate('对象图与关系质量', ['阶段9.4—9.5', '阶段9.8', '阶段12.1—12.2', '补充三.2'], 'partial',
         '38场景11日志六单因素配置各1508帧，9048次更新；召回/中心/IDF1/身份切换/关系/耗时已统计。新30场景10组图已保存。',
         '新封存主指标尚待统一计分；静动态/遮挡分层、错合并及全部坐标时效比较仍需逐项终审。旧关系resolved口径不可冒充包含unknown的精确率。',
         ['nuscenes_ablation/statistics_001/report.json', 'main_experiments/sealed_public_058/mapping/'])
    gate('真值辅助诊断', ['阶段9.9'], 'partial',
         '官方几何图诊断、真实感知候选覆盖上限及身份关联不足已单列，不计实际到达。',
         '全部正确对象/正确解析/候选上限条件与最终主表的对应覆盖需进一步核对，不能混入普通方法成绩。',
         ['qwen_adapter_fair_comparison/real_graph_decisions_001/report.json', 'qwen_adapter_fair_comparison/real_graph_authorization_audit_002/report.json'])
    gate('新训练与选模', ['阶段10.2', '阶段10.4—10.5', '补充三.4.训练'], 'completed_with_limits',
         '真实感知图459训练/171验证/108开发留出；日志互斥；seed17/29/43各240步，本次复核三份Adapter哈希。仅验证选checkpoint。',
         '新Adapter主要训练结构化候选排序，不是通用颜色/解析训练；未达到建议1000—3000样本，但建议预算不能伪称已达到。开发留出非新封存集。',
         ['qwen_adapter_fair_comparison/real_graph_dataset_003/manifest.json',
          'qwen_adapter_fair_comparison/training/real_graph_seed17_001/training_report.json',
          'qwen_adapter_fair_comparison/training/real_graph_seed29_001/training_report.json',
          'qwen_adapter_fair_comparison/training/real_graph_seed43_001/training_report.json'])
    gate('公平模型对照与真实GPU', ['阶段10.1', '阶段10.3', '阶段10.6', '阶段10.8—10.9', '阶段12.3', '补充三.4'], 'running',
         f'七模型108条开发比较及五模型20张真实crop已完成；新封存五模型已关闭核验{len(closed_models)}组，每组878条，逐条同ID/同提示哈希且未接GT。',
         '余下模型仍运行，全部完成后才统一主计分；所有负结果保留；不可把排序Adapter的crop格式失败解释为通用视觉能力变化。',
         ['qwen_adapter_fair_comparison/real_graph_completion_audit_001/report.json',
          'qwen_adapter_fair_comparison/human_crop_development_065/report.json',
          'main_experiments/sealed_execution_075/events.jsonl'])
    gate('独立模块消融', ['阶段10.7', '阶段12.5—12.7', '补充三.2', '补充三.4.消融'], 'partial',
         '六建图条件和开发集完整/减规则解析/减属性/减重排/减证据等真实输出后处理已运行；新封存同配置已冻结。',
         '新封存语义消融统一计分仍待运行；地图方向/连接、首个可达/联合选择、安全接近点的最新最终覆盖尚未全量核验；不能把诊断或历史自建仿真算官方闭环消融。',
         ['nuscenes_ablation/statistics_001/report.json', 'qwen_adapter_fair_comparison/real_graph_decisions_001/report.json',
          'main_experiments/sealed_public_058/release.json'])
    gate('方法比较与来源', ['阶段12.4', '补充三.3'], 'partial',
         'AB3DMOT官方核心共享前端适配2508帧；三种内部规划24真实快照；CLIP/受控稀疏网格/规则/Qwen等封存14方法已冻结。',
         '14方法新主表尚待统一计分；网格不是完整VLMaps复现，内部规划不是外部方法排行榜，全部方法覆盖/耗时说明需终稿核对。',
         ['method_comparison/ab3dmot_statistics_002/report.json', 'method_comparison/planning_official_snapshots_001/report.json',
          'main_experiments/sealed_public_058/release.json'])
    gate('车辆约束与执行', ['阶段11.1—11.4', '阶段12.8'], 'partial',
         '整车与扫掠检查、带航向规划、实际控制及独立终点评价已用于官方CARLA，能正确到达；规划与执行分开计。',
         '多任务仍选对后无法到达；当前道路边界与停车问题尚未通过修复后的新闭环验证；13份合法规划不能算13次到达。',
         ['method_comparison/planning_official_snapshots_001/report.json', 'carla_official/current_body_causes_066/report.json'])
    gate('官方CARLA交互闭环', ['阶段11.5', '阶段11.8', '补充三.5', '阶段12.9'], 'partial',
         '官方0.9.16实际RGB/深度/LiDAR→感知建图→语义→规划→控制→新传感器闭环；本轮前10回合已完整独立审计，现已停跑。',
         '基础Qwen/新Adapter同清单配对及五条件完整闭环尚未完成；旧五条件只有14/15完整，且无历史项实际为TTL诊断。未证明实时性。',
         ['carla_official/carla_ten_041/report.json', 'carla_official/ablation_statistics_004/report.json'])
    gate('正确到达与任务规模', ['阶段11.6—11.7', '原最终完成条件', '补充验收.CARLA'], 'partial_user_cap',
         '前10回合含6有目标/2无匹配/2阻挡；5个曾选对、曾锁定并有路线，严格安全到达1/6；清洁CPU链路同任务复现一次，不增加独立任务数。',
         '用户要求CARLA先只做10回合，保持停止；原50任务×3种子及5配置10不同正任务到达未满足，不能把1/6当原范围完成。',
         ['carla_official/carla_ten_041/report.json', 'reproduction/clean_cpu_navigation_readback_003/report.json'])
    gate('故障安全、确定性与缓存', ['阶段12.10', '工程验收.缓存/确定性/安全'], 'partial',
         '既有603请求五模型真实三任务推理及相同请求零新增调用缓存复跑；证据过期/冲突压力层与异步队列记录保留，独立整车检查已做。',
         '最终新请求三任务缓存、感知缺失/深度位姿噪声、真实时钟故障矩阵和轨迹差异仍需逐项确认；不以单元测试代替全部真实故障实验。',
         [str(OLD/'main_experiments/fair_three_task_full_validation_001/report.json'),
          'qwen_adapter_fair_comparison/real_graph_dataset_003/manifest.json', 'carla_official/ten_bottlenecks_043/report.json'])
    gate('参数、分组与泛化', ['阶段12.11'], 'not_fully_verified',
         '历史和开发参数/阈值保持原结果，新封存已冻结，不根据测试改阈值。',
         'Top-K、关联、深度、确认帧数、拒绝阈值全项及稀有类别/道路分组最终覆盖尚未逐项证明，不能因总指标已出就计为完成。',
         ['main_experiments/sealed_public_058/release.json'])
    gate('聚类统计与性能', ['阶段12.统计/性能段', '补充二'], 'partial',
         '开发集按log/序列4000次配对bootstrap；三种子保留；新封存统计/同log配对图表已预定，GPU逐调用耗时和显存保存。',
         '封存全部模型统计尚待执行；部分CPU并行/I-O共享，不能标为独占整机或端到端实时。没有独立参考不制造SPL。',
         ['statistics/kitti_expanded_full_001/report.json', 'nuscenes_ablation/statistics_001/report.json',
          'figures/sealed_public_078/figure_contract.json'])
    gate('浏览、图表与可编辑表', ['交付.1—交付.2', '补充六.7'], 'partial',
         '已有真实表Word/Excel/PDF；新浏览器1196深度显示与300图文件入口，KITTI10序列首末帧已实际浏览器验证；新模型表图等待器存活。',
         '新封存图表和Word/PDF尚未实际导出/视觉核验；统一显示官方实例/排序/轨迹全字段仍未全部证明。',
         ['visualizations/evidence_browser_082/browser_qa_083/report.json', 'tables/completed_office_002/',
          'tables/sealed_office_081/registration.json'])
    gate('论文正文与引用', ['交付.3—交付.4'], 'deferred_by_user',
         '遵守当前用户要求：仅实验与交付准备，不撰写论文正文，旧稿保护。',
         '原整论文Markdown/Word/PDF及最终引用核对未完成；不把代码索引当论文正文。', [])
    gate('测试与格式验收', ['工程验收.测试'], 'partial',
         '070全量472通过、5警告；新增071/074/078各必要测试另有记录；082和离线087真实浏览器检查完成。',
         '最终源码有新文件，070不是最新全部文件最终测试回执；最终全量测试/Ruff/格式/closed JSON核对尚待统一完成。',
         ['json/software_validation_070/report.json', 'annotation/natural_offline_087/browser_qa_088/report.json'])
    gate('历史保护与存储', ['边界.历史保护/存储', '补充五'], 'partial',
         '既有255170文件56.66GB完整SHA核验通过；用户已批准25GiB及未来传感器共享卷；本次没有修改或删除历史。',
         '最后一份完整历史核验为2026-11-15，最终交付前需刷新；最终新增总量和逐项共享入口/hash尚待终审。',
         ['json/protected_history_audit_001/report.json'])
    gate('复现与代码同步', ['交付.5', '补充六.9', '用户后续Git同步要求'], 'partial',
         '711文件旧快照独立展开与旧业务重放成功；19第三方模块隔离CPU闭环同轨迹复现；公开仓库只读访问已核对。',
         '073早于最新源码；任意路径全链路/主表干净环境复现、最终代码包与Git写入同步尚未完成；不覆盖远端历史、不上传数据/模型/凭据。',
         ['source_bundle/reproduction_073/receipt.json', 'reproduction/clean_cpu_navigation_readback_003/report.json',
          'source_bundle/git_preflight_077/report.json'])
    details = dict(kitti_sequences=dict(sequences), kitti_frame_evidence_path_checks=dict(file_checks),
                   kitti_index_sha256=sha(index), new_training=training,
                   same_request_models_closed=closed_models,
                   new_sealed_actual_GPU_calls_verified=sum(m['actual_GPU_calls'] for m in closed_models),
                   natural_original_records=natural_count, natural_qualifying_labels=0,
                   old_human_review_report_fields=list(humans),
                   seven_model_development_calls=dev['actual_upstream_GPU_requests'])
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), goal_complete=False,
                  artifact_role='验收缺口快照，非最终验收；所有原范围保留',
                  previous_goal_turn_classification='progress：离线真实画面采集页已交付并经浏览器验证',
                  source_sha256=sha(Path(__file__)), verification=details,
                  groups=groups, source_reports_sha256=SOURCES,
                  current_user_overrides=['论文正文暂缓', 'CARLA先10回合并保持停止',
                                          '项目产物预算25GiB，未来传感器大文件允许共享卷'],
                  unproven_items_are_not_complete=True,
                  new_accuracy_scores_computed=False, actual_new_GPU_calls=0,
                  actual_new_CARLA_calls=0, human_records_created=0)
    with (OUT/'report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with (ROOT/'validation_report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with (ROOT/'remaining_issues_090.md').open('x') as stream:
        stream.write('# 验收缺口快照090\n\ngoal_complete=false。未证明的项目不计完成；本快照不替代最终逐项验收。\n\n')
        for group in groups:
            stream.write(f"## {group['name']}（{group['status']}）\n\n已核实：{group['supported_by_evidence']}\n\n待完善/未证明：{group['remaining_or_not_proven']}\n\n")
    print(json.dumps(dict(groups=len(groups), kitti_frames=len(kitti), closed_models=len(closed_models),
                         verified_GPU_calls=details['new_sealed_actual_GPU_calls_verified'],
                         natural_original_records=natural_count, goal_complete=False)))


if __name__ == '__main__':
    main()
