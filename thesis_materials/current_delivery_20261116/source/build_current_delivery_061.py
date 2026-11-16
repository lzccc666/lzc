"""生成当前交付入口、证据矩阵及代码索引；不更改冻结实验和历史交付。"""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/lizichong')
ROOT = PROJECT / 'outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def symbol_line(path, name):
    tree = ast.parse(path.read_text())
    matches = []
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                full = prefix + node.name
                if full == name:
                    matches.append(node.lineno)
                visit(node.body, full + '.')
    visit(tree.body)
    if len(matches) != 1:
        raise ValueError(f'入口必须唯一存在：{path}:{name}:{matches}')
    return matches[0]


def main():
    dest = ROOT / 'source_bundle/current_delivery_061'
    dest.mkdir(exist_ok=False)
    kitti = load('kitti_tracking_expanded/completion_receipt_001.json')
    nu = load('nuscenes_ablation/statistics_001/report.json')
    qwen = load('qwen_adapter_fair_comparison/real_graph_completion_audit_001/report.json')
    human = load('annotation/offline_imports/user_lzc_20261116_001/review_summary_001.json')
    attrs = load('annotation/human_attribute_reference_003/report.json')
    carla = load('carla_official/carla_ten_041/report.json')
    sealed_root = ROOT / 'main_experiments/sealed_public_058'
    receipts = list((sealed_root / 'frontend').glob('scene-*/*/complete.json'))
    progress = dict(frontend_complete_frame_receipts=len(receipts), expected_frames=1196,
                    model_final_report_exists=(sealed_root / 'evaluation/report.json').exists())
    rows = [
        ('KITTI连续序列', '已完成离线开发实验', kitti['sequences'], kitti['frames'],
         'kitti_tracking_expanded/completion_receipt_001.json',
         '10个序列各100帧窗口；不是完整序列或闭环导航，实际到达为0'),
        ('nuScenes严格模块消融', '已完成开发实验', nu['scenes'], nu['frames_per_condition'],
         'nuscenes_ablation/statistics_001/report.json',
         '6条件×1508帧=9048次真实建图；38场景含训练/验证日志，非新封存'),
        ('修正后的全量语义消融', '已完成开发实验', 38, 1508,
         'nuscenes_ablation/full_semantics_near6_002/report.json',
         '9048次合作决策，实际Qwen调用1846次；选对正目标与拒答分列'),
        ('其他方法比较', '已完成受控适配实验', 48, 2508,
         'method_comparison/ab3dmot_statistics_002/report.json',
         '38个nuScenes场景+10个KITTI序列；共享前端AB3DMOT适配，不是作者榜单复现'),
        ('新Adapter训练与公平开发对照', '已完成开发实验', 2, 12,
         'qwen_adapter_fair_comparison/real_graph_completion_audit_001/report.json',
         '3训练种子；7模型×108任务=756实际GPU请求，不能混称独立封存成绩'),
        ('人工图像审核', '已收到图像属性记录', 20, 20,
         'annotation/human_attribute_reference_003/report.json',
         '38原始记录去重后20图；不是38独立图，无人工三维身份或自然指令记录'),
        ('官方CARLA首10回合', '完成运行，仍有执行缺口', 6, 10,
         'carla_official/carla_ten_041/report.json',
         '同一Town10的6配置；6正目标中1严格到达，另2阻挡和2无匹配；未完成同清单全消融'),
        ('封存标准答案', '独立核验完成', 30, 1196,
         'annotation/reserved_semantic_audit_053/report.json',
         '878模板指令，422有匹配/456无匹配；不是人工自然指令'),
        ('新封存方法比较', '运行中', 30, 1196,
         'main_experiments/sealed_public_058/release.json',
         '204冻结文件、10建图条件、5模型；只有evaluation/report.json生成并核验后才报告最终指标'),
    ]
    evidence = []
    for item, status, scenes, frames, path, boundary in rows:
        p = ROOT / path
        evidence.append(dict(item=item, status=status, scene_or_sequence_units=scenes,
            frame_or_episode_units=frames, evidence=str(p), evidence_sha256=sha(p), boundary=boundary))
    with (dest / 'evidence_matrix.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    descriptions = [
        ('检测分割及图像特征', PROJECT/'conceptgraph/scripts/streamlined_detections.py', 'detect_rgb_file', '传感器图像变为框、掩码和CLIP特征', '正在用于封存共享前端'),
        ('深度质量', PROJECT/'conceptgraph/completion/depth.py', 'depth_support', '去除无效深度，保留有支持的点簇', '已开发验证，封存固定'),
        ('跨帧匹配', PROJECT/'conceptgraph/completion/association.py', 'gated_assignment', '约束一条检测只匹配一个已有对象', '已开发验证，封存固定'),
        ('运动估计', PROJECT/'conceptgraph/completion/motion.py', 'robust_xy_speed', '从过去的观测位置估计平面速度', '已开发验证，封存固定'),
        ('连续建图', PROJECT/'conceptgraph/slam/streamlined_mapping.py', 'StreamlinedMapper.update', '逐帧更新对象位置、证据、身份和寿命', '已开发验证，封存固定'),
        ('关系维护', PROJECT/'conceptgraph/scenegraph/incremental_graph.py', 'IncrementalSceneGraph.update_snapshot', '维护当前对象间左右前后和邻近关系', 'near=6m版本已开发验证，封存固定'),
        ('统一模型输入', ROOT/'source/real_graph_contract_003.py', 'build_completion_prompt', '只向Qwen提供文字和预测候选/关系证据', '各模型共用同一提示与预算'),
        ('指令句式入口', ROOT/'source/public_query_intake_055.py', 'parse', '区分目标、参照和排除类别及静止要求', '738原任务/328探针验证后接入封存'),
        ('规则与模型合作', ROOT/'source/public_query_intake_055.py', 'choose', '清晰任务规则处理，复杂任务使用模型响应，执行授权另查证据', '新版本用于封存，不改写历史实验'),
        ('封存注册', ROOT/'source/sealed_public_benchmark_058.py', 'prepare', '运行前固定全部版本、输入、模型和评测预算', '已冻结，禁止改源码追逐测试成绩'),
        ('封存前端', ROOT/'source/sealed_public_benchmark_058.py', 'frontend', '一次GPU检测分割供十种配置共享', '运行中，以真实进程/回执为准'),
        ('封存建图', ROOT/'source/sealed_public_benchmark_058.py', 'map_condition', '只用当前及过去观测建图，不读对象答案', '冻结阶段入口'),
        ('共享候选', ROOT/'source/sealed_public_benchmark_058.py', 'requests', '构造所有Qwen版本共用的8候选/16关系请求', '没有正确对象身份'),
        ('实际模型调用', ROOT/'source/sealed_public_benchmark_058.py', 'gpu', '保存真实响应、格式检查和逐请求延迟', '不能用回读消融充当新的GPU调用'),
        ('隔离评价', ROOT/'source/sealed_public_benchmark_058.py', 'evaluate', '方法运行完再读取官方身份及允许答案计算指标', '无控制执行或实际到达信用'),
        ('官方CARLA运行', ROOT/'source/carla_expansion_episode_010.py', 'main', '传感器、感知、建图、选择、规划与实际物理控制', '首10回合已结束，不启动新批次'),
        ('身份与到达复查', ROOT/'source/audit_carla_identity_geometry_046.py', 'main', '把关联未知、几何进入、停车和严格到达分开', '未确认的身份不能记成确定选错另一辆车'),
        ('停车距离候选', ROOT/'source/carla_stock_stopping_profile_049.py', 'StockStoppingProfile.distance', '按测得速度及滑行/制动响应估算接近速度', '仅离线候选，尚未部署或闭环验证'),
        ('控制响应核查', ROOT/'source/audit_carla_actuator_alignment_051.py', 'main', '回读1020标定时刻和4131导航时刻，核查控制衔接', '不增加真实到达次数'),
    ]
    modules = [dict(module=name, source=str(path), entry=entry, line=symbol_line(path, entry),
                    meaning=meaning, boundary=boundary, source_sha256=sha(path))
               for name, path, entry, meaning, boundary in descriptions]
    with (dest/'modules.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(modules[0]))
        writer.writeheader()
        writer.writerows(modules)
    live = {}
    for pid in [143176, 143177]:
        try:
            os.kill(pid, 0)
            live[str(pid)] = True
        except ProcessLookupError:
            live[str(pid)] = False
    snapshot = dict(UTC=datetime.now(timezone.utc).isoformat(), progress_snapshot=progress,
        observed_processes=live, original_user_human_raw_records=human['raw_records'],
        unique_human_images=human['unique_reviewed_images'],
        human_3D_identity_records=human['human_3d_identity_records'],
        human_natural_instruction_records=human['human_natural_instruction_records'],
        human_single_color_images=attrs['known_single_color_images'], human_color_model_accuracy=None,
        fair_development_Qwen={x['model']: dict(correct=x['strict_correct'], denominator=x['actual_GPU_calls']) for x in qwen['models']},
        CARLA_positive_denominator=carla['positive_task_denominator'],
        CARLA_strict_arrivals=carla['strict_safe_arrival'],
        CARLA_wrong_lock_metric_superseded_by='carla_official/identity_geometry_audit_046/report.json',
        global_pending=['封存十条件/五模型全量结果及验收', '原要求的人工自然表达覆盖未齐，模板不能替代',
            '颜色/细属性实际模型评价和三维歧义身份核验范围仍不足',
            'CARLA同任务消融及控制候选闭环验证未完成，保持10回合不扩跑',
            '全链路实时性未证实', '最终全证据交付、干净环境重跑及代码同步尚未完成'],
        module_entries=len(modules), all_referenced_files_exist=True,
        source_sha256=sha(Path(__file__)), files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        goal_complete=False)
    write(dest/'status_snapshot.json', snapshot)
    text = ['# 当前实验与交付入口（061，阶段快照）', '',
        '本页汇总当前可用证据，不是论文正文或课题完成证明。旧交付说明原样保留；其中“人工记录为0”、早期CARLA数量和旧配额不能继续作为当前状态。', '',
        f"生成时，封存共享前端已有{len(receipts)}/1196帧完成回执。进度会继续变化，请查看真实进程或series_059/events.jsonl；此数不是最终模型成绩。", '',
        '## 已完成、运行中及证据边界', '',
        '| 工作 | 状态 | 证据入口 | 解释 |', '|---|---|---|---|']
    for row in evidence:
        text.append(f"| {row['item']} | {row['status']} | [结果]({row['evidence']}) | {row['boundary']} |")
    text += ['', '## Qwen结果如何理解', '',
        '同一108条开发诊断任务：基础Qwen正确6条，旧Adapter正确10条，新Adapter种子17/29/43分别正确49/47/46条。这里包括选择、无匹配和受控故障拒答，不能称作“找对真实物体49次”，也不能当独立封存结论。', '',
        '同一批任务的完整几何规则为61/108。新Adapter相对基础模型有提升，并不证明规则合作后的完整系统优于强规则。完整系统的正目标选对率和拒答必须分开；负结果保留。正在运行的新封存实验不得提前填入成绩。', '',
        '## 人工审核和CARLA', '',
        '用户已经提供38条原始记录，去重为20张图，并确认过1条公交车类别误选修正。16图单一已知颜色、2图多颜色、2图未知。这些图像属性不是20条三维身份审核，也不是人工自然指令；没有实际模型颜色预测时，颜色准确率保留空值。', '',
        'CARLA最新首10回合来自同一Town10的6配置：6个正目标任务中，5个曾选对、锁定并产生路线，1个满足严格到达；另2个阻挡任务和2个无匹配任务分列。旧wrong_lock计数混入关联未知，具体更正见046；不能说已经证实锁到了另一辆车。停车修正仅完成离线回放，未部署，不增加到达成绩。', '',
        '## 代码和逐帧入口', '',
        f"- [模块代码、函数与真实行号]({dest/'modules.csv'})", 
        f"- [已完成证据矩阵及哈希]({dest/'evidence_matrix.csv'})",
        f"- [KITTI逐帧目录说明]({ROOT/'kitti_tracking_expanded/结果与逐帧目录说明.md'})",
        f"- [封存运行目录和恢复规则]({sealed_root/'运行说明.md'})",
        f"- [最新CARLA 10回合说明]({ROOT/'carla_official/carla_ten_041/说明.md'})", '',
        '## 尚未完成', '',
        *['- '+s+'。' for s in snapshot['global_pending']], '',
        '全局goal_complete=false。不得把已有局部结果、运行中注册表、无碰撞或单个成功视频代替完整验收。']
    with (ROOT/'当前交付入口_061.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(text)+'\n')
    write(dest/'manifest.json', dict(artifact_sha256={str(p): sha(p) for p in list(dest.iterdir())+[ROOT/'当前交付入口_061.md'] if p.is_file()},
        old_files_modified=0, old_files_deleted=0, thesis_body_written=False, frozen_experiment_changed=False, goal_complete=False))
    print(json.dumps(dict(evidence_items=len(evidence), module_entries=len(modules),
                         frontend_snapshot_frames=len(receipts), goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
