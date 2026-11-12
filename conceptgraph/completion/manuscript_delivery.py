"""从冻结开发结果生成中文阶段整合稿与可编辑三线表，不宣称论文最终完成。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import re
import shutil
import sys
from pathlib import Path

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

METHOD_LABELS={
    'original_fusion':'原融合','timestamp_only':'仅真实时间戳',
    'one_to_one_no_expiry':'一对一（无过期）','one_to_one_timestamp':'一对一＋时间戳',
    'one_to_one_depth_quality':'一对一＋深度支持','one_to_one_depth_robust_motion':'深度＋稳健运动',
    'single_frame':'单帧对象','base_fullschema':'原始Qwen','old_adapter_fullschema':'旧Adapter',
    'new_adapter_seed17':'新Adapter（17）','new_adapter_seed29':'新Adapter（29）','new_adapter_seed43':'新Adapter（43）',
    'hybrid_astar':'Hybrid A*','point_astar':'二维A*','category_nearest_current_detection':'类别最近检测',
    'clip_current_crop':'当前crop CLIP','vlmaps_style_sparse_clip_grid':'稀疏CLIP网格适配',
    'existing_object_joint_rule':'联合规则选择','reachable_positive':'独立可达正任务',
    'no_match':'无匹配','inaccessible':'独立不可接近',
    'unperturbed':'无扰动','detection_dropout_30pct':'检测缺失30%',
    'lidar_dropout_80pct':'LiDAR点缺失80%','depth_noise_020m':'深度噪声0.20m',
    'pose_noise_025m_2deg':'位姿噪声0.25m/2°','three_frame_perception_blackout':'连续3帧感知缺失',
    'full_joint_rule':'完整联合规则','without_depth_evidence_gate':'减选择端深度门限',
    'without_relation_edges':'减预测关系边',
    'free_space':'自由空间','drivable_area':'可行驶区域','lane_area':'车道区域',
    'lane_direction':'车道＋方向','lane_connection':'车道＋连接',
    'lane_direction_connection':'车道＋方向＋连接',
    'full':'完整动态维护','without_dynamic_maintenance':'减动态维护',
    'without_short_term_expiry':'减短期过期',
    'joint_safe_approach':'联合＋接近环',
    'first_reachable_same_safe_pool':'首达（相同接近点池）',
    'joint_without_safe_approach':'联合（去安全接近点）',
}

def load(relative):
    path = OUTPUT / relative
    return json.loads(path.read_text()), dict(path=relative, sha256=digest(path))


def percent(value):
    return f'{value * 100:.2f}' if value is not None else '—'


def frozen_tables():
    tables, sources = [], []
    def add(number, title, columns, rows, evidence, note):
        rows=[[METHOD_LABELS.get(row[0],row[0]),*row[1:]] for row in rows]
        tables.append(dict(number=number, title=title, columns=columns, rows=rows,
            evidence=evidence, note=note))
    mapping, source = load('statistics/corrected_mapping_001/report.json')
    sources.append(source)
    add('6-1', '开发建图质量：38场景、1508帧、5723次官方对象观测',
        ['方法', '匹配数', '预测数', '召回率(%)', '匹配精确率(%)', '中心误差(m)', '重复节点', '身份切换/转换'],
        [[k, v['true_positive'], v['predictions'], percent(v['recall']), percent(v['precision']),
          f"{v['center_error_m_mean']:.3f}", v['duplicate_nodes'],
          f"{v['identity_switches']}/{v['adjacent_matched_transitions']}"] for k, v in mapping['metrics'].items()],
        [source], '同类别三米地面距离内一对一匹配；未匹配预测可能包含评价关联不足。开发诊断，不是封存测试。')
    motion, source = load('mapping_evaluation/motion_robust_001/report.json')
    sources.append(source)
    add('6-2', '匹配对象中的运动状态评价',
        ['方法', '官方运动组', '官方观测数', '匹配数', '运动判断正确', '条件准确率(%)'],
        [[method, group, v['truth'], v['matched'], v['classified_correct'], percent(v['matched_motion_accuracy'])]
         for method, groups in motion['groups'].items() if method != 'single_frame'
         for group, v in groups.items() if group != 'unknown'], [source],
        '条件准确率以已匹配对象为分母；不与整体召回率混用，单帧默认静态没有运动判断信用。')
    relation, source = load('mapping_evaluation/relations_depth_both_axes_001/summary.json')
    sources.append(source)
    rows = []
    for method in ['direct_geometry', 'structured', 'incremental']:
        totals = {key: sum(s['methods'][method][key] for s in relation['scenes'])
                  for key in ['true_positive', 'predicted', 'truth', 'unresolved_edges']}
        tp, pred, truth = (totals[k] for k in ['true_positive', 'predicted', 'truth'])
        precision, recall = tp / pred if pred else 0, tp / truth if truth else 0
        f1 = 2 * tp / (pred + truth) if pred + truth else 0
        rows.append([method, tp, pred, truth, percent(precision), percent(recall), f'{f1:.4f}', totals['unresolved_edges']])
    add('6-3', '驾驶员坐标双轴关系评价',
        ['方法', '正确边', '预测边', '官方边', '精确率(%)', '召回率(%)', 'F1', '端点未解边'],
        rows, [source], '端点关联不足的官方边保留在召回分母；增量/全量一致率不作为关系准确率。')
    fair, source = load('main_experiments/fair_calibrated_full_validation_001/report.json')
    sources.append(source)
    add('6-4', '统一v3提示的真实CUDA验证：首版Adapter负结果',
        ['模型', '类别正确/243', '类别准确率(%)', '解析完整正确/180', '解析准确率(%)', '解析格式通过/180', '缓存新增调用'],
        [[name, v['metrics']['object_attributes']['semantic_correct'],
          percent(v['metrics']['object_attributes']['accuracy']), v['metrics']['query_parse']['semantic_correct'],
          percent(v['metrics']['query_parse']['accuracy']), v['metrics']['query_parse']['schema_valid'],
          v['cache_replay_added_calls']] for name, v in fair['variants'].items()], [source],
        '每模型423个相同请求。类别标签为独立官方投影自动对应，非人工颜色标签。此表新Adapter为首版两任务训练；后续三任务版另测。')
    vehicle_version = 'kinematic_oracle_002' if (OUTPUT/'statistics/kinematic_oracle_002/report.json').exists() else 'kinematic_oracle_001'
    vehicle, source = load(f'statistics/{vehicle_version}/report.json')
    sources.append(source)
    add('6-5', '给定正确对象的车辆执行上限：每方法90次正任务',
        ['规划', '有路线/90', '实际到达/90', '到达率(%)', '95%配置聚类CI(%)', '执行审计通过/执行数'],
        [[name, int(v['positive_route']['numerator']), int(v['positive_arrival']['numerator']),
          percent(v['positive_arrival']['estimate']),
          '—'.join(percent(x) for x in v['positive_arrival']['ci95']),
          f"{v['independent_audits_passed']}/{v['executed_trials']}"] for name, v in vehicle['methods'].items()],
        [source], '10配置、50任务、3固定种子、2方法共300次；目标身份由GT提供，不计完整感知导航成绩。共享3个道路家族。')
    annotation, source = load('annotation/reserved_official_annotation_001/manifest.json')
    sources.append(source)
    add('6-6', '预留数据准备与测试开放状态', ['项目', '数值或状态', '计数范围'],
        [['新预留场景', annotation['scenes'], 'scene'], ['新采集log', annotation['logs'], 'log'],
         ['传感器帧', annotation['frames'], 'frame'], ['主集构造指令', annotation['main_queries'], 'scene-query'],
         ['挑战指令', annotation['challenge_queries'], 'scene-query'], ['预定连续回放指令', annotation['replay_queries'], 'scene-query'],
         ['每场景主集最少指令', min(s['main_queries'] for s in annotation['per_scene']), 'scene内'],
         ['每场景连续回放最少指令', min(s['replay_queries'] for s in annotation['per_scene']), 'scene内'],
         ['新测试方法评测', '未开启', '全部方法与运行协议冻结前禁止运行']], [source],
        '指令由声明语法构造，不是人工自然语言分布；颜色、细属性与可达性标注尚有缺口。')
    if (OUTPUT/'statistics/interactive_rule_full_002/report.json').exists():
        navigation,source = load('statistics/interactive_rule_full_002/report.json')
        sources.append(source)
        rows=[]
        for kind,v in navigation['summary']['by_kind'].items():
            outcome=v['four_outcomes']
            rows.append([kind,v['trials'],int(outcome['route_generated']['numerator']),
                int(outcome['selected_object_evaluated_correct']['numerator']) if kind!='no_match' else '不适用',
                int(outcome['stable_correct_lock']['numerator']) if kind!='no_match' else '不适用',
                int(outcome['actual_full_perception_arrival']['numerator']),int(v['any_lock']['numerator']),
                v['actually_moving_trials']])
        add('6-7','完整规则感知导航：50任务、10配置、3种子',
            ['任务组','试验数','目标路线','选对对象','稳定正确锁定','正确到达','任意锁定','实际运动试验'],rows,[source],
            '独立可达正任务全部90次保留；有路径、选对、锁定与实际到达分别统计。无匹配任意锁定为误锁；不可接近的原地证据不足不增加运动安全信用。')
    retrieval_path='statistics/retrieval_baselines_dev_full_001/report.json'
    if (OUTPUT/retrieval_path).exists():
        retrieval,source=load(retrieval_path)
        sources.append(source)
        rows=[]
        for name,values in retrieval['methods'].items():
            positive,negative=values['positive_selection'],values['no_match_false_selection']
            rows.append([name,f"{positive['numerator']}/{positive['denominator']}",
                percent(positive['estimate']),'—'.join(percent(x) for x in positive['ci95']),
                f"{negative['numerator']}/{negative['denominator']}",percent(negative['estimate']),
                '—'.join(percent(x) for x in negative['ci95'])])
        add('6-8','开发检索基线：同前端38场景、1115查询',
            ['方法','正目标选对/正查询','选对率(%)','95%log聚类CI(%)','无匹配误选/负查询','误选率(%)','95%log聚类CI(%)'],
            rows,[source],
            '11个采集log配对bootstrap4000次。未知关联保留为正任务失败；检测/网格与图节点实例关联协议不同。稀疏CLIP二维网格为VLMaps式简化适配，不是完整复现，不计路线/锁定/到达。')
    second_path='statistics/fair_three_task_full_validation_001/report.json'
    if (OUTPUT/second_path).exists():
        second,source=load(second_path)
        sources.append(source)
        rows=[]
        for name in second['tasks']['query_parse']['models']:
            values=[second['tasks'][task]['models'][name] for task in
                ['object_attributes','query_parse','candidate_rerank']]
            rows.append([name,*[f"{v['correct']}/{v['requests']} ({percent(v['accuracy'])})" for v in values],
                percent(second['detailed_groups'][name]['candidate_permutation']['identity_consistency'])])
        add('6-9','第二版三任务公平验证：五模型相同603请求',
            ['模型','crop类别 正确/243 (%)','完整解析 正确/180 (%)','受控排序 正确/180 (%)','置换身份一致性(%)'],
            rows,[source],
            'crop/解析仅2验证log，排序为6独立受控符号bank；去置换44案例、43可比较组。全部种子保留；完整请求缓存复跑0新增调用。不是可靠颜色或真实道路排序成绩。')
    fault_path='statistics/recorded_fault_full_001/report.json'
    if (OUTPUT/fault_path).exists():
        fault,source=load(fault_path)
        sources.append(source)
        rows=[]
        for name in ['unperturbed','detection_dropout_30pct','lidar_dropout_80pct',
                     'depth_noise_020m','pose_noise_025m_2deg','three_frame_perception_blackout']:
            values=fault['conditions'][name]
            rows.append([name,percent(values['recall']['estimate']),percent(values['precision']['estimate']),
                f"{values['center_error_m_mean']['estimate']:.3f}",
                percent(values['identity_switch_rate']['estimate']),
                f"{values['duplicates_per_gt_observation']['estimate']:.4f}"])
        add('6-10','录制感知扰动：38场景、3种子、6条件，共684次',
            ['条件','召回率(%)','匹配精确率(%)','中心误差(m)','身份切换率(%)','重复节点/官方观测'],rows,[source],
            '1508个不同输入帧，11log配对bootstrap4000次；随机种子不作为独立log。无扰动重复仅验证复现。基于保存的修正前端扰动，不包含重新检测、车辆执行或动态交通安全。')
    ablation_path='ablations/relation_evidence_dev_001/report.json'
    if (OUTPUT/ablation_path).exists():
        ablation,source=load(ablation_path)
        sources.append(source)
        rows=[]
        for name,values in ablation['conditions'].items():
            positive,negative=values['positive_selection'],values['no_match_false_selection']
            rows.append([name,f"{positive['numerator']}/{positive['denominator']}",percent(positive['estimate']),
                '—'.join(percent(x) for x in positive['ci95']),
                f"{negative['numerator']}/{negative['denominator']}",percent(negative['estimate'])])
        add('6-11','开发选择消融：相同1115查询与融合对象图',
            ['条件','正目标选对/733','选对率(%)','95%log聚类CI(%)','无匹配误选/382','误选率(%)'],rows,[source],
            '只移除已有图上的预测关系边或选择端深度门限，不等价于关闭建图深度处理、动态维护或安全接近点。移除关系边时查询关系条件仍保留；没有执行关闭车辆保护的试验。')
    selection_path='performance/selection_parameters_dev_001/report.json'
    if (OUTPUT/selection_path).exists():
        values,source=load(selection_path)
        sources.append(source)
        rows=[]
        aliases={'baseline':'基线K3/确认3/支持0.4/拒绝0', 'top_k_1':'Top-K=1','top_k_5':'Top-K=5',
            'minimum_observations_1':'确认观测=1','minimum_observations_5':'确认观测=5',
            'minimum_depth_support_fraction_0.2':'深度支持=0.2','minimum_depth_support_fraction_0.6':'深度支持=0.6',
            'rejection_score_0.5':'拒绝分数=0.5','rejection_score_0.7':'拒绝分数=0.7'}
        for name,v in values['conditions'].items():
            metrics=[v[k] for k in ['positive_selection','no_match_false_selection','positive_topk_coverage']]
            rows.append([aliases.get(name,name),*[f"{m['count']}/{m['denominator']}" for m in metrics],
                percent(v['positive_selection']['estimate'])])
        add('6-12','选择端单因素参数：相同1115条开发查询',
            ['设置','正目标选对/733','无匹配误选/382','正确候选覆盖/733','正目标选对率(%)'],rows,[source],
            '固定融合图，只改变列示一项参数；不重新建图。Top-K为后续排序的候选覆盖，不等于首选变化。拒绝0.7将全部正查询拒绝，不获得总体高性能信用；未改动冻结导航。')
    association_path='statistics/association_parameters_dev_full_001/report.json'
    if (OUTPUT/association_path).exists():
        values,source=load(association_path)
        sources.append(source)
        add('6-13','关联门限敏感性：38场景、114条件记录',
            ['关联门限','召回率(%)','匹配精确率(%)','中心误差(m)','身份切换率(%)','重复节点/官方观测'],
            [[name,percent(v['recall']['estimate']),percent(v['precision']['estimate']),
              f"{v['center_error_m_mean']['estimate']:.3f}",percent(v['identity_switch_rate']['estimate']),
              f"{v['duplicates_per_gt_observation']['estimate']:.4f}"] for name,v in values['conditions'].items()],
            [source], 'seed17，相同1508帧；0.25/0.45各实际重建38场景，0.35复用同配置无扰动基线。只改推理相似度门限，评价三米匹配门限不变；11log配对统计，非新封存主表。')
    map_path='statistics/map_constraints_dev_full_001/report.json'
    if (OUTPUT/map_path).exists():
        values,source=load(map_path)
        sources.append(source)
        rows=[]
        for name,v in values['conditions'].items():
            metrics=[v[k] for k in ['route_generated','full_lane_legal_all_queries','full_road_legal_all_queries',
                'direction_violation_among_routes','connection_violation_among_routes','static_rectangle_legal_among_routes']]
            rows.append([name,*[f"{m['count']}/{m['denominator']}" if m['denominator'] else '—/0' for m in metrics]])
        add('6-14','六种地图约束的正确对象离线诊断',
            ['条件','路线/全查询','完整车道合法/全查询','车道及可行驶合法/全查询','方向违规/路线','连接违规/路线','矩形姿态合法/路线'],
            rows,[source], '逐场景SHA预选最多6条正查询，共212条、176个不同几何查询、38场景11log；六条件共同端点/已观测障碍/欧氏代价。GT给定正确对象的投影参考点，不证明安全接近或可达。全部路径接受完整独立规则审计；矩形仅静态姿态，不代表控制执行、扫掠安全或到达。')
    dynamic_path='statistics/mapping_modules_dev_full_001/report.json'
    if (OUTPUT/dynamic_path).exists():
        values,source=load(dynamic_path)
        sources.append(source)
        dynamic_evidence=[source]
        motion_path='statistics/mapping_modules_motion_dev_full_001/report.json'
        if (OUTPUT/motion_path).exists():
            _,motion_source=load(motion_path)
            sources.append(motion_source)
            dynamic_evidence.append(motion_source)
        add('6-15','动态维护与短期过期策略独立消融',
            ['条件','召回率(%)','匹配精确率(%)','中心误差(m)','身份切换率(%)','重复节点/官方观测'],
            [[name,percent(v['recall']['estimate']),percent(v['precision']['estimate']),
              f"{v['center_error_m_mean']['estimate']:.3f}",percent(v['identity_switch_rate']['estimate']),
              f"{v['duplicates_per_gt_observation']['estimate']:.4f}"] for name,v in values['conditions'].items()],
            dynamic_evidence, '38场景1508帧，seed17，完整条件复用38记录，其余实际重建76次。关闭动态模块也关闭运动元数据更新，unknown不能解释为static。关闭TTL为单独条件；无车辆执行或新封存成绩。')
    selection_path='ablations/goal_selection_dev_full_001/report.json'
    if (OUTPUT/selection_path).exists():
        values,source=load(selection_path)
        sources.append(source)
        rows=[]
        for name,v in values['conditions'].items():
            positive,negative=v['positive_selection'],v['no_match_false_selection']
            geometry=[v[k] for k in ['route_generated','endpoint_2m_from_predicted_bbox','static_body_pose_legal']]
            rows.append([name,f"{positive['numerator']}/{positive['denominator']}",
                f"{negative['numerator']}/{negative['denominator']}",
                *[f"{g['count']}/{g['denominator']}" if g['denominator'] else '—/0' for g in geometry]])
        add('6-16','接近点与对象选择策略的离线消融',
            ['条件','正目标选对/733','无匹配误选/382','路线/全查询','端点距预测框≥2m/已选','矩形姿态合法/已选'],rows,[source],
            '相同1115查询和预测对象，Top-K=3，每对象12个接近环点；首达与联合共享实际规划池，只改对象顺序。去接近点条件改为预测中心投影，取消距离/净空门限，仅离线不执行。统一有界点车道规划和独立审计，不等同运行中的Hybrid车辆闭环；零路径不计安全成功。')
    region_path=OUTPUT/'annotation/reference_regions_reserved_full_003/acceptance_001.json'
    if region_path.exists():
        regions,source=load('annotation/reference_regions_reserved_full_003/acceptance_001.json')
        if not regions['static_region_certificates_accepted']:
            raise ValueError('static region table requires accepted independent certificates')
        sources.append(source)
        query_path=OUTPUT/'annotation/reference_regions_reserved_full_003/queries.jsonl'
        query_source=dict(path=str(query_path.relative_to(OUTPUT)),sha256=digest(query_path))
        sources.append(query_source)
        queries=[json.loads(line) for line in query_path.open()]
        coverage=[]
        for cohort,label in [('distribution_preserving','主集构造指令'),('challenge','挑战构造指令')]:
            values=[q for q in queries if q['cohort']==cohort]
            coverage.append([label,len(values),sum(not q['no_matching_target'] for q in values),
                sum(bool(q['safe_regions']) for q in values),
                sum(q['endpoint_status']=='unknown_no_certificate' for q in values),
                sum(q['no_matching_target'] for q in values)])
        add('6-17','预留指令的独立静态终点区域覆盖',
            ['指令组','全部查询','正查询','有区域的正查询','无证书的正查询','无匹配查询'],
            coverage, [source,query_source],
            '30场景14log；206个正查询帧、603条目标帧记录、216不同instance、1647个区域。独立地图/盒/163053姿态核验通过；该静态区域证书本身不提供可达结论，后续条件可达参考另列于表6-19。未执行封存方法，非自然语言人工审核。')
    time_path=OUTPUT/'data_manifest/development_timestamp_audit_001/report.json'
    if time_path.exists():
        development,ds=load('data_manifest/development_timestamp_audit_001/report.json')
        reserved,rs=load('data_manifest/reserved_timestamp_audit_001/report.json')
        sources.extend([ds,rs])
        rows=[]
        for label,result in [('原开发关键帧配对',development),('原预留关键帧配对',reserved)]:
            rows.append([label,result.get('frames',result.get('cameras')),0,0,result['future_lidar_pairs'],
                f"{result['minimum_lidar_minus_camera_s']*1000:.3f}—{result['maximum_lidar_minus_camera_s']*1000:.3f}"])
        evidence=[ds,rs]
        for run,label in [('reserved_causal_inputs_smoke_001','过去扫描试跑'),('reserved_causal_inputs_full_001','过去扫描正式转换')]:
            if not (OUTPUT/f'data_manifest/{run}/audit_001/report.json').exists():
                continue
            audit,source=load(f'data_manifest/{run}/audit_001/report.json')
            if not audit['passed']:
                raise ValueError('causal conversion table requires independent audit')
            complete,cs=load(f'data_manifest/{run}/complete.json')
            sources.extend([source,cs])
            evidence.extend([source,cs])
            rows.append([label,complete['cameras'],complete['real_past_lidar_pairs'],
                complete['missing_past_lidar'],complete['future_lidar_pairs'],
                f"−{complete['maximum_past_lidar_age_s']*1000:.3f}—−{complete['minimum_past_lidar_age_s']*1000:.3f}"])
        add('6-18','帧内传感器时间核查与严格过去扫描转换',
            ['输入组','相机帧','真实过去LiDAR','缺失LiDAR','未来LiDAR','LiDAR−相机(ms)'],rows,evidence,
            '旧离线配对包含相机之后的同关键帧LiDAR，不能作为严格因果在线证据。所有旧文件和结果保留；新转换缺失测量显式空掩码、不计真实深度。相机图像与位姿保持相同；不是新的方法成绩。')
    reach_path=OUTPUT/'annotation/reference_reachability_reserved_full_002/acceptance_001.json'
    if reach_path.exists():
        reach,source=load('annotation/reference_reachability_reserved_full_002/acceptance_001.json')
        if not reach['conditional_static_reference_certificates_accepted'] or reach['actual_vehicle_executions']:
            raise ValueError('requires independent conditional reference without actual-method credit')
        sources.append(source)
        rows=[]
        labels={'distribution_preserving':'主集构造指令','challenge':'挑战构造指令'}
        for row in reach['coverage_by_cohort']:
            rows.append([labels[row['cohort']],row['queries'],
                row['conditionally_reachable_static_from_rest'],row['unknown']])
        rows.append(['合计查询',reach['queries'],reach['query_status']['conditionally_reachable_static_from_rest'],
            reach['query_status']['unknown']])
        add('6-19','预留查询的独立条件静态可达参考',
            ['指令组','全部查询','有条件可达参考','可达性未知'],rows,[source],
            '105条控制参考、46142个状态独立审计通过，81条查询分布于10场景；参考共享对象时不增加独立任务数。已知后轴起点、零速度/零转角、当前官方sample冻结障碍、90秒内进入独立区域并保持2秒。未知不判不可接近；实际感知导航到达、方法调用与人工审核新增均为0。')
    return tables, sources


def table_md(table):
    lines = [f"表{table['number']} {table['title']}", '',
             '| ' + ' | '.join(table['columns']) + ' |',
             '| ' + ' | '.join(['---'] * len(table['columns'])) + ' |']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in table['rows']]
    lines += ['', '注：' + table['note'], '']
    return '\n'.join(lines)


def chapter_six(tables):
    parts = ['# 第六章 实验设计与阶段结果',
        '## 6.1 评价范围与统计单位',
        '本章只汇总已完成并冻结的开发实验。新封存测试尚未开放；完整规则感知闭环的结果由独立报告读取，三模型完整导航比较和三任务Adapter公平评价尚未验收。所有未完成条件保留为未验收，不以缺失结果作正面结论。',
        '录制数据按场景和采集log统计，运动学执行按仿真配置聚类。帧、模型调用和缓存复跑不增加独立样本数。对象、关系、语义和车辆到达分别使用独立分母。运行设备为RTX 4090；并行前端与语义任务的耗时不作为独占实时性能成绩。',
        '## 6.2 对象与关系质量']
    parts += [table_md(t) for t in tables[:3]]
    parts += ['深度支持簇条件的可匹配覆盖和中心误差优于原融合，但身份切换没有同步改善。稳健运动估计提高静态匹配对象的条件准确率，同时降低动态组准确率；因此不能宣称其全面优于原运动估计。关系指标仍受到对象端点缺失与身份对应不足影响，双轴表达的一致性不能消除上述瓶颈。',
        '## 6.3 公平语义比较与训练负结果', table_md(tables[3]),
        '首版新Adapter在自身训练提示下的验证分数较高，但切换到训练侧校准并统一给全部模型的v3提示后发生退化。旧Adapter在完整解析上的点估计高于基础模型，不能由此推断其实际选对对象或车辆到达得到改善。首版三个种子全部保留，不挑选种子43的属性结果代表全部新模型。',
        '当前证据支持提示敏感性问题，而不支持新Adapter泛化优势。三任务第二版使用统一提示重新训练，并新增候选置换与无匹配监督；其公平对照结果完成后应单列版本，不覆盖表6-4。可靠颜色准确率、候选排序、置换身份一致性和独立语义消融尚未形成完整主表。',
        '## 6.4 车辆运动学执行', table_md(tables[4]),
        '二维A*正任务均生成路线，其中28次没有完成到达；控制与安全保护阻止了无效路线被解释为执行成功。Hybrid A*在90次正任务中20次规划失败，这些任务保留在分母。到达率配对差为8.89个百分点，95%配置聚类区间为−7.78至25.56个百分点，包含零，不能宣称统计显著优势。',
        '所有执行日志均经过独立连续车身审计。没有开始执行的任务不计为安全成功；给定GT目标的无匹配与不可接近条件也不作为真实语义识别能力的证据。',
        '## 6.5 新数据、完整闭环与未完成主表', table_md(tables[5]),
        '预留场景来自14个此前未参与训练和调参的log，按与历史同城相机轨迹至少50m距离选择，精确图像和全图近似审计另存。该距离规则不证明所有道路拓扑语义完全独立。图像crop近似、表达家族与人工属性审计需要与测试release一并核对。',
        '可交互开发环境已真实运行相机渲染、稀疏LiDAR、YOLO/SAM/CLIP、增量对象图与车辆控制。完整运行与中断或smoke条件分别保存；固定逻辑时钟结果不作为真实时钟异步Qwen性能。',
        '## 6.6 尚待验收项目与复现证据',
        '尚待完成：封存测试主比较与至少每场景六条连续回放，来源明确的外部适配基线的封存验证，独立语义消融及选择策略的封存主比较，真实时钟后台Qwen完整比较与故障恢复，独占性能统计、封存道路泛化，人工颜色及歧义标签，以及全部新试验的统一可视化。开发地图、动态维护和参数检查已有结果，但不能替代新封存主表。论文终稿只有在上述必需条件核对后才能形成。',
        '本稿所有表格均从保存的结果文件读取；每个表的来源和SHA-256见同目录tables.json，原始计数另以CSV、Excel和可编辑Word提供。结果缺口不是数值零，表中未计算项不参与均值或改善率。']
    full=next((table for table in tables if table['number']=='6-7'),None)
    if full is not None:
        stats,_=load('statistics/interactive_rule_full_002/report.json')
        outcome=stats['summary']['by_kind']['reachable_positive']['four_outcomes']['actual_full_perception_arrival']
        function=stats['summary']['functional_evidence']
        functional_parts = ['## 6.6 完整规则感知闭环与最低功能证据',table_md(full),
            f"完整规则闭环在90次可达正试验中正确到达{function['arrivals']}次，到达率{percent(outcome['estimate'])}%，按配置聚类的95%区间为{percent(outcome['ci95'][0])}%—{percent(outcome['ci95'][1])}%。这些到达覆盖{function['distinct_positive_tasks']}个不同正任务和{function['scene_configs']}个配置，满足最低十任务、五配置的功能条件；总体失败仍多，不能将最低功能门槛解释为总体性能充分。",
            '90次正试验均曾有正确首选对象，其中79次有连续三次观测的独立正确锁定，68次生成目标接近路线，最终21次到达。47次因感知或路径失效终止、22次规划失败。所有失败保留在正任务分母，不能在规划失败后改标为不可达。',
            '30次无匹配试验均未误锁，经过实际搜索后耗尽；该结果只限冻结范围，不证明所有未知道路没有目标。30次独立不可接近任务均因证据不足留在原地，既未识别出正确目标，也不能据此宣称系统准确识别了不可达。120次实际运动的轨迹均通过独立整车扫掠审计，另30次原地审计单列，安全范围限于静态mesh和保存的道路/观测。',
            '开发用四秒停车恢复等待另跑四任务，两个正任务仍分别超时和安全停止，没有形成到达改善证据。真实墙钟规则实验也出现超时，保留传感器TTL制动记录。原始Qwen、新Adapter与独立语义消融仍需相同完整任务比较。']
        position=parts.index('## 6.6 尚待验收项目与复现证据')
        parts[position]='## 6.7 尚待验收项目与复现证据'
        parts[position:position]=functional_parts
    retrieval=next((table for table in tables if table['number']=='6-8'),None)
    if retrieval is not None:
        position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
        parts[position]='## 6.8 尚待验收项目与复现证据'
        parts[position:position]=['## 6.7 开发检索主基线及外部方法适配',table_md(retrieval),
            '类别最近、当前crop CLIP和稀疏CLIP网格读取当前及过去帧索引的前端证据，不读取后续帧缓存。原开发帧内LiDAR实际比相机晚33.136—42.356ms，因此这项帧索引限制不等于严格因果在线输入。中文指令使用同一确定性规则转换为英文CLIP文字，不读取GT正确解析；网格以0.5m二维单元累积实际crop特征，按5秒时间戳过期并限制当前相机可见范围。VLMaps[45]使用更完整的视觉语言地图，本适配未复现其LSeg密集特征、完整三维表示或原规划器，因此只在本项目相同离线开发输入上比较。',
            '类别最近在733条正查询中选对211条，但在382条无匹配中误选141条。CLIP和稀疏网格分别选对149、132条正查询，无匹配误选308、330条。联合规则误选8条，同时只选对45条，显示严格证据筛查明显限制覆盖。不能因拒绝多而认定整体方法更优；检测/网格和对象图独立身份关联口径不同，也限制模块间归因。',
            '全部查询逐方法输出保存；四种方法完整配对，4000次按log聚类区间不把帧或模型调用当作独立样本。本表尚缺相同协议的原始Qwen和新Adapter，不是封存测试最终检索主表。']
        parts=[p.replace('来源明确的外部适配基线，','外部适配的封存验证，') for p in parts]
    second=next((table for table in tables if table['number']=='6-9'),None)
    if second is not None:
        position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
        parts[position]='## 6.9 尚待验收项目与复现证据'
        stats,_=load('statistics/fair_three_task_full_validation_001/report.json')
        selected=stats['tasks']['query_parse']['models']['new_adapter_seed17']
        parts[position:position]=['## 6.8 第二版三任务完整公平验证',table_md(second),
            f"第二版三个正式种子均已完成相同603请求的验证和三任务缓存复跑，所选种子17完整解析{selected['correct']}/{selected['requests']}。选模在完整公平验证结束前按训练侧固定验证规则完成，不使用本表挑选最佳种子。各字段、类别宏平均、受控查询类型及按log/bank配对差的区间另存statistics/fair_three_task_full_validation_001。",
            '该实验对解析与crop仅有2个采集log，4000次聚类bootstrap不能弥补独立log过少；排序为6个受控符号bank，而非真实道路对象检索。单独提高置换身份一致性也不保证选中了正确候选。首版在统一提示下的退化结果继续保留，不能由第二版结构化验证反推首版或实际导航获益。']
        parts=[p.replace('三任务Adapter公平评价尚未验收','三任务Adapter结构化公平验证已完成，真实道路语义收益尚未验收')
            .replace('三任务第二版Adapter公平评价尚未验收','三任务第二版完整公平验证已完成、真实道路语义收益尚未验收')
            .replace('三任务第二版使用统一提示重新训练，并新增候选置换与无匹配监督；其公平对照结果完成后应单列版本，不覆盖表6-4。可靠颜色准确率、候选排序、置换身份一致性和独立语义消融尚未形成完整主表。',
                '三任务第二版使用统一提示重新训练，并新增候选置换与无匹配监督；完整验证另列为表6-9，不覆盖表6-4。可靠颜色准确率、真实道路排序和独立语义消融仍需补齐。') for p in parts]
    for number,heading,paragraphs in [
        ('6-10','录制感知故障与误差扰动',[
            '检测缺失30%使召回率相对无扰动下降5.13个百分点，配对95%log聚类区间为−6.26至−3.74个百分点。LiDAR点缺失80%使召回率下降16.79个百分点，区间为−21.10至−13.19个百分点。后者匹配精确率点估计上升，但其配对区间包含零，且同时漏掉更多对象，不能解释为整体稳健性提升。',
            '位姿噪声使中心误差增加0.210米，配对95%区间为0.187至0.253米。深度噪声0.20米条件下，召回差和中心误差差的区间均包含零。连续三帧感知缺失使召回下降1.59个百分点，区间为−2.07至−1.15个百分点。检测缺失时身份切换率下降伴随覆盖下降，不能单独证明身份维护更好。',
            '扰动幅度和分布是在运行前冻结的压力测试条件，未标定为真实车辆传感器故障频率。该实验只评价保存的录制前端与对象维护，不为实际到达、碰撞避免或完整动态交通安全提供信用。']),
        ('6-11','关系与选择端证据门限的独立消融',[
            '移除预测关系边后，正目标选对由45/733变为59/733，无匹配误选保持8/382。按11log配对的选对率差为1.91个百分点，95%区间为0至3.88个百分点。当前开发分布不支持预测关系边提高目标选择的结论，端点漏检和关系错误可能限制其用途，具体原因仍需分组诊断。',
            '关闭选择端深度证据门限后逐查询选择完全相同。该结果说明此开关在现有融合图和查询分布中没有产生下游变化，不能据此断言深度支持建图没有价值。动态维护与地图的开发消融另列；首个可达与联合选择、安全接近点及Qwen模块的其余独立消融仍未验收。'])]:
        table=next((t for t in tables if t['number']==number),None)
        if table is not None:
            position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
            section=int(parts[position].split()[1].split('.')[1])
            parts[position]=f'## 6.{section+1} 尚待验收项目与复现证据'
            parts[position:position]=[f'## 6.{section} {heading}',table_md(table),*paragraphs]
    additions=[('6-12','选择端参数与候选覆盖',[
        '该实验固定全部1115查询，只有开发参数发生单因素变化。确认观测从3提高为5时，正目标选对由45/733降至40/733；Top-K=1、3、5的首选均为45/733，候选覆盖分别为45、52、52。放宽候选数量不能补回前端未检出的目标。',
        '拒绝分数0.7同时消除了全部正目标选择与无匹配误选，因此不得把零误选解释为总体更好的导航。类别、可见性和距离探索分组另存逐组CSV，不把同帧观测当独立道路样本；本检查未修改在运行的导航参数。']),
        ('6-13','关联门限与身份覆盖权衡',[
        '将关联门限从0.35调为0.25，匹配精确率增加1.82个百分点，11log配对95%区间为0.51至2.69个百分点；中心误差和身份切换差未显示明确改善。0.45使召回增加0.77个百分点，却使匹配精确率下降13.77个百分点、身份切换率增加15.47个百分点。',
        '此处只改变推理门限，固定三米评价距离。未匹配预测不都经过人工确认，不能直接将其全部称为真实误检；关联门限也未根据新封存测试选择。']),
        ('6-14','地图方向与连接约束的离线作用',[
        '六条件只改变地图约束，同一查询使用相同当前窗口、预测障碍和共同投影端点。自由空间或仅车道区域能产生路线，不等于其方向和连接合法；方向与连接由完整独立检查分别计数。地图孔洞按官方多边形保留，车道与可行驶先验不一致的情况另列。',
        '实际预选212条正查询，部分场景不足六条，不补造重复查询。六条件共1272次规划，含176个不同几何查询。仅车道区域产生158条路径，其中49条通过完整车道检查；方向与连接同时约束时产生91条路径、88条通过完整车道检查，合法覆盖增加18.40个百分点，11log配对95%区间为14.29至22.40个百分点。若同时要求官方可行驶多边形覆盖，合法数为38与47，差为4.25个百分点，区间为1.43至7.23个百分点。',
        '方向约束条件的已生成路径未见方向违规，但仅加方向仍有61/111条连接违规。仅加连接有32/118条方向违规；同时约束方向与连接仍有3/91条被独立连接审计判为违规，不能宣称全部合法。六条件的路径矩形姿态全部未通过完整检查；点路径、离散转向与目标参考点并不足以证明整车接近可行。',
        '全部查询保留在路线和完整合法覆盖分母，规划超时及搜索失败不作为不可达真值。此处GT给定对象，且未执行控制；选对对象、稳定锁定与正确到达均未评价，不能从离线路径数量推算。路径切线上的矩形姿态检查也不能代替车辆运动学与连续扫掠审计。']),
        ('6-15','动态维护与过期策略的建图影响',[
        '完整配置与两个独立关闭条件使用相同真实前端、时间戳、关联和深度处理。关闭动态模块后，旧几何会长期累积，运动状态保留为unknown；单独关闭短期过期则继续估计运动，只取消超龄删除。两条件不能混为同一个功能开关。',
        '关闭动态维护使召回由31.94%降至27.01%，配对差为−4.93个百分点，11log的95%区间为−6.76至−2.61个百分点；中心误差增加0.076米，区间为0.036至0.120米。身份切换率差的区间包含零，不支持身份一致性明显改善。只关闭短期过期使匹配精确率下降3.05个百分点，区间为−4.27至−1.80个百分点。',
        '官方动态组3022次观测的匹配数由869降至609，静态组2556次由911降至896；关闭动态模块后的匹配对象运动状态均为unknown，不能解释为准确判断了静止。完整动态组条件运动准确率为535/869，另有246次unknown，未知和漏检分别保留。',
        '指标由全部38场景逐帧重建和相同三米关联评价得到，完整条件复用同配置seed17记录。按11采集log配对统计，不用缓存基线或重复帧增加独立样本数；本实验尚不能证明下游车辆到达改善。'])]
    for number,heading,paragraphs in additions:
        table=next((t for t in tables if t['number']==number),None)
        if table is None:
            continue
        position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
        section=int(parts[position].split()[1].split('.')[1])
        parts[position]=f'## 6.{section+1} 尚待验收项目与复现证据'
        parts[position:position]=[f'## 6.{section} {heading}',table_md(table),*paragraphs]
    selection=next((t for t in tables if t['number']=='6-16'),None)
    if selection is not None:
        values,_=load('ablations/goal_selection_dev_full_001/report.json')
        first=values['conditions']['first_reachable_same_safe_pool']
        joint=values['conditions']['joint_safe_approach']
        removed=values['conditions']['joint_without_safe_approach']
        position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
        section=int(parts[position].split()[1].split('.')[1])
        parts[position]=f'## 6.{section+1} 尚待验收项目与复现证据'
        parts[position:position]=[f'## 6.{section} 接近点与首达策略的离线比较',table_md(selection),
            f"联合与首达在733条正查询中分别选对{joint['positive_selection']['numerator']}与{first['positive_selection']['numerator']}条，无匹配误选分别为{joint['no_match_false_selection']['numerator']}与{first['no_match_false_selection']['numerator']}（分母382）。两者共享同一批实际规划的接近点和独立准入结果，仅替换对象选择顺序，不用重新规划的耗时差异混入策略效果。",
            '联合相对首达多选对2条，差为0.27个百分点，11log配对95%区间为0至0.62个百分点；去安全接近点相对完整条件少选对11条，差为−1.50个百分点，区间为−2.97至0个百分点。区间均包含零，不作确定优势结论。三个条件的矩形姿态通过数均为零，因此点车道规划和预测框外距离不能据此获得整车安全信用。',
            f"去安全接近点后，在{removed['selected']}个已选终点中，距预测对象框至少2米的有{removed['endpoint_2m_from_predicted_bbox']['count']}个。此距离只针对不完整预测框，不能作为真实物体安全间隔；所有矩形姿态结果与失败分别保存，未实际执行去保护条件。",
            '该实验复用生产接近环生成和联合效用，以当前观测窗口及预测障碍进行有界点车道规划，并用独立连续规则拒绝不合法路径。它是录制数据上的离线适配诊断，既没有重新产生策略各自的控制轨迹，也未评价稳定锁定或实际到达；封存主比较与运行中Hybrid闭环仍需另行完成。']
    if (OUTPUT/'performance/road_groups_dev_001/report.json').exists():
        position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
        parts[position:position]=[
            '道路探索分组使用当前位姿所在的官方车道多边形、中心线局部转角和声明连接，不读取方法是否成功。1115条开发查询中，联合规则在直路正查询选对33/279，在连接区域选对12/449，在弯道选对0/5；无匹配误选分别为4/131、4/242、0/9。弯道正查询仅贡献2个log，零选对及其聚类区间不能证明对所有弯道均无效，也不能当作组间显著性检验。',
            '分组来源、阈值、四方法逐log计数和区间见performance/road_groups_dev_001。这里的道路标签是地图派生代理，未新增独立场景或打开封存测试，不支持道路泛化已经完成的结论。']
    for number,title,paragraphs in [
        ('6-17','预留标签的静态区域与可达性边界',[
            '静态区域由隔离评价端的完整官方三维盒投影与HD地图产生，不读取检测、候选排序或待测规划器结果。0.4m×0.2m区域表示后轴中心，名义车身外包络覆盖位置及±5°航向误差，再增加0.1m净空；区域内参考点到目标完整投影盒距离保证为2—8m。相机位置不可直接当后轴初始位置。',
            '全部1647区域经独立地图多边形/孔洞、8620官方盒和163053个扰动姿态核验通过，原始标签集合和878条指令没有删减。缺失地图引用不填充，动态与unknown运动盒不被遗漏；但参考只对应官方sample时刻冻结几何，不能保证未来动态交通安全。',
            '307条正查询有静态区域，115条未找到证书，后者保持unknown而不改标不可接近。静态区域证书本身不证明从初始状态可达，后续条件控制参考另见表6-19。静态区域、合法路线、正确锁定与实际到达必须继续分开。真实到达入口需要显式5°航向容差，不能套用默认15°；目前没有据此运行任何封存方法。']),
        ('6-18','传感器采集时间与因果性修订',[
            '逐文件时间核查发现，38场景1508帧开发数据与30场景1196帧预留数据均采用同关键帧配对，但LiDAR采集晚于相机。原开发建图、关系、故障和检索结果仍作为相同离线输入的配对诊断保留，不能将当前/过去帧缓存限制解释为帧内所有测量也来自过去。本稿修正该边界，不估计未测的时间偏差性能损失。',
            '新预留输入另从完整原始sample_data表选择不晚于相机、年龄不超过100ms的最新LiDAR，使用各自时刻的已知ego_pose和标定作传感器运动补偿后稀疏投影。无合格历史扫描时保留原RGB和指令，显式标为缺失与空掩码，不借未来返回、插值、GT运动或删除困难首帧来补齐。只对实际转换完成且独立审计通过的版本报告数字。',
            '自然指令360帧输入已使用过去扫描，与本次原关键帧配对问题分开。可交互三维导航使用其独立真实时钟传感器链，冻结源码和已有执行记录未因该数据修订改变。新的严格因果关键帧版本还需纳入方法、缺失处理与最终release后才能开启封存主表。']),
        ('6-19','独立控制参考及其条件范围',[
            '隔离标签端从真实地面后轴起点出发，独立枚举官方有向车道序列并生成低速控制积分；没有调用待测规划器、语义模型或控制器的成功标志。审计器另行复算车辆积分、矩形车身及连续扫掠，要求原始地图与孔洞保持不变、全类别当前官方盒均避让，并存在连续的官方有向车道分配。',
            '正式参考保留全部603条目标帧记录，105条控制参考通过46142个状态的独立审计。878查询中81条至少对应一个正确实例的条件可达参考，其余797条保持unknown；其中主构造集为53/600，挑战构造集为28/278。多个查询可共享参考对象或路径，105条参考和状态数均不作为独立导航样本。',
            '终态需进入已有独立安全区域，航向误差不超过5度、速度不超过0.02米/秒，并连续保持至少2秒，总时长不超过既有90秒协议。首轮124候选中1条被独立车道连接审计拒绝，且旧保持窗口仅0.9秒；修订仅继承逐条已通过的控制积分、追加制动保持，并剔除18条超出90秒的参考，保留全部失败来源。',
            '这些证书明确假设静止和零转角起步、官方sample时刻障碍物冻结；记录车辆的实际非零速度仍未知，也未保证未来动态交通安全。没有获得证书的查询不能因有限搜索失败就改标不可接近。这里没有执行感知导航方法，选对对象、稳定锁定与实际到达均不获信用，新封存主表尚未开启。'])]:
        table=next((t for t in tables if t['number']==number),None)
        if table is not None:
            position=next(i for i,p in enumerate(parts) if p.endswith('尚待验收项目与复现证据'))
            section=int(parts[position].split()[1].split('.')[1])
            parts[position]=f'## 6.{section+1} 尚待验收项目与复现证据'
            parts[position:position]=[f'## 6.{section} {title}',table_md(table),*paragraphs]
    return '\n\n'.join(parts) + '\n'


def add_figures(chapter,destination,figure_run):
    if not figure_run:
        return chapter,[]
    source=OUTPUT/'thesis'/figure_run
    qa=json.loads((source/'qa_visual_pdf_002.json').read_text())
    automatic=json.loads((source/'qa_automatic.json').read_text())
    if not qa['passed'] or not automatic['passed']:
        raise ValueError('only QA-passed scientific figures may enter manuscript')
    manifest=json.loads((source/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(source/name)!=sha:
            raise ValueError('figure package changed')
    asset=destination/'figures'
    asset.mkdir()
    numbering={'6-1':'6-1','6-3':'6-2','6-2':'6-3','6-4':'6-4','6-5':'6-5','6-6':'6-6','6-7':'6-7'}
    for p in source.iterdir():
        if p.suffix in {'.png','.pdf','.svg','.tiff','.csv'}:
            target=('图'+numbering[p.stem[1:]]+p.suffix) if p.stem.startswith('图') and p.stem[1:] in numbering else p.name
            shutil.copy2(p,asset/target)
    captions=[c.strip() for c in (source/'图注与统计说明.md').read_text().split('\n\n')]
    blocks={}
    caption_overrides={}
    for i in range(1,8):
        old=f'6-{i}'
        if not (source/f'图{old}.png').exists():
            continue
        new=numbering[old]
        caption=next(c for c in captions if c.startswith(f'图{old} ')).replace(f'图{old} ',f'图{new} ',1)
        original_caption=caption
        caption=caption.replace('全部方法读取当前及过去证据；',
            '全部方法读取当前及过去帧索引的离线配对证据；帧内LiDAR晚于相机，不能作为严格因果在线输入；')
        if caption!=original_caption:
            caption_overrides[new]=dict(original=original_caption,revised=caption,
                evidence='data_manifest/development_timestamp_audit_001/report.json',
                reason='修正帧索引与帧内采集因果性的区别；图像和原始计数不变')
        blocks[old]=f'![图{new}](figures/图{new}.png)\n\n{caption}\n\n'
    dump(asset/'figure_number_mapping.json',dict(source_package=figure_run,source_to_manuscript=numbering,
        note='原图源数据CSV保留源包编号；正文与复制的图片按首次出现顺序统一编号',
        caption_overrides=caption_overrides))
    for marker,number in [('## 6.3','6-1'),('## 6.4','6-3'),('## 6.7','6-2'),('## 6.8','6-4')]:
        chapter=chapter.replace(marker,blocks[number]+marker,1)
    if (source/'图6-5.png').exists():
        chapter=chapter.replace('## 6.9',blocks['6-5']+'## 6.9',1)
    for number,marker in [('6-6','## 6.13 '),('6-7','## 6.14 ')]:
        if number in blocks and marker in chapter:
            chapter=chapter.replace(marker,blocks[number]+marker,1)
    return chapter,[dict(path=str((source/'manifest.json').relative_to(OUTPUT)),sha256=digest(source/'manifest.json')),
        dict(path=str((source/'qa_visual_pdf_002.json').relative_to(OUTPUT)),sha256=digest(source/'qa_visual_pdf_002.json'))]


def revised_front(original):
    text = original.split('# 第一章 绪论', 1)[1].split('# 参考文献', 1)[0]
    text = '# 第一章 绪论' + text
    text = re.sub(r'【图[^\n]+建议[^\n]*】.*?(?=\n\n|\Z)', '', text, flags=re.S)
    replacement = '''## 1.4 主要研究内容与证据边界

本研究围绕已知位姿与声明地图先验下的静止对象接近任务展开，研究内容包括车载稀疏深度支持的对象维护、带时间及坐标来源的关系表示，以及结合语言目标、道路约束和车辆执行的导航。YOLO-World、MobileSAM、CLIP、ConceptGraphs、Qwen2-VL和Hybrid A*均有明确上游来源，本文不将这些基础方法归为自行提出。

具体系统改动包括真实时间戳与深度支持诊断、带门限的一对一关联、静动态证据维护、显式无匹配和安全候选内语义推荐、整车运动约束及独立到达评价。开发实验支持部分定位改善，也显示身份、运动估计和提示泛化的局限。是否形成可推广的方法贡献，需由新封存测试、独立消融和完整感知导航共同检验，不能仅由模块组合或单次成功演示确定。

本文范围不包括独立SLAM、自动泊车和完整交通规则决策；真实时钟异步与完整主比较尚未验收。本整合稿属于阶段稿，不宣称整个课题已经完成。

'''
    text = re.sub(r'## 1\.4.*?(?=## 1\.5)', replacement, text, flags=re.S)
    text = text.split('# 第三章 ', 1)[0]
    text = text.replace('点、框或文本等提示', '点、框等提示')
    text = text.replace('LM-Nav 和 SayCan 等研究表明', 'LM-Nav[77]和SayCan[78]等研究表明')
    text = text.replace('不过，这类方法多在室内或桌面机器人场景中验证，面向道路车辆时需要额外处理交通规则、动态目标、可通行区域和安全距离。',
        '上述方法的验证环境各不相同，LM-Nav已包含室外移动机器人长程导航；面向道路车辆时，仍需另行验证车辆运动学、车道方向与连接、动态风险和安全接近区域，不能将已有机器人结果直接当作本系统的道路车辆性能。')
    for before,after in [
        ('Kimera 构建了从度量层到语义层的多层次三维场景图结构，为机器人理解和规划提供了较完整的中间表示[31]。',
         'Kimera提供视觉惯性估计、位姿图优化、网格与度量语义重建模块[31]；后续三维动态场景图另见文献[33]。'),
        ('Armeni 等提出的 3D Scene Graph 将建筑环境中的对象、房间、楼层和建筑层级组织为统一图结构[32]。',
         'Armeni等提出的3D Scene Graph将对象、房间、建筑与相机信息及其关系组织为统一三维图结构[32]。'),
        ('Road Scene Graph 等工作尝试用图结构表达道路参与者、车道和交通关系[74]',
         'Road Scene Graph尝试用图结构表达道路对象及其成对语义关系[74]'),
        ('Active Neural SLAM、SemExp 等方法将语义地图构建与目标搜索结合，利用语义先验提高未知环境中的探索效率[43,44]。',
         'Active Neural SLAM将学习的SLAM模块、解析规划器及层级策略结合用于环境探索[43]；SemExp进一步结合语义地图和目标导向探索策略处理对象目标导航[44]。'),
        ('LanguageMPC 则尝试将语言指令融入模型预测控制，使车辆规划能够响应自然语言偏好[82]。',
         'LanguageMPC以大语言模型承担高层驾驶决策，并通过控制器参数调整连接到低层控制[82]。')]:
        if before not in text:
            raise ValueError('reviewed citation passage changed: '+before)
        text=text.replace(before,after)
    return text.strip()


def word_export(path, markdown, only_tables=None):
    sys.path.insert(0, str(OUTPUT / 'source_bundle/document_runtime'))
    from docx import Document
    from docx.shared import Mm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from conceptgraph.completion.manuscript_math import append_word_math, word_inline
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin, section.right_margin = Mm(30), Mm(25)
    section.top_margin = section.bottom_margin = Mm(25)
    for style in document.styles:
        if style.type == 1:
            style.font.name = 'Noto Serif CJK SC'
            style.font.size = Pt(10.5)
            style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Serif CJK SC')
    normal = document.styles['Normal'].paragraph_format
    normal.line_spacing = 1.5
    normal.space_after = Pt(6)
    footer = section.footer.paragraphs[0]
    footer.alignment = 1
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    def add_table(columns, rows):
        table = document.add_table(rows=1, cols=len(columns))
        for cell, value in zip(table.rows[0].cells, columns):
            cell.text = str(value).replace(r'\*', '*')
        header = OxmlElement('w:tblHeader')
        table.rows[0]._tr.get_or_add_trPr().append(header)
        for row in rows:
            for cell, value in zip(table.add_row().cells, row):
                cell.text = str(value).replace(r'\*', '*')
        borders = OxmlElement('w:tblBorders')
        for edge in ['top', 'bottom', 'left', 'right', 'insideH', 'insideV']:
            tag = OxmlElement('w:' + edge)
            tag.set(qn('w:val'), 'single' if edge in {'top', 'bottom'} else 'nil')
            tag.set(qn('w:sz'), '8')
            borders.append(tag)
        table._tbl.tblPr.append(borders)
        for cell in table.rows[0].cells:
            border = OxmlElement('w:tcBorders')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '4')
            border.append(bottom)
            cell._tc.get_or_add_tcPr().append(border)
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(8)
        # 表末行与紧随其后的说明同页，避免脚注只剩半句落到下一页。
        for cell in table.rows[-1].cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True
    if only_tables:
        for table in only_tables:
            document.add_heading('表' + table['number'] + ' ' + table['title'], 2)
            add_table(table['columns'], table['rows'])
            document.add_paragraph('注：' + table['note'])
    else:
        lines = markdown.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if line == r'\[':
                end=index+1
                while end<len(lines) and lines[end].strip()!=r'\]':
                    end+=1
                if end==len(lines):
                    raise ValueError('unclosed display formula')
                from docx.enum.text import WD_TAB_ALIGNMENT
                paragraph=document.add_paragraph()
                paragraph.paragraph_format.tab_stops.add_tab_stop(Mm(77.5),WD_TAB_ALIGNMENT.CENTER)
                paragraph.paragraph_format.tab_stops.add_tab_stop(Mm(155),WD_TAB_ALIGNMENT.RIGHT)
                paragraph.add_run('\t')
                append_word_math(paragraph,'\n'.join(lines[index+1:end]),display=True)
                index=end+1
                continue
            if line.startswith('|') and index + 1 < len(lines) and re.match(r'^\|[\s|:-]+$', lines[index + 1]):
                columns = [s.strip() for s in line.strip('|').split('|')]
                index += 2
                rows = []
                while index < len(lines) and lines[index].strip().startswith('|'):
                    rows.append([s.strip() for s in lines[index].strip().strip('|').split('|')])
                    index += 1
                add_table(columns, rows)
                continue
            if line and line != '---':
                image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
                if image:
                    document.add_picture(str(path.parent/image[2]),width=Mm(150))
                    document.paragraphs[-1].paragraph_format.keep_with_next=True
                    index+=1
                    continue
                heading = re.match(r'^(#{1,3})\s+(.+)', line)
                if heading:
                    document.add_heading(heading[2], min(len(heading[1]), 3))
                else:
                    paragraph=document.add_paragraph()
                    word_inline(paragraph,re.sub(r'\*\*(.+?)\*\*', r'\1', line).replace(r'\*','*'))
                    if line.startswith('表6-'):
                        paragraph.paragraph_format.keep_with_next=True
            index += 1
    document.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--figure-run')
    args = parser.parse_args()
    destination = OUTPUT / 'thesis' / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    tables, sources = frozen_tables()
    dump(destination/'method_aliases.json',METHOD_LABELS)
    dump(destination / 'tables.json', tables)
    sixth,figure_sources=add_figures(chapter_six(tables),destination,args.figure_run)
    sources+=figure_sources
    (destination / '第六章.md').write_text(sixth.replace('A*',r'A\*'))
    original_path = PROJECT / '论文1-3章初稿.md'
    original = original_path.read_text()
    front = revised_front(original)
    third = (OUTPUT / 'thesis/manuscript_working_001/第三章修订.md').read_text()
    chapters = [(OUTPUT / f'thesis/manuscript_working_001/第{n}章.md').read_text() for n in ['四', '五', '七']]
    chapters[0]=chapters[0].replace('后续语言检索只访问Gₜ及当前、过去的传感器证据。',
        '后续语言检索只访问Gₜ及当前、过去帧索引的数据包。原离线开发包的LiDAR实际比相机晚33.136—42.356ms，帧索引约束不等于帧内采集完全因果；严格在线输入应另要求LiDAR时间不晚于相机。第六章单列时间核查与新的过去扫描转换，不把原离线结果当作严格因果性能。')
    chapters[2]=chapters[2].replace('确认至少十个不同正任务、五个配置的实际正确到达。',
        '完整规则感知闭环已完成150次试验，90次正试验中有路线68次、选对对象90次、稳定正确锁定79次、正确到达21次，覆盖13个不同正任务与8个配置，满足最低功能门槛；仍需公平三模型主比较。')
    chapters[2]=chapters[2].replace('该实验属于真值辅助上限，不能替代完整感知到达证据。',
        '该实验属于真值辅助上限，不能替代完整感知到达证据。另已完成同前端完整规则闭环，实际到达21/90，95%配置聚类区间12.22%—35.58%，失败与拒绝全部保留。')
    corrected=OUTPUT/'thesis/references_corrected_005.md'
    references = corrected.read_text() if corrected.exists() else '# 参考文献' + original.split('# 参考文献', 1)[1]
    if corrected.exists():
        sources.append(dict(path=str(corrected.relative_to(OUTPUT)),sha256=digest(corrected)))
    for relative in ['thesis/citation_corrections_007.json', 'thesis/claim_citation_review_008.json',
                     'thesis/claim_citation_review_015.json', 'performance/road_groups_dev_001/report.json']:
        if (OUTPUT/relative).exists():
            sources.append(dict(path=relative,sha256=digest(OUTPUT/relative)))
    references += '''
[83] Wang P, Bai S, Tan S, et al. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution[EB/OL]. arXiv:2409.12191, 2024. https://arxiv.org/abs/2409.12191.

[84] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models[EB/OL]. arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685.

[85] Dolgov D, Thrun S, Montemerlo M, Diebel J. Practical Search Techniques in Path Planning for Autonomous Driving[C]//Proceedings of the First International Symposium on Search Techniques in Artificial Intelligence and Robotics (STAIR-08). Chicago: AAAI, 2008. https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf.

'''
    preface = '''# 面向智能车辆的三维语义场景图构建与导航方法研究

## 阶段整合稿说明

本稿采用通用中文硕士论文版式，学校模板未提供。原第1—3章及旧实验保持只读；本文件修订研究范围并新增第4—7章。所有实验结论仅限已冻结证据；整个课题尚未完成。开发图表逐步补充，完整主对比、新封存测试、人工属性和实时异步评价仍需补齐，参考文献为原稿条目与已核验新增来源的阶段整合，不能视为全部正文主张已全文核验。

## 中文摘要（阶段稿）

针对车载对象观测稀疏、跨帧关联不稳定及语言目标与车辆可执行终点不一致的问题，本文研究已知位姿与道路先验下的三维对象和关系维护，以及静止目标接近导航。系统将相机与投影LiDAR深度作为观测，结合深度支持、一对一关联和时间运动证据形成对象图；通过规则或结构化多模态语义任务检索目标，在确定性道路与整车约束下规划和执行。独立评价区分路线、对象、锁定和实际到达。开发实验表明深度支持改善对象位置与可匹配覆盖，同时存在身份、动静态判断和Adapter提示泛化瓶颈。完整规则感知闭环完成50任务三种子共150次试验，其中90次可达正试验有21次正确到达，覆盖13个不同正任务和8个配置，满足最低功能门槛；失败、无匹配和不可接近任务完整保留。三模型导航对比与新封存主比较仍在验收，本文暂不提出整体性能优于基线的最终结论。

关键词：智能车辆；三维语义场景图；稀疏深度；静止目标接近；车辆运动约束
'''
    manuscript = '\n\n'.join([preface, front, third, chapters[0], chapters[1],
        (destination / '第六章.md').read_text(), chapters[2], references])
    # Noto中文字体缺少꜀字形；明确使用相机到世界的ASCII记号，避免PDF方框。
    manuscript = manuscript.replace('Tᵂ꜀,ₜ', 'T_WC(t)').replace('Tᵂ꜀', 'T_WC')
    manuscript = manuscript.replace('pᶜ', 'p_C').replace('pᵂ', 'p_W')
    manuscript=manuscript.replace('[86]','[45]')
    manuscript=manuscript.replace('A*',r'A\*')
    review_path=OUTPUT/'thesis/claim_citation_review_015.json'
    if review_path.exists():
        for change in json.loads(review_path.read_text())['changes']:
            if manuscript.count(change['before']) != 1:
                raise ValueError('reviewed citation passage changed: '+change['before'])
            for evidence in change['evidence']:
                if digest(OUTPUT/evidence['path']) != evidence['sha256']:
                    raise ValueError('changed primary citation evidence')
            manuscript=manuscript.replace(change['before'],change['after'],1)
    (destination / '完整论文阶段整合稿.md').write_text(manuscript)
    table_text = '\n\n'.join(table_md(t) for t in tables).replace('A*',r'A\*')
    (destination / '第六章三线表.md').write_text(table_text)
    import openpyxl
    from openpyxl.styles import Border, Side, Font, Alignment
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for table in tables:
        sheet = workbook.create_sheet('表' + table['number'])
        sheet.append(table['columns'])
        for row in table['rows']:
            sheet.append(row)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.border = Border(top=Side(style='medium'), bottom=Side(style='thin'))
        for cell in sheet[sheet.max_row]:
            cell.border = Border(bottom=Side(style='medium'))
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(34, max(15, max(len(str(c.value)) for c in column) * 1.3))
            for cell in column:
                cell.alignment = Alignment(wrap_text=True, vertical='center')
        sheet.freeze_panes = 'B2'
        with (destination / f"表{table['number']}.csv").open('x') as f:
            writer = csv.writer(f)
            writer.writerow(table['columns'])
            writer.writerows(table['rows'])
    workbook.save(destination / '第六章三线表.xlsx')
    word_export(destination / '完整论文阶段整合稿.docx', manuscript)
    word_export(destination / '第六章三线表.docx', '', tables)
    import markdown as md
    from playwright.sync_api import sync_playwright
    css = '''@page{size:A4;margin:25mm 22mm}body{font-family:"Noto Serif CJK SC",serif;font-size:10.5pt;line-height:1.65;color:#111}h1{page-break-before:always;font-size:20pt}h1:first-child{page-break-before:auto}h2{font-size:15pt}h3{font-size:12pt}h1,h2,h3{break-after:avoid}p{orphans:3;widows:3;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:8pt;line-height:1.4;border-top:1.2pt solid;border-bottom:1.2pt solid;table-layout:auto}table{break-inside:avoid;break-after:avoid}p:has(+table){break-after:avoid}thead{display:table-header-group;border-bottom:.6pt solid}th,td{padding:5px 4px;overflow-wrap:anywhere}tr{break-inside:avoid}pre{white-space:pre-wrap}img{max-width:100%;height:auto}p:has(img){break-inside:avoid;break-after:avoid}'''
    css += '.table-group{break-inside:avoid}p:has(+.table-group){break-after:avoid}.table-group>p{break-inside:avoid}'
    css += '.equation{display:flex;align-items:center;justify-content:center;position:relative;min-height:3em;break-inside:avoid}.equation-number{position:absolute;right:0}math{font-family:"STIX Two Math","DejaVu Math TeX Gyre",serif;font-size:1.05em}'
    from conceptgraph.completion.manuscript_math import protect_html_math, PATTERN, DISPLAY_NUMBERS, key
    dump(destination/'math_formulas.json',[
        dict(source=match[1] or match[2],display=match[1] is not None,
             number=DISPLAY_NUMBERS.get(key(match[1] or match[2])))
        for match in PATTERN.finditer(manuscript)])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
        page = browser.new_page()
        for name, text in [('完整论文阶段整合稿', manuscript), ('第六章三线表', table_text)]:
            protected,replacements=protect_html_math(text)
            rendered=md.markdown(protected,extensions=['tables','toc'])
            for token,formula in replacements.items():
                rendered=rendered.replace(token,formula)
            rendered=re.sub(r'(<table>.*?</table>)(\s*<p>注：.*?</p>)',
                r'<div class="table-group">\1\2</div>',rendered,flags=re.S)
            rendered=re.sub(r'(src="figures/[^\"]+)\.png"',r'\1.svg"',rendered)
            content = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>' + html.escape(name) + '</title><style>' + css + '</style><body>' + rendered + '</body></html>'
            (destination / f'{name}.html').write_text(content)
            page.goto((destination/f'{name}.html').as_uri())
            page.evaluate('document.fonts.ready')
            page.pdf(path=str(destination / f'{name}.pdf'), print_background=True, display_header_footer=True,
                header_template='<span></span>', footer_template='<div style="font-size:9px;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    with (destination/'generator_source.py').open('x') as handle:
        handle.write(Path(__file__).read_text())
    shutil.copy2(Path(__file__).with_name('manuscript_math.py'),destination/'math_generator_source.py')
    dump(destination / 'delivery_manifest.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        source_original_sha256=digest(original_path), result_sources=sources,
        source_sha256=digest(__file__), output_files={p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        editable_word=True, generic_thesis_layout=True, school_template_supplied=False,
        source_original_modified=False, manuscript_final=False, all_citations_claim_verified=False,
        figure_package=args.figure_run,missing=['其余论文图件','封存测试主表','完整独立消融','完整感知导航比较','人工属性标签','真实时钟异步','最终文献主张核对'],
        goal_complete=False))
    print(json.dumps(dict(output=str(destination), tables=len(tables), characters=len(manuscript), manuscript_final=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
