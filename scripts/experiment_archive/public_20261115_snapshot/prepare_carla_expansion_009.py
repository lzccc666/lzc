"""性能运行前冻结10配置50任务；方法/消融复用相同任务，不把种子算新任务。"""
import copy
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    bank=copy.deepcopy(json.loads((R/'carla_official/guarded_complex_registration_008.json').read_text()))
    auth=R/'json/sensor_bulk_authorization_003.json';policy=json.loads(auth.read_text())
    configs=[];tasks=[]
    for spawn in range(5):
        for weather in ['ClearNoon','CloudyNoon']:
            config=f'town10_spawn{spawn:03d}_{weather}'
            configs.append(dict(config_id=config,spawn_index=spawn,weather=weather,
                geometry_inventory_sha256=digest(R/f'carla_official/task_geometry_inventory_001/spawn_{spawn:03d}.json')))
            variants=[('car18','positive',18,'靠近本车道内最近的小汽车'),
                ('car20','positive',20,'靠近本车道内最近的小汽车'),
                ('relation22','positive',22,'以卡车为参照，选择本车道内左侧的小汽车'),
                ('no_bus','no_match',20,'靠近本车道内的公交车'),
                ('blocked_truck','blocked_target',20,'靠近本车道内的卡车')]
            for label,kind,index,query in variants:
                task=dict(episode=config+'_'+label,config_id=config,spawn_index=spawn,weather=weather,
                    task_kind=kind,target_waypoint_index=index,query=query,positive=kind!='no_match',
                    timeout_s=30 if kind=='no_match' else 90,
                    target_blueprint='vehicle.carlamotors.carlacola' if kind=='blocked_target' else 'vehicle.tesla.model3',
                    target_category='truck' if kind=='blocked_target' else 'car',
                    distractor_blueprint='vehicle.carlamotors.carlacola' if label=='relation22' else 'vehicle.tesla.model3',
                    distractor_category='truck' if label=='relation22' else 'car',
                    distractor_waypoint_index=index if label=='relation22' else (9 if kind=='blocked_target' else 45),
                    distractor_lateral_right_m=4.3 if label=='relation22' else (0. if kind=='blocked_target' else 10.),
                    distractor_yaw_offset_deg=90. if kind=='blocked_target' else 0.,
                    speed_mps=0.,goal_region_radius_m=1.3,target_center_goal_offset_m=8.,
                    arrival_max_speed_mps=.08,arrival_max_heading_error_deg=15.,arrival_hold_s=2.,
                    eligibility_requires_independent_scope_and_geometry_certificate=True)
                tasks.append(task)
    seeds=[17,29,43]
    primary=[dict(episode=t['episode'],seed=seed,mode='adapter',condition='full_rule_geometry') for seed in seeds for t in tasks]
    priorities=['town10_spawn001_ClearNoon_car20','town10_spawn002_ClearNoon_car18',
        'town10_spawn003_ClearNoon_blocked_truck','town10_spawn004_ClearNoon_no_bus']
    first=[next(j for j in primary if j['episode']==episode and j['seed']==17) for episode in priorities]
    primary=first+[j for j in primary if j not in first]
    paired=[t for t in tasks if t['weather']=='ClearNoon' and t['episode'].endswith(('_car20','_relation22'))]
    assert len(paired)==10
    methods=[dict(episode=t['episode'],seed=seed,mode=mode,condition='full_rule_geometry') for t in paired for seed in seeds for mode in ['rule','base']]
    removed=['current_frame_only','without_relations','without_history_expiry','without_evidence_check']
    ablations=[dict(episode=t['episode'],seed=seed,mode='adapter',condition=condition) for t in paired for seed in seeds for condition in removed]
    bank.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='官方CARLA开发扩展；共享地图相近道路，非独立城市/封存测试；008三项负结果继续保留',
        configurations=configs,episodes=tasks,seeds=seeds,
        seed_perturbations={'17':dict(target_lateral_m=0.),'29':dict(target_lateral_m=-.04),'43':dict(target_lateral_m=.04)},
        seed_effect='固定目标横向偏移0/-4/+4cm及Python/NumPy/Torch种子；不是三个独立场景',
        modes=['rule','base','adapter'],planned_tasks=50,planned_configurations=10,
        planned_reachable_positive_tasks=30,planned_no_match_tasks=10,planned_blocked_target_tasks=10,
        independent_truth_certificates_must_be_checked_before_credit=True,
        primary_system='adapter模式完整合作：清楚条件用规则、复杂参照调用真实Adapter；简单任务成功不能归因Qwen',
        execution_phases=dict(primary=primary,methods=methods,ablations=ablations),
        actual_new_episodes_at_registration=0,planned_primary_physical_episodes=150,
        planned_additional_method_comparison_episodes=60,planned_additional_ablation_episodes=120,
        matched_task_ids=[t['episode'] for t in paired],
        maximum_wall_s=1800,maximum_reserve_per_episode_bytes=512*2**20,
        bulk_sensor_root=policy['authorized_sensor_root'],storage_authorization_sha256=digest(auth),
        condition_definitions={
            'full_rule_geometry':'完整合作链：真实当前传感器、跨帧一对一建图、运动/短期维护、关系、证据门闸、规则与选定Qwen；名称沿用类接口，不表示只运行规则。',
            'current_frame_only':'每次感知重置mapper及属性，不积累历史；其余证据门限和控制不改。',
            'without_relations':'关系生成及输入均为空；其余坐标与候选保留，允许方法从坐标重新推断；不承诺一定降低成绩。',
            'without_history_expiry':'只禁用mapper短期对象2s过期（TTL=1e9）；其余观测更新/运动状态/控制授权保留，不冒充去掉全部历史维护。',
            'without_evidence_check':'关闭rank和桥内观测来源/像素/支持率/运动/时效门闸；保留节点存在、稳定身份连续性、任务类别和整车几何/运动保护。'},
        common_development_changes_from_008=['精确障碍并集加独立逐障碍扫掠审计','声明车道形状先从当前状态积分审计，失败仍调用原Hybrid','停稳但距感知目标超过1m时可恢复接近','本车道任务对目标类别按公开道路先验过滤；参照物不滤','新逐帧原始传感器写共享卷并保留项目入口及SHA256','JSONL流式gzip逐字节无损'],
        limits='可达/无匹配/阻断标签须以每个实际回合独立证书确认；没有证书不计验收。短期过期消融仍不能替代完整历史维护消融。')
    sourcefiles=[R/'source'/n for n in [
        'prepare_carla_expansion_009.py','carla_expansion_episode_009.py','carla_expansion_truth_009.py',
        'carla_lane_seed_bridge_006.py','carla_aggregated_obstacle_bridge_005.py','carla_persistent_intent_bridge_004.py',
        'carla_real_graph_bridge_003.py','real_graph_cooperation_002.py','carla_lossless_jsonl_001.py',
        'run_carla_expansion_batch_009.py','run_official_expansion_017.py','audit_carla_expansion_009.py',
        'audit_carla_guarded_execution_002.py','test_carla_expansion_009.py']]
    bank['source_sha256'].update({str(p.resolve()):digest(p) for p in sourcefiles})
    bank['source_sha256'][str(auth.resolve())]=digest(auth)
    assert len(tasks)==50 and len(configs)==10 and len(primary)==150 and len(methods)==60 and len(ablations)==120
    assert len({(j['episode'],j['seed'],j['mode'],j['condition']) for phase in bank['execution_phases'].values() for j in phase})==330
    dump(R/'carla_official/expansion_registration_009.json',bank)
    print('已冻结：10配置/50任务，主系统150回合，额外方法60/消融120；此时实际运行0，独立真值证书尚待每回合核验。')


if __name__=='__main__':main()
