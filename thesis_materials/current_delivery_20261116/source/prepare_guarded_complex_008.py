"""由冻结007生成新008；保留旧实验，先验冻结共同修复后再实跑三方法。"""
import ast
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    old = R/'source/carla_complex_semantic_episode_007.py'
    new = R/'source/carla_guarded_complex_episode_008.py'
    assert not new.exists()
    calibration = R/'carla_official/prius_cutoff_readback_003/report.json'
    proof = json.loads(calibration.read_text())
    assert proof['actual_cutoff_calibration_passed']
    assert proof['nominal_stopping_distance_bound_covers_observed_calibration']
    text = old.read_text()
    changes = {
        'import argparse,copy,gzip,json,math,pickle,queue,time,traceback':
            'import argparse,copy,gzip,hashlib,json,math,pickle,queue,time,traceback,zlib',
        'from carla_ablation_agent_001 import Agent,CONDITIONS':
            'from carla_ablation_agent_001 import Agent as PreviousAgent,CONDITIONS',
        'from carla_prius_braking_pilot_007 import AccelerationFeedback':
            'from carla_prius_cutoff_pilot_008 import AccelerationFeedback',
        'from carla_real_graph_bridge_002 import ActualQwenBackend,RealGraphNavigationBridge':
            'from carla_persistent_intent_bridge_003 import ActualQwenBackend,PersistentIntentBridge,hold_search_until_semantic_delivery\nfrom carla_motion_guard_001 import current_full_lidar_obstacles\nfrom carla_guarded_control_001 import guarded_control',
        "ROOT=Path(__file__).resolve().parents[1];OUT=None":
            "class Agent(PreviousAgent):\n def perceive(self,*args,**kwargs):\n  result=super().perceive(*args,**kwargs)\n  return hold_search_until_semantic_delivery(self,result)\n\nROOT=Path(__file__).resolve().parents[1];OUT=None",
        "ROOT/'carla_official/complex_semantic_registration_007.json'":
            "ROOT/'carla_official/guarded_complex_registration_008.json'",
        "ROOT/'carla_official/complex_semantic_navigation_007'":
            "ROOT/'carla_official/guarded_complex_navigation_008'",
        "bridge=RealGraphNavigationBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.)":
            "bridge=PersistentIntentBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.,maximum_intent_shift_m=1.5)",
        "release['model_sha256']={":
            "release['actuator_policy']='原厂Prius，冻结cutoff008反馈控制；停车立即切油，速度<0.15时刹车；最大加速1/减速2，终区与保持门限不变'\n  release['ego_physics_control']['reason']='原厂物理，立即切油控制经过两组官方标定，不代表完整导航成功'\n  release['ego_physics_control']['calibration_sha256']=digest(ROOT/'carla_official/prius_cutoff_pilot_008/report.json')\n  release['steering_calibration']['throttle_down_on_stop']='immediate_zero'\n  release['steering_calibration']['coasting_calibration_sha256']=digest(ROOT/'carla_official/prius_cutoff_readback_003/report.json')\n  release['controller_calibration_note']='008两组实际切油停车+逐条重算；仅两条开发转向轨迹，不证明泛化'\n  release['motion_guard']=bank['motion_guard']\n  release['persistent_intent']=bank['persistent_intent']\n  release['comparison_scope']='共同修复控制提前返回和语义运动绑定后，同一开发场景三方法重跑；保留007三项失败，不把多项修复差值归因单个模块'\n  release['model_sha256']={",
        "body=Polygon([(v.x,-v.y) for v in ego.bounding_box.get_world_vertices(tf)]).convex_hull;lane_violation|=not road.buffer(.05).covers(body)":
            "body=Polygon([(v.x,-v.y) for v in ego.bounding_box.get_world_vertices(tf)]).convex_hull;lane_violation|=not road.buffer(.05).covers(body)\n    current_lidar=np.frombuffer(data['lidar'].raw_data,np.float32).reshape(-1,4);current_pose=np.asarray(data['lidar'].transform.get_matrix());current_obstacles,current_observation=current_full_lidar_obstacles(current_lidar,current_pose,state,spec,sensor_frame=data['lidar'].frame,snapshot_frame=snapshot.frame)\n    if tick%2==0:guard_lidar_path=OUT/'frames'/f'{tick//2:06d}'/'lidar.bin';guard_encoding='raw_float32_xyzi'\n    else:\n     guard_lidar_path=OUT/'guard_lidar'/f'{tick:06d}.bin.zlib';guard_lidar_path.parent.mkdir(exist_ok=True);guard_lidar_path.write_bytes(zlib.compress(data['lidar'].raw_data,6));guard_encoding='zlib_lossless_float32_xyzi'\n     assert zlib.decompress(guard_lidar_path.read_bytes())==data['lidar'].raw_data\n    current_observation.update(raw_file=str(guard_lidar_path.relative_to(OUT)),raw_encoding=guard_encoding,raw_uncompressed_sha256=hashlib.sha256(data['lidar'].raw_data).hexdigest(),lidar_pose_carla=current_pose.tolist(),obstacle_polygons_xy=[list(p.exterior.coords) for p in current_obstacles])\n    agent.current_lidar_observation=current_observation;agent.obstacles=current_obstacles;agent.safety_update=lambda observed_agent:setattr(observed_agent,'obstacles',current_obstacles)",
        "if collisions or lane_violation:acceleration=-2.":
            "acceleration,steering,tracking=guarded_control(agent,state,spec,acceleration,steering,tracking)\n    tracking['current_motion_observation']=current_observation\n    if collisions or lane_violation:acceleration=-2.",
        "or agent.guard_latched or collisions or lane_violation":
            "or agent.guard_latched or tracking.get('current_motion_guard_stop',False) or collisions or lane_violation",
        "stock_controller_version='pilot007'": "stock_controller_version='cutoff008'",
    }
    for before, after in changes.items():
        count = text.count(before)
        # registration路径出现读入和release绑定两次，其他替换均应只有一处。
        expected = 2 if before == "ROOT/'carla_official/complex_semantic_registration_007.json'" else 1
        assert count == expected, (before, count, expected)
        text = text.replace(before, after)
    # 本轮summary与独立选择审计都对两个受控对象作同一一对一关联。
    single = "[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')]"
    both = "[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car'),dict(instance_id=str(distractor.id),center_xy=other_center.tolist(),category='car')]"
    assert text.count(single) == 3
    text = text.replace(single, both)
    text = text.replace('官方CARLA开发episode：传感器→原有真实前端→对象/关系→规则选择→Hybrid A*→物理控制。',
        '官方CARLA008：整圈LiDAR每物理步保护、持续身份核验和异步复杂语义；三方法同一修复。')
    ast.parse(text)
    with new.open('x') as stream:
        stream.write(text)
    previous = json.loads((R/'carla_official/complex_semantic_registration_007.json').read_text())
    sources = {p: h for p, h in previous['source_sha256'].items()}
    assert all(digest(p) == h for p, h in sources.items())
    for p in [Path(__file__), new, R/'source/carla_persistent_intent_bridge_003.py',
            R/'source/test_carla_persistent_intent_bridge_003.py', R/'source/carla_motion_guard_001.py',
            R/'source/carla_guarded_control_001.py', R/'source/test_carla_guarded_control_001.py',
            R/'source/carla_prius_cutoff_pilot_008.py', calibration,
            R/'source/audit_carla_complex_choices_008.py']:
        sources[str(p)] = digest(p)
    bank = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='修复007失败原因后的固定开发场景；并非单因素消融、封存或规模验收',
        official_version='0.9.16', map='Town10HD_Opt', seed=17, initial_spawn_index=0,
        weather='ClearNoon', traffic_density=0, episodes=previous['episodes'],
        modes=['rule', 'base', 'adapter'], planned_tasks=1, planned_physical_episodes=3,
        conditions=['full_rule_geometry'], graph_config=previous['graph_config'],
        adapter=previous['adapter'], base_model_binding=previous['base_model_binding'],
        semantic_protocol=dict(previous['semantic_protocol'],
            safety='同一原厂Prius/切油008控制/每物理步整圈LiDAR名义滑行保护/当前证据检查/规划扫掠检查',
            only_common_change_vs_pilot005=None,
            performance_scope='同步前端与物理步进、异步模型，未声明端到端实时',
            commit_policy='初次模型响应保持20sTTL和全上下文绑定；真实状态机锁定后只保留对象身份，每帧重验新证据和路线，不计新模型调用或输出'),
        motion_guard=dict(source='current official full360 LiDAR at every physics tick',
            nominal_passive_deceleration_mps2=.03, reaction_s=.3, brake_gate_mps=.15,
            ground_z_m=0., flat_road_prior=True, route_and_current_steering_forecasts=True,
            invoked_after_original_control_including_no_path=True, GT_in_obstacles=False,
            observed_calibration=str(calibration), physical_safety_generalization_proven=False,
            raw_storage='偶数tick复用frames原LiDAR；奇数tick独立zlib无损保存，逐时刻原始SHA和位姿及障碍多边形'),
        persistent_intent=dict(maximum_static_identity_shift_m=1.5,
            requires_real_machine_lock=True, identical_for_all_modes=True,
            current_evidence_and_safe_plan_required_every_frame=True,
            pending_semantics_hold='有相关观测对象且初始模型未投递时抑制搜索路线，无对象仍允许有界搜索',
            model_result_TTL_wall_s=20, backend_output_cache=False),
        vehicle='vehicle.toyota.prius', physics_modified=False, dt_s=.1,
        maximum_throttle=.24, maximum_desired_speed_mps=.55, perception_every_ticks=2,
        arrival=previous['arrival'], maximum_reserve_per_episode_bytes=1152*1024**2,
        maximum_total_project_growth_reserve_bytes=3456*1024**2,
        comparison_rules=['强规则、基础Qwen、新Adapter使用同一候选协议、规划/控制/等待策略和90s任务',
            '保持2048/256预算和提示词，不降低基线，不改终区、速度、保持和控制限值',
            '所有失败和原始前缀保留，旧007不覆盖，不按CARLA表现换查询/种子/Adapter',
            '三方法动作不同导致后续观测不同，闭环调用次数不能当配对冻结准确率'],
        source_sha256=sources, previous_failure_retained='carla_official/complex_semantic_navigation_007',
        whole_project_sealed_test=False, goal_complete=False)
    dump(R/'carla_official/guarded_complex_registration_008.json', bank)
    print('008新源码和共同协议已冻结，尚无本轮物理导航信用', flush=True)


if __name__ == '__main__':
    main()
