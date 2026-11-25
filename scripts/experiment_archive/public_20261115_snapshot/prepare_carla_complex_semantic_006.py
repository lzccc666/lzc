"""从冻结Prius导航005派生复杂语义开发实验；旧代码/旧结果均不覆盖。"""
import ast
import difflib
import json
from pathlib import Path

from conceptgraph.completion.stage9 import PROJECT, digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    old = R/'source/carla_prius_navigation_episode_005.py'
    new = R/'source/carla_complex_semantic_episode_006.py'
    original = old.read_text()
    source = original

    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    replace('from carla_prius_braking_pilot_007 import AccelerationFeedback',
        'from carla_prius_braking_pilot_007 import AccelerationFeedback\nfrom carla_real_graph_bridge_001 import ActualQwenBackend,RealGraphNavigationBridge')
    replace("parser.add_argument('--condition',choices=CONDITIONS,required=True);args=parser.parse_args()",
        "parser.add_argument('--mode',choices=['rule','base','adapter'],required=True);args=parser.parse_args()")
    replace("prius_navigation_registration_005.json').read_text());PARAMS=next(p for p in bank['episodes'] if p['episode']==args.episode);condition=args.condition;OUT=ROOT/'carla_official/prius_controller_navigation_005'/args.episode/condition",
        "complex_semantic_registration_006.json').read_text());PARAMS=next(p for p in bank['episodes'] if p['episode']==args.episode);condition='full_rule_geometry';OUT=ROOT/'carla_official/complex_semantic_navigation_006'/args.episode/args.mode")
    replace('actors=[];collisions=[];rows=[];perceptions=[]',
        'actors=[];collisions=[];rows=[];perceptions=[];bridge=None')
    replace("target.set_simulate_physics(False)", """target.set_simulate_physics(False)
  side_index=PARAMS['distractor_waypoint_index'];side_t=carla.Transform(waypoints[side_index].transform.location,waypoints[side_index].transform.rotation);side_t.location.z+=.3;side_right=side_t.get_right_vector();side_t.location.x+=side_right.x*PARAMS['distractor_lateral_right_m'];side_t.location.y+=side_right.y*PARAMS['distractor_lateral_right_m']
  distractor=w.spawn_actor(bp.find('vehicle.tesla.model3'),side_t);actors.append(distractor);distractor.set_simulate_physics(False)""")
    replace('certificate=[]', """other_t=distractor.get_transform();other_center=np.array([other_t.location.x,-other_t.location.y]);other_polygon=Polygon([(v.x,-v.y) for v in distractor.bounding_box.get_world_vertices(other_t)]).convex_hull
  assert np.linalg.norm(other_center-origin)<np.linalg.norm(truth_center-origin), '预选干扰车应比指定目标近；失败不换场景'
  certificate=[]""")
    replace('not body.intersects(target_polygon)',
        'not body.intersects(target_polygon) and not body.intersects(other_polygon)')
    replace("dump(OUT/'evaluation_only/truth.json',gt)", """gt['other_controlled_actors']=[dict(instance_id=str(distractor.id),category='car',center_xy=other_center.tolist(),polygon_xy=list(other_polygon.exterior.coords))]
  gt['controlled_object_scope']='两辆预注册官方车辆；最近/第二近按当前自车距离定义。未把这些actor身份或位置发给智能体；不证明世界内所有未标注静态道具均已分类。'
  gt['initial_target_distance_m']=float(np.linalg.norm(truth_center-origin));gt['initial_distractor_distance_m']=float(np.linalg.norm(other_center-origin))
  dump(OUT/'evaluation_only/truth.json',gt)""")
    source = source.replace('prius_navigation_registration_005.json', 'complex_semantic_registration_006.json')
    replace("release['comparison_scope']='官方CARLA清晰类别规则几何闭环消融；Qwen复杂模块未在该清单激活，不冒充完整语义闭环'",
        "release['comparison_scope']='官方CARLA开发实验：相同两车场景/传感器/预算下比较强规则、基础Qwen、Adapter；复杂否定/第二近实际交给后台模型，未证明封存或大规模成功'\n  release['semantic_mode']=args.mode;release['semantic_ttl_wall_s']=20.;release['semantic_registration']=bank['semantic_protocol'];release['maximum_wall_s']=900")
    replace("agent=Agent(PARAMS['query'],prior,classes,OUT,condition);evaluator=ArrivalEvaluator(truth);cfg=SimpleNamespace(device='cuda:0');models=load_detection_models(cfg)", """backend=None
  if args.mode!='rule':
   binding=json.loads(Path(bank['base_model_binding']).read_text());assert all(digest(p['path'])==p['sha256'] for p in binding['base_model_files'])
   backend=ActualQwenBackend(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct',bank['adapter'] if args.mode=='adapter' else None,OUT/'semantic_backend',seed=bank['seed'])
  bridge=RealGraphNavigationBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.)
  agent=Agent(PARAMS['query'],prior,classes,OUT,condition);agent.semantic_bridge=bridge;evaluator=ArrivalEvaluator(truth);cfg=SimpleNamespace(device='cuda:0');models=load_detection_models(cfg)""")
    replace('if time.monotonic()-started>600:', 'if time.monotonic()-started>900:')
    replace('GT_in_agent=False,rule_only=True,Qwen_evaluated=False,',
        "GT_in_agent=False,rule_only=args.mode=='rule',Qwen_evaluated=bool(backend and backend.calls),semantic_mode=args.mode,actual_Qwen_calls=backend.calls if backend else 0,")
    replace('finally:\n  for a in reversed(actors):',
        "finally:\n  if bridge is not None:dump(OUT/'semantic_queue_final.json',bridge.close(timeout_s=30.))\n  for a in reversed(actors):")
    ast.parse(source)
    new.open('x').write(source)
    prep = R/'carla_official/complex_semantic_preparation_006'
    prep.mkdir(exist_ok=False)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                      fromfile=str(old), tofile=str(new)))
    (prep/'source_delta.diff').write_text(diff)
    training = R/'qwen_adapter_fair_comparison/training/real_graph_seed17_001/training_report.json'
    adapter = Path(json.loads(training.read_text())['best_checkpoint']['path'])
    prior = json.loads((R/'carla_official/prius_navigation_registration_005.json').read_text())
    paths = [new, Path(__file__), R/'source/carla_real_graph_bridge_001.py',
        R/'source/real_graph_contract_003.py', R/'source/real_graph_cooperation_001.py',
        training, adapter/'adapter_model.safetensors', adapter/'adapter_config.json']
    sources = {**prior['source_sha256'], **{str(p): digest(p) for p in paths}}
    bank = dict(prior)
    bank.update(source_sha256=sources, seed=17, adapter=str(adapter),
        base_model_binding=str(R/'nuscenes_ablation/full_semantics_near6_002/base_model_binding_001.json'),
        modes=['rule','base','adapter'], planned_tasks=1, planned_physical_episodes=3,
        maximum_reserve_per_episode_bytes=1*2**30,
        episodes=[dict(episode='complex_second_nearest20', target_waypoint_index=20,
            query='不要最近的小汽车，选择第二近的小汽车', positive=True, timeout_s=90,
            distractor_waypoint_index=12, distractor_lateral_right_m=4.3)],
        semantic_protocol=dict(same_candidate_pool=True, candidate_cap=8, relation_cap=16,
            same_input_prompt=True, same_model_budget=True, max_input_tokens=2048,
            max_new_tokens=256, dtype='bfloat16', do_sample=False,
            pretrained_adapter_selection='只根据既有独立验证集已选step160，不根据CARLA结果选择',
            clear_tasks='基础Qwen与Adapter均先走同一清晰规则，复杂分支均使用相同后台协议',
            rule='既有受限句法强规则，支持否定/第二近；未削弱基线',
            safety='同一原厂Prius、pilot007控制、当前静止/深度/过期检查及整车扫掠检查',
            inference_queue='单工作线程、最多1个等待请求、单请求20s真实时钟TTL',
            only_common_change_vs_pilot005='为三种方法共同增加一个近处车道外干扰车，完整安全检查一致',
            performance_scope='同步感知步进，后台语言推理；不是端到端实时声明'),
        scope='三方法同一复杂开发场景，保留全部失败；不等于全目标配置数/任务数/种子数完成',
        whole_project_sealed_test=False, goal_complete=False)
    registration = R/'carla_official/complex_semantic_registration_006.json'
    dump(registration, bank)
    dump(prep/'receipt.json', dict(base=str(old), base_sha256=digest(old), derived=str(new),
        derived_sha256=digest(new), source_diff_sha256=digest(prep/'source_delta.diff'),
        registration_sha256=digest(registration), old_source_unchanged=digest(old)==sources[str(old)],
        goal_complete=False))
    print(json.dumps(dict(registration=str(registration), planned_episodes=3), ensure_ascii=False))


if __name__ == '__main__':
    main()
