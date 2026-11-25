"""派生新版本，不改008及其源码；替换必须逐项命中，随后预注册所有源码哈希。"""
from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R/'source'
changes = []


def derive(old, new, replacements):
    original = (S/old).read_text()
    content = original
    for before, after, count in replacements:
        assert content.count(before) == count, (old, before[:100], content.count(before), count)
        content = content.replace(before, after)
    if (S/new).exists():
        assert (S/new).read_text() == content, '已有派生内容不同，不能覆盖：'+new
    else:
        with (S/new).open('x') as stream:
            stream.write(content)
    changes.append(dict(source=old, derived=new,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(content.encode()).hexdigest(), replacements=len(replacements)))


def main():
    derive('real_graph_cooperation_001.py', 'real_graph_cooperation_002.py', [
        ('def complete_geometry_rule(query,nodes,relations):',
         'def complete_geometry_rule(query,nodes,relations,*,evidence_check_enabled=True):', 1),
        ('if explicit:allowed=[n for n in allowed if reliable(n)]',
         'if explicit and evidence_check_enabled:allowed=[n for n in allowed if reliable(n)]', 1),
        ('proposal if chosen is not None and reliable(chosen) else None',
         'proposal if chosen is not None and (not evidence_check_enabled or reliable(chosen)) else None', 1)])
    derive('carla_real_graph_bridge_002.py', 'carla_real_graph_bridge_003.py', [
        ('from real_graph_cooperation_001 import categories, choose, complete_geometry_rule',
         'from real_graph_cooperation_002 import categories, choose, complete_geometry_rule', 1),
        ('clock=time.monotonic, planner=None):',
         'clock=time.monotonic, planner=None, evidence_check_enabled=True):', 1),
        ('self.planner = planner',
         'self.planner = planner\n        self.evidence_check_enabled = bool(evidence_check_enabled)', 1),
        ("pre = choose(self.query, row['candidate_nodes'], row['relation_metadata'], None)",
         "policy = 'full' if self.evidence_check_enabled else 'without_evidence_check'\n        pre = choose(self.query, row['candidate_nodes'], row['relation_metadata'], None, condition=policy)", 1),
        ("decision = complete_geometry_rule(self.query, row['candidate_nodes'], row['relation_metadata'])",
         "decision = complete_geometry_rule(self.query, row['candidate_nodes'], row['relation_metadata'], evidence_check_enabled=self.evidence_check_enabled)", 1),
        ("decision = choose(self.query, row['candidate_nodes'], row['relation_metadata'], prediction)",
         "decision = choose(self.query, row['candidate_nodes'], row['relation_metadata'], prediction, condition=policy)", 1),
        ("evidence_pass = bool(node is not None and reliable(node) and original is not None\n            and not original.is_short_term and latest_depth.get('units') == 'm')",
         "strict_evidence_pass = bool(node is not None and reliable(node) and original is not None\n            and not original.is_short_term and latest_depth.get('units') == 'm')\n        evidence_pass = strict_evidence_pass if self.evidence_check_enabled else bool(node is not None and original is not None)", 1),
        ("current_evidence_passed=evidence_pass, authorized_node_id=authorized,",
         "current_evidence_passed=strict_evidence_pass, evidence_check_enabled=self.evidence_check_enabled, authorized_node_id=authorized,", 1)])
    derive('carla_persistent_intent_bridge_003.py', 'carla_persistent_intent_bridge_004.py', [
        ('from carla_real_graph_bridge_002 import', 'from carla_real_graph_bridge_003 import', 1),
        ("if not reliable(node) or original.is_short_term or latest.get('units')!='m':",
         "if self.evidence_check_enabled and (not reliable(node) or original.is_short_term or latest.get('units')!='m'):", 1),
        ("elif original.metadata.get('timestamp_s') is None or state.time_s-float(original.metadata['timestamp_s'])>2.:",
         "elif self.evidence_check_enabled and (original.metadata.get('timestamp_s') is None or state.time_s-float(original.metadata['timestamp_s'])>2.):", 1),
        ("current_evidence_passed=passed,authorized_node_id=authorized,",
         "current_evidence_passed=passed,evidence_check_enabled=self.evidence_check_enabled,authorized_node_id=authorized,", 1)])
    derive('carla_aggregated_obstacle_bridge_004.py', 'carla_aggregated_obstacle_bridge_005.py', [
        ('from carla_persistent_intent_bridge_003 import', 'from carla_persistent_intent_bridge_004 import', 1)])
    derive('carla_guarded_complex_episode_008.py', 'carla_expansion_episode_009.py', [
        ('from carla_persistent_intent_bridge_003 import ActualQwenBackend,PersistentIntentBridge,hold_search_until_semantic_delivery',
         'from carla_persistent_intent_bridge_004 import ActualQwenBackend,hold_search_until_semantic_delivery\nfrom carla_lane_seed_bridge_006 import LaneSeedBridge\nfrom carla_lossless_jsonl_001 import LosslessJsonl\nfrom carla_expansion_truth_009 import finalize_truth', 1),
        ("parser.add_argument('--mode',choices=['rule','base','adapter'],required=True);args=parser.parse_args()",
         "parser.add_argument('--mode',choices=['rule','base','adapter'],required=True);parser.add_argument('--seed',required=True,type=int);parser.add_argument('--condition',choices=CONDITIONS,required=True);args=parser.parse_args()", 1),
        ('guarded_complex_registration_008.json', 'expansion_registration_009.json', 2),
        (";condition='full_rule_geometry';OUT=ROOT/'carla_official/guarded_complex_navigation_008'/args.episode/args.mode",
         ";condition=args.condition;assert args.seed in bank['seeds'];bank['seed']=args.seed;OUT=ROOT/'carla_official/expansion_navigation_009'/args.episode/f'seed{args.seed}'/args.mode/condition", 1),
        ("OUT.mkdir(parents=True,exist_ok=False);started=", "OUT.mkdir(parents=True,exist_ok=False);bulk=Path(bank['bulk_sensor_root'])/args.episode/f'seed{args.seed}'/args.mode/condition;bulk.mkdir(parents=True,exist_ok=False);(bulk/'frames').mkdir();(bulk/'guard_lidar').mkdir();(OUT/'frames').symlink_to(bulk/'frames',target_is_directory=True);(OUT/'guard_lidar').symlink_to(bulk/'guard_lidar',target_is_directory=True);started=", 1),
        ('w.set_weather(carla.WeatherParameters.ClearNoon)', "w.set_weather(getattr(carla.WeatherParameters,PARAMS['weather']))", 1),
        ('get_spawn_points()[0]', "get_spawn_points()[PARAMS['spawn_index']]", 1),
        ("target_t.location.z+=.3", "target_t.location.z+=.3;target_right=target_t.get_right_vector();jitter=bank['seed_perturbations'][str(args.seed)]['target_lateral_m'];target_t.location.x+=jitter*target_right.x;target_t.location.y+=jitter*target_right.y", 1),
        ("bp.find('vehicle.tesla.model3'),target_t", "bp.find(PARAMS['target_blueprint']),target_t", 1),
        ("distractor=w.spawn_actor(bp.find('vehicle.tesla.model3'),side_t)", "side_t.rotation.yaw+=PARAMS['distractor_yaw_offset_deg'];distractor=w.spawn_actor(bp.find(PARAMS['distractor_blueprint']),side_t)", 1),
        ("assert np.linalg.norm(other_center-origin)<np.linalg.norm(truth_center-origin), '预选干扰车应比指定目标近；失败不换场景'",
         "assert PARAMS['task_kind'] in ['positive','no_match','blocked_target']", 1),
        ("category='car',target_polygon_xy=", "category=PARAMS['target_category'],target_polygon_xy=", 1),
        ("category='car',center_xy=other_center.tolist()", "category=PARAMS['distractor_category'],center_xy=other_center.tolist()", 1),
        ("  dump(OUT/'evaluation_only/truth.json',gt)",
         "  truth,gt=finalize_truth(w,gt,truth,PARAMS,road,region,origin,other_polygon);reachable=truth.reachability=='reachable'\n  dump(OUT/'evaluation_only/truth.json',gt)", 1),
        ("weather='ClearNoon',seed=17", "weather=PARAMS['weather'],seed=args.seed", 1),
        ("bridge=PersistentIntentBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.,maximum_intent_shift_m=1.5)",
         "bridge=LaneSeedBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.,maximum_intent_shift_m=1.5,evidence_check_enabled=condition!='without_evidence_check')", 1),
        ("agent.graph=nav.IncrementalSceneGraph(bank['graph_config'])", "agent.graph=agent.graph if condition=='without_relations' else nav.IncrementalSceneGraph(bank['graph_config'])", 1),
        ("with (OUT/'execution.jsonl').open('x') as execution,(OUT/'perception.jsonl').open('x') as perception:",
         "with LosslessJsonl(OUT/'execution.jsonl.gz') as execution,LosslessJsonl(OUT/'perception.jsonl.gz') as perception:", 1),
        ("guard_lidar_path.parent.mkdir(exist_ok=True)", "guard_lidar_path.parent.mkdir(exist_ok=True)", 1),
        ("category='car'),dict(instance_id=str(distractor.id)", "category=PARAMS['target_category']),dict(instance_id=str(distractor.id)", 3),
        ("center_xy=other_center.tolist(),category='car'", "center_xy=other_center.tolist(),category=PARAMS['distractor_category']", 3),
        ("digest(OUT/'execution.jsonl')", "digest(OUT/'execution.jsonl.gz')", 1),
        ("digest(OUT/'perception.jsonl')", "digest(OUT/'perception.jsonl.gz')", 1),
        ("release['comparison_scope']='共同修复控制提前返回和语义运动绑定后，同一开发场景三方法重跑；保留007三项失败，不把多项修复差值归因单个模块'",
         "release['comparison_scope']='009扩展开发集；10配置50任务预注册，种子/方法/消融同一任务参数；008失败保留。道路候选形状先做当前车辆积分和整车扫掠检查，不使用对象真值。'\n  release['bulk_sensor_root']=str(bulk);release['storage_authorization_sha256']=bank['storage_authorization_sha256'];release['logs']='lossless_streaming_gzip_jsonl';release['ablation_definition']=bank['condition_definitions'][condition];release['ablation_switches']=dict(current_frame_only=condition=='current_frame_only',relations_enabled=condition!='without_relations',short_term_expiry_enabled=condition!='without_history_expiry',evidence_check_enabled=condition!='without_evidence_check');release['planner_candidate_seed_budget_s']=1.;release['source_sha256'].update(bank['source_sha256'])", 1),
        ("  ego.apply_control(carla.VehicleControl(brake=1));w.tick(30)",
         "  dump(OUT/'sensor_files_manifest.json',dict(bulk_root=str(bulk),files={str(p.relative_to(bulk)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in bulk.rglob('*') if p.is_file()},authorization_sha256=bank['storage_authorization_sha256'],goal_complete=False))\n  ego.apply_control(carla.VehicleControl(brake=1));w.tick(30)", 1)])
    derive('audit_carla_guarded_execution_001.py', 'audit_carla_guarded_execution_002.py', [
        ('from carla_motion_guard_001 import', 'from carla_lossless_jsonl_001 import read_rows, actual_log\nfrom carla_motion_guard_001 import', 1),
        ("independent_motion_guard_audit_001", "independent_motion_guard_audit_002", 1),
        ("observations = {r['frame_idx']*2: r for r in map(json.loads,\n        (episode/'perception.jsonl').open())}", "observations = {r['frame_idx']*2:r for r in read_rows(episode,'perception.jsonl')}", 1),
        ("rows = list(map(json.loads, (episode/'execution.jsonl').open()))", "rows = read_rows(episode,'execution.jsonl')", 1),
        ('assert path_to_raw.is_relative_to(episode.resolve())',
         "bulk=Path(release['bulk_sensor_root']).resolve()\n        policy=json.loads((R/'json/sensor_bulk_authorization_003.json').read_text())\n        assert digest(R/'json/sensor_bulk_authorization_003.json')==release['storage_authorization_sha256']\n        assert bulk.is_relative_to(Path(policy['authorized_sensor_root']).resolve())\n        assert path_to_raw.is_relative_to(bulk)", 1),
        ("episode/'execution.jsonl', episode/'perception.jsonl',", "actual_log(episode,'execution.jsonl'), actual_log(episode,'perception.jsonl'),", 1)])
    with (R/'json/runtime_derivation_009.json').open('x') as stream:
        json.dump(dict(derived_sources=changes,old_sources_unmodified=True,goal_complete=False),stream,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    main()
