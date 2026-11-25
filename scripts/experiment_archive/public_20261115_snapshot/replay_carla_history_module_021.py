"""三个历史维护配置读相同已闭合真实传感器；机制诊断不计新闭环或方法效果。"""
from collections import Counter
import copy
import datetime
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from carla_agent_isolated_002 import mapping_config
from carla_compact_evidence_001 import load_compact
from carla_history_lifecycle_017 import lifecycle_config, lifecycle_observation
from carla_lossless_jsonl_001 import read_rows
from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.scripts.run_incremental_semantic_navigation import mapping_objects_to_nodes
from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]
PROJECT = R.parents[3]
CONDITIONS = ['full', 'without_short_term_expiry', 'without_history_maintenance']


def save(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    out = R / 'carla_official/history_module_recorded_diagnostic_021'
    out.mkdir(exist_ok=False)
    begun = time.monotonic()
    bankpath = R / 'carla_official/expansion_registration_011.json'
    bank = json.loads(bankpath.read_text())
    jobs = bank['execution_phases']['primary'][:2]
    assert [j['episode'] for j in jobs] == ['town10_spawn001_ClearNoon_car20', 'town10_spawn002_ClearNoon_car18']
    sources = [Path(__file__).resolve(), R/'source/carla_history_lifecycle_017.py',
               R/'source/carla_agent_isolated_002.py', R/'source/carla_compact_evidence_001.py',
               PROJECT/'conceptgraph/slam/streamlined_mapping.py', PROJECT/'conceptgraph/slam/utils.py',
               PROJECT/'conceptgraph/slam/mapping.py', PROJECT/'conceptgraph/completion/evidence.py']
    source_sha = {str(p): digest(p) for p in sources}
    registered = []
    inputs = {}
    for job in jobs:
        episode = (R/'carla_official/expansion_navigation_010'/job['episode']
                   / ('seed'+str(job['seed'])) / job['mode'] / job['condition'])
        assert (episode/'independent_expansion_audit_009/report.json').exists()
        observations = read_rows(episode, 'perception.jsonl')
        inputs[str(episode/'perception.jsonl.gz')] = digest(episode/'perception.jsonl.gz')
        for p in observations:
            folder = episode/'frames'/f"{p['frame_idx']:06d}"
            for name in ['rgb.png', 'lidar_depth_mm.png', 'detections_compact.pkl.gz']:
                file = folder/name
                assert digest(file) == p['files_sha256'][str(file)]
        registered.append(dict(episode=str(episode), frames=len(observations)))
    configs = {'full': mapping_config(), 'without_short_term_expiry': mapping_config(),
               'without_history_maintenance': lifecycle_config(mapping_config(), False)}
    configs['without_short_term_expiry'].dynamic_ttl_s = 1.e9
    cfgs = {name: OmegaConf.to_container(c) for name, c in configs.items()}
    different = {k for k in cfgs['full'] if cfgs['full'][k] != cfgs['without_history_maintenance'][k]}
    assert different == {'enable_dynamic_object_handling', 'dynamic_ttl_s'}
    save(out/'registration.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         selection='冻结主队列最先两项；全部真实前端帧，相同原传感器，不依据成绩筛帧。该队列前两项已查看成绩，因此明确开发机制诊断，非封存/正式全场景效果比较。',
         registered=registered, conditions=CONDITIONS, configs=cfgs, input_sha256=inputs,
         source_sha256=source_sha, seed_per_frame='17+frame_idx，三配置相同',
         evidence_policy=dict(enabled=True, minimum_observations=3, minimum_lidar_pixels=5,
                              minimum_depth_support_fraction=.4),
         maximum_wall_s=1800, maximum_new_bytes=32*1024**2, actual_new_GPU_calls=0,
         actual_new_physics_calls=0, whole_project_sealed_test=False, goal_complete=False))
    classes = (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines()
    K = np.array([[320., 0, 320], [0, 320., 180], [0, 0, 1.]])
    reports = []
    for item in registered:
        episode = Path(item['episode'])
        observations = read_rows(episode, 'perception.jsonl')
        for condition in CONDITIONS:
            mapper = StreamlinedMapper(copy.deepcopy(configs[condition]), classes)
            times = []
            totals = Counter()
            case = out/episode.parents[2].name/condition
            case.mkdir(parents=True, exist_ok=False)
            with (case/'frames.jsonl').open('x') as stream:
                for obs in observations:
                    if time.monotonic()-begun > 1800 or shutil.disk_usage(R).free < 20*2**30:
                        raise RuntimeError('冻结时间/空间门限触发，保留已有前缀')
                    idx = obs['frame_idx']
                    folder = episode/'frames'/f'{idx:06d}'
                    rgb = np.asarray(Image.open(folder/'rgb.png').convert('RGB'))
                    depth = np.asarray(Image.open(folder/'lidar_depth_mm.png'), dtype=float)/1000.
                    gobs = load_compact(folder/'detections_compact.pkl.gz')
                    np.random.seed(17+idx)
                    tick = time.monotonic()
                    mapper.update(MappingFrame(idx, rgb, depth, K, np.asarray(obs['camera_pose_cv']),
                                  gobs, str(folder/'rgb.png'), timestamp_s=obs['sim_time_s']))
                    elapsed = time.monotonic()-tick
                    times.append(elapsed)
                    nodes = mapping_objects_to_nodes(mapper.objects, classes, idx)
                    decisions = {str(n): target_evidence_decision(node, dict(enabled=True,
                        minimum_observations=3, minimum_lidar_pixels=5,
                        minimum_depth_support_fraction=.4)) for n, node in nodes.items()}
                    evidence = lifecycle_observation(mapper, obs['sim_time_s'])
                    count = Counter(node.motion_state for node in nodes.values())
                    totals.update(node_frame_observations=len(nodes), eligible_node_frame_observations=sum(v[0] for v in decisions.values()), **count)
                    row = dict(frame_idx=idx, sensor_frame=obs['sensor_frame'], sim_time_s=obs['sim_time_s'],
                               condition=condition, mapper_elapsed_s=elapsed,
                               lifecycle=evidence, evidence_decisions=decisions,
                               nodes=[dict(node_id=n, center=node.bbox_center.tolist(),
                                           motion=node.motion_state, metadata=dict(
                                               num_detections=node.metadata['num_detections'],
                                               object_identity=node.metadata['object_identity'],
                                               timestamp_s=node.metadata['timestamp_s'],
                                               first_seen_frame=node.metadata['first_seen_frame'],
                                               last_seen_frame=node.metadata['last_seen_frame'],
                                               depth_evidence_count=len(node.metadata['depth_evidence']),
                                               latest_depth_evidence=(node.metadata['depth_evidence'][-1]
                                                   if node.metadata['depth_evidence'] else None)))
                                      for n, node in nodes.items()])
                    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'
                    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                    if used+len(encoded.encode('utf-8'))+128*1024 > 32*1024**2:
                        raise RuntimeError('32MiB诊断日志保护停止；保留已写前缀')
                    stream.write(encoded)
                stream.flush()
            report = dict(episode=str(episode), condition=condition, frames=len(observations),
                          totals=dict(totals), mapper_wall_s=sum(times),
                          process_RSS_high_water_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          process_peak_not_isolated_per_config=True,
                          frame_records_sha256=digest(case/'frames.jsonl'), goal_complete=False)
            save(case/'report.json', report)
            reports.append(report)
            print(json.dumps(dict(episode=episode.parents[2].name, condition=condition,
                                  frames=len(observations), totals=dict(totals)), ensure_ascii=False), flush=True)
    assert all(digest(p) == h for p, h in source_sha.items())
    save(out/'report.json', dict(recorded_episodes=len(registered), unique_sensor_frames=sum(r['frames'] for r in registered),
         actual_CPU_mapping_updates=sum(r['frames'] for r in reports), reports=reports,
         wall_s=time.monotonic()-begun,
         interpretation='验证关闭后的实际对象状态与证据门限交互，不计算三方法准确率，不用本诊断推断完整闭环消融优势。未知运动状态和缺失时间更新不得用手工补值掩盖。',
         actual_new_GPU_calls=0, actual_new_simulation_calls=0, actual_new_arrivals=0,
         full_closed_loop_history_ablation_complete=False, new_sealed_test_opened=False, goal_complete=False))


if __name__ == '__main__':
    main()
