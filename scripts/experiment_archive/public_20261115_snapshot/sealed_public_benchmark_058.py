"""冻结后运行30场景公开数据离线比较。GT只进入evaluate，不进入front/map/requests/gpu。"""
# ruff: noqa: E402
from __future__ import annotations

import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import resource
import shutil
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

PROJECT = Path('/root/autodl-tmp/lizichong')
OLD = PROJECT / 'outputs/project_v2/thesis_completion_v1'
ROOT = OLD / 'public_experiment_completion_20261115'
DEST = ROOT / 'main_experiments/sealed_public_058'
INPUT = OLD / 'data_manifest/reserved_causal_inputs_full_001'
LABEL = OLD / 'annotation/reserved_official_annotation_001'
MODEL = PROJECT / 'checkpoint/Qwen2-VL-2B-Instruct'
CLIP = PROJECT / 'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors'
sys.path.insert(0, str(PROJECT))
from conceptgraph.completion.fault_robustness import mapping_config  # noqa: E402
from conceptgraph.completion.stage9 import predict  # noqa: E402
from public_query_intake_055 import choose, parse, strong_rule  # noqa: E402
from real_graph_contract_003 import build_completion_prompt, validate  # noqa: E402

SETTINGS = dict(full={}, without_depth_quality={'depth_policy': 'legacy'},
    without_one_to_one={'assignment_cardinality': 'greedy_many_to_one'},
    without_dynamic_maintenance={'enable_dynamic_object_handling': False},
    without_short_term_expiry={'dynamic_ttl_s': 1e9}, without_relations={})
CONDITIONS = list(SETTINGS) + ['original_fusion', 'single_frame', 'one_to_one_only',
                              'one_to_one_depth_quality']
VARIANTS = ['base', 'old_adapter', 'real_graph_seed17', 'real_graph_seed29', 'real_graph_seed43']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        yield from map(json.loads, f)


def frozen():
    release = read(DEST / 'release.json')
    for path, expected in release['frozen_files_sha256'].items():
        if sha(path) != expected:
            raise ValueError('冻结文件变化，禁止续跑：' + path)
    if sha(DEST / 'agent_frames.jsonl') != release['agent_frames_sha256']:
        raise ValueError('观测清单变化')
    if sha(DEST / 'query_inputs.jsonl') != release['query_inputs_sha256']:
        raise ValueError('文字指令清单变化')
    return release


def guard(begun, budget_s):
    if time.monotonic() - begun > budget_s:
        raise TimeoutError('达到冻结单阶段时间预算，保留全部前缀')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise OSError('物理空闲少于20GiB，停止新增大任务')
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if size + 32*2**20 >= 25*2**30:
        raise OSError('本轮新增产物达到25GiB守卫，保留全部文件')


def config(condition):
    full = mapping_config()
    full.assignment_cardinality = 'one_to_one'
    if condition in SETTINGS:
        for key, value in SETTINGS[condition].items():
            setattr(full, key, value)
        return full
    original = OmegaConf.load(PROJECT / 'conceptgraph/hydra_configs/base_mapping.yaml')
    original.merge_with(dict(device='cpu', min_points_threshold=4, obj_min_detections=1,
        dbscan_remove_noise=False, match_method='sim_sum', sim_threshold=.35,
        phys_bias=.5, spatial_sim_type='iou', frame_dt=1., skip_bg=True))
    if condition in {'one_to_one_only', 'one_to_one_depth_quality'}:
        original.merge_with(dict(association_policy='one_to_one', assignment_cardinality='one_to_one',
            dynamic_ttl_s=2., initial_motion_unknown=True, static_confirmation_observations=3,
            depth_policy='legacy' if condition == 'one_to_one_only' else 'largest_supported_cluster'))
    return original


def english(text):
    q = parse(text)
    target = ' or '.join(q.targets) or 'object'
    result = 'a ' + ('parked ' if q.requires_static else '') + target
    if q.relation:
        result += ' ' + dict(left_of='to the left of', right_of='to the right of',
                            in_front_of='in front of', behind='behind')[q.relation]
        result += ' a ' + ' or '.join(q.anchors)
    if q.excluded:
        result += ', not ' + ' or '.join(q.excluded)
    return result


def prepare():
    """仅读取观测、文字和既有核验回执；不访问任何待测模型输出。"""
    DEST.mkdir(parents=True, exist_ok=False)
    assert read(INPUT / 'complete.json')['input_conversion_complete']
    for receipt in [ROOT / 'annotation/reserved_semantic_audit_053/report.json',
                    ROOT / 'annotation/historical_log_readback_056/report.json']:
        assert read(receipt)['passed']
    selection = read(OLD / 'splits/reserved_test_scenes_v2.json')['scenes']
    observations = []
    sources = {Path(__file__), ROOT / 'source/public_query_intake_055.py',
        ROOT / 'source/run_sealed_public_series_059.py',
        ROOT / 'source/real_graph_contract_003.py', ROOT / 'source/mapping_assignment_isolated_001.py',
        ROOT / 'source/train_real_graph_adapter_004.py',
        PROJECT / 'scripts/train_qwen_lora_semantic_delegate_v2.py',
        PROJECT / 'conceptgraph/hydra_configs/base_mapping.yaml',
        PROJECT / 'conceptgraph/vehicle_classes.txt', INPUT / 'complete.json', INPUT / 'release.json',
        INPUT / 'audit_001/report.json', OLD / 'splits/reserved_test_scenes_v2.json',
        LABEL / 'manifest.json', LABEL / 'evaluation_only_frames.jsonl',
        ROOT / 'annotation/reserved_semantic_audit_053/report.json',
        ROOT / 'annotation/historical_log_readback_056/report.json',
        OLD / 'splits/reserved_crop_audit_001/report.json',
        ROOT / 'json/resource_policy_002.json',
        ROOT / 'nuscenes_ablation/graph_near6_correction_001/release.json'}
    for folder in ['slam', 'utils', 'scenegraph', 'navigation', 'vlm', 'completion']:
        sources.update((PROJECT / 'conceptgraph' / folder).rglob('*.py'))
    sources.add(PROJECT / 'conceptgraph/scripts/streamlined_detections.py')
    for scene in selection:
        folder = INPUT / 'converted' / scene['scene'] / 'inputs'
        paths = sorted(folder.glob('*.json'))
        assert len(paths) == scene['frame_count']
        for path in paths:
            row = read(path)
            assert row['gt_in_agent'] is False and row['future_lidar_used'] is False
            assert row['frame_idx'] == len([x for x in observations if x['scene'] == row['scene']])
            if row['lidar_timestamp_s'] is not None:
                assert 0 <= row['camera_timestamp_s'] - row['lidar_timestamp_s'] <= .100001
            row['input_sha256'] = {str(path): sha(path), row['rgb']: sha(row['rgb']), row['depth']: sha(row['depth'])}
            observations.append(row)
    assert len(observations) == 1196
    with (DEST / 'agent_frames.jsonl').open('x') as f:
        for row in observations:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    # 这里只复制允许给推理器的文字/开始时刻，不把semantic_truth或对象答案放进去。
    query_rows = []
    for name in ['main_queries.jsonl', 'challenge_queries.jsonl']:
        sources.add(LABEL / name)
        for q in lines(LABEL / name):
            query_rows.append({key: q[key] for key in ['query_id', 'scene', 'frame_idx',
                'timestamp_s', 'query_text', 'cohort', 'online_replay_selected']})
    with (DEST / 'query_inputs.jsonl').open('x') as f:
        for q in query_rows:
            assert parse(q['query_text']).supported
            f.write(json.dumps(q, ensure_ascii=False, separators=(',', ':')) + '\n')
    adapters = {'base': None, 'old_adapter': PROJECT / 'outputs/project_v2/qwen_eval/finetune_v2/best_adapter'}
    for seed in [17, 29, 43]:
        report = ROOT / f'qwen_adapter_fair_comparison/training/real_graph_seed{seed}_001/training_report.json'
        sources.add(report)
        adapters[f'real_graph_seed{seed}'] = Path(read(report)['best_checkpoint']['path'])
    for p in [PROJECT / 'checkpoint/yolov8l-world.pt', PROJECT / 'checkpoint/mobile_sam.pt', CLIP]:
        sources.add(p)
    sources.update(p for p in MODEL.iterdir() if p.is_file() and p.name != 'README.md')
    for path in adapters.values():
        if path:
            sources.update(p for p in path.iterdir() if p.is_file() and p.name.startswith('adapter_'))
    graph = read(ROOT / 'nuscenes_ablation/graph_near6_correction_001/release.json')['new_actual_graph_config']
    assert graph['relation_distance_thresholds_m']['near'] == 6
    release = dict(UTC=datetime.now(timezone.utc).isoformat(),
        scope='固定公开数据离线感知建图/目标选择版本；不声称CARLA、实时执行或实际到达',
        scenes=[dict(scene=s['scene'], log_token=s['log_token'], frames=s['frame_count']) for s in selection],
        frames=1196, queries=878, cohorts=dict(Counter(q['cohort'] for q in query_rows)),
        selected_replay_queries=sum(q['online_replay_selected'] for q in query_rows),
        natural_language_status='既有声明模板主集/挑战集；不是人工自然指令测试',
        conditions=CONDITIONS, strict_single_module_conditions=list(SETTINGS),
        configs={c: OmegaConf.to_container(config(c), resolve=True) for c in CONDITIONS},
        graph_config=graph, assignment_isolation='复用开发验证的mapping_assignment_isolated_001，统一替换merge函数',
        variants={k: str(v) if v else None for k, v in adapters.items()},
        Qwen_same_candidates_prompt_and_budget=True,
        model_selection='每个种子仅用各自validation最佳检查点；种子17为预定合作模块主种子，不按封存结果选种子',
        retrieval_methods=['category_nearest', 'complete_geometry_rule', 'clip_current_crop',
            'controlled_sparse_clip_grid', *VARIANTS, 'cooperation_seed17',
            'without_rule_query_resolution', 'without_attribute_check',
            'without_candidate_reranking', 'without_evidence_check'],
        semantic_comparison_scope='模型与合作消融只在full共享候选运行；十种建图条件另作对象/身份/关系主表，不声称封存十条件全语义GPU消融',
        representation_policy='Qwen各版本及合作消融共享同一full预测图的8候选/16关系；当前crop与稀疏网格另列表示诊断，不冒充同节点公平Qwen比较或完整外部方法复现',
        candidate_limit=8, relation_limit=16, candidate_order='SHA256(query_id,node_id)，全部方法共用',
        graph_direction_radius_m=12, label_relation_radius_m=15, strong_rule_relation_radius_m=15,
        spatial_label_note='官方语义关系半径15m，预测关系图半径12m，真实覆盖差异保留；所有Qwen还获得同一观测中心坐标',
        grid=dict(resolution_m=.5, ttl_s=5., clip_threshold=None),
        prompt_profile='real_perception_graph_proposal_v3_empty_supported', max_input_tokens=2048,
        max_new_tokens=256, do_sample=False, seed=17, dtype='bfloat16',
        overlength_policy='不截断、不删样本，全部方法该模型请求记输入超限失败',
        fair_Qwen_actual_calls_planned=5*878, GPU_execution='一个模型一个进程，独占串行，不与前端并发',
        source_cache='同一固定请求的模型输出仅推理一次供合作消融回读；回读不计新的GPU调用或延迟样本',
        thresholds=dict(object_match_m=3., unambiguous_margin_m=.5, nearest_tie_m=.5),
        statistics=dict(bootstrap_unit='log', repetitions=4000, seed=17, separate_cohorts=True,
                        missing_positive_target_counts_as_failure=True),
        budgets=dict(frontend_s=7200, mapping_each_condition_s=7200, requests_s=1200,
                     GPU_each_model_s=21600, evaluate_s=1800, new_project_GiB=25, minimum_free_GiB=20),
        frozen_files_sha256={str(p): sha(p) for p in sorted(sources)},
        agent_frames_sha256=sha(DEST / 'agent_frames.jsonl'),
        query_inputs_sha256=sha(DEST / 'query_inputs.jsonl'),
        missing_depth_frames_included=sum(not x['depth_available'] for x in observations),
        methods_frozen=True, model_test_inference_executed=False, correct_arrival_credit=False,
        CARLA_limit=10, do_not_start_CARLA=True, goal_complete=False)
    dump(DEST / 'release.json', release)
    with tarfile.open(DEST / 'source_snapshot.tar.gz', 'x:gz') as archive:
        for p in sorted(sources):
            if p.suffix in {'.py', '.yaml', '.txt'} and p.stat().st_size < 2**20:
                archive.add(p, arcname=str(p.relative_to(PROJECT)), recursive=False)
    print(json.dumps(dict(frozen=True, scenes=30, frames=1196, queries=878,
                         conditions=len(CONDITIONS), variants=VARIANTS)), flush=True)


def frontend():
    release = frozen()
    import cv2
    import torch
    from conceptgraph.scripts.streamlined_detections import detect_rgb_file, load_detection_models
    begun = time.monotonic()
    dest = DEST / 'frontend'
    dest.mkdir(exist_ok=True)
    if (dest / 'report.json').exists():
        raise FileExistsError('前端已完成，禁止重复推理')
    if not (DEST / 'test_opened.json').exists():
        dump(DEST / 'test_opened.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
            release_sha256=sha(DEST / 'release.json'), first_stage='shared_frontend',
            frozen_before_first_inference=True, correct_arrival_credit=False, goal_complete=False))
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    models = load_detection_models(SimpleNamespace(device='cuda:0'))
    for model in models[:2]:
        model.to('cuda:0')
        model.overrides['device'] = 'cuda:0'
    classes = [x.strip() for x in (PROJECT / 'conceptgraph/vehicle_classes.txt').read_text().splitlines() if x.strip()]
    reports = []
    for i, row in enumerate(lines(DEST / 'agent_frames.jsonl')):
        if i % 10 == 0:
            guard(begun, release['budgets']['frontend_s'])
        folder = dest / row['scene'] / f"{row['frame_idx']:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / 'complete.json'
        if receipt.exists():
            record = read(receipt)
            assert all(sha(p) == h for p, h in record['files_sha256'].items())
            reports.append(record)
            continue
        if any(folder.iterdir()):
            raise FileExistsError('帧有未完成前缀，禁止自动重复推理或覆盖：' + str(folder))
        assert all(sha(p) == h for p, h in row['input_sha256'].items())
        t = time.monotonic()
        result, image = detect_rgb_file(SimpleNamespace(device='cuda:0'), models, row['rgb'], classes)
        torch.cuda.synchronize()
        assert result['classes'] == classes and len(result['xyxy']) == len(result['mask'])
        assert result['frame_provenance']['rgb_sha256'] == row['input_sha256'][row['rgb']]
        assert str(next(models[0].model.parameters()).device) == 'cuda:0'
        assert str(next(models[1].model.parameters()).device) == 'cuda:0'
        detection = folder / 'detections.pkl.gz'
        with gzip.open(detection, 'xb') as f:
            pickle.dump(result, f)
        visual = folder / 'detections.jpg'
        if visual.exists() or not cv2.imwrite(str(visual), image):
            raise OSError('检测可视化已有或写入失败')
        record = dict(scene=row['scene'], frame_idx=row['frame_idx'], detections=str(detection),
            visualization=str(visual), boxes=len(result['xyxy']), elapsed_s=time.monotonic()-t,
            files_sha256={str(p): sha(p) for p in [detection, visual]}, gt_used=False, actual_GPU_call=True)
        dump(receipt, record)
        reports.append(record)
        if (i+1) % 10 == 0:
            print(json.dumps(dict(frontend=i+1, total=1196, scene=row['scene'])), flush=True)
    # 图像特征与所有文字特征使用同一个固定CLIP模型；不读取任何目标答案。
    texts = sorted({english(q['query_text']) for q in lines(DEST / 'query_inputs.jsonl')})
    vectors = []
    with torch.inference_mode():
        for i in range(0, len(texts), 16):
            z = models[2].encode_text(models[4](texts[i:i+16]).to('cuda:0')).float()
            z /= z.norm(dim=-1, keepdim=True)
            vectors.extend(z.cpu().numpy())
    if (dest / 'clip_text_features.npz').exists():
        raise FileExistsError('已有文字特征但无完整回执，保留前缀，禁止覆盖')
    np.savez_compressed(dest / 'clip_text_features.npz', texts=np.asarray(texts), features=np.asarray(vectors))
    dump(dest / 'report.json', dict(frames=len(reports), actual_GPU_frame_calls=len(reports),
        actual_CLIP_texts=len(texts), text_features_sha256=sha(dest / 'clip_text_features.npz'),
        receipts_sha256={str(dest / x['scene'] / f"{x['frame_idx']:06d}" / 'complete.json'):
            sha(dest / x['scene'] / f"{x['frame_idx']:06d}" / 'complete.json') for x in reports},
        wall_s=time.monotonic()-begun, peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
        gt_used=False, actual_arrival_credit=False, goal_complete=False))
    print('frontend complete1196', flush=True)


def map_condition(condition):
    release = frozen()
    assert condition in CONDITIONS and read(DEST / 'frontend/report.json')['frames'] == 1196
    import conceptgraph.slam.streamlined_mapping as sm
    from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
    from conceptgraph.scenegraph.relations import RelationNode
    from conceptgraph.completion.retrieval_baselines import SparseFeatureGrid, observation_objects
    spec = importlib.util.spec_from_file_location('isolated_sealed_mapping', ROOT / 'source/mapping_assignment_isolated_001.py')
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    sm.merge_detections_to_objects = isolated.merge_detections_to_objects
    begun = time.monotonic()
    dest = DEST / 'mapping' / condition
    dest.mkdir(parents=True, exist_ok=True)
    classes = [x.strip() for x in (PROJECT / 'conceptgraph/vehicle_classes.txt').read_text().splitlines() if x.strip()]
    all_inputs = list(lines(DEST / 'agent_frames.jsonl'))
    qmap = defaultdict(list)
    for q in lines(DEST / 'query_inputs.jsonl'):
        qmap[q['scene'], q['frame_idx']].append(q)
    embeddings = np.load(DEST / 'frontend/clip_text_features.npz')
    vectors = dict(zip(embeddings['texts'], embeddings['features']))
    completed = []
    for selected in release['scenes']:
        scene = selected['scene']
        folder = dest / scene
        folder.mkdir(exist_ok=True)
        if (folder / 'report.json').exists():
            record = read(folder / 'report.json')
            assert sha(folder / 'frames.jsonl.gz') == record['frames_sha256']
            completed.append(record)
            continue
        mapper = sm.StreamlinedMapper(OmegaConf.create(release['configs'][condition]), classes)
        graph = IncrementalSceneGraph(release['graph_config'])
        registry = {}
        grid = SparseFeatureGrid(resolution=.5, ttl_s=5.)
        count = 0
        times = []
        with gzip.open(folder / 'frames.jsonl.gz', 'xt', compresslevel=6) as output:
            for row in [x for x in all_inputs if x['scene'] == scene]:
                if count % 10 == 0:
                    guard(begun, release['budgets']['mapping_each_condition_s'])
                assert all(sha(p) == h for p, h in row['input_sha256'].items())
                front = read(DEST / 'frontend' / scene / f"{row['frame_idx']:06d}" / 'complete.json')
                assert all(sha(p) == h for p, h in front['files_sha256'].items())
                with gzip.open(front['detections'], 'rb') as f:
                    observation = pickle.load(f)
                image = np.asarray(Image.open(row['rgb']).convert('RGB'))
                depth = np.asarray(Image.open(row['depth']), float) / row['depth_scale']
                pose, intrinsic = np.asarray(row['camera_pose']), np.asarray(row['intrinsic'])
                np.random.seed(17+row['frame_idx'])
                tick = time.monotonic()
                mapper.update(sm.MappingFrame(row['frame_idx'], image, depth, intrinsic, pose,
                    copy.deepcopy(observation), row['rgb'], timestamp_s=row['camera_timestamp_s']))
                collection = mapper.last_observations if condition == 'single_frame' else mapper.objects
                objects = predict(collection, classes, pose=pose, intrinsic=intrinsic)
                if condition == 'single_frame':
                    for obj in objects:
                        obj['node_id'] = f"frame{row['frame_idx']}:{obj['node_id']}"
                for obj in objects:
                    registry.setdefault(obj['node_id'], len(registry)+1)
                nodes = {registry[o['node_id']]: RelationNode(registry[o['node_id']], o['center'],
                    dynamic=o['motion_state'] == 'dynamic', observation_count=o['num_detections'],
                    last_seen_frame=o['last_seen_frame'], metadata={'timestamp_s': o.get('timestamp_s')}) for o in objects}
                edges = [] if condition == 'without_relations' else graph.update_snapshot(nodes,
                    frame_index=row['frame_idx'], ego_heading=pose[:2, 2]).edges
                elapsed = time.monotonic()-tick
                representation_tick = time.monotonic()
                extra = []
                if condition == 'full':
                    # 独立表示基线使用同一当前前端，稀疏网格仅累积当前及过去帧。
                    current = observation_objects(depth, intrinsic, pose, observation)
                    grid.update(current, row['camera_timestamp_s'])
                    cells = grid.visible(pose, intrinsic, row['width'], row['height'])
                    for q in qmap[scene, row['frame_idx']]:
                        vector = vectors[english(q['query_text'])]
                        for method, candidates in [('clip_current_crop', current), ('controlled_sparse_clip_grid', cells)]:
                            chosen = max(candidates, key=lambda x: float(x['feature'] @ vector)) if candidates else None
                            extra.append(dict(query_id=q['query_id'], method=method,
                                prediction=None if chosen is None else dict(category=chosen['category'], center=chosen['center'].tolist()),
                                gt_used=False))
                times.append(elapsed)
                record = dict(scene=scene, frame_idx=row['frame_idx'], condition=condition,
                    camera_timestamp_s=row['camera_timestamp_s'], camera_pose=row['camera_pose'],
                    intrinsic=row['intrinsic'], rgb=row['rgb'], depth=row['depth'], detections=front['detections'],
                    objects=objects, semantic_ids={o['node_id']: registry[o['node_id']] for o in objects},
                    edges=[e.to_dict() for e in edges], representation_baselines=extra,
                    elapsed_s=elapsed, representation_baseline_elapsed_s=time.monotonic()-representation_tick,
                    gt_in_inference=False, actual_arrival_credit=False)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                count += 1
        assert count == selected['frames']
        report = dict(scene=scene, condition=condition, frames=count, total_mapper_graph_time_s=sum(times),
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            frames_sha256=sha(folder / 'frames.jsonl.gz'), gt_in_inference=False)
        dump(folder / 'report.json', report)
        completed.append(report)
        print(json.dumps(dict(condition=condition, scenes=len(completed), total=30, frames=count)), flush=True)
    dump(dest / 'report.json', dict(condition=condition, scenes=30, frames=sum(x['frames'] for x in completed),
        wall_s=time.monotonic()-begun, reports=completed, gt_in_inference=False,
        new_Qwen_calls=0, actual_arrival_credit=False, goal_complete=False))


def requests():
    release = frozen()
    assert read(DEST / 'mapping/full/report.json')['frames'] == 1196
    begun = time.monotonic()
    dest = DEST / 'requests'
    dest.mkdir(exist_ok=False)
    qmap = defaultdict(list)
    for q in lines(DEST / 'query_inputs.jsonl'):
        qmap[q['scene'], q['frame_idx']].append(q)
    count = 0
    with gzip.open(dest / 'inputs.jsonl.gz', 'xt') as output:
        for selected in release['scenes']:
            assert sha(DEST / 'mapping/full' / selected['scene'] / 'frames.jsonl.gz') == read(
                DEST / 'mapping/full' / selected['scene'] / 'report.json')['frames_sha256']
            for frame in lines(DEST / 'mapping/full' / selected['scene'] / 'frames.jsonl.gz'):
                guard(begun, release['budgets']['requests_s'])
                pose = np.asarray(frame['camera_pose'])
                forward = pose[:2, 2].copy()
                forward /= np.linalg.norm(forward)
                left = np.array([-forward[1], forward[0]])
                nodes = []
                for obj in frame['objects']:
                    d = np.asarray(obj['center'][:2])-pose[:2, 3]
                    e = (obj.get('depth_evidence') or [{}])[-1]
                    nodes.append(dict(node_id=frame['semantic_ids'][obj['node_id']], category=obj['category'],
                        center_xy=[round(float(d@forward), 3), round(float(d@left), 3)],
                        distance_m=round(float(np.linalg.norm(d)), 3), motion_state=obj['motion_state'],
                        version=frame['frame_idx']+1, now_s=frame['camera_timestamp_s'],
                        last_seen_s=obj.get('timestamp_s') if obj.get('timestamp_s') is not None else -1,
                        evidence=dict(source=e.get('source', 'unknown'), version=frame['frame_idx']+1,
                            observations=obj['num_detections'], lidar_pixels=int(e.get('retained_pixels', 0)),
                            support_fraction=float(e.get('support_fraction', 0)), conflict=not bool(e.get('sufficient_support', False)))))
                for q in qmap[frame['scene'], frame['frame_idx']]:
                    parsed = parse(q['query_text'])
                    requested = set(parsed.targets+parsed.anchors)
                    pool = sorted(nodes, key=lambda n: (n['category'] not in requested, n['distance_m'], n['node_id']))[:8]
                    pool.sort(key=lambda n: hashlib.sha256(f"{q['query_id']}/{n['node_id']}".encode()).hexdigest())
                    ids = {n['node_id'] for n in pool}
                    edges = sorted((e for e in frame['edges'] if e['source'] in ids and e['target'] in ids),
                                   key=lambda e: (e['source'], e['target'], e['relation']))[:16]
                    row = dict(task='candidate_rerank', sample_id=q['query_id'], scene_id=q['scene'],
                        frame_idx=q['frame_idx'], query_text=q['query_text'], candidate_nodes=pool,
                        candidate_ids=[n['node_id'] for n in pool],
                        relation_metadata=[{k: e[k] for k in ['source', 'target', 'relation']} for e in edges])
                    prompt = build_completion_prompt('candidate_rerank', row)
                    row['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    count += 1
    assert count == 878
    dump(dest / 'report.json', dict(requests=count, input_sha256=sha(dest / 'inputs.jsonl.gz'),
        official_truth_read=False, new_model_calls=0, goal_complete=False))


def gpu(variant):
    release = frozen()
    assert variant in VARIANTS
    assert sha(DEST / 'requests/inputs.jsonl.gz') == read(DEST / 'requests/report.json')['input_sha256']
    import torch
    from transformers import AutoProcessor
    from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
    from train_real_graph_adapter_004 import inputs_for
    dest = DEST / 'models' / variant
    dest.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    adapter = release['variants'][variant]
    model = _load_model(MODEL, Path(adapter) if adapter else None, trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    load_s = time.monotonic()-begun
    count = actual = 0
    with gzip.open(dest / 'outputs.jsonl.gz', 'xt') as output:
        for row in lines(DEST / 'requests/inputs.jsonl.gz'):
            if count % 10 == 0:
                guard(begun, release['budgets']['GPU_each_model_s'])
            assert hashlib.sha256(build_completion_prompt('candidate_rerank', row).encode()).hexdigest() == row['prompt_sha256']
            x = inputs_for(processor, row).to(model.device)
            tokens = int(x['input_ids'].shape[-1])
            raw, latency, valid = None, None, None
            if tokens <= 2048:
                torch.cuda.synchronize()
                tick = time.monotonic()
                with torch.inference_mode():
                    y = model.generate(**x, max_new_tokens=256, do_sample=False, temperature=None,
                        top_p=None, top_k=None, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
                torch.cuda.synchronize()
                latency = time.monotonic()-tick
                raw = processor.tokenizer.decode(y[0, tokens:], skip_special_tokens=True).strip()
                valid = validate('candidate_rerank', raw, allowed_candidate_ids=row['candidate_ids'])
                actual += 1
            record = dict(sample_id=row['sample_id'], variant=variant, input_tokens=tokens,
                prompt_sha256=row['prompt_sha256'], raw_output=raw, GPU_latency_s=latency,
                schema_valid=bool(valid and valid.valid), normalized_output=valid.normalized_output if valid and valid.valid else None,
                errors=valid.validation_errors if valid else ['TOKEN_LIMIT_EXCEEDED'],
                actual_GPU_call=tokens <= 2048, inference_received_official_truth=False)
            output.write(json.dumps(record, ensure_ascii=False)+'\n')
            output.flush()
            count += 1
            if count % 10 == 0:
                print(json.dumps(dict(variant=variant, completed=count, total=878, actual_GPU=actual)), flush=True)
    assert count == 878
    dump(dest / 'report.json', dict(variant=variant, requests=count, actual_GPU_calls=actual,
        wall_s=time.monotonic()-begun, load_s=load_s,
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
        output_sha256=sha(dest / 'outputs.jsonl.gz'), gt_received=False, goal_complete=False))


def evaluate():
    """全部模型结束后统一隔离评价，不把评价配对回灌到推理输入。"""
    release = frozen()
    from conceptgraph.completion.quality import object_frame_metrics, identity_metrics, relations_from_centers
    from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
    from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes
    from conceptgraph.scenegraph.relations import inverse_relation
    from conceptgraph.completion.retrieval_baselines import associated_identity
    begun = time.monotonic()
    for c in CONDITIONS:
        assert read(DEST / 'mapping' / c / 'report.json')['frames'] == 1196
    for v in VARIANTS:
        assert read(DEST / 'models' / v / 'report.json')['requests'] == 878
        assert sha(DEST / 'models' / v / 'outputs.jsonl.gz') == read(
            DEST / 'models' / v / 'report.json')['output_sha256']
    dest = DEST / 'evaluation'
    dest.mkdir(exist_ok=False)
    truths = {(f['scene'], f['frame_idx']): f for f in lines(LABEL / 'evaluation_only_frames.jsonl')}
    qtruth = {q['query_id']: q for name in ['main_queries.jsonl', 'challenge_queries.jsonl']
              for q in lines(LABEL / name)}
    mapping_reports = []
    full_frames = {}
    with gzip.open(dest / 'mapping_metrics.jsonl.gz', 'xt') as output:
        for condition in CONDITIONS:
            for scene in release['scenes']:
                guard(begun, release['budgets']['evaluate_s'])
                assert sha(DEST / 'mapping' / condition / scene['scene'] / 'frames.jsonl.gz') == read(
                    DEST / 'mapping' / condition / scene['scene'] / 'report.json')['frames_sha256']
                sequence, identities = [], []
                totals = Counter()
                for f in lines(DEST / 'mapping' / condition / scene['scene'] / 'frames.jsonl.gz'):
                    key = (f['scene'], f['frame_idx'])
                    truth = truths[key]['objects']
                    predictions = f['objects']
                    metric = object_frame_metrics(predictions, truth)
                    by_id = {f['semantic_ids'][predictions[m['prediction_index']]['node_id']]:
                             truth[m['truth_index']]['instance_token'] for m in metric['matches']}
                    predicted_edges = set()
                    unresolved = 0
                    for e in f['edges']:
                        if e['source'] in by_id and e['target'] in by_id:
                            a, b = by_id[e['source']], by_id[e['target']]
                            predicted_edges.add((a, e['relation'], b))
                            predicted_edges.add((b, inverse_relation(e['relation']), a))
                        else:
                            unresolved += 2
                    true_edges = relations_from_centers([dict(id=o['instance_token'], center=o['center']) for o in truth],
                        forward_xy=np.asarray(f['camera_pose'])[:2, 2], max_distance_m=12., near_m=6.)
                    for k in ['predictions', 'truth', 'true_positive', 'unmatched_predictions', 'false_negative', 'duplicate_nodes']:
                        totals[k] += metric[k]
                    totals['center_error_sum_m'] += sum(x['distance_m'] for x in metric['matches'])
                    totals['relation_true_positive'] += len(predicted_edges & true_edges)
                    totals['relation_predicted_resolved'] += len(predicted_edges)
                    totals['relation_predicted_unresolved'] += unresolved
                    totals['relation_truth'] += len(true_edges)
                    totals['mapper_graph_time_s'] += f['elapsed_s']
                    pairs = [(predictions[m['prediction_index']]['node_id'], truth[m['truth_index']]['instance_token'])
                             for m in metric['matches']]
                    identities.append((f['frame_idx'], pairs))
                    sequence.append(dict(frame_idx=f['frame_idx'], camera_timestamp_s=f['camera_timestamp_s'],
                        predictions={condition: predictions}, evaluation_only_official=truth, metrics={condition: metric}))
                    output.write(json.dumps(dict(scene=f['scene'], frame_idx=f['frame_idx'], condition=condition,
                        object_metrics=metric, relation_true_positive=len(predicted_edges & true_edges),
                        relation_truth=len(true_edges), relation_predicted_resolved=len(predicted_edges),
                        relation_predicted_unresolved=unresolved), ensure_ascii=False)+'\n')
                    if condition == 'full':
                        full_frames[key] = f
                identity = identity_counts(prepare_sequence(sequence, condition), 'all_gated')['counts']
                mapping_reports.append(dict(scene=scene['scene'], log=scene['log_token'], condition=condition,
                    frames=len(sequence), totals=dict(totals), identity=identity, switches=identity_metrics(identities)))
    requests_by_id = {r['sample_id']: r for r in lines(DEST / 'requests/inputs.jsonl.gz')}
    predictions = {v: {r['sample_id']: r for r in lines(DEST / 'models' / v / 'outputs.jsonl.gz')} for v in VARIANTS}
    results = []
    for qid, query in qtruth.items():
        row = requests_by_id[qid]
        key = (query['scene'], query['frame_idx'])
        frame = full_frames[key]
        truth = truths[key]['objects']
        objects = [dict(o, node_id=frame['semantic_ids'][o['node_id']]) for o in frame['objects']]
        association = associate_nodes(objects, truth, maximum_distance_m=3., ambiguity_margin_m=.5)
        binding = {m['node_id']: m['instance_token'] for m in association['matches']}
        accepted = set(query['accepted_instance_tokens'])
        nodes = row['candidate_nodes']
        covered = any(binding.get(n['node_id']) in accepted for n in nodes)
        parsed = parse(row['query_text'])
        def record(method, selected, *, selected_instance=None, identity_status=None, valid=True, **extra):
            instance = selected_instance if selected_instance is not None else binding.get(selected)
            no_answer = selected is None and selected_instance is None
            correct = bool(valid and ((instance in accepted) if accepted else no_answer))
            results.append(dict(query_id=qid, scene=query['scene'], log=truths[key]['log_token'],
                cohort=query['cohort'], method=method, positive=bool(accepted), selected_node_id=selected,
                selected_instance=instance, schema_valid=valid, correct=correct,
                no_match_false_selection=bool(not accepted and not no_answer),
                candidate_target_coverage=covered, identity_status=identity_status or (
                    'no_selection' if no_answer else 'associated' if instance else 'unassociated_or_ambiguous'),
                actual_lock_credit=False, actual_arrival_credit=False, **extra))
        same = [n for n in nodes if n['category'] in parsed.targets]
        nearest = min(same, key=lambda n: (n['distance_m'], n['node_id']))['node_id'] if same else None
        record('category_nearest', nearest)
        rule = strong_rule(row['query_text'], nodes, relation_radius_m=15)
        record('complete_geometry_rule', rule['proposed_node_id'], valid=rule['valid'])
        for base in frame['representation_baselines']:
            if base['query_id'] != qid:
                continue
            candidate = base['prediction']
            if candidate is not None:
                candidate = dict(candidate, center=np.asarray(candidate['center']))
            instance, status = associated_identity(candidate, truth)
            # 仅用于标记“确有选择”，不能把一个无身份关联的选择记成拒答。
            record(base['method'], -1 if candidate is not None else None,
                   selected_instance=instance, identity_status=status)
        for variant in VARIANTS:
            prediction = predictions[variant][qid]
            answer = prediction['normalized_output']['selected_node_id'] if prediction['schema_valid'] else None
            record(variant, answer, valid=prediction['schema_valid'], GPU_latency_s=prediction['GPU_latency_s'],
                   actual_GPU_call=prediction['actual_GPU_call'])
        prediction = predictions['real_graph_seed17'][qid]
        for condition in ['full', 'without_rule_query_resolution', 'without_attribute_check',
                          'without_candidate_reranking', 'without_evidence_check']:
            answer = choose(row['query_text'], nodes, prediction, condition)
            selected = answer['authorized_node_id'] if answer['explicit_authorization_query'] else answer['proposed_node_id']
            record('cooperation_seed17' if condition == 'full' else condition, selected,
                   valid=answer['valid'], backend_output_used=answer['backend_output_used'],
                   authorized_node_id=answer['authorized_node_id'])
    assert len(results) == 878*14
    with gzip.open(dest / 'query_metrics.jsonl.gz', 'xt') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    # 所有比例都按log聚类，主集与挑战集分列，不把同场景多指令当独立场景。
    table = []
    logs = sorted({r['log'] for r in results})
    draws = np.random.default_rng(17).integers(0, len(logs), (4000, len(logs)))
    for cohort in ['distribution_preserving', 'challenge']:
        for method in release['retrieval_methods']:
            rr = [x for x in results if x['method'] == method and x['cohort'] == cohort]
            for name, subset, field in [
                ('task_correct', rr, 'correct'),
                ('positive_target_correct', [x for x in rr if x['positive']], 'correct'),
                ('no_match_false_selection', [x for x in rr if not x['positive']], 'no_match_false_selection')]:
                a = np.array([[sum(bool(x[field]) for x in subset if x['log'] == log),
                               sum(x['log'] == log for x in subset)] for log in logs])
                boot = a[draws].sum(1)
                values = np.divide(boot[:, 0], boot[:, 1], out=np.full(4000, np.nan), where=boot[:, 1] > 0)
                low, high = np.nanpercentile(values, [2.5, 97.5]) if subset else (None, None)
                table.append(dict(cohort=cohort, method=method, metric=name,
                    numerator=int(a[:, 0].sum()), denominator=len(subset),
                    value=float(a[:, 0].sum()/len(subset)) if subset else None,
                    ci95_low=float(low) if low is not None else None, ci95_high=float(high) if high is not None else None))
    dump(dest / 'mapping_summary.json', mapping_reports)
    dump(dest / 'retrieval_summary.json', table)
    dump(dest / 'report.json', dict(scenes=30, logs=14, frames_each=1196, conditions=len(CONDITIONS),
        queries_each=878, retrieval_methods=14, actual_query_method_results=len(results),
        actual_GPU_calls=sum(read(DEST / 'models' / v / 'report.json')['actual_GPU_calls'] for v in VARIANTS),
        inference_GT_isolation_preserved=True, actual_lock_credit=False, actual_arrival_credit=False,
        methods_frozen_before_test=True, new_test_opened=True,
        files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        limitations=['模板主集不代表人工自然指令分布', '仅离线对象/关系建图和目标提议，不是完整实时导航',
            '相机与官方sample标注存在约33—40ms时间差，未插值',
            '语义只比较full共享图，十条件建图消融不冒充十条件语义GPU消融',
            '所有无候选正目标保留失败；unknown身份未补成正确',
            'crop/稀疏格是不同表示基线；网格为受控简化，不是完整VLMaps复现',
            '总任务准确率含无匹配，必须同时查看正目标和无匹配分层'], goal_complete=False))
    print('sealed evaluation complete; whole goal remains incomplete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'frontend', 'map', 'requests', 'gpu', 'evaluate'])
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    try:
        if args.action == 'map':
            map_condition(args.condition)
        elif args.action == 'gpu':
            gpu(args.variant)
        else:
            globals()[args.action]()
    except BaseException as exc:
        if DEST.exists():
            import traceback
            dump(DEST / f"failure_{args.action}_{time.time_ns()}.json", dict(error=repr(exc),
                traceback=traceback.format_exc(), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
