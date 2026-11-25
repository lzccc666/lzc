"""只在已访问的开发帧实际运行十种建图配置，封存方法运行前检查兼容性。"""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import sealed_public_benchmark_058 as runner
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.relations import RelationNode
from conceptgraph.completion.retrieval_baselines import observation_objects, SparseFeatureGrid
import conceptgraph.slam.streamlined_mapping as sm


def main():
    dest = runner.ROOT / 'method_comparison/sealed_runner_preflight_058'
    dest.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('preflight_isolated', runner.ROOT / 'source/mapping_assignment_isolated_001.py')
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    sm.merge_detections_to_objects = isolated.merge_detections_to_objects
    sources = [runner.OLD / 'mapping_evaluation/causal_mapping_full_001' / s / 'frames.jsonl'
               for s in ['scene-0012', 'scene-0014']]
    graph_cfg = runner.read(runner.ROOT / 'nuscenes_ablation/graph_near6_correction_001/release.json')['new_actual_graph_config']
    source_files = [Path(runner.__file__), Path(__file__), runner.ROOT / 'source/run_sealed_public_series_059.py']
    before = {str(p): runner.sha(p) for p in sources+source_files}
    runner.dump(dest / 'release.json', dict(input_and_source_sha256=before,
        input_selection='两个历史开发场景各取真实时间顺序前3帧，不按前端或方法成败挑帧',
        all_conditions=runner.CONDITIONS, new_test_opened=False, new_GPU_calls=0, goal_complete=False))
    started = time.monotonic()
    records = []
    for source in sources:
        frames = list(runner.lines(source))[:3]
        with gzip.open(frames[0]['detections'], 'rb') as f:
            classes = pickle.load(f)['classes']
        for condition in runner.CONDITIONS:
            cfg = runner.config(condition)
            # 解析插值后再重建，与将来release内冻结配置走同一条入口。
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            mapper = sm.StreamlinedMapper(cfg, classes)
            graph = IncrementalSceneGraph(graph_cfg)
            registry = {}
            grid = SparseFeatureGrid()
            for frame in frames:
                pose, intrinsic = np.asarray(frame['camera_pose']), np.asarray(frame['intrinsic'])
                image = np.asarray(Image.open(frame['rgb']).convert('RGB'))
                depth = np.asarray(Image.open(frame['depth']), float)/1000
                with gzip.open(frame['detections'], 'rb') as f:
                    observation = pickle.load(f)
                np.random.seed(17+frame['frame_idx'])
                mapper.update(sm.MappingFrame(frame['frame_idx'], image, depth, intrinsic, pose,
                    copy.deepcopy(observation), frame['rgb'], timestamp_s=frame['camera_timestamp_s']))
                collection = mapper.last_observations if condition == 'single_frame' else mapper.objects
                objects = runner.predict(collection, classes, pose=pose, intrinsic=intrinsic)
                for o in objects:
                    if condition == 'single_frame':
                        o['node_id'] = f"frame{frame['frame_idx']}:{o['node_id']}"
                    registry.setdefault(o['node_id'], len(registry)+1)
                assert len({o['node_id'] for o in objects}) == len(objects)
                nodes = {registry[o['node_id']]: RelationNode(registry[o['node_id']], o['center'],
                    dynamic=o['motion_state'] == 'dynamic', observation_count=o['num_detections'],
                    last_seen_frame=o['last_seen_frame'], metadata={'timestamp_s': o.get('timestamp_s')}) for o in objects}
                edges = [] if condition == 'without_relations' else graph.update_snapshot(nodes,
                    frame_index=frame['frame_idx'], ego_heading=pose[:2, 2]).edges
                grid_cells = None
                if condition == 'full':
                    detections = observation_objects(depth, intrinsic, pose, observation)
                    grid.update(detections, frame['camera_timestamp_s'])
                    grid_cells = len(grid.visible(pose, intrinsic, image.shape[1], image.shape[0]))
                records.append(dict(scene=source.parent.name, frame_idx=frame['frame_idx'], condition=condition,
                    objects=len(objects), edges=len(edges), sparse_grid_cells=grid_cells, gt_in_inference=False))
    assert len(records) == 60
    assert all(runner.sha(Path(p)) == sha for p, sha in before.items())
    with (dest / 'records.jsonl').open('x') as f:
        for row in records:
            f.write(json.dumps(row)+'\n')
    runner.dump(dest / 'report.json', dict(passed=True, actual_CPU_mapper_calls=60,
        history_scenes=2, unique_history_frames=6, conditions=10,
        actual_grid_update_calls=6, input_source_hashes_unchanged=True,
        elapsed_s=time.monotonic()-started, new_GPU_calls=0, new_CARLA_calls=0,
        new_test_opened=False, source_sha256=runner.sha(Path(runner.__file__)), goal_complete=False))
    print('preflight passed60 mapper updates / 6 grid updates', flush=True)


if __name__ == '__main__':
    main()
