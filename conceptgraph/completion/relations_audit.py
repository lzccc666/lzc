"""用官方三维位置独立导出的关系，评价直接几何、结构化和增量关系。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
import importlib.util
from pathlib import Path

import numpy as np

from conceptgraph.completion.quality import relations_from_centers, relation_metrics
from conceptgraph.completion.stage9 import OUTPUT, dump, digest
from conceptgraph.scenegraph.relations import RelationNode, build_relations, inverse_relation
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph


def expand(edges):
    result=set()
    for edge in edges:
        result.add((edge.source,edge.relation,edge.target))
        result.add((edge.target,inverse_relation(edge.relation),edge.source))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mapping-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--legacy-source')
    parser.add_argument('--coordinate-frame',choices=['ego_heading','world'],default='ego_heading')
    parser.add_argument('--position-threshold',type=float,default=0.1)
    parser.add_argument('--method',default='one_to_one_timestamp')
    parser.add_argument('--directional-mode',choices=['primary_axis','all_supported_axes'],default='primary_axis')
    args=parser.parse_args()
    dest=OUTPUT/'mapping_evaluation'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    config=dict(coordinate_frame=args.coordinate_frame,coordinate_frame_fallback=None,
        max_distance_m=12.0,near_distance_m=6.0,position_update_threshold_m=args.position_threshold,
        directional_mode=args.directional_mode,minimum_direction_component_m=1.0)
    graph_class=IncrementalSceneGraph
    implementation=Path(__file__).parents[1]/'scenegraph/incremental_graph.py'
    if args.legacy_source:
        implementation=Path(args.legacy_source).resolve()
        spec=importlib.util.spec_from_file_location('legacy_incremental',implementation)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph_class=module.IncrementalSceneGraph
    dump(dest/'release.json',dict(config=config,implementation=str(implementation),
        implementation_sha256=digest(implementation),runner_sha256=digest(Path(__file__)),
        mapping_run=args.mapping_run,method=args.method,ground_truth_in_inference=False,
        relations_source_sha256=digest(Path(__file__).parents[1]/'scenegraph/relations.py')))
    records=[]
    summaries=[]
    for path in sorted((OUTPUT/'mapping_evaluation'/args.mapping_run).glob('scene-*/frames.jsonl')):
        incremental=graph_class(config)
        registry={}
        grouped=defaultdict(lambda:dict(true_positive=0,predicted=0,truth=0,unresolved_edges=0))
        for frame in map(json.loads,path.open()):
            objects=frame['predictions'][args.method]
            for obj in objects:
                registry.setdefault(obj['node_id'],len(registry))
            forward=np.asarray(frame['camera_pose'])[:2,2]
            if args.coordinate_frame=='world':
                forward=np.array([0.0,1.0])
            gt_objects=frame['evaluation_only_official']
            gt_rel=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in gt_objects],
                forward_xy=forward,max_distance_m=12,near_m=6)
            pred_direct=relations_from_centers([dict(id=registry[o['node_id']],center=o['center']) for o in objects],
                forward_xy=forward,max_distance_m=12,near_m=6)
            nodes={registry[o['node_id']]:RelationNode(registry[o['node_id']],o['center'],
                dynamic=o['motion_state']=='dynamic',observation_count=o['num_detections'],
                last_seen_frame=o['last_seen_frame'],metadata={'timestamp_s':o.get('timestamp_s')}) for o in objects}
            rebuilt=build_relations(nodes,config,frame_index=frame['frame_idx'],ego_heading=forward)
            inc=incremental.update_snapshot(nodes,frame_index=frame['frame_idx'],ego_heading=forward)
            mapping={registry[objects[m['prediction_index']]['node_id']]:gt_objects[m['truth_index']]['instance_token']
                for m in frame['metrics'][args.method]['matches']}
            scores={}
            for name,edges in [('direct_geometry',pred_direct),('structured',expand(rebuilt.edges)),
                               ('incremental',expand(inc.edges))]:
                resolved={(mapping[s],r,mapping[t]) for s,r,t in edges if s in mapping and t in mapping}
                unresolved=sum(s not in mapping or t not in mapping for s,r,t in edges)
                metric=relation_metrics(resolved,gt_rel)
                metric['unresolved_edges']=unresolved
                metric['precision_lower_bound_including_unresolved']=metric['true_positive']/(len(resolved)+unresolved) if resolved or unresolved else None
                scores[name]=metric
                for key in grouped[name]:
                    grouped[name][key]+=metric[key]
            scores['incremental_full_content_consistency']=expand(inc.edges)==expand(rebuilt.edges)
            records.append(dict(scene=frame['scene'],frame_idx=frame['frame_idx'],scores=scores))
        summaries.append(dict(scene=path.parent.name,methods=dict(grouped),source=str(path),sha256=digest(path)))
    dump(dest/'summary.json',dict(scenes=summaries,frame_count=len(records),
        truth_source='官方实例位置、可见范围与预定几何关系定义；独立于预测图，非人工关系标签',
        coordinate_frame='相机+Z投影的当前驾驶员前向，左为其逆时针垂向；世界坐标另评',
        relation_truth_definition=dict(max_distance_m=12,near_m=6,component_min_m=1),
        unresolved_policy='关联未知单列；同时给出把未知全计错误的精确率下界',
        incremental_consistent_frames=sum(r['scores']['incremental_full_content_consistency'] for r in records),
        stage9_complete=False))
    with (dest/'frames.jsonl').open('x') as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
