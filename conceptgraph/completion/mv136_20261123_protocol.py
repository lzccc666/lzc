"""固定1/3/6视角公平时钟、统一全分母评价。GT仅供运行后评测。"""
from collections import Counter, defaultdict
import gzip
import json
import numpy as np

from conceptgraph.completion.quality import object_frame_metrics, relations_from_centers
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.scenegraph.paper_graph_20261123_metrics import edge_counts, scores
from conceptgraph.slam.multiview_20261123_fusion import in_view
from scripts.multiview_20261123_run_graph import node_scores, write_csv

FRONT3 = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
SIX = [*FRONT3, 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
SUBSETS = {'A_front_context':['CAM_FRONT'], 'C_3':FRONT3, 'C_6':SIX,
    'T3_1':['CAM_FRONT'], 'T3_3':FRONT3, 'T3_6':SIX}


def clock_evidence(group):
    if set(group['views']) != set(SIX):
        raise ValueError('每组必须六路完整且没有额外相机')
    times = {c:float(group['views'][c]['camera_timestamp_s']) for c in SIX}
    reference = float(group['reference_timestamp_s'])
    if not np.isfinite([reference,*times.values()]).all() or reference != max(times.values()):
        raise ValueError('共同地图时钟必须等于六路最新实际曝光')
    for camera in SIX:
        lidar = group['views'][camera].get('lidar_timestamp_s')
        if lidar is not None and not float(lidar) < times[camera]:
            raise ValueError('LiDAR必须严格过去于自身相机，不可相对组时钟选未来点')
    ready = {m:max(times[c] for c in cams) for m,cams in SUBSETS.items()}
    lidar_by_subset = {}
    for method,cameras in SUBSETS.items():
        tokens = {group['views'][c].get('lidar_token') for c in cameras}
        tokens.discard(None)
        timestamps = [group['views'][c]['lidar_timestamp_s'] for c in cameras
            if group['views'][c].get('lidar_timestamp_s') is not None]
        lidar_by_subset[method] = dict(unique_scan_count=len(tokens),scan_tokens=sorted(tokens),
            missing_views=sum(group['views'][c].get('lidar_token') is None for c in cameras),
            earliest_lidar_timestamp_s=min(timestamps) if timestamps else None,
            latest_lidar_timestamp_s=max(timestamps) if timestamps else None)
    return dict(common_reference_timestamp_s=reference,camera_capture_times_s=times,
        subset_ready_timestamp_s=ready,
        lidar_by_subset=lidar_by_subset,
        subset_ready_delay_from_front_ms={m:(t-times['CAM_FRONT'])*1000 for m,t in ready.items()},
        common_evaluation_wait_after_ready_ms={m:(reference-t)*1000 for m,t in ready.items()},
        official_sample_minus_reference_ms=(group['official_sample_timestamp_s']-reference)*1000)


def evaluate(run, manifest, ground_truth, dest, prefix):
    all_gt = list(map(json.loads, ground_truth.open()))
    ground = {(r['scene'],r['frame_idx']):r for r in all_gt}
    expected = {(s['scene'],g['frame_idx']) for s in manifest['scenes'] for g in s['groups']}
    assert len(ground) == len(all_gt) and ground.keys() == expected
    totals, relations = defaultdict(Counter), defaultdict(Counter)
    scene_rows, frame_rows, coverage_rows = [], [], []
    for scene in manifest['scenes']:
        name = scene['scene']
        sequence = list(map(json.loads,gzip.open(run/name/(prefix+'predictions.jsonl.gz'),'rt')))
        assert len(sequence) == len(scene['groups']) == 20
        local, prepared = defaultdict(Counter), defaultdict(list)
        with gzip.open(run/name/(prefix+'evaluation.jsonl.gz'),'wt') as stream:
            for row,group in zip(sequence,scene['groups']):
                assert (row['frame_idx'],row['sample_token']) == (group['frame_idx'],group['sample_token'])
                official = ground[name,row['frame_idx']]
                assert official['sample_token'] == group['sample_token']
                union = official['objects']
                assert len(union) == len({o['instance_token'] for o in union})
                truth = dict(union6=union,
                    front3=[o for o in union if set(o['visible_cameras']) & set(FRONT3)],
                    front=[o for o in union if 'CAM_FRONT' in o['visible_cameras']])
                cameras_by_scope = dict(union6=SIX,front3=FRONT3,front=['CAM_FRONT'])
                forward = np.asarray(group['views']['CAM_FRONT']['camera_pose'])[:2,2]
                result = {}
                for method, raw in row['predictions'].items():
                    result[method] = {}
                    for scope,cameras in cameras_by_scope.items():
                        pred = [p for p in raw if any(in_view(p['center'],group['views'][c]) for c in cameras)]
                        metric = object_frame_metrics(pred,truth[scope],threshold_m=3.)
                        c = local[method,scope]
                        for k in ['true_positive','truth','predictions','unmatched_predictions','false_negative','duplicate_nodes']:
                            c[k] += metric[k]
                        c['error_sum_m'] += sum(m['distance_m'] for m in metric['matches'])
                        prepared[method,scope].append(dict(frame_idx=row['frame_idx'],camera_timestamp_s=row['reference_timestamp_s'],
                            predictions={method:pred},evaluation_only_official=truth[scope],metrics={method:metric}))
                        pe = relations_from_centers([dict(id=i,center=p['center']) for i,p in enumerate(pred)],forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                        ge = relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in truth[scope]],forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                        mapping = {m['prediction_index']:truth[scope][m['truth_index']]['instance_token'] for m in metric['matches']}
                        edges = edge_counts(pe,ge,mapping)
                        relations[method,scope].update(edges)
                        result[method][scope] = dict(predictions=pred,metrics=metric,relation_counts=dict(edges),
                            predicted_relations=sorted(pe),gt_relations=sorted(ge),node_mapping=mapping)
                        frame_rows.append(dict(scene=name,log_token=scene['log_token'],frame_idx=row['frame_idx'],method=method,scope=scope,
                            raw_output_predictions=len(raw),**{k:metric[k] for k in ['truth','predictions','true_positive','unmatched_predictions','false_negative','duplicate_nodes']},
                            center_error_m=metric['center_error_m_mean'],**scores(edges)))
                        if scope == 'union6':
                            matched = {m['truth_index'] for m in metric['matches']}
                            front = {i for i,o in enumerate(union) if 'CAM_FRONT' in o['visible_cameras']}
                            front3 = {i for i,o in enumerate(union) if set(o['visible_cameras']) & set(FRONT3)}
                            rear = set(range(len(union)))-front3
                            coverage_rows.append(dict(scene=name,log_token=scene['log_token'],frame_idx=row['frame_idx'],method=method,
                                union6_GT=len(union),union6_TP=len(matched),common_front_GT=len(front),common_front_TP=len(front & matched),
                                front3_GT=len(front3),front3_TP=len(front3 & matched),rear_only_GT=len(rear),rear_only_TP=len(rear & matched)))
                stream.write(json.dumps(dict(scene=name,frame_idx=row['frame_idx'],evaluation_only_official=truth,result=result),allow_nan=False)+'\n')
        for (method,scope),seq in prepared.items():
            identity = identity_counts(prepare_sequence(seq,method),'all_gated')['counts']
            local[method,scope].update(identity)
            totals[method,scope].update(local[method,scope])
            scene_rows.append(dict(scene=name,log_token=scene['log_token'],method=method,scope=scope,
                **local[method,scope],**node_scores(local[method,scope])))
    node_main = [dict(method=m,scope=s,**c,**node_scores(c),duplicate_per_prediction=c['duplicate_nodes']/max(1,c['predictions']),
        duplicate_per_GT=c['duplicate_nodes']/max(1,c['truth'])) for (m,s),c in totals.items()]
    relation_main = [dict(method=m,scope=s,**c,**scores(c)) for (m,s),c in relations.items()]
    coverage = defaultdict(Counter)
    for row in coverage_rows:
        coverage[row['method']].update({k:v for k,v in row.items() if k.endswith('_GT') or k.endswith('_TP')})
    coverage_main = [dict(method=m,**c) for m,c in coverage.items()]
    for name,records in dict(node_main=node_main,relation_main=relation_main,coverage_main=coverage_main,
            scene_metrics=scene_rows,frame_metrics=frame_rows,coverage_frame=coverage_rows).items():
        write_csv(dest/(prefix+name+'.csv'),records)
    return dict(node_main=node_main,relation_main=relation_main,coverage_main=coverage_main)
