"""P0：冻结R1的完整身份、关系双分母与增量一致性，不调用任何感知模型。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.quality import object_frame_metrics, relations_from_centers
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.completion.temporal_identity_evaluation import (
    identity_counts, metric_values, prepare_sequence, return_diagnostics)
from conceptgraph.project_paths import experiment_directory
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.paper_graph_20261123_metrics import (
    RELATION_CONFIG, edge_counts, expand, scores, semantic_content, temporal_content)
from conceptgraph.scenegraph.relations import RelationConfig, RelationNode, build_relations


PREFIX = 'paper_graph_20261123_'
R1 = PROJECT / 'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
REPORT_ROOT = PROJECT / 'outputs/reports/paper_campaign_20261123/graph'
METHODS = ('direct_geometry', 'structured_full', 'structured_incremental')
RELATIONS = (None, 'near', 'left_of', 'right_of', 'in_front_of', 'behind')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-budget-s', type=float, default=2100.)
    parser.add_argument('--input-run', default='recall_refinement_R1_full_20261120_v2')
    parser.add_argument('--report-tag', default='P0')
    args = parser.parse_args()
    start = time.monotonic()
    input_root = PROJECT/'outputs/results/nuscenes/validation/iterations'/args.input_run
    release_path = input_root / 'release.json'
    report_path = input_root / 'report.json'
    if not release_path.exists():
        release_path = input_root/(PREFIX+'release.json')
        report_path = input_root/(PREFIX+'report.json')
    release = json.loads(release_path.read_text())
    original = json.loads(report_path.read_text())
    method_key = release['condition']
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    source_paths = [Path(__file__).resolve(), PROJECT/'conceptgraph/scenegraph/paper_graph_20261123_metrics.py',
        PROJECT/'conceptgraph/scenegraph/relations.py', PROJECT/'conceptgraph/scenegraph/incremental_graph.py',
        PROJECT/'conceptgraph/completion/temporal_identity_evaluation.py', PROJECT/'conceptgraph/completion/quality.py',
        PROJECT/'conceptgraph/completion/association.py', PROJECT/'conceptgraph/project_paths.py']
    snapshots = {}
    for source in source_paths:
        target = dest / (PREFIX + 'source_' + source.name)
        shutil.copy2(source, target)
        snapshots[str(source)] = dict(sha256=digest(source), snapshot=str(target))
    input_files = [release_path, report_path]
    if (input_root/'input_fingerprints.json').exists():
        input_files.append(input_root/'input_fingerprints.json')
    inputs = {str(p): digest(p) for p in input_files}
    for scene in release['scenes']:
        frozen = input_root/scene['scene']/'frames.jsonl.gz'
        if not frozen.exists():
            frozen = input_root/scene['scene']/(PREFIX+'frames.jsonl.gz')
        for p in [frozen, OUTPUT/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl']:
            inputs[str(p)] = digest(p)
    effective = RelationConfig.from_mapping(RELATION_CONFIG).to_dict()
    if effective['relation_distance_thresholds_m']['near'] != 6.:
        raise AssertionError('近关系实际阈值错误')
    dump(dest/(PREFIX+'release.json'), dict(mode='frozen_graph_paper_evaluation', input_run=str(input_root),condition=method_key,
        input_sha256=inputs, source_sha256=snapshots, seed=20261123, wall_budget_s=args.wall_budget_s,
        campaign_wall_budget_s=2700, maximum_new_bytes=256*2**20, minimum_free_bytes=20*2**30,
        node_match_threshold_m=3., relation_config_supplied=RELATION_CONFIG, relation_config_effective=effective,
        scope=dict(scenes=38, logs=11, frames=1508, GT=5723),
        predicate_counting='ordered expanded inverse facts; near represented twice consistently in GT/prediction',
        primary_identity_support='all_gated', supplementary_identity_support='mutually_unique_gated',
        conditional_truth='all GT edges induced by matched GT endpoints; all resolved predictions retained',
        full_truth='all original GT edges, missing endpoints remain FN; unresolved predictions count FP',
        temporal_equivalence='source/target observed frame and timestamp; historical first_seen/count intentionally separate',
        unknown_adapter='RelationNode only supports dynamic boolean; unknown encoded dynamic=False plus explicit metadata, never claimed static truth',
        sparse_last_seen='exact source camera timestamp->frame index; missing timestamp is fatal',
        no_new_frontend_calls=True, new_mapping_inference=False, GT_in_inference=False, test_accessed=False,
        historical_protocol_note='legacy near_distance_m key is not consumed by RelationConfig; this new run explicitly sets effective near6, no old scores relabelled'))
    totals, identity_total, node_total, counters = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    relation_scene_rows, identity_scene_rows, node_scene_rows, frame_rows, timing_rows, source_rows = [], [], [], [], [], []
    groups = defaultdict(Counter)
    for scene in release['scenes']:
        name, log = scene['scene'], scene['log_token']
        frozen_path = input_root/name/'frames.jsonl.gz'
        if not frozen_path.exists():
            frozen_path = input_root/name/(PREFIX+'frames.jsonl.gz')
        frames = list(map(json.loads, gzip.open(frozen_path, 'rt')))
        raw_path = OUTPUT/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
        raw = list(map(json.loads, raw_path.open()))
        if len(raw) != len(frames):
            raise AssertionError('帧数变化')
        prepared = prepare_sequence(frames, method_key)
        returns = return_diagnostics(prepared)
        for support in ['all_gated', 'mutually_unique_gated']:
            identity = identity_counts(prepared, support)
            count = Counter(identity['counts'])
            count.update(returns['counts'])
            identity_total[support].update(count)
            identity_scene_rows.append(dict(scene=name, log_token=log, support=support, **count, **metric_values(count)))
        case = dest/name
        case.mkdir()
        with gzip.open(case/(PREFIX+'identity_events.jsonl.gz'), 'wt') as stream:
            for event in returns['events']:
                stream.write(json.dumps(dict(scene=name, **event), allow_nan=False)+'\n')
            for event in returns['censored_tails']:
                stream.write(json.dumps(dict(scene=name, kind='censored_tail', **event), allow_nan=False)+'\n')
        registry, first_output, clock_to_frame = {}, {}, {r['camera_timestamp_s']: r['frame_idx'] for r in frames}
        incremental = IncrementalSceneGraph(RELATION_CONFIG)
        scene_rel, scene_node = defaultdict(Counter), Counter()
        prior_forward = None
        with gzip.open(case/(PREFIX+'frame_relations.jsonl.gz'), 'wt') as out:
            for row, source in zip(frames, raw):
                if time.monotonic()-start > args.wall_budget_s or shutil.disk_usage(dest).free < 20*2**30:
                    raise RuntimeError('达到预算或磁盘保护线，保留真实前缀')
                idx = row['frame_idx']
                if source['frame_idx'] != idx or source['camera_timestamp_s'] != row['camera_timestamp_s'] or source['evaluation_only_official'] != row['evaluation_only_official']:
                    raise AssertionError('源数据join变化')
                predictions, truth = row['predictions'][method_key], row['evaluation_only_official']
                measured = object_frame_metrics(predictions, truth, threshold_m=3.)
                if measured != row['metrics'][method_key]:
                    raise AssertionError('原R1节点评价未精确复现')
                for key in ['true_positive','truth','predictions','false_negative','unmatched_predictions','duplicate_nodes']:
                    scene_node[key] += measured[key]
                scene_node['center_error_sum_m'] += sum(m['distance_m'] for m in measured['matches'])
                forward = np.asarray(source['camera_pose'])[:2,2]
                forward = forward / np.linalg.norm(forward)
                heading_changed = prior_forward is not None and not np.allclose(forward,prior_forward,rtol=0,atol=1e-9)
                prior_forward = forward.copy()
                for obj in predictions:
                    registry.setdefault(obj['node_id'], len(registry))
                    first_output.setdefault(obj['node_id'], idx)
                nodes = {}
                for obj in predictions:
                    sparse = obj.get('shape_available') is False
                    motion = obj.get('motion_state','unknown')
                    last_seen = obj.get('last_seen_frame')
                    if last_seen is None:
                        timestamp = obj.get('timestamp_s')
                        if timestamp not in clock_to_frame:
                            raise ValueError('稀疏源测量时间无法对应真实输入帧')
                        last_seen = clock_to_frame[timestamp]
                    if sparse:
                        counters['sparse_output_node_frames'] += 1
                    counters['motion_state:'+motion] += 1
                    node_id=registry[obj['node_id']]
                    nodes[node_id]=RelationNode(node_id,obj['center'],dynamic=motion=='dynamic',
                        observation_count=obj['num_detections'], first_seen_frame=first_output[obj['node_id']],
                        last_seen_frame=last_seen, metadata=dict(timestamp_s=obj.get('timestamp_s'),
                            motion_state=motion, shape_available=not sparse, source_node_id=obj['node_id']))
                t=time.perf_counter()
                direct=relations_from_centers([dict(id=registry[p['node_id']],center=p['center']) for p in predictions],
                    forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                direct_s=time.perf_counter()-t
                t=time.perf_counter()
                full=build_relations(nodes,RELATION_CONFIG,frame_index=idx,ego_heading=forward)
                full_s=time.perf_counter()-t
                t=time.perf_counter()
                inc=incremental.update_snapshot(nodes,frame_index=idx,ego_heading=forward)
                inc_s=time.perf_counter()-t
                if not full.success or not inc.success:
                    raise AssertionError('合法输入关系构建失败')
                actual={'direct_geometry':direct,'structured_full':expand(full.edges),'structured_incremental':expand(inc.edges)}
                gt_edges=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in truth],
                    forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                mapping={registry[predictions[m['prediction_index']]['node_id']]:truth[m['truth_index']]['instance_token'] for m in measured['matches']}
                per_method={}
                for method,pred_edges in actual.items():
                    per_method[method]={}
                    for rel in RELATIONS:
                        count=edge_counts(pred_edges,gt_edges,mapping,rel)
                        scene_rel[method,rel].update(count)
                        totals[method,rel].update(count)
                        per_method[method][rel or 'ALL']=dict(count)
                    frame_rows.append(dict(scene=name,log_token=log,frame_idx=idx,method=method,
                        **per_method[method]['ALL'],**scores(per_method[method]['ALL'])))
                fact_equal=actual['structured_full']==actual['structured_incremental']
                semantic_equal=semantic_content(full.edges)==semantic_content(inc.edges)
                temporal_equal=temporal_content(full.edges)==temporal_content(inc.edges)
                counters['frames']+=1
                counters['fact_equal_frames']+=fact_equal
                counters['semantic_equal_frames']+=semantic_equal
                counters['temporal_provenance_equal_frames']+=temporal_equal
                counters['heading_changed_frames']+=heading_changed
                counters['heading_changed_and_fact_different']+=bool(heading_changed and not fact_equal)
                stats=inc.update_statistics
                timing_rows.append(dict(scene=name,log_token=log,frame_idx=idx,nodes=len(nodes),
                    direct_s=direct_s,full_s=full_s,incremental_s=inc_s,
                    full_candidate_pairs=full.update_statistics.get('candidate_pair_count',0),
                    incremental_recomputed_pairs=stats.get('recomputed_pair_count',0),
                    reused_edges=stats.get('reused_edge_count',0),heading_changed=heading_changed,
                    facts_equal=fact_equal,semantic_content_equal=semantic_equal,temporal_provenance_equal=temporal_equal))
                if not fact_equal:
                    counters['full_only_expanded_facts']+=len(actual['structured_full']-actual['structured_incremental'])
                    counters['incremental_only_expanded_facts']+=len(actual['structured_incremental']-actual['structured_full'])
                record=dict(scene=name,frame_idx=idx,timestamp_s=row['camera_timestamp_s'],
                    gt_nodes=len(truth),predicted_nodes=len(predictions),matched_nodes=len(mapping),
                    node_id_registry={str(registry[p['node_id']]):p['node_id'] for p in predictions},
                    node_mapping=mapping,gt_relations=sorted(gt_edges),
                    predicted_relations={k:sorted(v) for k,v in actual.items()},metrics=per_method,
                    content_consistency=dict(facts=fact_equal,semantic=semantic_equal,temporal_provenance=temporal_equal),
                    only_full=sorted(actual['structured_full']-actual['structured_incremental']),
                    only_incremental=sorted(actual['structured_incremental']-actual['structured_full']),
                    full_edges=[e.to_dict() for e in full.edges],incremental_edges=[e.to_dict() for e in inc.edges],
                    incremental_statistics=stats)
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        for (method,rel),count in scene_rel.items():
            relation_scene_rows.append(dict(scene=name,log_token=log,method=method,relation=rel or 'ALL',**count,**scores(count)))
            groups[log,method,rel].update(count)
        node_total.update(scene_node)
        node_scene_rows.append(dict(scene=name,log_token=log,**scene_node))
        source_rows.append(dict(scene=name,log_token=log,frozen_R1=str(frozen_path),sha256=digest(frozen_path),raw_metadata=str(raw_path),raw_metadata_sha256=digest(raw_path)))
        print(name,'P0 complete',counters['frames'],flush=True)
    original_counts=original['result']['counts']
    if (len(source_rows),counters['frames'],node_total['truth'],node_total['true_positive'],node_total['predictions']) != (38,1508,5723,original_counts['true_positive'],original_counts['predictions']):
        raise AssertionError('P0范围不完整')
    for key in ['IDTP','IDFN','IDFP','GT_observations','prediction_observations']:
        if identity_total['all_gated'][key] != original['result']['counts'][key]:
            raise AssertionError('R1身份主计数未精确复现')
    relation_rows=[dict(method=m,relation=r or 'ALL',**c,**scores(c)) for (m,r),c in totals.items()]
    identity_rows=[dict(method=method_key,support=s,**c,**metric_values(c)) for s,c in identity_total.items()]
    log_rows=[dict(log_token=log,method=m,relation=r or 'ALL',**c,**scores(c)) for (log,m,r),c in groups.items()]
    for name,rows in [('relations_main',relation_rows),('identity_main',identity_rows),('relations_scene',relation_scene_rows),
        ('identity_scene',identity_scene_rows),('node_scene',node_scene_rows),('relations_log',log_rows),
        ('relations_frame',frame_rows),('incremental_frame',timing_rows),('sources',source_rows)]:
        csv_write(dest/(PREFIX+name+'.csv'),rows)
    report=dict(complete=True,method=method_key,scope=dict(scenes=38,logs=11,frames=1508,GT=5723),node_counts=dict(node_total),
        original_node_metrics=original['result']['metrics'],identity=identity_rows,relations=relation_rows,
        consistency=dict(counters),timing=dict(direct_total_s=sum(r['direct_s'] for r in timing_rows),
            full_total_s=sum(r['full_s'] for r in timing_rows),incremental_total_s=sum(r['incremental_s'] for r in timing_rows),
            full_candidate_pairs=sum(r['full_candidate_pairs'] for r in timing_rows),
            incremental_recomputed_pairs=sum(r['incremental_recomputed_pairs'] for r in timing_rows),
            measurement='one pass shared CPU Python module times; not end-to-end FPS'),
        wall_s=time.monotonic()-start,sources=source_rows,
        notes=['R1 identity and node counts exactly reproduced from frozen predictions',
            'relation facts use explicit effective near6/directional12 protocol; not re-labelled old scores',
            'all R1 sparse position nodes included, motion unknown preserved in metadata',
            'relation GT is official geometry-derived, no pixel/semantic relation manual annotation',
            'incremental historical metadata differs by design; semantic and current provenance checked separately'],
        outputs_sha256={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()})
    dump(dest/(PREFIX+'report.json'),report)
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    for name in ['relations_main','identity_main','sources','incremental_frame']:
        shutil.copy2(dest/(PREFIX+name+'.csv'),REPORT_ROOT/(PREFIX+args.report_tag+'_'+name+'.csv'))
    dump(REPORT_ROOT/(PREFIX+args.report_tag+'_result.json'),dict(run=str(dest),report=str(dest/(PREFIX+'report.json')),
        identity=identity_rows,relations=[r for r in relation_rows if r['relation']=='ALL'],
        consistency=dict(counters),timing=report['timing'],wall_s=report['wall_s']))
    print(json.dumps(dict(run=str(dest),identity=identity_rows,relations=[r for r in relation_rows if r['relation']=='ALL'],
        consistency=dict(counters),wall_s=report['wall_s']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
