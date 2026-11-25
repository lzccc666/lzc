"""复现冻结验证选择，隔离候选、非关系门、关系过滤和排序损失。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from conceptgraph.completion.navigation_diagnostics import to_nodes  # noqa: E402
from conceptgraph.completion.selection_ablation import select  # noqa: E402
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets  # noqa: E402

SOURCE = PROJECT / 'outputs/project_v2/thesis_completion_v1'
RETRIEVAL = SOURCE / 'main_experiments/causal_retrieval_full_001'
ABLATION = SOURCE / 'ablations/causal_selection_ablation_full_001'
VALIDATION = PROJECT / 'dataset/qwen/real_graph_v003/validation.jsonl'
CONDITIONS = ('full_joint_rule', 'without_relation_edges', 'without_depth_evidence_gate')
POLICY = dict(enabled=True, minimum_observations=3, minimum_lidar_pixels=5,
              minimum_depth_support_fraction=.4)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination
    started = time.monotonic()
    release = read(RETRIEVAL / 'release.json')
    original_report = read(ABLATION / 'report.json')
    sources = [Path(__file__), VALIDATION, RETRIEVAL / 'queries.jsonl',
               RETRIEVAL / 'release.json', ABLATION / 'queries.jsonl', ABLATION / 'report.json']
    sources += [PROJECT / name for name in (
        'conceptgraph/completion/selection_ablation.py',
        'conceptgraph/completion/navigation_diagnostics.py',
        'conceptgraph/navigation/semantic.py', 'conceptgraph/completion/evidence.py',
        'conceptgraph/scenegraph/relations.py')]
    fingerprints = {str(path): sha(path) for path in sources}
    dump(destination / 'diagnostic_release.json', dict(
        source_sha256=fingerprints, source_frames_sha256=release['source_frames_sha256'],
        methods=CONDITIONS, inference_uses_gt=False, independent_test=False,
        actual_arrival_evaluated=False, model_calls=0,
        scope='全部冻结1115验证任务与171条Qwen验证记录；两套分母分别报告'))
    for path in sources:
        if path.suffix == '.py':
            target = destination / 'source_snapshot' / path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    groups = defaultdict(list)
    for row in jsonl(RETRIEVAL / 'queries.jsonl'):
        if row['method'] == 'existing_object_joint_rule':
            groups[(row['scene'], row['frame_idx'])].append(row)
    frozen = {(r['scene'], r['frame_idx'], r['query_index'], r['method']): r
              for r in jsonl(ABLATION / 'queries.jsonl')}
    totals, losses = defaultdict(Counter), Counter()
    rows, cases = [], []
    scenes = set()
    for filename, expected in release['source_frames_sha256'].items():
        if sha(filename) != expected:
            raise ValueError(f'冻结预测改变：{filename}')
        for frame in jsonl(filename):
            queries = groups.get((frame['scene'], frame['frame_idx']), [])
            if not queries:
                continue
            predictions = frame['predictions']['one_to_one_depth_quality']
            pose = np.asarray(frame['camera_pose'])
            nodes = to_nodes(predictions)
            # 关联身份仅用于选择完成后的独立评价。
            truth = frame['evaluation_only_official']
            association = {m['prediction_index']: truth[m['truth_index']]['instance_token']
                           for m in frame['metrics']['one_to_one_depth_quality']['matches']}
            scenes.add(frame['scene'])
            for query in queries:
                ranked_methods = {}
                for method in CONDITIONS:
                    ranked = select(query['query_text'], predictions, pose, frame['frame_idx'], method)
                    ranked_methods[method] = ranked
                    selected = ranked[0].node_id if ranked else None
                    identity = association.get(selected)
                    previous = frozen[(frame['scene'], frame['frame_idx'], query['query_index'], method)]
                    if (selected != previous['selected_node_id'] or identity != previous['evaluation_identity']
                            or [c.node_id for c in ranked] != previous['candidate_ids']):
                        raise ValueError('原选择/排序未逐条复现')
                    positive = bool(query['positive'])
                    correct = positive and identity in query['acceptable_instances']
                    totals[method].update(queries=1, positive=int(positive), negative=int(not positive),
                                          correct=int(correct), false_selection=int(not positive and selected is not None),
                                          no_selection=int(selected is None),
                                          unknown_identity=int(selected is not None and identity is None))
                parsed = parse_navigation_query(query['query_text'], 8.)
                parsed.vehicle_heading_xy = pose[:2, 2]
                no_relation = replace(parsed, relation=None, anchor_text=None)
                eligible = rank_candidate_targets(no_relation, nodes, [], pose[:2, 3], .55, .3, .25,
                    relation_coordinate_frame='ego_heading', relation_coordinate_frame_fallback=None,
                    evidence_policy=POLICY)
                positive = bool(query['positive'])
                acceptable = set(query['acceptable_instances'])
                raw_correct = [i for i in nodes if association.get(i) in acceptable]
                eligible_correct = [c.node_id for c in eligible if association.get(c.node_id) in acceptable]
                full = ranked_methods['full_joint_rule']
                ranked_correct = [c.node_id for c in full if association.get(c.node_id) in acceptable]
                selected_correct = bool(full and association.get(full[0].node_id) in acceptable)
                if not positive:
                    reason = 'negative_false_selection' if full else 'negative_rejected'
                elif not raw_correct:
                    reason = 'target_absent_from_predicted_nodes'
                elif not eligible_correct:
                    reason = 'target_excluded_by_nonrelation_gates'
                elif not ranked_correct:
                    reason = 'target_excluded_by_relation_constraint'
                elif not selected_correct:
                    reason = 'target_present_but_ranked_below_wrong_choice'
                else:
                    reason = 'correct_target_selected'
                losses[reason] += 1
                audit = []
                for candidate in full:
                    matches = candidate.debug['relation_debug'].get('matches', [])
                    best = matches[0] if matches else None
                    if not best:
                        continue
                    anchor = nodes[best['anchor_id']]
                    delta = nodes[candidate.node_id].bbox_center[:2] - anchor.bbox_center[:2]
                    distance = float(np.linalg.norm(delta))
                    geometry = best.get('geometry_debug') or {}
                    component = geometry.get('left_component_m') if parsed.relation in {'left_of', 'right_of'} else geometry.get('forward_component_m')
                    audit.append(dict(node_id=candidate.node_id, anchor_id=best['anchor_id'],
                        edge_source=best['edge_source'], relation=parsed.relation,
                        relation_score=candidate.relation_score, distance_m=distance,
                        directional_component_m=component,
                        fallback_outside_graph_distance=(best['edge_source']=='geometry_fallback' and distance>12),
                        fallback_inside_graph_margin=(best['edge_source']=='geometry_fallback' and component is not None and abs(component)<=1),
                        best=best))
                record = {k: query[k] for k in ('scene', 'frame_idx', 'query_index', 'query_text', 'query_type', 'positive')}
                record.update(reason=reason, raw_correct_ids=raw_correct,
                              eligible_correct_ids=eligible_correct, ranked_correct_ids=ranked_correct,
                              full_candidate_ids=[c.node_id for c in full],
                              selected_node_id=full[0].node_id if full else None,
                              relation_audit=audit, gt_in_inference=False)
                rows.append(record)
                if parsed.relation is not None and (audit or eligible_correct):
                    cases.append(record)
            if time.monotonic() - started > 1200 or shutil.disk_usage(PROJECT).free < 20*1024**3:
                raise RuntimeError('诊断运行预算/磁盘边界，保留已写文件')
    for method, total in totals.items():
        old = original_report['conditions'][method]
        expected = [old['positive_queries'], old['negative_queries'], old['positive_selected_correct'],
                    old['no_match_false_selection']['numerator']]
        if [total[k] for k in ('positive', 'negative', 'correct', 'false_selection')] != expected:
            raise ValueError('原汇总未复现')
    val = list(jsonl(VALIDATION))
    ordinary = [r for r in val if r['ordinary_proposal_only'] and r['official_positive']]
    val_report = dict(rows=len(val), ordinary_positive=len(ordinary),
        positive_candidate_covered=sum(bool(r['candidate_target_coverage']) for r in ordinary),
        positive_reliable_evidence_covered=sum(bool(r['reliable_target_coverage']) for r in ordinary),
        scenes=len({r['scene_id'] for r in val}), logs=len({r['log_token'] for r in val}),
        source_frames=len({(r['scene_id'],r['frame_idx']) for r in val}),
        by_kind=dict(Counter(r['query_kind'] for r in val)), actual_new_model_calls=0,
        input_scope='候选覆盖诊断；不是171条真实模型推理成绩')
    raw_coverage = {}
    val_ids = {r['sample_id'] for r in val}
    for path in sorted((PROJECT/'outputs/results/qwen/development_evaluation').glob('*/raw_records.jsonl')):
        records = list(jsonl(path))
        ids = {r.get('sample_id') for r in records}
        raw_coverage[path.parent.name] = dict(rows=len(records), validation_id_intersection=len(ids & val_ids), sha256=sha(path))
    diagnostic = dict(original_1115_reproduced=True, conditions=totals,
        positive_loss_partition=losses, scenes=len(scenes), frozen_query_count=len(rows),
        validation_171=val_report, cached_model_validation_overlap=raw_coverage,
        relation_candidates=sum(len(r['relation_audit']) for r in rows),
        fallback_outside_graph_distance=sum(a['fallback_outside_graph_distance'] for r in rows for a in r['relation_audit']),
        fallback_inside_graph_margin=sum(a['fallback_inside_graph_margin'] for r in rows for a in r['relation_audit']),
        elapsed_s=time.monotonic()-started, inference_uses_gt=False,
        independent_test=False, actual_arrival_evaluated=False)
    for filename, data in [('diagnosis_rows.jsonl',rows),('relation_cases.jsonl',cases)]:
        with (destination/filename).open('x') as handle:
            for record in data:
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    if any(sha(path)!=value for path,value in fingerprints.items()):
        raise ValueError('运行中输入或源码改变')
    dump(destination/'diagnostic_report.json',diagnostic)
    print(json.dumps(diagnostic,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
