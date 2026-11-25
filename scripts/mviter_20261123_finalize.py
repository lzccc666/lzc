"""独立重计冻结结果、全部得失/身份切换与资源/历史保护审计。"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.completion.stage9 import PROJECT, digest, dump


P = 'mviter_20261123_'
REPORT = PROJECT/'outputs/reports/multiview_task_iteration_20261123/mapping'
RUN = PROJECT/'outputs/results/nuscenes/validation/iterations/mviter_20261123_candidate_v1'


def rows(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def main():
    result = json.loads((RUN/(P+'report.json')).read_text())
    release = json.loads((RUN/(P+'release.json')).read_text())
    start = json.loads((REPORT/(P+'start.json')).read_text())
    for path, sha in start['protected_sha256'].items():
        assert digest(path) == sha, path
    for path, sha in release['input_sha256'].items():
        assert digest(path) == sha, path
    for path, record in release['source_sha256'].items():
        assert digest(path) == digest(record['snapshot']) == record['sha256'], path
    for path, sha in result['outputs_sha256'].items():
        assert digest(RUN/path) == sha, path
    for path, sha in result['csv_sha256'].items():
        assert digest(path) == sha, path
    total, relation_total = defaultdict(Counter), defaultdict(Counter)
    changes, counts, identity_cases, identity_summary, canonical_cases = [], Counter(), [], [], []
    frame_scopes = 0
    for scene_path in sorted(RUN.glob('scene-*')):
        sequence = rows(scene_path/(P+'evaluation.jsonl.gz'))
        prediction_rows = rows(scene_path/(P+'predictions.jsonl.gz'))
        assert len(sequence) == len(prediction_rows) == 20
        potential, ids, scene_identity, previous = defaultdict(Counter), defaultdict(lambda: [set(), set()]), defaultdict(Counter), defaultdict(dict)
        for row, raw in zip(sequence, prediction_rows):
            assert row['frame_idx'] == raw['frame_idx']
            assert len(raw['predictions']['D_front_anchor']) == len({p['node_id'] for p in raw['predictions']['D_front_anchor']})
            for scope in ['front', 'union']:
                truth = row['evaluation_only_official'][scope]
                candidate = row['result']['D_front_anchor'][scope]
                for baseline in ['A_front', 'C_three_dedup']:
                    a = row['result'][baseline][scope]
                    ma, md = [{r['truth_index']: r for r in x['metrics']['matches']} for x in [a, candidate]]
                    for i in sorted(set(ma)^set(md)):
                        kind = 'gain' if i in md else 'loss'
                        group = 'front' if 'CAM_FRONT' in truth[i]['visible_cameras'] else 'side_only'
                        changes.append(dict(scene=row['scene'], frame_idx=row['frame_idx'], comparison='D_vs_'+baseline,
                            scope=scope, kind=kind, truth=truth[i], baseline_match=ma.get(i), D_match=md.get(i),
                            baseline_prediction=a['predictions'][ma[i]['prediction_index']] if i in ma else None,
                            D_prediction=candidate['predictions'][md[i]['prediction_index']] if i in md else None))
                        counts[baseline+':'+scope+':'+kind] += 1
                        counts[baseline+':'+scope+':'+kind+':'+group] += 1
                for method, scopes in row['result'].items():
                    item = scopes[scope]
                    pred = item['predictions']
                    a = np.asarray([o['center'][:2] for o in truth]).reshape(-1, 2)
                    b = np.asarray([o['center'][:2] for o in pred]).reshape(-1, 2)
                    distance = np.linalg.norm(a[:, None]-b[None, :], axis=-1)
                    gate = distance <= 3.
                    for i, gt in enumerate(truth):
                        for j, p in enumerate(pred):
                            gate[i, j] &= gt['category'] == p['category']
                    cost = np.full((len(truth), len(pred)+len(truth)), 1e6)
                    cost[:, :len(pred)] = np.where(gate, distance, 1e9)
                    gi, pi = linear_sum_assignment(cost)
                    tp = int(sum(j < len(pred) and gate[i, j] for i, j in zip(gi, pi)))
                    duplicate = int(gate.any(axis=0).sum())-tp
                    assert (tp, duplicate) == (item['metrics']['true_positive'], item['metrics']['duplicate_nodes'])
                    key = method, scope
                    total[key].update(TP=tp, GT=len(truth), pred=len(pred), duplicates=duplicate)
                    gids, pids = ids[key]
                    gids.update(o['instance_token'] for o in truth)
                    pids.update(o['node_id'] for o in pred)
                    scene_identity[key].update(GT=len(truth), pred=len(pred))
                    for i, j in zip(*np.nonzero(gate)):
                        potential[key][truth[i]['instance_token'], pred[j]['node_id']] += 1
                    for match in item['metrics']['matches']:
                        instance = truth[match['truth_index']]['instance_token']
                        node = pred[match['prediction_index']]['node_id']
                        prior = previous[key].get(instance)
                        if prior and prior[0] == row['frame_idx']-1:
                            scene_identity[key]['adjacent_transitions'] += 1
                            scene_identity[key]['adjacent_switches'] += prior[1] != node
                            if prior[1] != node:
                                identity_cases.append(dict(scene=row['scene'], frame_idx=row['frame_idx'], method=method,
                                    scope=scope, instance_token=instance, previous_node=prior[1], node=node))
                        previous[key][instance] = row['frame_idx'], node
                    gt_edges, pred_edges = [{tuple(e) for e in item[k]} for k in ['gt_relations', 'predicted_relations']]
                    mapping = {int(k): v for k, v in item['node_mapping'].items()}
                    resolved = {(mapping[s], rel, mapping[t]) for s, rel, t in pred_edges if s in mapping and t in mapping}
                    edge_tp = len(resolved & gt_edges)
                    assert edge_tp == item['relation_counts']['TP']
                    assert len(gt_edges) == item['relation_counts']['all_truth']
                    assert len(pred_edges) == item['relation_counts']['all_pred']
                    relation_total[key].update(TP=edge_tp, GT=len(gt_edges), pred=len(pred_edges))
                    frame_scopes += 1
            audit = raw['audit']['D_front_anchor']['canonicalization']
            for cluster in audit['clusters']:
                if len(cluster['members']) > 1:
                    canonical_cases.append(dict(scene=row['scene'], frame_idx=row['frame_idx'], cluster=cluster,
                        supporting_pairs=[p for p in audit['pairs'] if p['eligible'] and p['node_a'] in cluster['node_ids'] and p['node_b'] in cluster['node_ids']]))
        for key, (gids, pids) in ids.items():
            gids, pids = sorted(gids), sorted(pids)
            matrix = np.array([[potential[key][g, p] for p in pids] for g in gids], dtype=int).reshape(len(gids), len(pids))
            gi, pi = linear_sum_assignment(-matrix)
            idtp = int(matrix[gi, pi].sum())
            total[key]['IDTP'] += idtp
            identity_summary.append(dict(scene=scene_path.name, method=key[0], scope=key[1], IDTP=idtp, **scene_identity[key]))
    for row in result['results']['node_main']:
        measured = total[row['method'], row['scope']]
        assert [measured[k] for k in ['TP', 'GT', 'pred', 'duplicates', 'IDTP']] == [row[k] for k in ['true_positive', 'truth', 'predictions', 'duplicate_nodes', 'IDTP']]
    assert frame_scopes == 240
    dump(REPORT/(P+'gain_loss_cases.json'), dict(counts=dict(counts), cases=changes))
    dump(REPORT/(P+'identity_cases.json'), dict(summary=identity_summary, adjacent_switches=identity_cases,
        identity_policy='stable source namespace; no GT-derived alias; representative may switch when evidence changes'))
    dump(REPORT/(P+'canonicalization_cases.json'), canonical_cases)
    paths = [RUN, PROJECT/'outputs/results/nuscenes/validation/iterations/mviter_20261123_diagnosis_v1', REPORT]
    sizes = {str(p): sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in paths}
    dump(REPORT/(P+'final_verification.json'), dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        independently_recounted_frame_scopes=frame_scopes, node_and_duplicate_counts=True, full_IDTP_independent_assignment=True,
        relation_sets=True, protected_sha256_checked=len(start['protected_sha256']), input_sha256_checked=len(release['input_sha256']),
        source_and_snapshot_sha256_checked=len(release['source_sha256']), baseline_exact=result['baseline_exact'],
        candidate_second_replay=False, replay_limit='negative result; no second candidate replay; full independent stored-fact audit instead',
        gain_loss_counts=dict(counts), canonicalized_clusters=len(canonical_cases), bytes_by_directory=sizes,
        free_disk_bytes=shutil.disk_usage(PROJECT).free, source_sha256=digest(__file__),
        no_test=True, no_frontend_calls=True, no_default_change=True))
    print(dict(passed=True, frame_scopes=frame_scopes, gain_loss=dict(counts), identity=identity_summary))


if __name__ == '__main__':
    main()
