"""只读共同前视身份链与位置改动；不运行模型/建图，不修正预测。"""
import argparse
from collections import defaultdict, Counter
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_independent_scores import in_view, score_frame, score_identity, write_csv, digest, identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--ground-truth', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--candidate', default='L4_three')
    ap.add_argument('--out-id', required=True)
    ap.add_argument('--baseline-audit-run', type=Path)
    args = ap.parse_args()
    run = experiment_directory('nuscenes', 'validation', 'mvrepair_identity_'+args.out_id, create=True)
    manifest = json.loads(args.manifest.read_text())
    gt = {(r['scene'], r['frame_idx']): r['objects'] for r in map(json.loads, args.ground_truth.read_text().splitlines())}
    methods = ['P1_three', 'C_three_dedup', args.candidate]
    totals = defaultdict(Counter)
    summaries, chains, worse, all_rows, primary, primary_credit = [], [], [], [], [], []
    hashes = {str(args.manifest): digest(args.manifest), str(args.ground_truth): digest(args.ground_truth)}
    for scene in manifest['scenes']:
        name = scene['scene']
        path, = (args.run/name).glob('*predictions*.jsonl.gz')
        hashes[str(path)] = digest(path)
        rows = list(map(json.loads, gzip.open(path, 'rt')))
        if args.baseline_audit_run:
            baseline_file, = (args.baseline_audit_run/name).glob('*predictions*.jsonl.gz')
            baseline_rows = list(map(json.loads, gzip.open(baseline_file, 'rt')))
            for current, baseline in zip(rows, baseline_rows):
                assert current['frame_idx']==baseline['frame_idx'] and current['sample_token']==baseline['sample_token']
                current['predictions']['P1_three']=baseline['predictions']['P1_three']
                for key, value in baseline['audit'].items():
                    if key not in current['audit'] and key in current['predictions']:
                        assert current['predictions'][key] == baseline['predictions'][key]
                        current['audit'][key] = value
            hashes[str(baseline_file)] = digest(baseline_file)
        records, traces, full_records = defaultdict(list), defaultdict(list), defaultdict(list)

        def sources(method, prediction, frame_idx):
            if method == args.candidate:
                if prediction.get('front_position_source_key'):
                    return dict(status='explicit_front_position_source', keys=[prediction['front_position_source_key']])
                keys = []
                for row in rows[:frame_idx+1][::-1]:
                    life = row['audit'][method].get('lifecycle', {})
                    raw = life.get('output_representatives', {}).get(prediction['node_id'], prediction['node_id'])
                    for w in life.get('registry', {}).get('witnesses', []):
                        for side in ['a', 'b']:
                            if w.get('track_'+side) in [raw, prediction['node_id']] and w.get('source_key_'+side) not in [None, *keys]:
                                keys.append(w['source_key_'+side])
                    if keys:
                        return dict(status='latest_registry_witness_sources_not_forced_unique', keys=keys)
            # Historical baseline audit does not persist track-to-detection associations.
            # Exact category/timestamp/geometry evidence can resolve some dense observations;
            # unresolved cases stay unresolved instead of fabricating source keys.
            keys = []
            for row in rows[:frame_idx+1][::-1]:
                for a in row['audit'][method].get('mapping', []):
                    if (a.get('stage') == 'geometry' and a.get('reason') == 'accepted'
                            and a.get('category') == prediction['category']
                            and abs(a.get('camera_timestamp_s', -1)-(prediction.get('timestamp_s') or -100)) < .08
                            and np.linalg.norm(np.asarray(a['observation_center'])-prediction['center']) < 1e-5):
                        keys.append([a['camera'], a['camera_timestamp_s'], a['detection_index']])
                if keys:
                    return dict(status='exact_saved_observation_center_match', keys=keys)
            return dict(status='not_persisted_unresolved_no_inference_rerun', keys=[])

        for group, row in zip(scene['groups'], rows):
            idx = group['frame_idx']
            truth = [o for o in gt[name, idx] if 'CAM_FRONT' in o['visible_cameras']]
            matches = {}
            for method in methods:
                pred = [p for p in row['predictions'][method] if in_view(p['center'], group['views']['CAM_FRONT'])]
                counts, pairs, gate = score_frame(pred, truth)
                matches[method] = {truth[g]['instance_token']: pred[p] for g, p in pairs}
                full_records[method].append(dict(frame_idx=idx, truth=truth, pred=pred, pairs=pairs, gate=gate))
            common = set(matches['P1_three']) & set(matches[args.candidate])
            truth_by_id = {o['instance_token']: o for o in truth}
            tokens = sorted(common)
            for method in ['P1_three', args.candidate]:
                paired_pred = [matches[method][t] for t in tokens]
                records[method].append(dict(frame_idx=idx, truth=[truth_by_id[t] for t in tokens], pred=paired_pred,
                    pairs=[(i, i) for i in range(len(tokens))], gate=np.eye(len(tokens), dtype=bool)))
                for token, pred in zip(tokens, paired_pred):
                    source = sources(method, pred, idx)
                    traces[token, method].append(dict(frame_idx=idx, node_id=pred['node_id'], category=pred['category'],
                        center=pred['center'], source=source, position_source=pred.get('position_source'),
                        prediction_age_s=pred.get('prediction_age_s')))
                    all_rows.append(dict(scene=name, log_token=scene['log_token'], GT_instance=token, method=method,
                        frame_idx=idx, node_id=pred['node_id'], source_status=source['status'], source_keys=json.dumps(source['keys']),
                        position_source=pred.get('position_source')))
            for token in set(matches['C_three_dedup']) & set(matches[args.candidate]):
                c, n = matches['C_three_dedup'][token], matches[args.candidate][token]
                center = np.asarray(truth_by_id[token]['center'][:2])
                ec = float(np.linalg.norm(np.asarray(c['center'][:2])-center))
                en = float(np.linalg.norm(np.asarray(n['center'][:2])-center))
                if en > ec+1e-8 and np.linalg.norm(np.asarray(c['center'])-n['center']) > 1e-8:
                    decision = [d for d in row['audit'][args.candidate].get('front', {}).get('position_decisions', []) if d['canonical'] == n['node_id']]
                    worse.append(dict(scene=name, frame_idx=idx, GT_instance=token, baseline_prediction=c,
                        candidate_prediction=n, previous_error_m=ec, candidate_error_m=en, error_increase_m=en-ec,
                        front_position_decisions=decision, candidate_source=sources(args.candidate, n, idx)))
        for method, recs in records.items():
            counts = score_identity(recs)
            totals[method].update(counts)
            summaries.append(dict(scene=name, log_token=scene['log_token'], method=method, **counts,
                paired_chain_IDF1=2*counts['IDTP']/(counts['identity_GT']+counts['identity_predictions'])))
        for method in ['P1_three', args.candidate]:
            recs = full_records[method]
            full_counts = score_identity(recs)
            primary.append(dict(scene=name, log_token=scene['log_token'], method=method, **full_counts,
                full_denominator_IDF1=2*full_counts['IDTP']/(full_counts['identity_GT']+full_counts['identity_predictions'])))
            gids = sorted({identity(o['instance_token']) for rec in recs for o in rec['truth']})
            pids = sorted({identity(o['node_id']) for rec in recs for o in rec['pred']})
            gindex, pindex = {v:i for i,v in enumerate(gids)}, {v:i for i,v in enumerate(pids)}
            potential = np.zeros((len(gids), len(pids)), int)
            support_frames = defaultdict(list)
            for rec in recs:
                for g, p in zip(*np.nonzero(rec['gate'])):
                    gid, pid = identity(rec['truth'][g]['instance_token']), identity(rec['pred'][p]['node_id'])
                    potential[gindex[gid], pindex[pid]] += 1
                    support_frames[gid, pid].append(rec['frame_idx'])
            gi, pi = linear_sum_assignment(-potential)
            allocation = {gids[g]:pids[p] for g,p in zip(gi,pi)}
            for gid in gids:
                pid = allocation.get(gid)
                primary_credit.append(dict(scene=name, log_token=scene['log_token'], method=method,
                    GT_instance=json.loads(gid), assigned_ID=json.loads(pid) if pid is not None else None,
                    primary_IDTP_credit=int(potential[gindex[gid], pindex[pid]]) if pid is not None else 0,
                    credited_frames=support_frames.get((gid,pid), [])))
        assigned = {}
        for method in ['P1_three', args.candidate]:
            tokens_here = sorted({t for t, m in traces if m == method})
            ids_here = sorted({item['node_id'] for (t, m), seq in traces.items() if m == method for item in seq})
            support = np.asarray([[sum(item['node_id'] == pid for item in traces[token, method]) for pid in ids_here] for token in tokens_here])
            gi, pi = linear_sum_assignment(-support)
            assigned[method] = {tokens_here[g]: ids_here[p] for g, p in zip(gi, pi)}
        for token in sorted({t for t, m in traces}):
            aa, bb = traces[token, 'P1_three'], traces[token, args.candidate]
            def changes(seq):
                return sum(a['node_id'] != b['node_id'] for a, b in zip(seq, seq[1:]) if a['frame_idx']+1 == b['frame_idx'])
            ids_a, ids_b = {r['node_id'] for r in aa}, {r['node_id'] for r in bb}
            credit_a = sum(item['node_id'] == assigned['P1_three'].get(token) for item in aa)
            credit_b = sum(item['node_id'] == assigned[args.candidate].get(token) for item in bb)
            chains.append(dict(scene=name, log_token=scene['log_token'], GT_instance=token, common_frames=len(aa),
                P1_unique_IDs=len(ids_a), candidate_unique_IDs=len(ids_b), P1_adjacent_changes=changes(aa),
                candidate_adjacent_changes=changes(bb), P1_global_assigned_ID=assigned['P1_three'].get(token),
                candidate_global_assigned_ID=assigned[args.candidate].get(token), P1_credited_frames=credit_a,
                candidate_credited_frames=credit_b, candidate_minus_P1_uncredited_frames=credit_a-credit_b,
                P1_chain=aa, candidate_chain=bb))
    pooled = [dict(method=m, **c, paired_chain_IDF1=2*c['IDTP']/(c['identity_GT']+c['identity_predictions'])) for m, c in totals.items()]
    result = dict(candidate=args.candidate, paired_common_front=pooled, per_log=summaries,
        full_primary_front_identity=primary, full_primary_per_GT_credit=primary_credit,
        definitions='Intersection of P1 and candidate geometrically matched identical front GT object-frames. Diagnostic uses these fixed one-to-one geometric pairs only and equal GT/prediction denominators; paired_chain_IDF1 is NOT the primary all-gated full-denominator IDF1.',
        limitations='Automatic 3m matches may be ambiguous and are not human identity truth; source-key gaps are explicit. New logs already inspected: regression only.',
        chains=chains, center_worse_cases=sorted(worse, key=lambda r:-r['error_increase_m']), source_sha256=digest(__file__),
        input_sha256=hashes, model_calls=0, mapping_calls=0, modified_predictions=False)
    (run/'diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    report = PROJECT_ROOT/'outputs/reports/multiview_identity_repair_20261123/validation'
    write_csv(report/(args.out_id+'_common_front_identity.csv'), summaries+pooled)
    write_csv(report/(args.out_id+'_common_front_chains.csv'), all_rows)
    write_csv(report/(args.out_id+'_full_front_identity.csv'), primary)
    write_csv(report/(args.out_id+'_full_front_GT_credit.csv'), primary_credit)
    (report/(args.out_id+'_identity_diagnostics.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(pooled=pooled, worsening_centers=len(worse), fragmented_chains=[{k:r[k] for k in ['scene','GT_instance','common_frames','P1_unique_IDs','candidate_unique_IDs','P1_adjacent_changes','candidate_adjacent_changes']} for r in chains if r['candidate_unique_IDs']>r['P1_unique_IDs']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
