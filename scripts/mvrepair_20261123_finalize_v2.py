"""汇总冻结三批全分母表、逐帧得失和候选索引；不运行方法。"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from conceptgraph.project_paths import PROJECT_ROOT

BASE = PROJECT_ROOT / 'outputs/results/nuscenes/validation/iterations'
DEST = PROJECT_ROOT / 'outputs/reports/multiview_identity_repair_20261123/main'
P = 'mvrepair_20261123_'
VERSION = 'v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(name, records):
    name = name.removesuffix('_v2') + '_' + VERSION
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (DEST / (P + name + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(records)


def load_rows(run, prefix, name):
    out = {}
    for scene in sorted(run.glob('scene-*')):
        items = list(map(json.loads, gzip.open(scene / (prefix + name + '.jsonl.gz'), 'rt')))
        assert len(items) == 20
        for item in items:
            key = item['scene'], item['frame_idx']
            assert key not in out
            out[key] = item
    assert len(out) == 40
    return out


def main():
    global VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-nextlogs', action='store_true')
    parser.add_argument('--version', default='v2')
    args = parser.parse_args()
    VERSION = args.version
    sources = []
    for batch, suffix in [('pilot', 'pilot_v1'), ('regression', 'crosslog_v1')]:
        sources.extend([(batch, 'mvrepair_20261123_P1_' + suffix, P,
            {'A_front': 'A', 'C_three_dedup': 'C', 'P1_front': 'P1_front', 'P1_three': 'P1'}),
            (batch, 'mvrepair_20261123_T2_' + suffix, P, {'T2_front': 'T2_front', 'T2_three': 'T2'}),
            (batch, 'mvrepair_20261123_T3_' + suffix.replace('v1', 'v2'), P,
                {'T3_front': 'T3_front', 'T3_three': 'T3'})])
    sources.extend([
        ('pilot', 'mvrepair_lifecycle_20261123_A_physical_only_v1', 'mvrepair_lifecycle_20261123_', {'A_front_physical_only': 'A_physical'}),
        ('regression', 'mvrepair_A_physical_crosslog_regression_20261123_v1', 'mvrepair_lifecycle_20261123_', {'A_front_physical_only': 'A_physical'})])
    if args.include_nextlogs:
        for method, mapping, prefix in [
            ('P1', {'A_front':'A', 'C_three_dedup':'C', 'P1_front':'P1_front', 'P1_three':'P1'}, P),
            ('T2', {'T2_front':'T2_front', 'T2_three':'T2'}, P),
            ('T3', {'T3_front':'T3_front', 'T3_three':'T3'}, P),
            ('Aphysical', {'A_front_physical_only':'A_physical'}, 'mvrepair_fair_')]:
            sources.append(('nextlogs', f'mvrepair_{method}_nextlogs_confirmation_20261123_v1', prefix, mapping))
    nodes, relations, sides, costs, manifest, evaluations = [], [], [], [], [], {}
    for batch, run_id, prefix, methods in sources:
        run = BASE / run_id
        path = run / (prefix + 'report.json')
        report = json.loads(path.read_text())
        assert report['complete']
        manifest.append(dict(batch=batch, run=str(run), report=str(path), report_sha256=sha(path), methods=methods))
        for source_key, target in [('node_main', nodes), ('relation_main', relations), ('side_coverage', sides)]:
            for row in report['results'][source_key]:
                if row['method'] not in methods:
                    continue
                item = dict(batch=batch, variant=methods[row['method']], **row, source_report=str(path))
                if source_key == 'node_main':
                    item['duplicate_per_prediction'] = row['duplicate_nodes'] / max(1, row['predictions'])
                target.append(item)
        if 'cost_total_s' in report:
            for method, value in report['cost_total_s'].items():
                if method in methods:
                    costs.append(dict(batch=batch, variant=methods[method], update_wall_s=value,
                        full_run_wall_s=report['wall_s'], peak_co_resident_RSS_bytes=report['peak_process_RSS_bytes'],
                        RSS_scope=report['RSS_scope'], source_report=str(path)))
        if any(m in methods for m in ['P1_three', 'T3_three']):
            evaluations[batch, 'P1' if 'P1_three' in methods else 'T3'] = load_rows(run, prefix, 'evaluation')
    csv_write('three_batch_node_main_v2', nodes)
    csv_write('three_batch_relation_main_v2', relations)
    csv_write('three_batch_side_main_v2', sides)
    csv_write('three_batch_cost_v2', costs)
    # 累计保持原逐场景身份计数的可加口径；不把120组称独立测试。
    groups = defaultdict(list)
    for row in nodes:
        groups[row['variant'], row['scope']].append(row)
    pooled = []
    for (variant, scope), items in groups.items():
        row = dict(variant=variant, scope=scope, batches=len(items))
        for key in ['true_positive', 'truth', 'predictions', 'unmatched_predictions', 'false_negative',
                    'duplicate_nodes', 'error_sum_m', 'IDTP', 'IDFN', 'IDFP']:
            row[key] = sum(r[key] for r in items)
        tp, gt, pred = row['true_positive'], row['truth'], row['predictions']
        row.update(precision=tp/max(1,pred), recall=tp/max(1,gt), F1=2*tp/max(1,gt+pred),
            IDF1=2*row['IDTP']/max(1,2*row['IDTP']+row['IDFN']+row['IDFP']),
            center_error_m=row['error_sum_m']/max(1,tp), duplicate_per_prediction=row['duplicate_nodes']/max(1,pred))
        pooled.append(row)
    csv_write('pooled_node_main_v2', pooled)
    frames, changes = [], []
    for batch in sorted({b for b, _ in evaluations}):
        before, after = evaluations[batch,'P1'], evaluations[batch,'T3']
        assert before.keys() == after.keys()
        for key, a in before.items():
            b = after[key]
            for scope in ['front', 'union']:
                gt = a['evaluation_only_official'][scope]
                assert gt == b['evaluation_only_official'][scope]
                old, new = a['result']['P1_three'][scope], b['result']['T3_three'][scope]
                om = {m['truth_index']:m for m in old['metrics']['matches']}
                nm = {m['truth_index']:m for m in new['metrics']['matches']}
                frames.append(dict(batch=batch, scene=key[0], frame_idx=key[1], scope=scope,
                    TP_P1=len(om), TP_T3=len(nm), predictions_P1=len(old['predictions']), predictions_T3=len(new['predictions']),
                    duplicate_P1=old['metrics']['duplicate_nodes'], duplicate_T3=new['metrics']['duplicate_nodes'],
                    gained_GT=len(nm.keys()-om.keys()), lost_GT=len(om.keys()-nm.keys())))
                for index in sorted(om.keys() | nm.keys()):
                    op = old['predictions'][om[index]['prediction_index']] if index in om else None
                    np = new['predictions'][nm[index]['prediction_index']] if index in nm else None
                    if op is None or np is None or op['node_id'] != np['node_id'] or op['center'] != np['center']:
                        changes.append(dict(batch=batch, scene=key[0], frame_idx=key[1], scope=scope,
                            instance_token=gt[index]['instance_token'], category=gt[index]['category'],
                            change='gained' if op is None else 'lost' if np is None else 'identity_or_position_changed',
                            node_P1=op['node_id'] if op else '', node_T3=np['node_id'] if np else '',
                            error_P1=om[index]['distance_m'] if op else '', error_T3=nm[index]['distance_m'] if np else ''))
    csv_write('P1_T3_all_frame_deltas_v2', frames)
    csv_write('P1_T3_changed_GT_frames_v2', changes)
    (DEST/(P+'final_sources_'+VERSION+'.json')).write_text(json.dumps(dict(sources=manifest, include_nextlogs=args.include_nextlogs,
        scope='All object-frame denominators preserved. Adaptive debug selection; final nextlogs fixed once after source freeze. '
            'Pooled identity totals sum original per-scene ID accounting, not a cross-scene joint identity assignment.'), ensure_ascii=False, indent=2))
    print(json.dumps(dict(source_reports=len(manifest), node_rows=len(nodes), pooled_rows=len(pooled), frame_deltas=len(frames), changed_GT_frames=len(changes))))


if __name__ == '__main__':
    main()
