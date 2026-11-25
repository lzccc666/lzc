"""仅从保存预测/评价派生同形状集比较，并检查冻结来源和语义别名。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip, hashlib, json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from conceptgraph.project_paths import PROJECT_ROOT as ROOT

RUN = ROOT / 'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2'
DATA = ROOT / 'dataset/processed/node_motion_unified_20261125_v2'
REPORT = ROOT / 'outputs/reports/node_motion_unified_20261125'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)

def main():
    by_node = {}
    motion = defaultdict(Counter)
    aliases = {'estimated_motion_state', 'estimated_motion_state_source',
        'verified_motion_state', 'verified_motion_state_source', 'motion_state_semantics'}
    for scope in ('R1', 'S2'):
        old_path = RUN / (scope + '_task_nodes.jsonl.gz')
        new_path = RUN / (scope + '_semantic_task_nodes.jsonl.gz')
        old_rows = list(rows(old_path)); new_rows = list(rows(new_path))
        assert len(old_rows) == len(new_rows)
        for old, new in zip(old_rows, new_rows):
            for node in new['nodes']:
                assert node['estimated_motion_state'] == node['legacy_motion_state']
                assert node['verified_motion_state'] == node['motion_state']
                for key in aliases:
                    node.pop(key)
            new.pop('motion_alias_source_sha256'); new.pop('motion_alias_policy_changed')
            assert old == new
            for node in old['nodes']:
                key = (scope, old['scene'], old['frame_idx'], node['node_id'])
                assert key not in by_node
                by_node[key] = node['shape_available']
                label = scope + '|shape=' + str(node['shape_available'])
                motion[label]['nodes'] += 1
                motion[label]['legacy:' + node['legacy_motion_state']] += 1
                motion[label]['verified:' + node['motion_state']] += 1
    comparisons = defaultdict(Counter)
    for row in rows(RUN / 'motion_evaluation_rows.jsonl.gz'):
        if row['matched_node_id'] is None:
            continue
        key = (row['scope'], row['scene'], row['frame_idx'], row['matched_node_id'])
        shape = by_node[key]
        for reference, field in [('attribute', 'attribute_motion'),
                                 ('past_displacement', 'past_displacement_motion')]:
            label = row[field]
            group = row['scope'] + '|shape=' + str(shape) + '|' + reference
            comparisons[group]['matched_GT'] += 1
            if label == 'unknown':
                comparisons[group]['reference_unknown'] += 1
                continue
            comparisons[group]['reference_labeled_GT'] += 1
            for method, predfield in [('legacy', 'legacy_predicted_motion'),
                                       ('verified', 'predicted_motion')]:
                pred = row[predfield]
                comparisons[group][method + ':known'] += int(pred in ('static', 'dynamic'))
                comparisons[group][method + ':correct'] += int(pred == label)
                comparisons[group][method + ':unknown'] += int(pred == 'unknown')
    result = dict(passed=True, alias_previous_fields_exact=True,
        scope_note='same saved matched GT and same original shape subset; no rematching or threshold change',
        prediction_node_motion={k:dict(v) for k,v in motion.items()},
        matched_GT_shape_comparison={k:dict(v) for k,v in comparisons.items()})
    with (RUN / 'same_shape_and_alias_checks.json').open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    checks = {}
    for section in ('source_sha256', 'input_sha256', 'evaluation_only_sha256'):
        checked = 0
        for path, expected in manifest[section].items():
            assert sha(path) == expected, path
            checked += 1
        checks[section] = dict(passed=True, checked=checked)
    for freeze, filename in [('direction_extension_freeze.json', 'node_motion_unified_20261125_views.py'),
                              ('report_derivation_freeze.json', 'node_motion_unified_20261125_report.py'),
                              ('semantic_alias_freeze.json', 'node_motion_unified_20261125_semantics.py')]:
        frozen = json.loads((DATA / freeze).read_text())
        expected = frozen.get('source_sha256', frozen.get('report_source_sha256'))
        assert sha(ROOT / 'scripts' / filename) == expected, filename
    result2 = dict(passed=True, checks=checks, GT_in_method=False, GPU_calls=0,
        parameter_search=False, source_sha256=sha(__file__), checked_utc=datetime.now(timezone.utc).isoformat(),
        original_predictions_exact={'R1':1508,'S2':120}, independent_reports=[
          str(ROOT/'outputs/reports/git_integration_20261125'/name) for name in [
          'node_independent_R1_v1.json','node_independent_S2_v1.json',
          'direction_independent_R1_v1.json','direction_independent_S2_v1.json',
          'motion_evaluation_independent_v1.json']])
    with (REPORT / 'frozen_sources_and_alias_checks_v2.json').open('x') as out:
        json.dump(result2, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result2, ensure_ascii=False))

if __name__ == '__main__':
    main()
