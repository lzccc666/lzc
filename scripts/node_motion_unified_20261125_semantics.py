"""为冻结节点增加运动估计/来源核实状态别名；不改变原判定与任务结果。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip, hashlib, json
from collections import Counter
from conceptgraph.project_paths import PROJECT_ROOT as ROOT

RUN = ROOT / 'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2'
DATA = ROOT / 'dataset/processed/node_motion_unified_20261125_v2'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    freeze = dict(source_sha256=sha(__file__),
        role='additive field semantics; legacy estimate and verified raw state stay distinct',
        motion_state_semantics='verified_raw_evidence', configuration_changed=False,
        inputs={scope: sha(RUN / (scope + '_task_nodes.jsonl.gz')) for scope in ('R1', 'S2')})
    with (DATA / 'semantic_alias_freeze.json').open('x') as out:
        json.dump(freeze, out, ensure_ascii=False, indent=2)
    summaries = {}
    for scope in ('R1', 'S2'):
        source = RUN / (scope + '_task_nodes.jsonl.gz')
        target = RUN / (scope + '_semantic_task_nodes.jsonl.gz')
        assert not target.exists()
        counts = Counter()
        with gzip.open(source, 'rt') as inp, gzip.open(target, 'wt', compresslevel=6) as out:
            for line in inp:
                frame = json.loads(line)
                for node in frame['nodes']:
                    estimated = node['legacy_motion_state']
                    verified = node['motion_state']
                    assert estimated in ('static', 'dynamic', 'unknown')
                    assert verified in ('static', 'dynamic', 'unknown')
                    node.update(estimated_motion_state=estimated,
                        estimated_motion_state_source='original_saved_motion_estimator',
                        verified_motion_state=verified,
                        verified_motion_state_source='frozen_causal_raw_measurement_confirmation',
                        motion_state_semantics='verified_raw_evidence')
                    counts['nodes'] += 1
                    counts['estimated:' + estimated] += 1
                    counts['verified:' + verified] += 1
                    assert node['vehicle_safety_authorized'] is False
                frame['motion_alias_source_sha256'] = freeze['inputs'][scope]
                frame['motion_alias_policy_changed'] = False
                out.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + '\n')
        summaries[scope] = dict(counts=dict(counts), input_sha256=sha(source),
            output_sha256=sha(target), policy_or_previous_fields_changed=False)
    result = dict(passed=True, scopes=summaries,
        source_sha256=sha(__file__), GT_in_method=False, GPU_calls=0)
    with (RUN / 'semantic_alias_summary.json').open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
