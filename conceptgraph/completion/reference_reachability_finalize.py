"""保留首轮参考，按既有90秒/2秒协议加强已独立通过的控制见证。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
from dataclasses import asdict

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, step_vehicle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--input-reference', default='reference_reachability_reserved_full_001')
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'annotation'/args.input_reference
    prior_release, prior_summary = [json.loads((source/f'{name}.json').read_text()) for name in ['release', 'summary']]
    prior_audit = source/'audit_001'
    checks = json.loads((prior_audit/'witness_checks.json').read_text())
    accepted = {(r['scene'], r['frame_idx'], r['instance_token']) for r in checks if r['passed']}
    for name, sha in prior_summary['files_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('changed original reference artifact')
    dest = OUTPUT/'annotation'/args.run_id
    dest.mkdir(exist_ok=False)
    (dest/'witnesses').mkdir()
    spec = VehicleSpec()
    policy = dict(prior_release['policy'], hold_steps=21, minimum_hold_duration_s=2.,
        maximum_reference_time_s=90., maximum_steps=900,
        temporal_protocol_source='existing GoalTruth defaults: hold_s=2.0, timeout_s=90.0; not chosen from method outcomes')
    release = dict(prior_release, version='independent-static-directed-lane-control-reference-v2',
        policy=policy, source_reference_run=args.input_reference,
        source_sha256={str(p): digest(p) for p in [Path(__file__),
            PROJECT/'conceptgraph/navigation/vehicle_geometry.py', PROJECT/'conceptgraph/completion/protocol.py']},
        input_sha256=dict(prior_release['input_sha256'], **{str(p): digest(p) for p in
            [source/'release.json', source/'summary.json', prior_audit/'report.json', prior_audit/'witness_checks.json']}),
        wall_budget_s=180., storage_budget_bytes=500000000,
        source_failed_witnesses_preserved=True, no_new_search_or_method_calls=True)
    dump(dest/'release.json', release)
    shutil.copy2(__file__, dest/'generator_source.py')
    results = []
    for index, row in enumerate(map(json.loads, (source/'targets.jsonl').open())):
        if time.monotonic()-started > 180 or shutil.disk_usage(dest).free < 6000000000:
            raise RuntimeError('finalizer budget; partial result preserved')
        result = dict(row, source_reference=row['reference'], reference=None, reachability='unknown')
        key = row['scene'], row['frame_idx'], row['instance_token']
        if row['reference']:
            original = source/row['reference']['path']
            if digest(original) != row['reference']['sha256']:
                raise ValueError('changed original witness')
            if key not in accepted:
                result['reason'] = 'source_independent_audit_rejected_witness'
            else:
                witness = json.loads(original.read_text())
                rows = witness['rows']
                # 旧10个合格状态只有0.9s跨度，追加制动至21个状态覆盖完整2.0s。
                for _ in range(policy['hold_steps']-prior_release['policy']['hold_steps']):
                    before = VehicleState(**rows[-1]['state'])
                    after = step_vehicle(before, -spec.max_braking_mps2, before.steering_rad)
                    rows.append(dict(state=asdict(after), control=dict(
                        acceleration_mps2=-spec.max_braking_mps2, steering_command_rad=before.steering_rad)))
                if rows[-1]['state']['time_s'] > policy['maximum_reference_time_s']+1e-8:
                    result['reason'] = 'reference_exceeds_existing_90s_horizon'
                else:
                    name = f'witnesses/target_{index:04d}.json'
                    dump(dest/name, dict(witness, rows=rows, source_witness_sha256=digest(original),
                        terminal_hold_duration_required_s=2., reference_horizon_s=90.))
                    result.update(reference=dict(path=name, sha256=digest(dest/name)),
                        reason=None, reachability='pending_independent_witness_audit')
        results.append(result)
    with (dest/'targets.jsonl').open('x') as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    if any(digest(name) != sha for name, sha in release['source_sha256'].items()):
        raise ValueError('finalizer source changed')
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    dump(dest/'summary.json', dict(target_records=len(results),
        proposed_witnesses=sum(r['reference'] is not None for r in results),
        status_counts=dict(Counter(r['reason'] or 'pending_independent_witness_audit' for r in results)),
        files_sha256=files, elapsed_s=time.monotonic()-started, independently_audited=False,
        source_failed_audit_preserved=True, actual_arrival_credit=False, full_perception_credit=False,
        model_calls=0, new_test_opened=False, goal_complete=False))


if __name__ == '__main__':
    main()
