"""Bounded CPU audit of new semantic tasks and validation-only repairs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'outputs/reports/navfollowup_20261124/independent'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def baseline():
    REPORT.mkdir(parents=True,exist_ok=True)
    old=read(ROOT/'outputs/reports/navbench_stage1_20261124/independent/initial_protection_and_budget.json')
    paths={Path(row['path']) for row in old['protected']}
    package=ROOT/'dataset/processed/navbench_stage1_20261124_v1'
    paths.update(package/p for p in ['manifest.json','preregistration.json','task_registry.jsonl','denominator_freeze.json','effective_amendment_v3.json','task_record.schema_v2.json','evaluation_only/certificates.jsonl','evaluation_only/conditional_reference_labels.jsonl','source_binding_overlay_v1.jsonl'])
    paths.update(ROOT/p for p in ['conceptgraph/navigation/navstage2_20261124.py','conceptgraph/navigation/navstage2_runner_20261124.py','scripts/navstage2_20261124.py'])
    for folder in ['dataset/processed/navstage2_20261124_v1','dataset/processed/navstage2_20261124_v2','outputs/results/nuscenes/test/iterations/navstage2_20261124_v1','outputs/results/nuscenes/test/iterations/navstage2_20261124_v2']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    protected=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)]
    old_changed=[r['path'] for r in old['protected'] if sha(r['path'])!=r['sha256']]
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),wall_budget_s=1800,new_file_budget_bytes=64*1024**2,minimum_disk_free_bytes=20*1024**3,GPU_calls=0,model_calls=0,planner_calls=0,prior_history_tests_rerun=False,protected=protected,old_protection_changes=old_changed,disk_free_bytes=shutil.disk_usage(ROOT).free)
    write(REPORT/'baseline_v1.json',value)
    protocol=dict(
        inference_boundary='Only instruction and measurements/predictions with timestamp <= issue time, known initial vehicle pose, and declared common HD map. No future sensor/target/ego states, official instance IDs, accepted target set, evaluator boxes or reference paths.',
        evaluation_boundary='Future official object/ego trajectories may support offline timed-follow evaluation and independent existence certificates; retain their identities/times and never export them to the method input.',
        certificate='Bind action, issue time, original target/region semantics, initial pose/VehicleSpec, path-vs-timed scope and independent witness. A recorded ego future is a witness only after passing this exact action contract and geometry; method paths need not imitate it.',
        proof='Finite search failure or missing evidence => pending. Proven infeasible requires independently verified complete-domain contradiction/proof. No parking slot or specified roadside region => pending for those tasks.',
        repair_split='Use fixed validation/debug tasks for diagnosis and implementation selection. Historical81 is separate readonly explanation; do not use it to choose thresholds, fixes or method branches. Record scene/log intersections before repair.',
        counting='Mutually exclusive task stages sum to the full fixed denominator; first-gate reasons, all failing gates, and node-record totals are distinct. 28 noncandidate tasks cannot all be called static failures from 49 node records.',
        user_input='Ask only about an actual ambiguous visible target or the intended task region/permission semantics; do not ask the user to invent geometry or termination phrases. Other official binding, evidence checks and deterministic labels proceed automatically.',
        legacy_tests_rerun=False)
    write(REPORT/'audit_protocol_v1.json',protocol)
    print(json.dumps(dict(protected=len(protected),bytes=sum(p['bytes'] for p in protected),old_changed=old_changed,report=str(REPORT)),ensure_ascii=False))


def protection():
    initial=read(REPORT/'baseline_v1.json')
    return dict(checked=len(initial['protected']),unchanged=all(Path(r['path']).is_file() and sha(r['path'])==r['sha256'] for r in initial['protected']),changed=[r['path'] for r in initial['protected'] if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['baseline','protection']);args=parser.parse_args()
    if args.action=='baseline':baseline()
    else:print(json.dumps(protection(),ensure_ascii=False))


if __name__=='__main__':main()
