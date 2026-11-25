"""当前有界批次收尾：完整确定性复现、冻结来源与旧保护指纹。"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from conceptgraph.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
BASE = ROOT/'outputs/results/nuscenes/validation/iterations'
DEST = ROOT/'outputs/reports/multiview_identity_repair_20261123/main'
P = 'mvrepair_20261123_'


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4*2**20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def main():
    first, repeat = [BASE/(P+suffix) for suffix in ['L4_v1', 'L4_reproduction_v1']]
    counts = dict(groups=0, method_prediction_records=0, method_audit_records=0, evaluation_records=0)
    for directory in sorted(first.glob('scene-*')):
        initial = read(directory/(P+'predictions.jsonl.gz'))
        reproduced = read(repeat/directory.name/(P+'predictions.jsonl.gz'))
        assert len(initial) == len(reproduced) == 20
        for a, b in zip(initial, reproduced):
            assert (a['scene'], a['frame_idx'], a['sample_token']) == (b['scene'], b['frame_idx'], b['sample_token'])
            assert a['predictions'] == b['predictions']
            assert a['audit'] == b['audit']
            counts['groups'] += 1
            counts['method_prediction_records'] += len(a['predictions'])
            counts['method_audit_records'] += len(a['audit'])
        ev_a, ev_b = read(directory/(P+'evaluation.jsonl.gz')), read(repeat/directory.name/(P+'evaluation.jsonl.gz'))
        assert ev_a == ev_b
        counts['evaluation_records'] += len(ev_a)
    assert counts['groups'] == 40
    protected = json.loads((DEST/(P+'start.json')).read_text())
    failures = [p for p, value in protected['protected_sha256'].items() if not Path(p).is_file() or digest(p) != value]
    assert not failures, failures
    frozen_checks = []
    for run in [first, repeat, BASE/(P+'L2_v2'), BASE/(P+'U1_v1')]:
        release = json.loads((run/(P+'release.json')).read_text())
        sources = release['source_sha256']
        assert all(digest(path) == value['sha256'] for path, value in sources.items())
        assert all(digest(path) == value for path, value in release['input_sha256'].items())
        frozen_checks.append(dict(run=str(run), source_count=len(sources), input_count=len(release['input_sha256']), unchanged=True))
    candidate_report = json.loads((first/(P+'report.json')).read_text())
    assert candidate_report['baseline_exact'] == dict(A_front=40, C_three_dedup=40)
    results = candidate_report['results']['node_main']
    for scope, truth in [('front', 77), ('union', 105)]:
        assert all(r['truth'] == truth for r in results if r['scope'] == scope)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), exact_reproduction=counts,
        old_protected_files=len(protected['protected_sha256']), old_protected_SHA_unchanged=True,
        baseline_exact_full40=True, frozen_checks=frozen_checks, fixed_front77_union105=True,
        free_disk_bytes=shutil.disk_usage(ROOT).free,
        source_sha256=digest(__file__), no_algorithm_or_evaluation_change=True)
    (DEST/(P+'final_verification.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
