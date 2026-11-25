"""T3 两批完整确定性、冻结输入源码及本轮旧保护指纹核验。"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from conceptgraph.project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
BASE = ROOT / 'outputs/results/nuscenes/validation/iterations'
DEST = ROOT / 'outputs/reports/multiview_identity_repair_20261123/main'
P = 'mvrepair_20261123_'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def main():
    reproductions, frozen = [], []
    for batch in ['pilot', 'crosslog']:
        first = BASE / (P + 'T3_' + batch + '_v2')
        repeat = BASE / (P + 'T3_' + batch + '_reproduction_v2')
        count = dict(groups=0, prediction_method_records=0, audit_method_records=0, evaluation_rows=0)
        for scene in sorted(first.glob('scene-*')):
            a, b = [rows(r / scene.name / (P + 'predictions.jsonl.gz')) for r in [first, repeat]]
            assert len(a) == len(b) == 20
            for x, y in zip(a, b):
                assert (x['scene'], x['frame_idx'], x['sample_token']) == (y['scene'], y['frame_idx'], y['sample_token'])
                assert x['predictions'] == y['predictions']
                assert x['audit'] == y['audit']
                count['groups'] += 1
                count['prediction_method_records'] += len(x['predictions'])
                count['audit_method_records'] += len(x['audit'])
            a, b = [rows(r / scene.name / (P + 'evaluation.jsonl.gz')) for r in [first, repeat]]
            assert a == b
            count['evaluation_rows'] += len(a)
        assert count['groups'] == 40
        reproductions.append(dict(batch=batch, first=str(first), repeat=str(repeat), exact=True, **count))
        for run in [first, repeat]:
            release = json.loads((run / (P + 'release.json')).read_text())
            sources = release['source_sha256']
            assert all(sha(path) == value['sha256'] for path, value in sources.items())
            assert all(sha(path) == value for path, value in release['input_sha256'].items())
            frozen.append(dict(run=str(run), source_count=len(sources), input_count=len(release['input_sha256']), unchanged=True))
    start = json.loads((DEST / (P + 'start.json')).read_text())
    failures = [p for p, digest in start['protected_sha256'].items() if not Path(p).is_file() or sha(p) != digest]
    assert not failures, failures
    integration = json.loads((DEST / 'T3' / (P + 'representative_integration_audit.json')).read_text())
    assert sorted(c['actual_points'] for c in integration['full_candidates']['candidates']) == [15, 21]
    assert len(integration['unique_output']) == 1
    assert [r['outputs'][0]['hits'] for r in integration['hits']] == [3, 4, 5, 6]
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), full_reproductions=reproductions,
        old_protected_count=len(start['protected_sha256']), old_protected_SHA_unchanged=True,
        frozen_checks=frozen, actual_representative_integration_passed=True,
        source_sha256=sha(__file__), free_disk_bytes=shutil.disk_usage(ROOT).free,
        interpretation='T3 v2 valid implementation; T3 v1 retained as invalid MRO implementation contrast. No default or manuscript changed.')
    (DEST / (P + 'final_verification_v2.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
