"""Freeze a bounded three-view mapping iteration; no inference or GT relabeling."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import PROJECT, digest, dump


def main():
    source = PROJECT/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json'
    gt = PROJECT/'dataset/processed/mv136_20261123_sixscene_v1/mv136_20261123_evaluation_ground_truth.jsonl'
    manifest = json.loads(source.read_text())
    dest = PROJECT/'dataset/processed/mapnext_20261124_inputs_v1'
    dest.mkdir(exist_ok=False)
    all_gt = list(map(json.loads, gt.open()))
    scopes = {'all': None, 'validation': ['scene-0012','scene-0013','scene-0019','scene-0035'],
              'regression': ['scene-0042','scene-0047']}
    records = {}
    for name, scenes in scopes.items():
        data = copy.deepcopy(manifest)
        data['scenes'] = [s for s in data['scenes'] if scenes is None or s['scene'] in scenes]
        data['iteration_role'] = 'previously_seen_validation_or_regression_no_new_test'
        data['primary_scope'] = 'front3 same official union, no GT in inference'
        rows = [r for r in all_gt if scenes is None or r['scene'] in scenes]
        mp, gp = dest/(name+'_manifest.json'), dest/(name+'_GT.jsonl')
        dump(mp, data)
        gp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        records[name] = dict(manifest=str(mp), manifest_sha256=digest(mp), ground_truth=str(gp),
            GT_sha256=digest(gp), groups=len(rows), scenes=[s['scene'] for s in data['scenes']])
    frozen = PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json'
    release = json.loads(frozen.read_text())
    protected = {p:v['sha256'] for p,v in release['source_sha256'].items()}
    for p in ['AGENTS.md','MEMORY.md','PAPER_MEMORY.md','paper/当前版_v4/sources/论文正文.md']:
        protected[str(PROJECT/p)] = digest(PROJECT/p)
    assert all(digest(Path(p)) == sha for p,sha in protected.items())
    report = PROJECT/'outputs/reports/mapping_iteration_20261124'
    report.mkdir(exist_ok=True)
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(), deadline_utc='2026-11-24T08:08:00+00:00',
        maximum_new_bytes=2**30, minimum_free_bytes=20*2**30, GPU_calls=0, seed=17,
        source_manifest_sha256={str(source):digest(source),str(gt):digest(gt)}, inputs=records,
        protected_sha256=protected, baseline='fixed T3v2, common max6 exposure clock, only front3 images consumed',
        evaluation='same-class horizontal XY one-to-one 3m; primary front3 GT687; strictfront GT385; all missing frames kept',
        candidate_selection='only registered validation80; freeze selected candidate before regression40; all scenes previously seen',
        candidate_limit='at most two small mechanisms; failed configurations retained',
        GT_use='evaluation and failure taxonomy only, never inference', default_changed=False,
        free_bytes=shutil.disk_usage(PROJECT).free)
    dump(report/'mapnext_20261124_registration.json', record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
