"""三批收尾：独立旧保护与T3冻结输入来源，复用已完成80组复跑证据。"""
from datetime import datetime, timezone
import json
import shutil
from scripts.mvrepair_20261123_verify_v2 import ROOT, BASE, DEST, P, sha, rows


def main():
    previous = DEST/(P+'final_verification_v2.json')
    checks, total_groups, union_gt, front_gt = [], 0, 0, 0
    for name in [P+'T3_pilot_v2', P+'T3_crosslog_v2', 'mvrepair_T3_nextlogs_confirmation_20261123_v1']:
        run = BASE/name
        release = json.loads((run/(P+'release.json')).read_text())
        report = json.loads((run/(P+'report.json')).read_text())
        assert report['complete'] and report['groups'] == 40
        assert not release['GT_in_inference'] and not release['default_changed']
        for path, entry in release['source_sha256'].items():
            assert sha(path) == entry['sha256']
        assert all(sha(path) == digest for path,digest in release['input_sha256'].items())
        assert all(sha(path) == digest for path,digest in release['ground_truth_sha256'].items())
        scenes = list(run.glob('scene-*'))
        assert len(scenes) == 2
        for scene in scenes:
            a = rows(scene/(P+'predictions.jsonl.gz'))
            b = rows(scene/(P+'evaluation.jsonl.gz'))
            assert len(a) == len(b) == 20
            assert len({x['frame_idx'] for x in a}) == 20
            assert all((x['scene'],x['frame_idx']) == (y['scene'],y['frame_idx']) for x,y in zip(a,b))
        total_groups += 40
        for scope in ['front','union']:
            r = next(r for r in report['results']['node_main'] if r['method']=='T3_three' and r['scope']==scope)
            if scope == 'front':
                front_gt += r['truth']
            else:
                union_gt += r['truth']
        checks.append(dict(run=str(run), source_count=len(release['source_sha256']), input_count=len(release['input_sha256']), frozen_unchanged=True))
    assert (total_groups,union_gt,front_gt) == (120,687,385)
    start = json.loads((DEST/(P+'start.json')).read_text())
    assert all(sha(path) == value for path,value in start['protected_sha256'].items())
    own_paths = [p for run in BASE.glob(P+'*') for p in run.rglob('*') if p.is_file()]
    own_bytes = sum(p.stat().st_size for p in own_paths)
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(),groups=120,union_GT=687,front_GT=385,
        frozen_checks=checks,old_protected_count=len(start['protected_sha256']),old_protected_SHA_unchanged=True,
        prior_full80_exact_reproduction=dict(path=str(previous),sha256=sha(previous)),
        third_batch_one_fixed_complete_run_no_reselection=True,
        own_mapping_run_bytes=own_bytes,own_mapping_run_bytes_scope='all mvrepair_20261123_* raw runs including preserved failures/repeats; excludes parallel frontend and sibling lifecycle runs',
        free_disk_bytes=shutil.disk_usage(ROOT).free,source_sha256=sha(__file__),default_changed=False,manuscript_changed=False)
    path = DEST/(P+'final_verification_v3.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    index_path = DEST/(P+'file_index_v3.json')
    index = json.loads(index_path.read_text())
    index['final_three_batch_protection_audit'] = str(path)
    index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
