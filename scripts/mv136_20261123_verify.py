"""固定两协议120组完整性、执行源码/输入/GT与旧保护指纹审计。"""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import PROJECT, digest, dump

P = 'mv136_20261123_'
BASE = PROJECT/'outputs/results/nuscenes/validation/iterations'
DEST = PROJECT/'outputs/reports/multiview_viewcount_20261123/mapping'


def main():
    original = json.loads((BASE/'mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json').read_text())
    checks = []
    for name in [P+'main_v1', P+'sharedscan_v1']:
        run = BASE/name
        release = json.loads((run/(P+'release.json')).read_text())
        report = json.loads((run/(P+'report.json')).read_text())
        assert report['complete'] and report['groups'] == 120
        assert all(release['source_sha256'][path]['sha256'] == entry['sha256']
            for path,entry in original['source_sha256'].items())
        assert all(digest(Path(path)) == entry['sha256'] for path,entry in release['source_sha256'].items())
        assert all(digest(Path(path)) == value for path,value in release['input_sha256'].items())
        assert all(digest(Path(path)) == value for path,value in release['ground_truth_sha256'].items())
        keys = set()
        conditions = 0
        for scene in sorted(run.glob('scene-*')):
            pred = list(map(json.loads,gzip.open(scene/(P+'predictions.jsonl.gz'),'rt')))
            ev = list(map(json.loads,gzip.open(scene/(P+'evaluation.jsonl.gz'),'rt')))
            assert len(pred) == len(ev) == 20
            for a,b in zip(pred,ev):
                assert (a['scene'],a['frame_idx']) == (b['scene'],b['frame_idx'])
                assert set(a['predictions']) == {'A_front_context','C_3','C_6','T3_1','T3_3','T3_6'}
                keys.add((a['scene'],a['frame_idx']))
                conditions += len(a['predictions'])
        assert len(keys) == 120 and conditions == 720
        for scope,gt in [('union6',1181),('front3',687),('front',385)]:
            assert all(r['truth']==gt for r in report['results']['node_main'] if r['scope']==scope)
        checks.append(dict(run=str(run),groups=120,method_group_conditions=conditions,
            original_T3_dependency_SHA_equal=len(original['source_sha256']),
            source_count=len(release['source_sha256']),input_count=len(release['input_sha256']),
            all_frozen_SHA_unchanged=True))
    files = json.loads((PROJECT/'outputs/reports/multiview_identity_repair_20261123/main/mvrepair_20261123_start.json').read_text())['protected_sha256']
    assert all(digest(Path(path)) == value for path,value in files.items())
    a,b = [json.loads((PROJECT/'dataset/processed'/directory/'manifest.json').read_text())
        for directory in ['mv136_20261123_sixscene_v1','mv136_20261123_sharedscan_v1']]
    assert [s['scene'] for s in a['scenes']] == [s['scene'] for s in b['scenes']]
    for sa,sb in zip(a['scenes'],b['scenes']):
        assert len(sa['groups']) == len(sb['groups']) == 20
        for ga,gb in zip(sa['groups'],sb['groups']):
            assert (ga['frame_idx'],ga['sample_token'],ga['reference_timestamp_s']) == (gb['frame_idx'],gb['sample_token'],gb['reference_timestamp_s'])
            assert ga['views'].keys() == gb['views'].keys()
            for c,va in ga['views'].items():
                vb = gb['views'][c]
                for field in ['rgb','detections','camera_pose','intrinsic','camera_timestamp_s']:
                    assert va[field] == vb[field]
    own_bytes = sum(p.stat().st_size for run in [BASE/(P+'main_v1'),BASE/(P+'sharedscan_v1')]
        for p in run.rglob('*') if p.is_file())
    assert own_bytes < 256*2**20
    dump(DEST/(P+'final_verification.json'),dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,old_protected_SHA_equal=len(files),main_control_same_GT_images_features_pose_capture_clock=True,
        own_raw_run_bytes=own_bytes,own_budget_bytes=256*2**20,free_bytes=shutil.disk_usage(PROJECT).free,
        source_sha256=digest(Path(__file__)),no_method_retuning=True,default_changed=False,manuscript_changed=False))
    print(json.dumps(dict(complete=True,checks=checks,old_protected=len(files),own_bytes=own_bytes)))


if __name__ == '__main__':
    main()
