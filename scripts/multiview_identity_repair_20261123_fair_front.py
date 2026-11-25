"""任意冻结manifest的公平A+同物理清理；原A旁路要求逐字段完全相同。"""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mvrepair_lifecycle_20261123_frontphysical import FrontPhysicalOnlyMapper
from scripts import multiview_20261123_run_graph as original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'ground-truth', 'baseline-run', 'report-dir']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    baseline = PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_L4_v1/mvrepair_20261123_release.json'
    release = json.loads(baseline.read_text())
    assert all(digest(p) == r['sha256'] for p,r in release['source_sha256'].items())
    expected = {'conceptgraph/slam/mvrepair_lifecycle_20261123_frontphysical.py':'0ef09f34a1d0f3f8d14ab653194539633c513f463de79f95021373eacde678e4',
        'conceptgraph/slam/mvrepair_lifecycle_20261123_crossclass.py':'61545917d82ee995296b84dc4ef00dcdec95e8716d8b0fc2ee3f54c9213e2935'}
    assert all(digest(PROJECT/p) == sha for p,sha in expected.items())
    run = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    source = [Path(__file__).resolve(), *[PROJECT/p for p in expected]]
    source_sha = {}
    for p in source:
        snapshot = run/'source_snapshot'/p.relative_to(PROJECT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,snapshot)
        source_sha[str(p)] = dict(sha256=digest(p), snapshot=str(snapshot))
    manifest = json.loads(args.manifest.read_text())
    dump(run/'mvrepair_fair_release.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
        manifest_sha256=digest(args.manifest), seed=17, reference_run=str(args.baseline_run),
        frozen_dependency_release_sha256=digest(baseline), GT_in_inference=False, method='A+same frozen physical cleanup only',
        max_new_bytes=64*2**20, minimum_free_bytes=20*2**30, model_calls=0))
    exact = 0
    for scene in manifest['scenes']:
        name = scene['scene']
        reference_path, = (args.baseline_run/name).glob('*predictions*.jsonl.gz')
        reference = list(map(json.loads,gzip.open(reference_path,'rt')))
        assert len(reference) == len(scene['groups'])
        audit = FrontPhysicalOnlyMapper(OmegaConf.create(release['config']),CLASSES,audit_only=True)
        candidate = FrontPhysicalOnlyMapper(OmegaConf.create(release['config']),CLASSES)
        destination = run/name
        destination.mkdir()
        with gzip.open(destination/'mvrepair_fair_predictions.jsonl.gz','wt') as out:
            for group, old in zip(scene['groups'], reference):
                assert shutil.disk_usage(PROJECT).free > 20*2**30
                loaded = original.load_frame(group,'CAM_FRONT')
                result = {}
                for method, mapper in [('A_audit',audit),('A_front_physical_only',candidate)]:
                    np.random.seed(17+group['frame_idx'])
                    mapper.capture_times = {'CAM_FRONT':group['views']['CAM_FRONT']['camera_timestamp_s']}
                    frame = copy.deepcopy(loaded)
                    mapper.update_group([('CAM_FRONT',frame)])
                    result[method] = mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
                assert result['A_audit'] == old['predictions']['A_front']
                exact += 1
                pred = dict(old['predictions'], A_front_physical_only=result['A_front_physical_only'])
                out.write(json.dumps(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=pred,
                    audit={'A_front_physical_only':candidate.physical_audit},GT_in_inference=False),allow_nan=False)+'\n')
    original.DEST, original.P = args.report_dir, 'mvrepair_fair_'
    result = original.evaluate(run,manifest,args.ground_truth)
    record = dict(complete=True, run=str(run), A_exact_reproduced_groups=exact, results=result, model_calls=0, GT_in_inference=False)
    dump(run/'mvrepair_fair_report.json',record)
    dump(args.report_dir/'result.json',record)
    print(json.dumps(dict(run=str(run),complete=True,A_exact_groups=exact)))


if __name__ == '__main__':
    main()
