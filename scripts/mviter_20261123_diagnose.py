"""等价重放上轮A/C并保存来源、重复节点与前视丢失证据。"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from scripts.multiview_20261123_run_graph import load_frame


P = 'mviter_20261123_'
OLD = PROJECT/'outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1'
REPORT = PROJECT/'outputs/reports/multiview_task_iteration_20261123/mapping'


def main():
    start = time.monotonic()
    manifest = json.loads((PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json').read_text())
    release = json.loads((OLD/'multiview_20261123_release.json').read_text())
    run = experiment_directory('nuscenes', 'validation', P+'diagnosis_v1', create=True)
    dump(run/(P+'release.json'), dict(purpose='exact A/C replay with inference source provenance only',
        source_sha256={str(p): digest(p) for p in [Path(__file__).resolve(), PROJECT/'conceptgraph/slam/mviter_20261123_diagnostics.py']},
        input_release_sha256=digest(OLD/'multiview_20261123_release.json'), config=release['config'],
        seed=17, GT_in_inference=False, maximum_wall_s=600, maximum_new_bytes=32*2**20))
    duplicates = json.loads((REPORT/(P+'frozen_duplicate_cases.json')).read_text())
    diagnoses = []
    frames_checked = 0
    for scene in manifest['scenes']:
        name = scene['scene']
        reference = list(map(json.loads, gzip.open(OLD/name/'multiview_20261123_predictions.jsonl.gz', 'rt')))
        mappers = {m: SourceAuditedMapper(OmegaConf.create(release['config']), CLASSES,
            deduplication=m != 'A_front') for m in ['A_front', 'C_three_dedup']}
        with gzip.open(run/(P+name+'.jsonl.gz'), 'wt') as stream:
            for group, old in zip(scene['groups'], reference):
                loaded = {c: load_frame(group, c) for c in group['views']}
                actual = {}
                for method, mapper in mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras = release['methods'][method]['cameras']
                    mapper.capture_times = {c: group['views'][c]['camera_timestamp_s'] for c in cameras}
                    frames = [(c, copy.deepcopy(loaded[c])) for c in cameras]
                    mapper.update_group(frames)
                    predictions = mapper.position_predictions(predict(mapper.confirmed_objects(), CLASSES), frames[0][1])
                    assert predictions == old['predictions'][method]
                    assert mapper.frame_audit == old['audit'][method]['mapping']
                    evidence = {p['node_id']: mapper.evidence_for_prediction(p) for p in predictions}
                    actual[method] = dict(predictions=predictions, evidence=evidence, internal=mapper.internal_snapshot())
                    for case in duplicates:
                        if (case['scene'], case['frame_idx'], case['method']) != (name, group['frame_idx'], method):
                            continue
                        a = evidence[case['prediction']['node_id']]
                        mates = []
                        for mate in case['matched_predictions']:
                            b = evidence[mate['node_id']]
                            aa, bb = np.asarray(a['world_points']), np.asarray(b['world_points'])
                            overlap = None
                            if len(aa) and len(bb):
                                small, large = (aa, bb) if len(aa) < len(bb) else (bb, aa)
                                overlap = float(np.mean(cKDTree(large).query(small)[0] <= .2))
                            mates.append(dict(prediction=mate, evidence=b, shared_support_0_2m=overlap))
                        diagnoses.append(dict(case=case, evidence=a, matched=mates))
                stream.write(json.dumps(dict(scene=name, frame_idx=group['frame_idx'],
                    reference_timestamp_s=group['reference_timestamp_s'], methods=actual), allow_nan=False)+'\n')
                frames_checked += 1
    dump(REPORT/(P+'duplicate_provenance.json'), diagnoses)
    dump(REPORT/(P+'diagnosis_result.json'), dict(exact_groups_per_method=frames_checked,
        methods=['A_front', 'C_three_dedup'], run=str(run), runtime_s=time.monotonic()-start,
        duplicates=len(diagnoses), output_sha256={str(p): digest(p) for p in run.glob('*.gz')}))
    print(dict(exact_groups=frames_checked, duplicates=len(diagnoses), runtime_s=time.monotonic()-start))


if __name__ == '__main__':
    main()
