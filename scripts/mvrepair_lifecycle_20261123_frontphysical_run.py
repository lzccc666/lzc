"""原A+同物理假设清理的完整40组公平消融，原A旁路精确复现。"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.stage9 import PROJECT,digest,dump,predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mvrepair_lifecycle_20261123_frontphysical import FrontPhysicalOnlyMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_crossclass import PhysicalHypothesisSettings
from scripts import multiview_20261123_run_graph as original


P='mvrepair_lifecycle_20261123_'
REPORT=PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'
BASE=PROJECT/'outputs/results/nuscenes/validation/iterations'
L1=BASE/(P+'candidate_L1_v1');L3=BASE/(P+'candidate_L3_v1')


def main():
    started=time.monotonic()
    freeze=json.loads((L1/(P+'release.json')).read_text());physical=json.loads((L3/(P+'release.json')).read_text())
    assert all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items())
    assert all(digest(p)==sha for p,sha in physical['source_sha256'].items())
    manifest=json.loads((PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json').read_text())
    run=experiment_directory('nuscenes','validation',P+'A_physical_only_v1',create=True)
    dest=REPORT/(P+'A_physical_only_v1');dest.mkdir()
    paths=[Path(__file__).resolve(),PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_frontphysical.py']
    source_sha={}
    for p in paths:shutil.copy2(p,run/p.name);source_sha[str(p)]=digest(p)
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),
        mechanism='original A_front + identical frozen L3 physical hypothesis cleanup only; no L1/lifecycle/L2/front auxiliary',
        seed=17,settings=asdict(PhysicalHypothesisSettings()),source_sha256=source_sha,
        frozen_L1_release_sha256=digest(L1/(P+'release.json')),frozen_L3_release_sha256=digest(L3/(P+'release.json')),
        scope='same40groups105union77front28side original3m',GT_in_inference=False,
        audit_only_A_exact_required_groups=40,parameters_changed_after_L3=0,wall_budget_s=1200,new_bytes_budget=64*2**20))
    exact=0;eligible=removed=0
    for scene in manifest['scenes']:
        name=scene['scene'];outdir=run/name;outdir.mkdir()
        references=list(map(json.loads,gzip.open(L3/name/(P+'predictions.jsonl.gz'),'rt')))
        audit=FrontPhysicalOnlyMapper(OmegaConf.create(freeze['original_config']),CLASSES,audit_only=True)
        candidate=FrontPhysicalOnlyMapper(OmegaConf.create(freeze['original_config']),CLASSES)
        with gzip.open(outdir/(P+'predictions.jsonl.gz'),'wt') as out:
            for group,old in zip(scene['groups'],references):
                assert time.monotonic()-started<1200 and shutil.disk_usage(PROJECT).free>20*2**30
                loaded=original.load_frame(group,'CAM_FRONT')
                result={}
                for key,mapper in [('A_audit',audit),('A_front_physical_only',candidate)]:
                    np.random.seed(17+group['frame_idx'])
                    mapper.capture_times={'CAM_FRONT':group['views']['CAM_FRONT']['camera_timestamp_s']}
                    frames=[('CAM_FRONT',copy.deepcopy(loaded))]
                    mapper.update_group(frames)
                    result[key]=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frames[0][1])
                assert result['A_audit']==old['predictions']['A_front'];exact+=1
                eligible+=sum(w['accepted'] for w in candidate.physical_audit['pairs']);removed+=candidate.physical_audit['removed']
                preds=dict(old['predictions']);preds['A_front_physical_only']=result['A_front_physical_only']
                out.write(json.dumps(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=preds,
                    audit={'A_front_physical_only':candidate.physical_audit},GT_in_inference=False),allow_nan=False)+'\n')
    original.DEST,original.P=dest,P
    truth=PROJECT/'dataset/processed/multiview_20261123_pilot_v1/multiview_20261123_evaluation_ground_truth.jsonl'
    result=original.evaluate(run,manifest,truth)
    report=dict(run=str(run),complete=True,results=result,A_exact_reproduced_groups=exact,
        eligible_competition_pairs=eligible,removed_current_hypotheses=removed,wall_s=time.monotonic()-started,
        frozen_L1_L3_sources_unchanged=True,GT_in_inference=False,model_GPU_training_CARLA_calls=0)
    dump(run/(P+'report.json'),report);dump(dest/'result.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
