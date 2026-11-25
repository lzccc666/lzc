"""全40组跨类别物理假设诊断，或独立冻结L3候选；不改L1/L2。"""
from __future__ import annotations

import argparse
from collections import Counter
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
from conceptgraph.slam.mvrepair_lifecycle_20261123_crossclass import CrossClassPhysicalMapper,PhysicalHypothesisSettings,competing_hypothesis_witness
from scripts import multiview_20261123_run_graph as original


P='mvrepair_lifecycle_20261123_'
REPORT=PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'
BASE=PROJECT/'outputs/results/nuscenes/validation/iterations'
L1=BASE/(P+'candidate_L1_v1')


def boundary_checks():
    a=dict(source_key=['front',1.,0],category='car',box=[0.,0.,10.,10.],mask=np.ones((10,10),bool),
           real_points=[[0.,0.,0.],[.1,0.,0.]],feature=[1.,0.],center=np.zeros(3),confidence=.3)
    b=dict(a,source_key=['front',1.,1],category='truck')
    assert competing_hypothesis_witness(a,b)['accepted']
    for category in ['car','pedestrian','bicycle','motorcycle']:
        assert not competing_hypothesis_witness(a,dict(b,category=category))['accepted']
    assert not competing_hypothesis_witness(a,dict(b,real_points=[]))['accepted']
    assert not competing_hypothesis_witness(a,dict(b,source_key=['front',1.5,1]))['accepted']
    return dict(nonexclusive_vulnerable_classes_preserved=True,no_real_support_rejected=True,
        different_exposure_rejected=True,synthetic_interface_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args();started=time.monotonic()
    freeze=json.loads((L1/(P+'release.json')).read_text())
    assert all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items())
    manifest=json.loads((PROJECT/'dataset/processed/multiview_20261123_pilot_v1/manifest.json').read_text())
    run=experiment_directory('nuscenes','validation',args.run_id,create=True)
    report_dir=REPORT/args.run_id;report_dir.mkdir()
    paths=[Path(__file__).resolve(),PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_crossclass.py']
    sources={}
    for p in paths:
        shutil.copy2(p,run/p.name);sources[str(p)]=digest(p)
    checks=boundary_checks()
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),
        mode='L3_L1_plus_same_exposure_competing_vehicle_hypotheses' if args.candidate else 'diagnosis_only_exact_L1',
        source_sha256=sources,frozen_L1_release_sha256=digest(L1/(P+'release.json')),
        settings=asdict(PhysicalHypothesisSettings()),representative='highest real detector confidence; all class/source hypotheses retained',
        main_comparison='same105union/77front/28side GT and original3m; A/C/L1/L3 all full denominator',
        GT_in_inference=False,seed=17,no_TTL_or_default_change=True,interface_checks=checks,
        wall_budget_s=1200,new_storage_budget_bytes=128*2**20))
    counts=Counter();patterns=[]
    for scene in manifest['scenes']:
        name=scene['scene'];outdir=run/name;outdir.mkdir()
        reference=list(map(json.loads,gzip.open(L1/name/(P+'predictions.jsonl.gz'),'rt')))
        mapper=CrossClassPhysicalMapper(OmegaConf.create(freeze['original_config']),CLASSES,audit_only=not args.candidate)
        with gzip.open(outdir/(P+'predictions.jsonl.gz'),'wt') as out:
            for group,old in zip(scene['groups'],reference):
                if time.monotonic()-started>1200 or shutil.disk_usage(PROJECT).free<20*2**30:raise RuntimeError('预算/磁盘保护')
                np.random.seed(17+group['frame_idx'])
                frames=[(c,original.load_frame(group,c)) for c in group['views']]
                mapper.capture_times={c:group['views'][c]['camera_timestamp_s'] for c,_ in frames}
                mapper.update_group(frames)
                predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frames[0][1])
                if not args.candidate:assert predictions==old['predictions']['L1_lifecycle']
                physical=mapper.physical_audit
                accepted=[w for w in physical['pairs'] if w['accepted']]
                counts['groups']+=1;counts['groups_with_eligible_competition']+=bool(accepted)
                counts['eligible_pairs']+=len(accepted);counts['redundant_hypotheses']+=physical['removed']
                for pair in accepted:
                    patterns.append(dict(scene=name,frame_idx=group['frame_idx'],**pair))
                pred=dict(old['predictions'])
                if args.candidate:pred['L3_physical_hypotheses']=predictions
                out.write(json.dumps(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=pred,
                    audit=dict(physical=physical,lifecycle=mapper.lifecycle_audit),GT_in_inference=False),allow_nan=False)+'\n')
    result=None
    if args.candidate:
        original.DEST,original.P=report_dir,P
        truth=PROJECT/'dataset/processed/multiview_20261123_pilot_v1/multiview_20261123_evaluation_ground_truth.jsonl'
        result=original.evaluate(run,manifest,truth)
    report=dict(run=str(run),complete=True,audit_only=not args.candidate,counts=dict(counts),patterns=patterns,
        results=result,wall_s=time.monotonic()-started,source_sha256=sources,
        frozen_L1_sources_unchanged=all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items()),
        GT_in_inference=False,model_GPU_training_CARLA_calls=0)
    dump(run/(P+'report.json'),report);dump(report_dir/'result.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
