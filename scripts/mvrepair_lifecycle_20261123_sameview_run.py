"""L5同视支持包含：旧40和新40均用本入口，各自独立全分母回归。"""
from __future__ import annotations

import argparse
from collections import Counter
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

from conceptgraph.completion.stage9 import PROJECT,digest,dump
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.mvrepair_lifecycle_20261123_sameview import SameViewCanonicalMapper,SameViewOriginalFrontMapper,SameViewSettings,sameview_groups
from scripts import multiview_20261123_run_graph as original


P='mvrepair_lifecycle_20261123_'


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def checks():
    a=dict(source_key=['front',1.,0],category='car',lidar_token='real_scan_fixture',selected_point_indices=list(range(10)),
        center=np.zeros(3),feature=[1.,0.],box=[0.,0.,10.,10.],raw_mask=np.ones((10,10),bool),confidence=.5,path='dense',path_index=0)
    b=dict(a,source_key=['front',1.,1],selected_point_indices=list(range(20)))
    c=dict(a,source_key=['front',1.,2],selected_point_indices=list(range(10,20)))
    kept,audit=sameview_groups([a,b,c]);assert len(kept)==2
    assert any(not w['accepted'] and w['reason']=='insufficient_original_point_containment' for w in audit['pairs'])
    from conceptgraph.slam.mvrepair_lifecycle_20261123_sameview import sameview_witness
    assert not sameview_witness(a,dict(b,lidar_token='different_scan'))['accepted']
    assert not sameview_witness(a,dict(b,source_key=['front',1.5,1]))['accepted']
    assert not sameview_witness(a,dict(b,box=[11.,0.,21.,10.]))['accepted']
    return dict(complete_link_bridge_rejected=True,different_scan_and_exposure_rejected=True,
        independent_image_risk_rejected=True,synthetic_interface_checks_only=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--ground-truth',type=Path,required=True)
    ap.add_argument('--baseline-run',type=Path,required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--report-dir',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic()
    l4=PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_L4_v1/mvrepair_20261123_release.json'
    freeze=json.loads(l4.read_text());assert all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items())
    manifest=json.loads(args.manifest.read_text());assert len(manifest['scenes'])==2 and all(len(s['groups'])==20 for s in manifest['scenes'])
    run=experiment_directory('nuscenes','validation',args.run_id,create=True);args.report_dir.mkdir(parents=True)
    paths=[Path(__file__).resolve(),PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_sameview.py']
    sources={}
    for p in paths:shutil.copy2(p,run/p.name);sources[str(p)]=digest(p)
    bound=checks()
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        frozen_L4_release_sha256=digest(l4),manifest_sha256=digest(args.manifest),baseline_run=str(args.baseline_run),
        settings=asdict(SameViewSettings()),checks=bound,seed=17,GT_in_inference=False,
        role='L5=L4+sameview current original support containment; both datasets now regression; no new unseen-test claim',
        methods=['L5_sameview_three','L5_sameview_front','A_front_sameview_only'],
        max_wall_s=1200,max_new_bytes=64*2**20,minimum_free_bytes=20*2**30))
    counts=Counter();witnesses=[]
    for scene in manifest['scenes']:
        name=scene['scene'];outdir=run/name;outdir.mkdir()
        files=list((args.baseline_run/name).glob('*predictions.jsonl.gz'));assert len(files)==1
        reference=list(map(json.loads,gzip.open(files[0],'rt')))
        mappers={'L5_sameview_three':SameViewCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'L5_sameview_front':SameViewCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'A_front_sameview_only':SameViewOriginalFrontMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'L4_audit':SameViewCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES,audit_only=True)}
        with gzip.open(outdir/(P+'predictions.jsonl.gz'),'wt') as out:
            for group,old in zip(scene['groups'],reference):
                assert time.monotonic()-start<1200 and shutil.disk_usage(PROJECT).free>20*2**30
                loaded={c:original.load_frame(group,c) for c in group['views']};pred=dict(old['predictions']);audits={}
                for method,mapper in mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras=['CAM_FRONT'] if method in ['L5_sameview_front','A_front_sameview_only'] else list(group['views'])
                    fs=[(c,copy.deepcopy(loaded[c])) for c in cameras]
                    result=mapper.update_group(fs,{c:group['views'][c]['camera_timestamp_s'] for c in cameras},
                        {c:group['views'][c] for c in cameras})
                    if method=='L4_audit':
                        assert result==old['predictions']['L4_three'];counts['L4_exact_groups']+=1;continue
                    pred[method]=result;audits[method]=mapper.audit
                    a=mapper.sameview_audit
                    counts[method+'_removed_current']+=a['removed']
                    for w in a['pairs']:
                        if w['accepted']:witnesses.append(dict(scene=name,frame_idx=group['frame_idx'],method=method,**w))
                out.write(json.dumps(serial(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=pred,audit=audits,GT_in_inference=False)),allow_nan=False)+'\n')
    original.DEST,original.P=args.report_dir,P
    result=original.evaluate(run,manifest,args.ground_truth)
    report=dict(run=str(run),complete=True,counts=dict(counts),witnesses=witnesses,results=result,wall_s=time.monotonic()-start,
        GT_in_inference=False,frozen_L4_sources_unchanged=all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items()),
        model_GPU_training_CARLA_calls=0)
    dump(run/(P+'report.json'),report);dump(args.report_dir/'result.json',report)
    print(json.dumps(dict(run=str(run),counts=dict(counts),node_main=result['node_main'],side=result['side_coverage'],wall_s=report['wall_s']),ensure_ascii=False))


if __name__=='__main__':main()
