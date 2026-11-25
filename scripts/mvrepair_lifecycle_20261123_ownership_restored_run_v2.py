"""O2同曝光实测LiDAR支持唯一归属；全40+新40均为回归。"""
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
from conceptgraph.slam.mvrepair_lifecycle_20261123_sameview import SameViewCanonicalMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_ownership_restored_v2 import RestoredUniqueSupportCanonicalMapper,RestoredUniqueSupportOriginalFrontMapper,allocate_unique_support
from scripts import multiview_20261123_run_graph as original


P='mvrepair_lifecycle_20261123_'


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def checks():
    a=dict(source_key=['front',1.,0],category='car',lidar_token='fixture_scan',selected_point_indices=[1,2,3],confidence=.5,actual_retained_points=3,point_provenance_status='fixture')
    b=dict(a,source_key=['front',1.,1],selected_point_indices=[2,3,4],confidence=.7)
    kept,audit=allocate_unique_support([a,b]);assert kept[tuple(a['source_key'])]=={1} and kept[tuple(b['source_key'])]=={2,3,4}
    for c in [dict(b,lidar_token='other_scan'),dict(b,source_key=['left',1.,1]),dict(b,source_key=['front',2.,1])]:
        _,q=allocate_unique_support([a,c]);assert q['collision_points']==0
    _,q=allocate_unique_support([a,dict(b,selected_point_indices=[4,5])]);assert q['collision_points']==0
    _,q=allocate_unique_support([a,dict(b,confidence=.5)]);assert all(x['owner_source_key']==a['source_key'] for x in q['point_collisions'])
    return dict(unique_original_support=True,independent_points_retained=True,separate_camera_scan_exposure=True,original_index_tiebreak=True,synthetic_interface_only=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--ground-truth',type=Path,required=True)
    ap.add_argument('--baseline-run',type=Path,required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--report-dir',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic()
    l4=PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_L4_v1/mvrepair_20261123_release.json'
    freeze=json.loads(l4.read_text());assert all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items())
    manifest=json.loads(args.manifest.read_text());assert len(manifest['scenes'])==2 and all(len(s['groups'])==20 for s in manifest['scenes'])
    run=experiment_directory('nuscenes','validation',args.run_id,create=True);args.report_dir.mkdir(parents=True)
    paths=[Path(__file__).resolve(),PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_ownership_restored_v2.py',PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_sameview.py']
    sources={}
    for p in paths:shutil.copy2(p,run/p.name);sources[str(p)]=digest(p)
    bound=checks()
    input_paths={str(args.manifest.resolve()),str(args.ground_truth.resolve()),str(l4.resolve())}
    for scene in manifest['scenes']:
        files=list((args.baseline_run/scene['scene']).glob('*predictions.jsonl.gz'));assert len(files)==1
        reference=list(map(json.loads,gzip.open(files[0],'rt')))
        assert len(reference)==len(scene['groups'])==20
        for g,r in zip(scene['groups'],reference):
            assert g['frame_idx']==r['frame_idx'] and g['sample_token']==r['sample_token']
        input_paths.add(str(files[0].resolve()))
        for g in scene['groups']:
            for v in g['views'].values():
                for k,value in v.items():
                    if isinstance(value,str) and Path(value).is_file():input_paths.add(str(Path(value).resolve()))
    all_input_sha256={p:digest(p) for p in sorted(input_paths)}
    assert all(digest(p)==r['sha256'] for p,r in freeze['source_sha256'].items())
    dependencies={p:r['sha256'] for p,r in freeze['source_sha256'].items()}
    dependencies[str(PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_sameview.py')]=digest(PROJECT/'conceptgraph/slam/mvrepair_lifecycle_20261123_sameview.py')
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        frozen_L4_release_sha256=digest(l4),all_input_sha256=all_input_sha256,dependency_sha256=dependencies,manifest_sha256=digest(args.manifest),baseline_run=str(args.baseline_run),
        settings=dict(owner='original detconf desc, detection_index asc',scope='samecamera exposure scan, actually retained original support',rerun='owner subset through original depth cluster and geometry',sameview_identity_merge=False),checks=bound,seed=17,GT_in_inference=False,
        role='O2=L4+unique original selected support ownership; no L5 identity merge; both datasets regression',
        methods=['O2_ownership_three','O2_ownership_front','A_front_ownership_only'],
        max_wall_s=1200,max_new_bytes=64*2**20,minimum_free_bytes=20*2**30))
    counts=Counter();witnesses=[]
    for scene in manifest['scenes']:
        name=scene['scene'];outdir=run/name;outdir.mkdir()
        files=list((args.baseline_run/name).glob('*predictions.jsonl.gz'));assert len(files)==1
        reference=list(map(json.loads,gzip.open(files[0],'rt')))
        assert len(reference)==len(scene['groups'])==20
        assert all(g['frame_idx']==r['frame_idx'] and g['sample_token']==r['sample_token'] for g,r in zip(scene['groups'],reference))
        mappers={'O2_ownership_three':RestoredUniqueSupportCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'O2_ownership_front':RestoredUniqueSupportCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'A_front_ownership_only':RestoredUniqueSupportOriginalFrontMapper(OmegaConf.create(freeze['config']),CLASSES),
                 'L4_audit':RestoredUniqueSupportCanonicalMapper(OmegaConf.create(freeze['config']),CLASSES,audit_only=True)}
        with gzip.open(outdir/(P+'predictions.jsonl.gz'),'wt') as out:
            for group,old in zip(scene['groups'],reference):
                assert time.monotonic()-start<1200 and shutil.disk_usage(PROJECT).free>20*2**30
                loaded={c:original.load_frame(group,c) for c in group['views']};pred=dict(old['predictions']);audits={}
                for method,mapper in mappers.items():
                    np.random.seed(17+group['frame_idx'])
                    cameras=['CAM_FRONT'] if method in ['O2_ownership_front','A_front_ownership_only'] else list(group['views'])
                    fs=[(c,copy.deepcopy(loaded[c])) for c in cameras]
                    result=mapper.update_group(fs,{c:group['views'][c]['camera_timestamp_s'] for c in cameras},
                        {c:group['views'][c] for c in cameras})
                    if method=='L4_audit':
                        assert result==old['predictions']['L4_three'];counts['L4_exact_groups']+=1;assert mapper.unique_support_audit['probe_and_second_pass_observations_exact'];continue
                    pred[method]=result;audits[method]=mapper.audit
                    a=mapper.unique_support_audit
                    for k in ['collision_points','lost_support_points','affected_sources','final_remaining_collision_points']:counts[method+'_'+k]+=a[k]
                    for w in a['source_decisions']:
                        if w['lost_indices']:witnesses.append(dict(scene=name,frame_idx=group['frame_idx'],method=method,**w))
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
