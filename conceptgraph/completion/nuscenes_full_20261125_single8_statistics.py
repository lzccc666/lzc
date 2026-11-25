"""事前固定五比较、log配对区间、附属分组和异步scan信息；不改变预测或评分。"""
from __future__ import annotations
import argparse,gzip,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion.nuscenes_full_20261125_single8_context import DATA,REPORT,RUN
from conceptgraph.completion.nuscenes_full_20261125_run import METHODS
from conceptgraph.completion.nuscenes_full_20261125_single8_evaluate import SCOPES
COMPARISONS=[('original_fusion','single_frame','front'),('R1','original_fusion','front'),('S2_1','R1','front'),('S2_3','S2_1','union6'),('S2_6','S2_3','union6')]
FIELDS=['true_positive','truth','predictions','IDTP','GT_observations','prediction_observations','duplicate_nodes','error_sum_m']


def freeze():
    record=dict(created_utc=source.now(),seed=17,source_sha256=source.sha(__file__),bootstrap_resamples=10000,
        paired_cluster_unit='official collection log_token; same log draw for both methods and all metrics',
        primary_comparisons=[dict(method=a,baseline=b,scope=s,metric='recall') for a,b,s in COMPARISONS],
        primary_interval=dict(confidence=.99,method='percentile paired-cluster bootstrap',family='5 predeclared recall contrasts',Bonferroni_familywise_alpha=.05),
        secondary_intervals=dict(confidence=.95,metrics=['precision','F1','IDF1','duplicate_fraction','duplicate_nodes','center_error_m'],
            scope='all5contrasts separately within each front/front3/union6 universe; descriptive, no simultaneous significance claim'),
        groups=dict(category='all6classes; full-scope original matches, P/R/F1',distance='GT worldXY distance to actual CAM_FRONT sensor: [0,10),[10,30),[30,65),[65,infinity); GT-conditioned recall only',
            visibility='official visibility tokens2/3/4; GT-conditioned recall only; not an independently measured occlusion percentage'),
        measurement_confound='all methods share6camera readiness reference, per-camera own most-recent strictly-past<=100ms LiDAR; report exact unique scanIDs and additional scans relative CAM_FRONT each frame; camera count ablation is not pure identical-LiDAR experiment',
        applicability='all850/34149 requests retained. Certified absent source is input applicability failure; existing source but algorithm rejection/empty prediction is algorithm result. No post-success selection',
        pending_source_certification=True,method_source_preseal_sha256=source.sha(DATA/'phaseB_source_preseal_v1.json'),GT_in_method=False)
    source.dump(DATA/'statistics_protocol_v1.json',record)
    source.dump(DATA/'phaseB_statistics_addendum_v1.json',dict(created_utc=source.now(),statistics_protocol_sha256=source.sha(DATA/'statistics_protocol_v1.json'),
        source_sha256={str(source.ROOT/p):source.sha(source.ROOT/p) for p in ['conceptgraph/completion/nuscenes_full_20261125_statistics.py','scripts/nuscenes_full_20261125_statistics.py']},
        ex_ante=True,method_math_unchanged=True,whole_scope_denominators_unchanged=True))


def metrics(values):
    tp,gt,pred,idtp,ig,ip,dup,err=np.moveaxis(values,-1,0)
    def ratio(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b!=0)
    return dict(recall=ratio(tp,gt),precision=ratio(tp,pred),F1=ratio(2*tp,gt+pred),IDF1=ratio(2*idtp,ig+ip),duplicate_fraction=ratio(dup,pred),duplicate_nodes=dup,center_error_m=ratio(err,tp))


def run():
    protocol=json.loads((DATA/'statistics_protocol_v1.json').read_text());assert source.sha(__file__)==protocol['source_sha256']
    summary=json.loads((RUN/'evaluation/summary.json').read_text());assert summary['complete']
    manifest=json.loads((DATA/'manifest.json').read_text());logs=sorted({s['log_token'] for s in manifest['scenes']});rng=np.random.default_rng(17)
    draws=rng.integers(0,len(logs),size=(10000,len(logs)));counts={(r['log'],r['method'],r['scope']):r['counts'] for r in summary['bylog']};arrays={};results=[]
    for scope in SCOPES:
        for method in METHODS:
            arrays[method,scope]=np.asarray([[counts[log,method,scope].get(f,0) for f in FIELDS] for log in logs],dtype=float)
        for current,baseline,primary_scope in COMPARISONS:
            a,b=arrays[current,scope],arrays[baseline,scope];point_a,point_b=metrics(a.sum(0)),metrics(b.sum(0))
            boot_a,boot_b=metrics(a[draws].sum(1)),metrics(b[draws].sum(1))
            for metric in point_a:
                primary=metric=='recall' and scope==primary_scope;alpha=.01 if primary else .05;differences=boot_a[metric]-boot_b[metric];valid=differences[np.isfinite(differences)]
                ci=np.quantile(valid,[alpha/2,1-alpha/2]).tolist() if len(valid) else None
                estimate=float(point_a[metric]-point_b[metric]);estimate=estimate if np.isfinite(estimate) else None
                results.append(dict(method=current,baseline=baseline,scope=scope,metric=metric,primary_family_member=primary,
                    confidence=1-alpha,paired_delta=estimate,CI=ci,valid_resamples=len(valid),clusters=len(logs),simultaneous_claim='Bonferroni5recall family' if primary else 'descriptive only'))
    target=RUN/'statistics';target.mkdir(exist_ok=False);groups=defaultdict(Counter);scan_counts=Counter();frame_count=0
    with gzip.open(target/'scan_support.jsonl.gz','xt',compresslevel=6) as scanout:
        for scene in manifest['scenes']:
            frames={r['frame_idx']:r for r in source.rows(RUN/scene['scene']/'frames.jsonl.gz')}
            for evaluation in source.rows(RUN/'evaluation'/(scene['scene']+'.jsonl.gz')):
                frame=frames[evaluation['frame_idx']];views=frame['views'];base={views['CAM_FRONT']['lidar_token']}-{None};scans={}
                for method,cameras in [('single_frame',['CAM_FRONT']),('original_fusion',['CAM_FRONT']),('R1',['CAM_FRONT']),('S2_1',['CAM_FRONT']),('S2_3',SCOPES['front3']),('S2_6',SCOPES['union6'])]:
                    ids={views[c]['lidar_token'] for c in cameras}-{None};extra=ids-base
                    scans[method]=dict(scanIDs=sorted(ids),scan_count=len(ids),additional_vs_front=sorted(extra),has_additional_scan=bool(extra))
                    scan_counts[method+':frames']+=1;scan_counts[method+':extra_scan_frames']+=int(bool(extra));scan_counts[method+':total_unique_scan_per_frame']+=len(ids)
                scanout.write(json.dumps(dict(scene=scene['scene'],frame_idx=frame['frame_idx'],sample_token=frame['sample_token'],methods=scans,not_a_pure_identical_LiDAR_camera_ablation=True),separators=(',',':'))+'\n');frame_count+=1
                truth=evaluation['evaluation_only_official'];origin=np.asarray(views['CAM_FRONT']['camera_pose'])[:2,3]
                for scope,cameras in SCOPES.items():
                    gt=[o for o in truth if set(o['visible_cameras'])&set(cameras)]
                    for method,record in evaluation['result'][scope].items():
                        matched={m['truth_index'] for m in record['metrics']['matches']}
                        by_id={p['node_id']:p for p in frame['methods'][method]['predictions']}
                        for node_id in record['prediction_node_ids']:groups[method,scope,'category',by_id[node_id]['category']]['predictions']+=1
                        for j,obj in enumerate(gt):
                            distance=float(np.linalg.norm(np.asarray(obj['center'])[:2]-origin));band='0-10m' if distance<10 else '10-30m' if distance<30 else '30-65m' if distance<65 else '>=65m'
                            for group,value in [('category',obj['category']),('distance',band),('visibility',str(obj['visibility_token']))]:
                                c=groups[method,scope,group,value];c['truth']+=1;c['TP']+=int(j in matched)
    group_rows=[]
    for (method,scope,group,value),c in groups.items():
        row=dict(method=method,scope=scope,group=group,value=value,counts=dict(c),recall=c['TP']/c['truth'] if c['truth'] else None)
        if group=='category':row.update(precision=c['TP']/c['predictions'] if c['predictions'] else None,F1=2*c['TP']/(c['truth']+c['predictions']) if c['truth']+c['predictions'] else None)
        group_rows.append(row)
    source.dump(target/'summary.json',dict(complete=True,frames=frame_count,logs=len(logs),bootstrap=results,groups=group_rows,scan_counts=dict(scan_counts),
        statistics_protocol_sha256=source.sha(DATA/'statistics_protocol_v1.json'),input_evaluation_sha256=source.sha(RUN/'evaluation/summary.json'),
        limits=['CIs quantify within-local-dataset log variation; no official-full-unseen-data or causal-generalization claim','Five primary99percent recall intervals only; other intervals are descriptive','Camera setting may add distinct past LiDAR scans; per-frame scan ledger provided, no scan-equal counterfactual invented']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','run']);a=p.parse_args();freeze() if a.command=='freeze' else run()

if __name__=='__main__':main()
