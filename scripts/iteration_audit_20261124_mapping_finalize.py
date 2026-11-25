"""Verify provenance and per-log effects after the selected saved-prediction rescore."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from conceptgraph.project_paths import PROJECT_ROOT as ROOT

OUT=ROOT/'outputs/reports/iteration_independent_audit_20261124'
RUNS=ROOT/'outputs/results/nuscenes/validation/iterations'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]


def main():
    initial=load(OUT/'initial_input_freeze.json')
    assert all(sha(p)==h for p,h in initial['protected_sha256'].items())
    freeze=load(ROOT/'outputs/reports/mapping_iteration_20261124/mapnext_20261124_selection_freeze.json')
    assert freeze['selected']=='S2' and all(sha(p)==h for p,h in freeze['source_sha256'].items())
    release=load(RUNS/'mapnext_20261124_S2_full_v1/mapnext_20261124_release.json')
    assert sha(release['selection_freeze'])==release['selection_freeze_sha256']
    assert all(sha(p)==h for p,h in release['input_sha256'].items())
    assert all(sha(p)==h for p,h in release['source_reports'].items())
    copies=[];exact_reference=0;regression_exact=0;identical_old_nodes=0;extra_nodes=0
    for scene,p in release['predictions_provenance'].items():
        full=RUNS/'mapnext_20261124_S2_full_v1'/scene/'mapnext_20261124_predictions.jsonl.gz'
        assert sha(full)==sha(p['source'])==p['sha256']
        copies.append(dict(scene=scene,sha256=p['sha256'],unchanged_byte_copy=True))
        baseline=gz(RUNS/'mapnext_20261124_baseline_v1'/scene/'mapnext_20261124_predictions.jsonl.gz')
        old=gz(RUNS/'mv136_20261123_main_v1'/scene/'mv136_20261123_predictions.jsonl.gz')
        candidate=gz(full)
        assert len(baseline)==len(old)==len(candidate)==20
        for b,o,c in zip(baseline,old,candidate):
            assert b['frame_idx']==o['frame_idx']==c['frame_idx']
            for new,previous in [('T3_three','T3_3'),('T3_front','T3_1')]:
                assert b['predictions'][new]==o['predictions'][previous];exact_reference+=1
            bp=b['predictions']['T3_three'];cp=c['predictions']['S2_three']
            old_by={p['node_id']:p for p in bp};new_by={p['node_id']:p for p in cp}
            assert all(new_by.get(i)==p for i,p in old_by.items())
            identical_old_nodes+=len(bp);extra_nodes+=len(cp)-len(bp)
            if scene in ['scene-0042','scene-0047']:
                assert bp==cp;regression_exact+=1
    assert exact_reference==240 and regression_exact==40 and extra_nodes==1
    independent=load(RUNS/'iteration_independent_audit_20261124_S2_all_v1/scores.json')
    production=load(ROOT/'outputs/reports/mapping_iteration_20261124/S2_full_v1/mapnext_20261124_result.json')
    agree=[]
    rename={'TP':'true_positive','GT':'truth','duplicates':'duplicate_nodes','predictions':'predictions','IDTP':'IDTP','IDFP':'IDFP','IDFN':'IDFN','precision':'precision','recall':'recall','F1':'F1','IDF1':'IDF1'}
    for r in independent['summaries']:
        if r['method']!='S2:S2_three':continue
        p=next(x for x in production['results']['node_main'] if x['method']=='S2_three' and x['scope']==r['scope'])
        assert all(abs(r[k]-p[v])<1e-12 for k,v in rename.items())
        assert r['GT']=={'front':385,'front3':687,'union6':1181}[r['scope']]
        agree.append(r['scope'])
    counts=defaultdict(Counter)
    keys=['TP','GT','predictions','duplicates','IDTP','IDFP','IDFN','error_sum_m','adjacent_matched_transitions','adjacent_ID_changes']
    with (OUT/'S2_all_v1_scene.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['method'] not in ['baseline:T3_three','S2:S2_three']:continue
            counts[r['log_token'],r['scope'],r['method']].update({k:float(r[k]) for k in keys})
    comparisons=[]
    for log,scope,method in sorted(counts):
        if method!='baseline:T3_three':continue
        a=counts[log,scope,method];b=counts[log,scope,'S2:S2_three']
        def metrics(c):return dict(precision=c['TP']/c['predictions'] if c['predictions'] else None,recall=c['TP']/c['GT'] if c['GT'] else None,F1=2*c['TP']/(c['GT']+c['predictions']),IDF1=2*c['IDTP']/(c['GT']+c['predictions']),mean_center_error_m=c['error_sum_m']/c['TP'] if c['TP'] else None)
        am,bm=metrics(a),metrics(b)
        comparisons.append(dict(log_token=log,scope=scope,baseline=dict(a,**am),candidate=dict(b,**bm),negative_metrics=[k for k in ['precision','recall','F1','IDF1'] if am[k] is not None and bm[k]<am[k]-1e-12]))
    assert not any(r['negative_metrics'] for r in comparisons)
    shared=independent['paired_position_summary']
    for scope in ['front','front3','union6']:
        a=next(r for r in shared if r['scope']==scope and r['method']=='baseline:T3_three')
        b=next(r for r in shared if r['scope']==scope and r['method']=='S2:S2_three')
        assert {k:v for k,v in a.items() if k!='method'}=={k:v for k,v in b.items() if k!='method'}
    losses=[r for r in independent['changes'] if r['candidate']=='S2:S2_three']
    assert all(r['gained_GT_frames']==1 and r['lost_GT_frames']==0 for r in losses)
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),production_score_agreement=agree,
        protected_unchanged=True,selected_sources_unchanged=True,GT_and_manifest_unchanged=True,baseline_exact_historical_conditions=exact_reference,
        full_copy_provenance=copies,regression_groups_exact_baseline=regression_exact,old_prediction_nodes_exact=identical_old_nodes,new_prediction_nodes=extra_nodes,
        per_log=comparisons,common_GT_position_and_jitter_unchanged=True,gain_loss=losses,
        limits=['Gain is one object-frame, not a general recall or stability breakthrough.','Duplicate counts remain 11 front3/3 strictfront; rate decreases slightly only because denominator gains one.','All scenes/logs were previously seen; regression is not an unseen test.','119 near-observation misses in posterior taxonomy are not 119 guaranteed recoverable objects.'],
        source_sha256=sha(__file__),scorer_sha256=sha(ROOT/'scripts/iteration_audit_20261124_mapping_scores.py'),model_calls=0,GPU_calls=0,new_mapping_calls=0)
    p=OUT/'mapping_independent_final_v1.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,report=str(p),baseline_exact_conditions=exact_reference,regression_exact=regression_exact,per_log_negative=0,new_prediction_nodes=extra_nodes)))


if __name__=='__main__':main()
