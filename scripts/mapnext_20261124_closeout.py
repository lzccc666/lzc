"""Close the bounded round: score tables, source checks, exact preservation, budgets."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import PROJECT,digest,dump
from scripts.multiview_20261123_run_graph import write_csv

P='mapnext_20261124_'


def main():
    root=PROJECT/'outputs/reports/mapping_iteration_20261124'
    registration=json.loads((root/(P+'registration.json')).read_text())
    freeze=json.loads((root/(P+'selection_freeze.json')).read_text())
    for p,sha in registration['protected_sha256'].items():assert digest(p)==sha
    for p,sha in registration['source_manifest_sha256'].items():assert digest(p)==sha
    for p,sha in freeze['source_sha256'].items():assert digest(p)==sha
    runs=sorted((PROJECT/'outputs/results/nuscenes/validation/iterations').glob(P+'*'))
    reports=[]
    for run in runs:
        report=json.loads((run/(P+'report.json')).read_text())
        release=json.loads((run/(P+'release.json')).read_text())
        if not release.get('aggregation_only'):
            for p,v in report['source_sha256'].items():
                assert digest(p)==v['sha256']==digest(v['snapshot'])
            for p,sha in report['input_sha256'].items():assert digest(p)==sha
        reports.append(dict(run=run.name,groups=report['groups'],tag=report['tag'],
            wall_s=report.get('wall_s'),method_cost_s=report.get('cost_total_s'),
            report_sha256=digest(run/(P+'report.json')),release_sha256=digest(run/(P+'release.json'))))
    baseline=next(r for r in runs if r.name==P+'baseline_v1')
    selected=next(r for r in runs if r.name==P+'S2_full_v1')
    manifest=json.loads((PROJECT/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json').read_text())
    exact_old,added=0,[]
    for scene in manifest['scenes']:
        name=scene['scene']
        sequences=[list(map(json.loads,gzip.open(run/name/(P+'predictions.jsonl.gz'),'rt')))
                   for run in [baseline,selected]]
        for a,b in zip(*sequences):
            old={p['node_id']:p for p in a['predictions']['T3_three']}
            new={p['node_id']:p for p in b['predictions']['S2_three']}
            assert all(node in new and value==new[node] for node,value in old.items())
            exact_old+=len(old)
            added.extend(dict(scene=name,frame_idx=a['frame_idx'],prediction=p)
                         for node,p in new.items() if node not in old)
    logs=[];grouped=defaultdict(list)
    for tag in ['baseline_v1','S2_full_v1']:
        path=root/tag/(P+'scene_metrics.csv')
        for row in csv.DictReader(path.open(encoding='utf-8-sig')):
            if not row['method'].endswith('_three'):continue
            grouped[row['log_token'],row['method'],row['scope']].append(row)
    for (log,method,scope),rows in grouped.items():
        value=dict(log_token=log,method=method,scope=scope,scenes='|'.join(r['scene'] for r in rows))
        for k in ['true_positive','truth','predictions','duplicate_nodes','error_sum_m','IDTP','IDFN','IDFP']:
            value[k]=sum(float(r[k]) for r in rows)
        tp,gt,pred=value['true_positive'],value['truth'],value['predictions']
        value.update(precision=tp/max(pred,1),recall=tp/max(gt,1),F1=2*tp/max(gt+pred,1),
            IDF1=2*value['IDTP']/max(2*value['IDTP']+value['IDFN']+value['IDFP'],1))
        logs.append(value)
    write_csv(root/(P+'log_metrics.csv'),logs)
    validation=[]
    for name in ['baseline_validation_v1','H1_validation_v1','S1_validation_v1','H2_validation_v1','S2_validation_v1','H2S2_validation_v1']:
        for row in csv.DictReader((root/name/(P+'node_main.csv')).open(encoding='utf-8-sig')):
            if row['method'].endswith('_three') and row['scope'] in ['front3','front']:
                validation.append(dict(configuration=name,**row))
    write_csv(root/(P+'all_validation_configurations.csv'),validation)
    paths=[root,PROJECT/'dataset/processed/mapnext_20261124_inputs_v1',*runs,
           *sorted((PROJECT/'scripts').glob(P+'*.py')),*sorted((PROJECT/'conceptgraph/slam').glob(P+'*.py'))]
    files={p.resolve() for path in paths for p in ([path] if path.is_file() else path.rglob('*')) if p.is_file()}
    total=sum(p.stat().st_size for p in files)
    assert total<registration['maximum_new_bytes']
    free=shutil.disk_usage(PROJECT).free
    assert free>=registration['minimum_free_bytes']
    independent=PROJECT/'outputs/reports/iteration_independent_audit_20261124/mapping_independent_final_v1.json'
    audit=json.loads(independent.read_text())
    ended=datetime.now(timezone.utc)
    record=dict(complete=True,ended_utc=ended.isoformat(),task_start_utc='2026-11-24T06:38:13+00:00',
        task_wall_s=(ended-datetime.fromisoformat('2026-11-24T06:38:13+00:00')).total_seconds(),
        deadline_utc=registration['deadline_utc'],finished_before_deadline=ended<datetime.fromisoformat(registration['deadline_utc']),
        new_files_total_bytes=total,maximum_new_bytes=registration['maximum_new_bytes'],free_disk_bytes=free,
        GPU_calls=0,new_frontend_calls=0,new_training_calls=0,default_changed=False,
        protected_sha256_checked=len(registration['protected_sha256']),candidate_source_sha256_checked=len(freeze['source_sha256']),
        reports=reports,exact_preserved_original_three_view_predictions=exact_old,added_predictions=added,
        independent_audit=str(independent),independent_audit_sha256=digest(independent),
        source_sha256=digest(__file__),source_checks_passed=True,
        limitations=['one GT-frame gain in one validation log only; no observed effect in two regression logs',
          'duplicate count and common-GT center, jitter, automatic adjacent ID changes unchanged',
          'strictfront remains below frozen single-front precision/IDF1; not a full repair',
          'all six scenes previously seen validation/regression; no unseen test result claimed'])
    dump(root/(P+'closeout.json'),record)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()
