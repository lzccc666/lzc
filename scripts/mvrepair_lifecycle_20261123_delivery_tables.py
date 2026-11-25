"""汇总冻结候选的全分母、逐来源支持和逐GT失得；不运行推理。"""
from pathlib import Path
import csv,gzip,json
from conceptgraph.completion.stage9 import PROJECT,dump

P='mvrepair_lifecycle_20261123_'
BASE=PROJECT/'outputs/results/nuscenes/validation/iterations'
REPORT=PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'

def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    scores=[];ledger=[];changes=[]
    for r in sorted(BASE.glob(P+'*')):
        result=r/(P+'report.json')
        if not result.exists():continue
        obj=json.loads(result.read_text());metric=(obj.get('results') or {}).get('node_main',[])
        for row in metric:scores.append(dict(run_id=r.name,source=str(result),**row))
        if not any(z in r.name for z in ['O2_ownership_pilot_v2','O2_ownership_regression_v2','O3_reliable_pilot_v2','O3_reliable_regression_v2']):continue
        for sf in sorted(r.glob('scene-*')):
            pf=sf/(P+'predictions.jsonl.gz');ef=sf/(P+'evaluation.jsonl.gz')
            if not pf.exists() or not ef.exists():continue
            for row in map(json.loads,gzip.open(pf,'rt')):
                for method,a in row['audit'].items():
                    u=a.get('unique_support')
                    if not u:continue
                    final={tuple(z['source_key']):z for z in a['sameview']['observations']}
                    original={tuple(z['source_key']):z for z in u['initial_trace_observations']}
                    for z in u['source_decisions']:
                        after=final.get(tuple(z['source_key']));before=original[tuple(z['source_key'])]
                        ledger.append(dict(run_id=r.name,scene=sf.name,frame_idx=row['frame_idx'],method=method,
                            camera=z['source_key'][0],capture_timestamp_s=z['source_key'][1],original_detection_index=z['source_key'][2],
                            lidar_token=z['lidar_token'],category=z['category'],original_confidence=z['original_confidence'],
                            reliable_current_dense=z.get('r1_reliable_current_dense'),before_path=before['path'],
                            before_actual_points=z['actual_retained_points'],before_original_contributors=len(z['original_selected_indices']),
                            kept_original_contributors=len(z['retained_indices']),lost_original_contributors=len(z['lost_indices']),
                            after_path=after['path'] if after else 'no_current_3d_geometry',after_actual_points=after['actual_retained_points'] if after else 0,
                            lost_indices=';'.join(map(str,z['lost_indices'])),retained_indices=';'.join(map(str,z['retained_indices'])),
                            source_prediction_file=str(pf)))
            for row in map(json.loads,gzip.open(ef,'rt')):
                truth={z['instance_token']:z for z in row['evaluation_only_official']['union']}
                for method in row['result']:
                    if not method.endswith('_three') or not method.startswith(('O2','O3')):continue
                    a=row['result']['L4_three']['union'];b=row['result'][method]['union']
                    old=set(a['node_mapping'].values());new=set(b['node_mapping'].values())
                    for gid in sorted(old^new):
                        g=truth[gid];kind='lost' if gid in old else 'gained'
                        z=a if kind=='lost' else b
                        idx=next(k for k,v in z['node_mapping'].items() if v==gid);pred=z['predictions'][int(idx)]
                        changes.append(dict(run_id=r.name,scene=sf.name,frame_idx=row['frame_idx'],method=method,change=kind,
                            evaluation_only_GT_instance=gid,category=g['category'],visibility_cameras=';'.join(g['visible_cameras']),
                            scope='front' if 'CAM_FRONT' in g['visible_cameras'] else 'side_only',matched_prediction_node=pred['node_id'],
                            prediction_center=pred['center'],GT_center=g['center'],source_evaluation_file=str(ef)))
    table(REPORT/'all_candidate_metrics_full_denominator.csv',scores)
    table(REPORT/'O2_O3_all_source_support_ledger.csv',ledger)
    dump(REPORT/'O2_O3_all_changed_GT_observations.json',changes)
    print(json.dumps(dict(score_rows=len(scores),source_rows=len(ledger),changed_GT_observations=len(changes))))

if __name__=='__main__':main()
