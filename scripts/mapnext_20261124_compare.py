"""Saved-output comparison: full denominator, per-node evidence, common-GT stability."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.stage9 import digest, dump
from scripts.multiview_20261123_run_graph import write_csv

P='mapnext_20261124_'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--report-dir',type=Path,required=True)
    ap.add_argument('--baseline-method',default='T3_three')
    ap.add_argument('--candidate-method',required=True)
    args=ap.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    common,adjacent,changes,node_frames=[],[],[],[]
    before={}
    hashes={}
    for scene in manifest['scenes']:
        name=scene['scene']
        paths=[run/name/(P+'evaluation.jsonl.gz') for run in [args.baseline,args.candidate]]
        raws=[run/name/(P+'predictions.jsonl.gz') for run in [args.baseline,args.candidate]]
        for p in paths+raws:
            hashes[str(p)]=digest(p)
        sequences=[list(map(json.loads,gzip.open(p,'rt'))) for p in paths]
        rawseq=[list(map(json.loads,gzip.open(p,'rt'))) for p in raws]
        assert len(sequences[0])==len(sequences[1])==len(scene['groups'])
        for idx,(a,b) in enumerate(zip(*sequences)):
            assert a['frame_idx']==b['frame_idx']
            frame=a['frame_idx']
            for scope in ['front3','front','union6']:
                truth=a['evaluation_only_official'][scope]
                assert truth==b['evaluation_only_official'][scope]
                rows=[a['result'][args.baseline_method][scope],b['result'][args.candidate_method][scope]]
                matches=[{m['truth_index']:m for m in r['metrics']['matches']} for r in rows]
                for status,indices in [('gained',matches[1].keys()-matches[0].keys()),
                                       ('lost',matches[0].keys()-matches[1].keys())]:
                    for i in indices:
                        changes.append(dict(scene=name,log_token=scene['log_token'],frame_idx=frame,scope=scope,
                            status=status,instance_token=truth[i]['instance_token'],category=truth[i]['category']))
                for i in matches[0].keys()&matches[1].keys():
                    obj=truth[i]
                    for method,row,matched in zip([args.baseline_method,args.candidate_method],rows,matches):
                        pred=row['predictions'][matched[i]['prediction_index']]
                        error=np.asarray(pred['center'][:2])-obj['center'][:2]
                        record=dict(scene=name,log_token=scene['log_token'],frame_idx=frame,scope=scope,
                            instance_token=obj['instance_token'],category=obj['category'],method=method,
                            node_id=pred['node_id'],error_x_m=float(error[0]),error_y_m=float(error[1]),
                            center_error_m=float(np.linalg.norm(error)))
                        common.append(record)
                        key=(name,scope,obj['instance_token'],method)
                        if key in before and before[key]['frame_idx']==frame-1:
                            old=before[key]
                            delta=error-np.asarray([old['error_x_m'],old['error_y_m']])
                            adjacent.append(dict(**record,previous_node_id=old['node_id'],
                                ID_changed=int(old['node_id']!=pred['node_id']),
                                GT_motion_subtracted_error_change_m=float(np.linalg.norm(delta))))
                        before[key]=record
                for method,row,raw in zip([args.baseline_method,args.candidate_method],rows,[s[idx] for s in rawseq]):
                    mapped={m['prediction_index']:m['truth_index'] for m in row['metrics']['matches']}
                    internal=raw['audit'][method]['mapnext_internal']
                    registry=raw['audit'][method]['lifecycle']['registry']['canonical_by_track']
                    aliases=internal['aliases']
                    for i,pred in enumerate(row['predictions']):
                        raw_ids=[s['node_id'] for s in internal['states'] if registry.get(
                            aliases.get(s['node_id'],s['node_id']),aliases.get(s['node_id'],s['node_id']))==pred['node_id']]
                        events=[e for e in internal['position_events'] if e['node_id'] in raw_ids]
                        distances=[np.linalg.norm((np.asarray(pred['center'])-o['center'])[:2])
                                   for o in truth if o['category']==pred['category']]
                        node_frames.append(dict(scene=name,log_token=scene['log_token'],frame_idx=frame,scope=scope,
                            method=method,node_id=pred['node_id'],category=pred['category'],matched=int(i in mapped),
                            matched_GT=truth[mapped[i]]['instance_token'] if i in mapped else None,
                            duplicate_metric=int(i not in mapped and bool(distances) and min(distances)<=3),
                            position_source=pred.get('position_source'),prediction_age_s=pred.get('prediction_age_s'),
                            actual_current_source_cameras='|'.join(sorted({e['camera'] for e in events if e.get('camera')})),
                            actual_current_points='|'.join(str(e['observation_points']) for e in events)))
    stability=[]
    for scope in ['front3','front','union6']:
        for method in [args.baseline_method,args.candidate_method]:
            xs=[r for r in common if r['scope']==scope and r['method']==method]
            pairs=[r for r in adjacent if r['scope']==scope and r['method']==method]
            errors=[r['center_error_m'] for r in xs]
            delta=[r['GT_motion_subtracted_error_change_m'] for r in pairs]
            stability.append(dict(scope=scope,method=method,common_GT_frames=len(xs),common_adjacent_GT_pairs=len(pairs),
                common_center_error_mean_m=float(np.mean(errors)) if errors else None,
                common_error_change_mean_m=float(np.mean(delta)) if delta else None,
                common_error_change_p90_m=float(np.quantile(delta,.9)) if delta else None,
                common_adjacent_automatic_ID_changes=sum(r['ID_changed'] for r in pairs)))
    bynode=defaultdict(list)
    for r in node_frames:
        bynode[r['scene'],r['scope'],r['method'],r['node_id']].append(r)
    nodes=[]
    for (scene,scope,method,node),rs in bynode.items():
        nodes.append(dict(scene=scene,scope=scope,method=method,node_id=node,category=rs[0]['category'],
            output_frames=len(rs),matched_frames=sum(r['matched'] for r in rs),
            duplicate_frames=sum(r['duplicate_metric'] for r in rs),
            matched_GT_instances='|'.join(sorted({r['matched_GT'] for r in rs if r['matched_GT']})),
            source_cameras='|'.join(sorted(set('|'.join(r['actual_current_source_cameras'] for r in rs).split('|'))))))
    for name,rs in dict(common_GT_frames=common,common_GT_adjacent_pairs=adjacent,paired_stability=stability,
                       GT_gains_losses=changes,node_frame_evidence=node_frames,node_summary=nodes).items():
        write_csv(args.report_dir/(P+name+'.csv'),rs)
    dump(args.report_dir/(P+'comparison_audit.json'),dict(source_sha256=digest(__file__),input_sha256=hashes,
        methods=[args.baseline_method,args.candidate_method],groups=sum(len(s['groups']) for s in manifest['scenes']),
        common_GT_scope='same GT-frame matched by both methods under same 3m same-class XY one-to-one gate',
        ID_changes='automatic adjacent raw ID changes, not human verified identity switches',
        GT_motion_subtracted_error_change='adjacent predicted-minus-GT XY vector difference; does not include unmatched GT',
        all_denominator_IDF1='separately in full scope score tables; common-GT stability does not replace it',
        GT_in_inference=False,no_method_calls=True))
    print(json.dumps(stability,ensure_ascii=False))


if __name__=='__main__':
    main()
