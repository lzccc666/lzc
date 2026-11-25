"""逐GT逐帧保留视角/算法收益与丢失，不调用模型或地图。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from scripts.multiview_20261123_run_graph import write_csv

P = 'mv136_20261123_'
PAIRS = [('T3_1','T3_3'),('T3_3','T3_6'),('C_3','T3_3'),('C_6','T3_6')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args = parser.parse_args()
    all_changes, frames, totals = [], [], Counter()
    for file in sorted(args.run.glob('scene-*/'+P+'evaluation.jsonl.gz')):
        sequence = list(map(json.loads,gzip.open(file,'rt')))
        assert len(sequence) == 20
        for row in sequence:
            for scope in ['union6','front3','front']:
                gt = row['evaluation_only_official'][scope]
                for old,new in PAIRS:
                    a,b = row['result'][old][scope],row['result'][new][scope]
                    am = {x['truth_index']:x for x in a['metrics']['matches']}
                    bm = {x['truth_index']:x for x in b['metrics']['matches']}
                    gained,lost = bm.keys()-am.keys(),am.keys()-bm.keys()
                    totals[scope,old,new,'gained'] += len(gained)
                    totals[scope,old,new,'lost'] += len(lost)
                    frames.append(dict(scene=row['scene'],frame_idx=row['frame_idx'],scope=scope,reference=old,candidate=new,
                        TP_reference=len(am),TP_candidate=len(bm),gained=len(gained),lost=len(lost),
                        predictions_reference=len(a['predictions']),predictions_candidate=len(b['predictions']),
                        duplicate_reference=a['metrics']['duplicate_nodes'],duplicate_candidate=b['metrics']['duplicate_nodes']))
                    for status,indices,result,matches in [('gained',gained,b,bm),('lost',lost,a,am)]:
                        for i in sorted(indices):
                            match = matches[i]
                            pred = result['predictions'][match['prediction_index']]
                            all_changes.append(dict(scene=row['scene'],frame_idx=row['frame_idx'],scope=scope,reference=old,candidate=new,
                                status=status,instance_token=gt[i]['instance_token'],category=gt[i]['category'],
                                visible_cameras='|'.join(gt[i]['visible_cameras']),node_id=pred['node_id'],
                                center_error_m=match['distance_m'],prediction_center=json.dumps(pred['center']),truth_center=json.dumps(gt[i]['center'])))
    summary = [dict(scope=k[0],reference=k[1],candidate=k[2],status=k[3],GT_frames=v) for k,v in totals.items()]
    for name,values in dict(all_frame_gains_losses=frames,changed_GT_cases=all_changes,gains_losses_summary=summary).items():
        write_csv(args.report_dir/(P+name+'.csv'),values)
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
