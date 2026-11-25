"""已保存结果上的三T3共同GT交集位置/身份诊断，不运行方法。"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from conceptgraph.completion.stage9 import digest, dump
from scripts.multiview_20261123_run_graph import write_csv

P = 'mv136_20261123_'
METHODS = ['T3_1','T3_3','T3_6']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = json.loads((args.run/(P+'report.json')).read_text())
    assert report['complete'] and report['groups'] == 120
    samples, adjacency, timing = [], [], []
    before = {}
    clocks = []
    primitive = []
    for scene in manifest['scenes']:
        rows = list(map(json.loads,gzip.open(args.run/scene['scene']/(P+'evaluation.jsonl.gz'),'rt')))
        assert len(rows) == 20
        raw = list(map(json.loads,gzip.open(args.run/scene['scene']/(P+'predictions.jsonl.gz'),'rt')))
        for x in raw:
            for method,t in x['clock']['subset_ready_timestamp_s'].items():
                scan = x['clock']['lidar_by_subset'][method]
                timing.append(dict(scene=scene['scene'],frame_idx=x['frame_idx'],method=method,
                    ready_timestamp_s=t,ready_delay_from_front_ms=x['clock']['subset_ready_delay_from_front_ms'][method],
                    common_wait_after_ready_ms=x['clock']['common_evaluation_wait_after_ready_ms'][method],
                    unique_lidar_scans=scan['unique_scan_count'],missing_lidar_views=scan['missing_views'],
                    official_sample_minus_reference_ms=x['clock']['official_sample_minus_reference_ms']))
            clocks.append(x['clock'])
        for row in rows:
            frame = row['frame_idx']
            for scope in ['front','front3','union6']:
                truth = row['evaluation_only_official'][scope]
                matches = {m:{v['truth_index']:v for v in row['result'][m][scope]['metrics']['matches']} for m in METHODS}
                common = set.intersection(*(set(v) for v in matches.values()))
                for i in common:
                    gt = truth[i]
                    for method in METHODS:
                        match = matches[method][i]
                        pred = row['result'][method][scope]['predictions'][match['prediction_index']]
                        error = np.asarray(pred['center'])[:2]-np.asarray(gt['center'])[:2]
                        r = dict(scene=scene['scene'],log_token=scene['log_token'],frame_idx=frame,scope=scope,
                            instance_token=gt['instance_token'],category=gt['category'],method=method,node_id=pred['node_id'],
                            error_x_m=float(error[0]),error_y_m=float(error[1]),center_error_m=float(np.linalg.norm(error)))
                        samples.append(r)
                        key = scene['scene'],scope,gt['instance_token'],method
                        if key in before and before[key]['frame_idx'] == frame-1:
                            old = before[key]
                            change = float(np.linalg.norm(error-np.array([old['error_x_m'],old['error_y_m']])))
                            adjacency.append(dict(**r,previous_frame_idx=old['frame_idx'],previous_node_id=old['node_id'],
                                ID_changed=int(old['node_id']!=pred['node_id']),GT_motion_subtracted_error_change_m=change))
                        before[key] = r
    groups = defaultdict(list)
    for r in samples:
        groups[r['scope'],r['method']].append(r)
    summary = []
    for (scope,method),values in groups.items():
        errors = np.asarray([r['center_error_m'] for r in values])
        pairs = [r for r in adjacency if (r['scope'],r['method'])==(scope,method)]
        delta = np.asarray([r['GT_motion_subtracted_error_change_m'] for r in pairs])
        summary.append(dict(scope=scope,method=method,common_GT_frames=len(values),common_adjacent_GT_pairs=len(pairs),
            common_center_error_mean_m=float(errors.mean()),common_center_error_median_m=float(np.median(errors)),
            common_error_change_mean_m=float(delta.mean()) if len(delta) else None,
            common_error_change_p90_m=float(np.quantile(delta,.9)) if len(delta) else None,
            common_adjacent_raw_ID_changes=sum(r['ID_changed'] for r in pairs)))
    for name,records in dict(common_GT_position_frames=samples,common_GT_adjacent_pairs=adjacency,
            paired_stability=summary,clock_and_scan_frame=timing).items():
        write_csv(args.report_dir/(P+name+'.csv'),records)
    scene_csv = args.report_dir/(P+'scene_metrics.csv')
    import csv
    scene_rows = list(csv.DictReader(scene_csv.open(encoding='utf-8-sig')))
    log_groups = defaultdict(list)
    for row in scene_rows:
        log_groups[row['log_token'],row['method'],row['scope']].append(row)
    for (log,method,scope),rows in log_groups.items():
        v = dict(log_token=log,method=method,scope=scope,scenes='|'.join(r['scene'] for r in rows))
        for k in ['true_positive','truth','predictions','duplicate_nodes','error_sum_m','IDTP','IDFN','IDFP']:
            v[k] = sum(float(r[k]) for r in rows)
        tp,gt,pred = v['true_positive'],v['truth'],v['predictions']
        v.update(precision=tp/max(1,pred),recall=tp/max(1,gt),F1=2*tp/max(1,gt+pred),
            IDF1=2*v['IDTP']/max(1,2*v['IDTP']+v['IDFN']+v['IDFP']))
        primitive.append(v)
    write_csv(args.report_dir/(P+'log_metrics.csv'),primitive)
    dump(args.report_dir/(P+'diagnostic_audit.json'),dict(groups=120,methods=METHODS,
        matching_intersection='same GT-frame matched by all T3_1/3/6 under each identical strict scope',
        error='world XY predicted-minus-GT vector; adjacent error change subtracts true GT displacement',
        identity='raw ID change on common matched adjacency; automatic 3m assignment ambiguity remains, not human-verified ID switch',
        source_sha256=digest(__file__),run_report_sha256=digest(args.run/(P+'report.json')),
        no_method_or_model_calls=True))
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
