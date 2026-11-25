"""Read-only explanation of saved duplicate proxies; no matcher/model rerun."""
from __future__ import annotations

from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/results/kitti/validation/iterations/kitti_local_all_20261125_tracking_v1'
REPORT=ROOT/'outputs/reports/kitti_local_all_20261125'


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mapping_path=RUN/'mapping_frames.jsonl.gz';evaluation_path=RUN/'evaluation_frames.jsonl.gz'
    seen=defaultdict(set);totals=defaultdict(Counter);by_seq=defaultdict(Counter);by_class=defaultdict(Counter)
    observations=[];unique=defaultdict(set);seqclass=defaultdict(Counter)
    for frame,evaluation in zip(rows(mapping_path),rows(evaluation_path)):
        key=frame['sequence'],frame['frame_idx']
        assert key==(evaluation['sequence'],evaluation['frame_idx'])
        truth=evaluation['official_truth']
        for method,predictions in frame['predictions'].items():
            history=seen[key[0],method]
            saved=evaluation['metrics'][method];matched={m['prediction_index'] for m in saved['matches']}
            duplicate_count=0
            for i,prediction in enumerate(predictions):
                if i in matched:continue
                neighbors=[(j,float(np.linalg.norm(np.asarray(prediction['center'][:2])-gt['center'][:2]))) for j,gt in enumerate(truth)
                    if prediction['category']==gt['category'] and np.linalg.norm(np.asarray(prediction['center'][:2])-gt['center'][:2])<=3.]
                if not neighbors:continue
                duplicate_count+=1
                old=prediction['node_id'] in history
                last=prediction.get('last_seen_frame')
                if isinstance(last,int) and last<key[1]:update='stale_explicit_last_seen'
                elif last==key[1]:update='current_update'
                else:update='unknown_update_time'
                emission='previously_exported_ID' if old else 'first_exported_ID_not_proven_new_creation'
                category=prediction['category'];first=sorted(neighbors,key=lambda x:(x[1],x[0]))[0]
                counts=dict(duplicate_occurrences=1,**{update:1,emission:1})
                totals[method].update(counts);by_seq[method,key[0]].update(counts);by_class[method,category].update(counts)
                seqclass[method,key[0],category].update(counts);unique[method].add((key[0],prediction['node_id']))
                observations.append(dict(sequence=key[0],frame_idx=key[1],original_frame_id=frame['original_frame_id'],method=method,
                    prediction_index=i,node_id=prediction['node_id'],category=category,last_seen_frame=last,
                    node_update_state=update,export_history=emission,num_detections=prediction.get('num_detections'),
                    nearest_official_instance=truth[first[0]]['instance_token'],nearest_center_distance_xy_m=first[1],
                    same_category_GT_within3m=len(neighbors),evaluation_only=True))
            assert duplicate_count==saved['duplicate_nodes']
            totals[method].update(prediction_node_frames=len(predictions),frames=1)
            history.update(p['node_id'] for p in frame['all_predictions'][method])
    saved=json.loads((RUN/'evaluation_summary_v2.json').read_text())
    for method,counts in totals.items():
        assert counts['duplicate_occurrences']==saved['methods'][method]['counts']['duplicate_nodes']
        assert counts['prediction_node_frames']==saved['methods'][method]['counts']['predictions']
    dest=REPORT/'duplicate_proxy_diagnosis_v1.json'
    report=dict(definition='unmatched by saved maximum one-to-one same-class XY3m assignment, but within3m of at least one same-class official GT',
        denominator='prediction-node frame occurrences; not unique physical objects',
        totals={m:dict(c,unique_exported_IDs_with_duplicate_proxy=len(unique[m]),duplicate_fraction=c['duplicate_occurrences']/c['prediction_node_frames']) for m,c in totals.items()},
        by_sequence=[dict(method=m,sequence=s,counts=dict(c)) for (m,s),c in sorted(by_seq.items())],
        by_class=[dict(method=m,category=c,counts=dict(v)) for (m,c),v in sorted(by_class.items())],
        by_sequence_class=[dict(method=m,sequence=s,category=c,counts=dict(v)) for (m,s,c),v in sorted(seqclass.items())],
        limitations=['first exported ID is not proven creation time; tentative states and exact association lineage were not saved',
            'current update is not proof of correct/wrong association; this ledger cannot certify the cause as erroneous matching',
            'nearby same-class centers in crowds and annotation gaps can trigger the duplicate proxy; no physical-duplicate human truth claim'],
        no_new_models_or_matching=True,source_sha256={str(p):sha(p) for p in [mapping_path,evaluation_path,RUN/'evaluation_summary_v2.json',Path(__file__),ROOT/'scripts/kitti_duplicate_diagnosis_20261125.py']})
    with dest.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    with (REPORT/'duplicate_proxy_occurrences_v1.jsonl').open('x') as f:
        for row in observations:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(report['totals']))


if __name__=='__main__':main()
