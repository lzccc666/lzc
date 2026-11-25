"""Post-hoc actual-state trace for near-measurement misses and duplicate metric."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from conceptgraph.completion.stage9 import PROJECT,dump,digest
from scripts.multiview_20261123_run_graph import write_csv

P='mapnext_20261124_'


def main():
    root=PROJECT/'outputs/reports/mapping_iteration_20261124'
    run=PROJECT/'outputs/results/nuscenes/validation/iterations/mapnext_20261124_baseline_v1'
    manifest=json.loads((PROJECT/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json').read_text())
    details,duplicates,bus=[] ,[],[]
    for scene in manifest['scenes']:
        name=scene['scene']
        evidence=list(map(json.loads,gzip.open(root/'diagnosis_v1'/(name+'_all_GT_evidence.jsonl.gz'),'rt')))
        raw=list(map(json.loads,gzip.open(run/name/(P+'predictions.jsonl.gz'),'rt')))
        evaluated=list(map(json.loads,gzip.open(run/name/(P+'evaluation.jsonl.gz'),'rt')))
        maximum_confidence,last_sources={},{}
        for row,evaluation in zip(raw,evaluated):
            frame=row['frame_idx'];audit=row['audit']['T3_three'];internal=audit['mapnext_internal']
            obs=audit['mapnext_observations']
            registry=audit['lifecycle']['registry']['canonical_by_track'];aliases=internal['aliases']
            def canonical(node):
                node=aliases.get(node,node)
                return registry.get(node,node)
            event_sources={}
            for event in internal['position_events']:
                candidates=[o for o in obs if o['camera']==event['camera'] and o['category']==event['category']
                    and np.linalg.norm(np.asarray(o['center'])-event['observation_center'])<1e-4]
                event_sources[event['node_id']]=candidates
                maximum_confidence[event['node_id']]=max(maximum_confidence.get(event['node_id'],0),
                    max((o['confidence'] for o in candidates),default=0))
                if candidates:last_sources[event['node_id']]=candidates
            result=evaluation['result']['T3_three']['front3'];truth=evaluation['evaluation_only_official']['front3']
            pred=result['predictions'];matches=result['metrics']['matches']
            matched_pred={m['prediction_index']:m['truth_index'] for m in matches}
            def source_records(node):
                states=[s for s in internal['states'] if canonical(s['node_id'])==node]
                sources=[o for s in states for o in last_sources.get(s['node_id'],[])]
                return states,sources
            for record in [r for r in evidence if r['frame_idx']==frame]:
                if record['classification']=='measurement_within_3m_but_output_unmatched':
                    item=record['GT_evaluation_only'];traces=[]
                    for observation in record['linked_measurements']:
                        if np.linalg.norm(np.asarray(observation['center'][:2])-item['center'][:2])>3:continue
                        raw_ids=[node for node,sources in event_sources.items()
                            if any(o['source_key']==observation['source_key'] for o in sources)]
                        for raw_id in raw_ids:
                            state=next((s for s in internal['states'] if s['node_id']==raw_id),None)
                            root_id=canonical(raw_id)
                            outputs=[(i,p) for i,p in enumerate(pred) if p['node_id']==root_id]
                            reason='state_not_output_unknown_or_canonical_retirement'
                            if outputs:
                                i,p=outputs[0]
                                distance=float(np.linalg.norm(np.asarray(p['center'][:2])-item['center'][:2]))
                                if distance>3:reason='associated_output_center_outside_3m'
                                elif i in matched_pred:reason='within_gate_output_assigned_to_other_GT'
                                else:reason='within_gate_unmatched_output_pairing_ambiguity'
                            elif state and raw_id.startswith('sparse-'):
                                needed=2 if maximum_confidence.get(raw_id,0)>=.2 else 3
                                if state['hits']<needed or state['total_points']<4:
                                    reason='sparse_confirmation_or_total_point_requirement'
                                else:reason='confirmed_sparse_not_output_proximity_or_canonical_filter'
                            traces.append(dict(source_key=observation['source_key'],raw_state_id=raw_id,canonical=root_id,
                                reason=reason,state=state,observed_maximum_confidence=maximum_confidence.get(raw_id)))
                        if not raw_ids:
                            traces.append(dict(source_key=observation['source_key'],reason='accepted_geometry_not_in_position_event',
                                possible='dense confirmation, association representative removal, or source-link ambiguity; unresolved'))
                    reasons=sorted({t['reason'] for t in traces})
                    details.append(dict(scene=name,frame_idx=frame,instance_token=item['instance_token'],
                        category=item['category'],reasons='|'.join(reasons),traces=traces,
                        GT_diagnostic_link='projected 2D overlap may be nonunique; these are source traces, not recoverable GT claims'))
                if record['classification']=='accepted_measurement_outside_3m' and record['category']=='bus':
                    item=record['GT_evaluation_only']
                    observation=min(record['linked_measurements'],key=lambda o:np.linalg.norm(np.asarray(o['center'][:2])-item['center'][:2]))
                    local=Quaternion(item['rotation_wxyz']).rotation_matrix.T@(np.asarray(observation['center'])-item['center'])
                    bus.append(dict(scene=name,frame_idx=frame,instance_token=item['instance_token'],
                        source_key=observation['source_key'],actual_points=observation['points'],reliable=observation['reliable'],
                        error_m=record['nearest_observation_error_m'],GT_length_m=item['size_wlh'][1],
                        longitudinal_offset_m=float(local[0]),lateral_offset_m=float(local[1]),
                        longitudinal_half_length_ratio=abs(float(local[0]))/(item['size_wlh'][1]/2),
                        interpretation='GT size/orientation used for diagnosis only; visible surface center can differ from full object center'))
            for i,p in enumerate(pred):
                if i in matched_pred:continue
                same=[(np.linalg.norm(np.asarray(p['center'][:2])-o['center'][:2]),j) for j,o in enumerate(truth) if o['category']==p['category']]
                if not same:continue
                distance,gi=min(same)
                if distance>3:continue
                owner=next((pi for pi,g in matched_pred.items() if g==gi),None)
                states,sources=source_records(p['node_id'])
                owner_states,owner_sources=source_records(pred[owner]['node_id']) if owner is not None else ([],[])
                duplicates.append(dict(scene=name,log_token=scene['log_token'],frame_idx=frame,
                    unmatched_node=p['node_id'],matched_node=pred[owner]['node_id'] if owner is not None else None,
                    category=p['category'],nearest_GT_instance=truth[gi]['instance_token'],distance_m=float(distance),
                    unmatched_position_source=p.get('position_source'),unmatched_age_s=p.get('prediction_age_s'),
                    raw_states=states,actual_source_records=sources,matched_raw_states=owner_states,
                    matched_actual_source_records=owner_sources,physical_identity_status='unknown; automatic 3m duplicate metric does not prove same physical instance'))
    dest=root/'diagnosis_details_v1';dest.mkdir(exist_ok=False)
    with gzip.open(dest/'near_measurement_state_traces.jsonl.gz','wt') as stream:
        for row in details:stream.write(json.dumps(row)+'\n')
    with gzip.open(dest/'all_11_duplicate_traces.jsonl.gz','wt') as stream:
        for row in duplicates:stream.write(json.dumps(row)+'\n')
    write_csv(dest/'bus_partial_surface_offsets.csv',bus)
    counts=Counter(reason for r in details for reason in r['reasons'].split('|'))
    dump(dest/'trace_summary.json',dict(near_measurement_GT_rows=len(details),reasons_nonexclusive=dict(counts),
        duplicate_metric_rows=len(duplicates),duplicate_scene_count=dict(Counter(r['scene'] for r in duplicates)),
        bus_offset_rows=len(bus),source_sha256=digest(__file__),GT_in_inference=False,
        attribution_limits='Some observation-GT links are ambiguous; each GT row can have multiple state reasons, not an exclusive causal partition. No count is a promised recoverable gain.'))
    print(json.dumps(dict(near_measurement_rows=len(details),reason_counts_nonexclusive=dict(counts),duplicates=len(duplicates))))


if __name__=='__main__':
    main()
