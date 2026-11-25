"""仅审计保存的实际选后点来源与L5身份链，不改变预测。"""
from collections import Counter
from itertools import combinations
import gzip,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import PROJECT,digest,dump
from conceptgraph.completion.temporal_identity_evaluation import identity_counts,prepare_sequence
from conceptgraph.project_paths import experiment_directory

P='mvrepair_lifecycle_20261123_'

def main():
    run=experiment_directory('nuscenes','validation',P+'support_collision_diagnosis_v1',create=True)
    report=PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle'
    collisions=[];special=[];counts=Counter();identities=[];chains=[]
    for split in ['pilot','regression']:
        source=PROJECT/'outputs/results/nuscenes/validation/iterations'/f'{P}L5_sameview_{split}_v1'
        for sf in sorted(source.glob('scene-*')):
            rows=list(map(json.loads,gzip.open(sf/(P+'predictions.jsonl.gz'),'rt')))
            for row in rows:
                a=row['audit']['L5_sameview_three']['sameview'];counts[split+'_groups']+=1
                obs=a['observations'];counts[split+'_current_observations']+=len(obs)
                for z in obs:counts[split+'_'+z['point_provenance_status']]+=1
                for x,y in combinations(obs,2):
                    if x['source_key'][:2]!=y['source_key'][:2] or not x['lidar_token'] or x['lidar_token']!=y['lidar_token']:continue
                    ix,iy=set(x['selected_point_indices']),set(y['selected_point_indices']);shared=ix&iy
                    record=dict(split=split,scene=sf.name,frame_idx=row['frame_idx'],
                        source_key_a=x['source_key'],source_key_b=y['source_key'],category_a=x['category'],category_b=y['category'],
                        lidar_token=x['lidar_token'],actual_retained_points_a=x['actual_retained_points'],actual_retained_points_b=y['actual_retained_points'],
                        original_point_count_a=len(ix),original_point_count_b=len(iy),shared_original_points=len(shared),shared_original_indices=sorted(shared),
                        independent_points_a=len(ix-iy),independent_points_b=len(iy-ix),
                        confidence_a=x['confidence'],confidence_b=y['confidence'])
                    if shared:collisions.append(record)
                    if sf.name=='scene-0035' and row['frame_idx']==10 and x['source_key'][0]=='CAM_FRONT_LEFT' and {x['source_key'][2],y['source_key'][2]}=={1,5}:
                        special.append(record|dict(original_point_indices_a=sorted(ix),original_point_indices_b=sorted(iy),
                            conclusion='actual selected supports disjoint; no evidence for same-point duplicate or identity merge'))
            evalrows=list(map(json.loads,gzip.open(sf/(P+'evaluation.jsonl.gz'),'rt')))
            for method in ['L4_three','L5_sameview_three']:
                seq=[dict(frame_idx=x['frame_idx'],camera_timestamp_s=y['reference_timestamp_s'],
                    predictions={method:x['result'][method]['union']['predictions']},evaluation_only_official=x['evaluation_only_official']['union'],
                    metrics={method:x['result'][method]['union']['metrics']}) for x,y in zip(evalrows,rows)]
                identities.append(dict(split=split,scene=sf.name,method=method,**identity_counts(prepare_sequence(seq,method),'all_gated')))
            if sf.name=='scene-0035':
                for x in evalrows[:5]:
                    evidence={}
                    for method in ['L4_three','L5_sameview_three']:
                        z=x['result'][method]['union'];evidence[method]=[dict(node_id=p['node_id'],category=p['category'],center=p['center'],
                            matched_instance=z['node_mapping'].get(str(i))) for i,p in enumerate(z['predictions'])]
                    chains.append(dict(scene=sf.name,frame_idx=x['frame_idx'],evaluation_only=True,**evidence))
    result=dict(counts=dict(counts),collision_pairs=collisions,f10_disjoint_actual_support=special,
        identity_counts_and_global_assignments=identities,early_chain=chains,
        interpretation='L5 f2 representative changes frame1 track1 from target35e2 to targete188 at frame2; the removed detection is not merely a new duplicate. Global assignment loses two identity overlaps; node TP unchanged does not prove identity preservation.',
        inference_GT_used=False,scope='all80 groups; actual retained support, not raw interval candidates; same camera+exposure+scan only',
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__).resolve())})
    dump(run/(P+'support_collision_audit.json'),result);dump(report/'sameview_support_collision_all80.json',result)
    print(json.dumps(dict(run=str(run),counts=dict(counts),collision_pairs=len(collisions),f10=special),ensure_ascii=False))

if __name__=='__main__':main()
