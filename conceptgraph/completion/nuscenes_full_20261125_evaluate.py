"""已封存六条件预测的隔离官方评价；不供方法侧导入。"""
from __future__ import annotations
import argparse,gzip,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion.nuscenes_full_20261125_run import RUN,METHODS,DATA,REPORT
from conceptgraph.completion.quality import object_frame_metrics,relations_from_centers
from conceptgraph.completion.temporal_identity_evaluation import prepare_sequence,identity_counts,return_diagnostics
from conceptgraph.scenegraph.paper_graph_20261123_metrics import edge_counts,scores
from conceptgraph.scenegraph.relations import inverse_relation
from conceptgraph.nuscenes.benchmark_ground_truth import official_category
from conceptgraph.slam.multiview_20261123_fusion import in_view
SCOPES={'front':['CAM_FRONT'],'front3':['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'],'union6':['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']}
LABEL_FILES=['category.json','instance.json','sample_annotation.json','attribute.json','sample.json']


def freeze():
    paths=[source.META/n for n in LABEL_FILES]
    record=dict(created_utc=source.now(),source_sha256=source.sha(__file__),label_sha256={str(p):source.sha(p) for p in paths},
        roles='evaluation_only; never imported/opened by mapping method',threshold_m=3.,class_same=True,one_to_one=True,
        common_GT_scopes=SCOPES,visibility_minimum=2,depth_interval_open_m=[1,65],in_image='3D annotation center projected inside actual image',
        GT_time='official current sample annotation positions; no motion compensation to latest6camera reference; offsets retained',
        identity='same original global maximal category-and-3m gated overlap counts, all observation denominators',
        relations='production current edges and unpruned geometric rules reported separately; all truth edges and conditional endpoints separate',
        motion='official attributes and past-interval displacement separate; unknown/missing kept in full labeled denominator; same matched GT legacy estimate vs verified state',
        past_motion_threshold_mps=.5,GT_never_in_method=True,negative_results_preserved=True,body_heading_not_evaluated='method body heading unknown')
    source.dump(DATA/'evaluation_protocol.json',record);return record


def attributes_motion(names):
    a=bool(set(names)&{'vehicle.moving','pedestrian.moving'});b=bool(set(names)&{'vehicle.stopped','vehicle.parked','pedestrian.standing','pedestrian.sitting_lying_down'})
    return 'dynamic' if a and not b else 'static' if b and not a else 'unknown'


def official_tables():
    categories={r['token']:r['name'] for r in source.iter_records(source.META/'category.json')}
    instances={r['token']:official_category(categories[r['category_token']]) for r in source.iter_records(source.META/'instance.json')}
    attributes={r['token']:r['name'] for r in source.iter_records(source.META/'attribute.json')}
    stamps={r['token']:r['timestamp']/1e6 for r in source.iter_records(source.META/'sample.json')}
    anns={};by_sample=defaultdict(list)
    for r in source.iter_records(source.META/'sample_annotation.json'):
        category=instances[r['instance_token']]
        if category is None:continue
        a=dict(annotation_token=r['token'],instance_token=r['instance_token'],sample_token=r['sample_token'],prev=r['prev'],category=category,
            center=r['translation'],size_wlh=r['size'],rotation_wxyz=r['rotation'],visibility_token=int(r['visibility_token']),num_lidar_pts=r['num_lidar_pts'],
            official_attributes=[attributes[t] for t in r['attribute_tokens'] if t in attributes])
        a['attribute_motion']=attributes_motion(a['official_attributes']);anns[r['token']]=a
        if int(r['visibility_token'])>=2:by_sample[r['sample_token']].append(a)
    for a in anns.values():
        prev=anns.get(a['prev']);dt=stamps[a['sample_token']]-stamps[prev['sample_token']] if prev else 0
        speed=float(np.linalg.norm((np.asarray(a['center'])-prev['center'])[:2])/dt) if prev and prev['instance_token']==a['instance_token'] and dt>0 else None
        a.update(past_displacement_speed_mps=speed,past_interval_s=dt if dt>0 else None,past_displacement_motion='unknown' if speed is None else 'static' if speed<=.5 else 'dynamic')
    return by_sample


def production_facts(payload,pred):
    selected={p['node_id'] for p in pred};registry=payload['relation_node_registry'];facts=set()
    for edge in payload['edges']:
        a,b=registry[str(edge['source'])],registry[str(edge['target'])]
        if a in selected and b in selected:
            relation=edge['relation'];facts.add((a,relation,b));facts.add((b,inverse_relation(relation),a))
    return facts


def node_scores(c):
    tp,gt,pred=c['true_positive'],c['truth'],c['predictions']
    return dict(precision=tp/pred if pred else None,recall=tp/gt if gt else None,F1=2*tp/(gt+pred) if gt+pred else None,
        center_error_m=c['error_sum_m']/tp if tp else None,IDF1=2*c['IDTP']/(c['GT_observations']+c['prediction_observations']) if c['GT_observations']+c['prediction_observations'] else None,
        common_object_center_error_m=c['common_error_sum_m']/c['common_objects'] if c['common_objects'] else None,
        common_object_temporal_residual_m=c['common_jitter_sum_m']/c['common_adjacent_pairs'] if c['common_adjacent_pairs'] else None)


def evaluate():
    started=time.monotonic();frozen=json.loads((DATA/'evaluation_protocol.json').read_text());manifest=json.loads((DATA/'manifest.json').read_text())
    assert source.sha(__file__)==frozen['source_sha256']
    for p,h in frozen['label_sha256'].items():assert source.sha(p)==h,p
    assert json.loads((RUN/'inference_complete.json').read_text())['complete'];scenes=manifest['scenes'];target=RUN/'evaluation'
    target.mkdir(exist_ok=False)
    official=official_tables();globalc=defaultdict(Counter);globaledge=defaultdict(Counter);globalmotion=defaultdict(Counter);logc=defaultdict(Counter);rolec=defaultdict(Counter);scene_rows=[]
    overlay=json.loads((DATA/'scene_role_access_overlay_v2.json').read_text())['scenes']
    for ordinal,scene in enumerate(scenes):
        name=scene['scene'];completion=json.loads((RUN/name/'complete.json').read_text());assert source.sha(RUN/name/'frames.jsonl.gz')==completion['frames_sha256']
        cc=defaultdict(Counter);ee=defaultdict(Counter);mm=defaultdict(Counter);prepared=defaultdict(list);last_common={};frames=0;outpath=target/(name+'.jsonl.gz')
        with gzip.open(outpath,'xt',compresslevel=6) as out:
            for frame in source.rows(RUN/name/'frames.jsonl.gz'):
                truth=[]
                for obj in official[frame['sample_token']]:
                    visible=[c for c,v in frame['views'].items() if in_view(obj['center'],v)]
                    if visible:truth.append(dict(obj,visible_cameras=visible))
                truth.sort(key=lambda o:o['instance_token']);assert len(truth)==len({o['instance_token'] for o in truth})
                row=dict(scene=name,frame_idx=frame['frame_idx'],sample_token=frame['sample_token'],evaluation_only_official=truth,result={})
                forward=np.asarray(frame['views']['CAM_FRONT']['camera_pose'])[:2,2]
                for scope,cameras in SCOPES.items():
                    gt=[o for o in truth if set(o['visible_cameras'])&set(cameras)]
                    gt_edges=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in gt],forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                    scoped={m:[p for p in frame['methods'][m]['predictions'] if any(in_view(p['center'],frame['views'][c]) for c in cameras)] for m in METHODS}
                    metrics={m:object_frame_metrics(scoped[m],gt,threshold_m=3.) for m in METHODS}
                    common=set.intersection(*[{x['truth_index'] for x in metrics[m]['matches']} for m in METHODS])
                    result={}
                    for method in METHODS:
                        key=(method,scope);pred=scoped[method];metric=metrics[method];counts=cc[key]
                        for field in ['predictions','truth','true_positive','unmatched_predictions','false_negative','duplicate_nodes']:counts[field]+=metric[field]
                        counts['frames']+=1;counts['error_sum_m']+=sum(x['distance_m'] for x in metric['matches'])
                        bygt={x['truth_index']:x for x in metric['matches']};mapping={pred[x['prediction_index']]['node_id']:gt[x['truth_index']]['instance_token'] for x in metric['matches']}
                        node_by_id={n['node_id']:n for n in frame['methods'][method]['nodes']}
                        edges=relations_from_centers([dict(id=p['node_id'],center=p['center']) for p in pred],forward_xy=forward,max_distance_m=12.,near_m=6.,component_margin_m=1.)
                        geom=edge_counts(edges,gt_edges,mapping);prod=edge_counts(production_facts(frame['methods'][method],pred),gt_edges,mapping)
                        ee[method,scope,'geometry'].update(geom);ee[method,scope,'production'].update(prod)
                        prepared[key].append(dict(frame_idx=frame['frame_idx'],camera_timestamp_s=frame['render_timestamp_s'],predictions={method:pred},evaluation_only_official=gt,metrics={method:metric}))
                        for j in common:
                            p=pred[bygt[j]['prediction_index']];obj=gt[j];counts['common_objects']+=1;counts['common_error_sum_m']+=bygt[j]['distance_m']
                            history=(method,scope,obj['instance_token']);prior=last_common.get(history)
                            if prior and prior['frame']==frame['frame_idx']-1:
                                delta=(np.asarray(p['center'])-prior['pred'])[:2]-(np.asarray(obj['center'])-prior['GT'])[:2]
                                counts['common_adjacent_pairs']+=1;counts['common_jitter_sum_m']+=float(np.linalg.norm(delta))
                            last_common[history]=dict(frame=frame['frame_idx'],pred=np.asarray(p['center']),GT=np.asarray(obj['center']))
                        motion=mm[key]
                        matched_ids={p for p in mapping}
                        for p in pred:
                            n=node_by_id[p['node_id']];motion['prediction:'+n['verified_motion_state']]+=1
                            if p['node_id'] not in matched_ids:motion['unmatched_prediction:'+n['verified_motion_state']]+=1
                        for j,obj in enumerate(gt):
                            match=bygt.get(j);node=node_by_id[pred[match['prediction_index']]['node_id']] if match else None
                            motion['matched_GT' if node else 'unmatched_GT']+=1
                            for reference in ['attribute_motion','past_displacement_motion']:
                                label=obj[reference];motion[reference+':GT:'+label]+=1
                                for statekind in ['estimated_motion_state','verified_motion_state']:
                                    state=node[statekind] if node else 'missing_node';prefix=reference+':'+statekind
                                    motion[prefix+':confusion:'+label+'->'+state]+=1
                                    if label!='unknown':
                                        motion[prefix+':all_labeled']+=1;motion[prefix+':correct']+=int(state==label);motion[prefix+':known']+=int(state in ['static','dynamic'])
                                        if node:motion[prefix+':matched_labeled']+=1;motion[prefix+':matched_correct']+=int(state==label)
                            if node and node.get('motion_speed_xy_mps') is not None and obj['past_displacement_speed_mps'] is not None:
                                motion['speed_N']+=1;motion['speed_absolute_error_sum_mps']+=abs(node['motion_speed_xy_mps']-obj['past_displacement_speed_mps'])
                        result[method]=dict(metrics=metric,prediction_node_ids=[p['node_id'] for p in pred],geometric_relations=dict(geom),production_relations=dict(prod),motion_scope='see all denominator aggregation')
                    row['result'][scope]=result
                out.write(json.dumps(row,separators=(',',':'))+'\n');frames+=1
        assert frames==scene['frames']
        for key,sequence in prepared.items():
            identity=identity_counts(prepare_sequence(sequence,key[0]),'all_gated')['counts'];cc[key].update(identity)
            ret=return_diagnostics(prepare_sequence(sequence,key[0]))['counts'];cc[key].update(ret)
        access=overlay.get(name,scene['role'])['access_role']
        for (method,scope),counts in cc.items():
            globalc[method,scope].update(counts);logc[scene['log_token'],method,scope].update(counts);rolec[access,method,scope].update(counts)
            scene_rows.append(dict(scene=name,log=scene['log_token'],access_role=access,method=method,scope=scope,counts=dict(counts),scores=node_scores(counts)))
        for key,c in ee.items():globaledge[key].update(c)
        for key,c in mm.items():globalmotion[key].update(c)
        source.dump(target/(name+'_complete.json'),dict(complete=True,scene=name,frames=frames,predictions_sha256=completion['frames_sha256'],evaluation_sha256=source.sha(outpath),GT_in_method=False))
        print(json.dumps(dict(scene=name,completed_scenes=ordinal+1,expected=len(scenes),elapsed_s=time.monotonic()-started)),flush=True)
    source.dump(target/'summary.json',dict(complete=True,scenes=len(scenes),frames=sum(s['frames'] for s in scenes),threshold_m=3.,GT_only_evaluation=True,
        node=[dict(method=m,scope=s,counts=dict(c),scores=node_scores(c)) for (m,s),c in globalc.items()],
        relations=[dict(method=m,scope=s,kind=k,counts=dict(c),scores=scores(c)) for (m,s,k),c in globaledge.items()],
        motion=[dict(method=m,scope=s,counts=dict(c)) for (m,s),c in globalmotion.items()],
        bylog=[dict(log=l,method=m,scope=s,counts=dict(c),scores=node_scores(c)) for (l,m,s),c in logc.items()],
        by_access_role=[dict(access_role=r,method=m,scope=s,counts=dict(c),scores=node_scores(c)) for (r,m,s),c in rolec.items()],
        elapsed_s=time.monotonic()-started,evaluation_protocol_sha256=source.sha(DATA/'evaluation_protocol.json'),source_sha256=source.sha(__file__),
        limits=['Project-specific object-map3m diagnostic, not official nuScenes leaderboard','Official sample position and past-interval speed not exact exposure-time/instantaneous truth','Body orientation remains unknown; prediction geometry and graph relations are not vehicle safety/closed-loop arrival','All data roles retained; no method selection from used-test or fixed new scenes']))
    with gzip.open(target/'scene_summary.jsonl.gz','xt') as f:
        for r in scene_rows:f.write(json.dumps(r,separators=(',',':'))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','evaluate']);a=p.parse_args()
    if a.command=='freeze':freeze()
    else:evaluate()

if __name__=='__main__':main()
