"""第7sharedscan辅助独立表；只读主六封存GT评价，不更改主结果与统计家庭。"""
from __future__ import annotations
import gzip,json,time
from collections import Counter,defaultdict
import numpy as np
from conceptgraph.completion import nuscenes_full_20261125 as source
from conceptgraph.completion.nuscenes_full_20261125_single8_context import configure
configure()
from conceptgraph.completion import nuscenes_full_20261125_run as base
from conceptgraph.completion import nuscenes_full_20261125_single8_evaluate as ev
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.nuscenes_full_20261125_single8_evaluate import relations_from_centers,edge_counts,scores
from conceptgraph.completion.temporal_identity_evaluation import prepare_sequence,identity_counts,return_diagnostics

from conceptgraph.slam.multiview_20261123_fusion import in_view
CONTROL='S2_6_sharedscan'


def main():
    tick=time.monotonic();protocol=json.loads((base.DATA/'sharedscan_auxiliary_protocol_v1.json').read_text())
    for p,h in protocol['source_sha256'].items():assert source.sha(p)==h,p
    assert json.loads((base.RUN/'inference_complete.json').read_text())['complete']
    assert json.loads((base.RUN/'evaluation/summary.json').read_text())['complete']
    manifest=json.loads((base.DATA/'manifest.json').read_text());target=base.RUN/'sharedscan_evaluation';target.mkdir(exist_ok=False)
    totals=defaultdict(Counter);edges=defaultdict(Counter);motions=defaultdict(Counter);bylog=defaultdict(Counter);scenes=[];pending=[];source_counts=Counter()
    for scene in manifest['scenes']:
        name=scene['scene'];done=base.RUN/name/'sharedscan_complete.json'
        if not done.exists():pending.append(dict(scene=name,frames=scene['frames'],reason='auxiliary output not sealed; full original scope retained'));continue
        receipt=json.loads(done.read_text());assert receipt['complete'] and source.sha(base.RUN/name/'sharedscan_frames.jsonl.gz')==receipt['frames_sha256']
        primaries={r['frame_idx']:r for r in source.rows(base.RUN/name/'frames.jsonl.gz')};truths={r['frame_idx']:r for r in source.rows(base.RUN/'evaluation'/(name+'.jsonl.gz'))}
        cc=defaultdict(Counter);ee=defaultdict(Counter);mm=defaultdict(Counter);sequences=defaultdict(list);last={};rows=0
        with gzip.open(target/(name+'.jsonl.gz'),'xt',compresslevel=6) as out:
            for control in source.rows(base.RUN/name/'sharedscan_frames.jsonl.gz'):
                index=control['frame_idx'];frame=primaries[index];truthrow=truths[index]
                assert control['sample_token']==frame['sample_token']==truthrow['sample_token'];assert control['render_timestamp_s']==frame['render_timestamp_s']
                predictions=control['payload']['predictions'];node_by_id={n['node_id']:n for n in control['payload']['nodes']};truth=truthrow['evaluation_only_official'];result={}
                source_counts['frames']+=1;source_counts['input_differs_from_main_groups']+=int(any(v['applied_lidar_token']!=v['main_protocol_lidar_token'] for v in control['view_source_protocol'].values()))
                source_counts['groups_without_earliest_scan']+=int(control['shared_scan_requested_token'] is None)
                for scope,cameras in ev.SCOPES.items():
                    gt=[o for o in truth if set(o['visible_cameras'])&set(cameras)];pred=[p for p in predictions if any(in_view(p['center'],frame['views'][c]) for c in cameras)]
                    metric=object_frame_metrics(pred,gt,threshold_m=3.);c=cc[scope]
                    for field in ['truth','predictions','true_positive','unmatched_predictions','false_negative','duplicate_nodes']:c[field]+=metric[field]
                    c['frames']+=1;c['error_sum_m']+=sum(m['distance_m'] for m in metric['matches']);mapping={pred[m['prediction_index']]['node_id']:gt[m['truth_index']]['instance_token'] for m in metric['matches']};bygt={m['truth_index']:m for m in metric['matches']}
                    mainmetric=truthrow['result'][scope]['S2_6']['metrics'];common={m['truth_index'] for m in mainmetric['matches']}&set(bygt)
                    for j in common:
                        p=pred[bygt[j]['prediction_index']];obj=gt[j];c['common_objects']+=1;c['common_error_sum_m']+=bygt[j]['distance_m'];key=(scope,obj['instance_token']);previous=last.get(key)
                        if previous and previous['frame']==index-1:
                            c['common_adjacent_pairs']+=1;c['common_jitter_sum_m']+=float(np.linalg.norm((np.asarray(p['center'])-previous['pred'])[:2]-(np.asarray(obj['center'])-previous['truth'])[:2]))
                        last[key]=dict(frame=index,pred=np.asarray(p['center']),truth=np.asarray(obj['center']))
                    forward=np.asarray(frame['views']['CAM_FRONT']['camera_pose'])[:2,2]
                    gt_edges=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in gt],forward_xy=forward,max_distance_m=12.)
                    geometric=relations_from_centers([dict(id=p['node_id'],center=p['center']) for p in pred],forward_xy=forward,max_distance_m=12.)
                    geom=edge_counts(geometric,gt_edges,mapping);production=edge_counts(ev.production_facts(control['payload'],pred),gt_edges,mapping)
                    ee[scope,'geometry'].update(geom);ee[scope,'production'].update(production)
                    motion=mm[scope]
                    for p in pred:motion['prediction:'+node_by_id[p['node_id']]['verified_motion_state']]+=1
                    for j,obj in enumerate(gt):
                        match=bygt.get(j);node=node_by_id[pred[match['prediction_index']]['node_id']] if match else None
                        for reference in ['attribute_motion','past_displacement_motion']:
                            label=obj[reference]
                            for kind in ['estimated_motion_state','verified_motion_state']:
                                state=node[kind] if node else 'missing_node';prefix=reference+':'+kind
                                motion[prefix+':confusion:'+label+'->'+state]+=1
                                if label!='unknown':motion[prefix+':all_labeled']+=1;motion[prefix+':correct']+=int(label==state);motion[prefix+':known']+=int(state in ['static','dynamic'])
                    sequences[scope].append(dict(frame_idx=index,camera_timestamp_s=frame['render_timestamp_s'],predictions={CONTROL:pred},evaluation_only_official=gt,metrics={CONTROL:metric}))
                    result[scope]=dict(metrics=metric,prediction_node_ids=[p['node_id'] for p in pred],geometry=dict(geom),production=dict(production),common_GT_with_main6=len(common))
                out.write(json.dumps(dict(scene=name,frame_idx=index,sample_token=frame['sample_token'],result=result),separators=(',',':'))+'\n');rows+=1
        assert rows==scene['frames']
        for scope,seq in sequences.items():
            prepared=prepare_sequence(seq,CONTROL);cc[scope].update(identity_counts(prepared,'all_gated')['counts']);cc[scope].update(return_diagnostics(prepared)['counts'])
        for scope,c in cc.items():totals[scope].update(c);bylog[scene['log_token'],scope].update(c);scenes.append(dict(scene=name,scope=scope,counts=dict(c),scores=ev.node_scores(c)))
        for k,c in ee.items():edges[k].update(c)
        for k,c in mm.items():motions[k].update(c)
    source.dump(target/'summary.json',dict(complete=not pending,method=CONTROL,expected_scenes=850,expected_frames=34149,evaluated_scenes=850-len(pending),pending=pending,full_denominator_not_claimed_if_pending=bool(pending),
        node=[dict(scope=s,counts=dict(c),scores=ev.node_scores(c)) for s,c in totals.items()],relations=[dict(scope=s,kind=k,counts=dict(c),scores=scores(c)) for (s,k),c in edges.items()],
        motion=[dict(scope=s,counts=dict(c)) for s,c in motions.items()],bylog=[dict(log=l,scope=s,counts=dict(c),scores=ev.node_scores(c)) for (l,s),c in bylog.items()],source_counts=dict(source_counts),
        auxiliary_only=True,primary6_tables_and5tests_unchanged=True,elapsed_s=time.monotonic()-tick,protocol_sha256=source.sha(base.DATA/'sharedscan_auxiliary_protocol_v1.json'),limits=['Full same-scan control; not the previously used120frame result','Unknown motion/shape preserved; no GT enters mapping','Auxiliary missing technical output remains pending, not a selected-success result']))
    with gzip.open(target/'scene_summary.jsonl.gz','xt') as f:
        for r in scenes:f.write(json.dumps(r,separators=(',',':'))+'\n')

if __name__=='__main__':main()
