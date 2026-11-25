"""旧8真实节点逐值保留，仅CPU重建单八方向边；不运行前端/建图/GT。"""
from __future__ import annotations
import gzip,hashlib,json,shutil,time
from pathlib import Path
from collections import Counter
import numpy as np
from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as legacy
from conceptgraph.scenegraph import relation_single8_20261125 as geometry
from conceptgraph.project_paths import experiment_directory
source=legacy.source
DATA=source.ROOT/'dataset/processed/nuscenes_full_20261125_single8_pilot_v1'
RUN_ID='nuscenes_full_20261125_single8_pilot_v1'
RUN=legacy.RUN.parent/RUN_ID
CONTRACT='single8_v1_source_relative_to_target'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def reedge(payload,forward):
    snapshot=geometry.build_edge_snapshot([dict(id=p['node_id'],center=p['center']) for p in payload['predictions']],forward)
    local={str(v):int(k) for k,v in payload['relation_node_registry'].items()};new=dict(payload)
    new['edges']=[dict(source=local[str(e['source'])],target=local[str(e['target'])],relation=e['relation'],distance_m=e['distance_m']) for e in snapshot['edges']]
    new['relation_contract']=CONTRACT;new['relation_statistics']=snapshot['statistics'];new['relation_provider_sha256']=source.sha(geometry.__file__)
    old_nonedge={k:v for k,v in payload.items() if k!='edges'};new_nonedge={k:v for k,v in new.items() if k not in ['edges','relation_contract','relation_statistics','relation_provider_sha256']};assert old_nonedge==new_nonedge
    return new,snapshot['statistics'],digest(old_nonedge)

def run():
    tick=time.monotonic();DATA.mkdir(exist_ok=False);experiment_directory('nuscenes','validation',RUN_ID,create=True)
    own=[Path(__file__).resolve(),source.ROOT/'scripts/nuscenes_full_20261125_single8_reedge.py',Path(geometry.__file__).resolve()]
    old_manifest=json.loads((legacy.DATA/'manifest.json').read_text());old_method=json.loads((legacy.DATA/'method_manifest.json').read_text());core_sha=source.sha(geometry.__file__)
    protocol=dict(created_utc=source.now(),contract=CONTRACT,source_sha256={str(p):source.sha(p) for p in own},canonical='str(stable_id) ascending source/target; relation states source position relative to target',labels=list(geometry.LABELS),reference_frame=geometry.FRAME_SOURCE,sector_width_deg=45.,front_interval_deg=[-22.5,22.5],interval='lower inclusive upper exclusive',boundary_numeric_rule='8 ULP representational boundary normalization only',max_distance_m=12.,coincident_threshold_m=1e-6,near_edges=False,inverse_edges_stored=False,confidence_or_topk_pruning=False,all_finite_XY_nodes_include_dynamic_and_unknown=True,old_multirelation_pilot_preserved=True,GT_not_read=True,node_fusion_frontend_changes=False)
    source.dump(DATA/'relation_protocol_v1.json',protocol);source.dump(DATA/'manifest.json',dict(old_manifest,phase='CPU_relation_reencoding_only',original_manifest_sha256=source.sha(legacy.DATA/'manifest.json'),relation_protocol_sha256=source.sha(DATA/'relation_protocol_v1.json')))
    source.dump(DATA/'method_manifest.json',dict(old_method,relation_contract=CONTRACT,relation_provider_sha256=core_sha,original_method_manifest_sha256=source.sha(legacy.DATA/'method_manifest.json'),GPU_inference_reused_exactly=True,new_GPU_calls=0))
    totals=Counter();per_method={};scene_summaries=[]
    for scene in old_manifest['scenes']:
        name=scene['scene'];origin=legacy.RUN/name;old_complete=json.loads((origin/'complete.json').read_text());assert source.sha(origin/'frames.jsonl.gz')==old_complete['frames_sha256'] and old_complete['auxiliary_sharedscan']['complete']
        root=RUN/name;out=root/'attempts/reedge_v1';out.mkdir(parents=True);fronts={};checks=[]
        with gzip.open(out/'frames.jsonl.gz','xt',compresslevel=6) as output:
            for frame in source.rows(origin/'frames.jsonl.gz'):
                forward=np.asarray(frame['views']['CAM_FRONT']['camera_pose'])[:2,2];fronts[frame['sample_token']]=forward
                row=dict(frame);row['relation_contract']=CONTRACT;row['relation_provider_sha256']=core_sha;row['methods']={}
                for method,payload in frame['methods'].items():
                    new,stats,nodehash=reedge(payload,forward);row['methods'][method]=new;totals.update({method+':'+k:v for k,v in stats.items() if isinstance(v,int) and not isinstance(v,bool)})
                    checks.append(dict(scene=name,frame_idx=frame['frame_idx'],method=method,all_nonedge_payload_sha256=nodehash,predictions=len(payload['predictions']),new_edges=len(new['edges'])))
                output.write(json.dumps(row,separators=(',',':'))+'\n')
        with gzip.open(out/'sharedscan_frames.jsonl.gz','xt',compresslevel=6) as output:
            for frame in source.rows(origin/'sharedscan_frames.jsonl.gz'):
                row=dict(frame);new,stats,nodehash=reedge(frame['payload'],fronts[frame['sample_token']]);row['payload']=new;row['relation_contract']=CONTRACT;row['relation_provider_sha256']=core_sha
                totals.update({'S2_6_sharedscan:'+k:v for k,v in stats.items() if isinstance(v,int) and not isinstance(v,bool)})
                checks.append(dict(scene=name,frame_idx=frame['frame_idx'],method='S2_6_sharedscan',all_nonedge_payload_sha256=nodehash,predictions=len(frame['payload']['predictions']),new_edges=len(new['edges'])));output.write(json.dumps(row,separators=(',',':'))+'\n')
        for filename in ['source_members.json','frontend_receipt.json']:(out/filename).symlink_to(origin/filename)
        auxiliary=dict(old_complete['auxiliary_sharedscan'],frames_sha256=source.sha(out/'sharedscan_frames.jsonl.gz'),relation_contract=CONTRACT,relation_provider_sha256=core_sha,new_GPU_calls=0)
        source.dump(out/'sharedscan_complete.json',auxiliary)
        for filename in ['frames.jsonl.gz','source_members.json','frontend_receipt.json','sharedscan_frames.jsonl.gz','sharedscan_complete.json']:(root/filename).symlink_to((out/filename).relative_to(root))
        record=dict(old_complete,frames_sha256=source.sha(out/'frames.jsonl.gz'),auxiliary_sharedscan=auxiliary,accepted_attempt=str(out),relation_contract=CONTRACT,relation_provider_sha256=core_sha,original_closed_node_source_sha256=old_complete['frames_sha256'],new_GPU_calls=0)
        source.dump(root/'complete.json',record)
        with gzip.open(out/'nonedge_invariance.jsonl.gz','xt') as f:
            for check in checks:f.write(json.dumps(check,separators=(',',':'))+'\n')
        scene_summaries.append(dict(scene=name,frames=scene['frames'],method_frames=len(checks),main_frame_sha256=record['frames_sha256'],auxiliary_frame_sha256=auxiliary['frames_sha256'],all_nonedge_fields_exact=True,output_physical_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())))
        print(json.dumps(scene_summaries[-1]),flush=True)
    attempts=RUN/'runtime_attempts';attempts.mkdir()
    for prior in (legacy.RUN/'runtime_attempts').iterdir():assert (prior/'GPU_release.json').exists();(attempts/prior.name).symlink_to(prior,target_is_directory=True)
    source.dump(RUN/'manifest.json',dict(created_utc=source.now(),source_pilot_run=str(legacy.RUN),source_method_manifest_sha256=source.sha(legacy.DATA/'method_manifest.json'),relation_protocol_sha256=source.sha(DATA/'relation_protocol_v1.json'),scenes=old_manifest['scenes'],model_and_node_settings=old_method['configs'],source_sha256=protocol['source_sha256'],new_GPU_calls=0,GT_not_read=True))
    for p in own:
        target=RUN/'source_snapshot'/p.relative_to(source.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    source.dump(RUN/'summary.json',dict(passed=True,scenes=8,frames=sum(s['frames'] for s in scene_summaries),method_frames=sum(s['method_frames'] for s in scene_summaries),methods=7,totals=dict(totals),scene_summaries=scene_summaries,all_nonedge_fields_exact=True,node_predictions_motion_ID_source_unchanged=True,new_GPU_calls=0,GT_not_read=True,relation_provider_sha256=core_sha,relation_protocol_sha256=source.sha(DATA/'relation_protocol_v1.json'),elapsed_s=time.monotonic()-tick))
    print('SINGLE8_REEDGE_DONE',flush=True)
if __name__=='__main__':run()
