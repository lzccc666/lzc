"""O2负结果的被动状态重放；六条件预测逐条等于冻结结果，不重评分。"""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from conceptgraph.project_paths import experiment_directory
from conceptgraph.completion.stage9 import PROJECT,digest,dump,predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_20261123_combined import CanonicalPhysicalFrontMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_ownership_restored_v2 import RestoredUniqueSupportCanonicalMapper,RestoredUniqueSupportOriginalFrontMapper
from scripts.multiview_20261123_run_graph import load_frame
from scripts.mvrepair_lifecycle_20261123_sameview_run import serial

P='mvrepair_lifecycle_20261123_'

def state_record(s):
    return {k:serial(s[k]) for k in ['node_id','category','center','last_measured_center','variance','velocity','timestamp',
        'hits','total_points','confidence','feature','_last_camera','_observation_keys','_last_world_points'] if k in s}

class TraceMixin:
    def update_group(self,*args,**kwargs):
        layer=self.position_layer
        before=[state_record(s) for s in [*layer.tracks,*layer.dense_states.values()]]
        out=super().update_group(*args,**kwargs)
        self.state_trace=dict(before=before,after=[state_record(s) for s in [*layer.tracks,*layer.dense_states.values()]],
            aliases=dict(layer.aliases),events=serial(layer.events),
            objects=[dict(node_id=o['stable_object_id'],category=o['class_name'][-1],n_detections=o['num_detections'],
                motion_state=o.get('motion_state'),is_short_term=o.get('is_short_term'),last_seen_frame=o.get('last_seen_frame'),
                timestamp_s=o.get('timestamp_s'),image_idx=o['image_idx'],mask_idx=o['mask_idx'],
                confidence=serial(o['conf']),current_source=self.measurement_sources[o['mask_idx'][-1]].get('_observation_key')
                if o['image_idx'][-1]==self.group_frames[0][1].frame_idx else None) for o in self.objects])
        return out

class TraceL4(TraceMixin,CanonicalPhysicalFrontMapper):pass
class TraceO2(TraceMixin,RestoredUniqueSupportCanonicalMapper):pass
class TraceOriginalOwnership(TraceMixin,RestoredUniqueSupportOriginalFrontMapper):pass
class OriginalFrontAdapter(SourceAuditedMapper):
    def __init__(self,cfg,classes):super().__init__(cfg,classes,deduplication=False)
    def update_group(self,frames,capture_times,projections=None):
        self.capture_times=capture_times;super().update_group(frames)
        return self.position_predictions(predict(self.confirmed_objects(),self.classes),frames[0][1])
class TraceOriginal(TraceMixin,OriginalFrontAdapter):pass

def main():
    start=time.monotonic();base=PROJECT/'outputs/results/nuscenes/validation/iterations'
    reference=base/(P+'O2_ownership_regression_v2')
    manifest_path=PROJECT/'dataset/processed/multiview_identity_repair_20261123_validation_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text());release=base/'mvrepair_20261123_L4_v1/mvrepair_20261123_release.json'
    frozen=json.loads(release.read_text());assert all(digest(p)==v['sha256'] for p,v in frozen['source_sha256'].items())
    run=experiment_directory('nuscenes','validation',P+'O2_state_causal_audit_v1',create=True)
    dump(run/'release.json',dict(source_sha256={str(Path(__file__).resolve()):digest(Path(__file__).resolve())},
        parent_release_sha256=digest(reference/(P+'release.json')),manifest_sha256=digest(manifest_path),
        role='passive diagnosis only, all predictions must equal saved output; no GT read',seed=17))
    exact=0
    for scene in manifest['scenes']:
        outdir=run/scene['scene'];outdir.mkdir()
        rows=list(map(json.loads,gzip.open(reference/scene['scene']/(P+'predictions.jsonl.gz'),'rt')))
        assert len(rows)==len(scene['groups'])==20
        classes={'L4_three':TraceL4,'O2_ownership_three':TraceO2,'L4_front':TraceL4,
            'O2_ownership_front':TraceO2,'A_front':TraceOriginal,'A_front_ownership_only':TraceOriginalOwnership}
        mappers={k:c(OmegaConf.create(frozen['config']),CLASSES) for k,c in classes.items()}
        with gzip.open(outdir/'state_trace.jsonl.gz','wt') as f:
            for group,row in zip(scene['groups'],rows):
                assert group['frame_idx']==row['frame_idx'] and group['sample_token']==row['sample_token']
                loaded={c:load_frame(group,c) for c in group['views']};trace={}
                for method,mapper in mappers.items():
                    cameras=list(group['views']) if method.endswith('_three') else ['CAM_FRONT']
                    frames=[(c,copy.deepcopy(loaded[c])) for c in cameras];np.random.seed(17+group['frame_idx'])
                    result=mapper.update_group(frames,{c:group['views'][c]['camera_timestamp_s'] for c in cameras},
                        {c:group['views'][c] for c in cameras})
                    assert result==row['predictions'][method],(scene['scene'],group['frame_idx'],method)
                    exact+=1;trace[method]=mapper.state_trace
                f.write(json.dumps(dict(scene=scene['scene'],frame_idx=group['frame_idx'],state_trace=trace),allow_nan=False)+'\n')
    report=dict(run=str(run),passive_exact_method_groups=exact,expected=240,wall_s=time.monotonic()-start,GT_read=False)
    dump(run/'report.json',report);dump(PROJECT/'outputs/reports/multiview_identity_repair_20261123/lifecycle/O2_state_replay_verification.json',report)
    print(json.dumps(report))

if __name__=='__main__':main()
