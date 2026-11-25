"""独立解码720冻结检测缓存，核原图hash、mask、CLIP和类别字段完整性。"""
import gzip,json,pickle
import numpy as np
from conceptgraph.project_paths import PROJECT_ROOT
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump

def main():
 root=PROJECT_ROOT;p=root/'dataset/processed/mv136_20261123_sixscene_v1';m=json.loads((p/'manifest.json').read_text());rows=[]
 for s in m['scenes']:
  for g in s['groups']:
   for c,v in g['views'].items():
    with gzip.open(v['detections'],'rb') as f:d=pickle.load(f)
    n=len(d['xyxy']);assert len(d['mask'])==len(d['class_id'])==len(d['confidence'])==len(d['image_feats'])==n
    assert d['frame_provenance']['rgb_sha256']==digest(v['rgb'])
    if n:
     assert d['mask'].shape[1:]==(v['height'],v['width']) and np.isfinite(d['image_feats']).all() and np.isfinite(d['confidence']).all();assert np.all((d['class_id']>=0)&(d['class_id']<6))
    assert np.isfinite(d['text_feats']).all() and 'image_crops' not in d
    rows.append(dict(scene=s['scene'],frame_idx=g['frame_idx'],channel=c,reused=v['reused_frontend'],detections=n,mask_shape=list(d['mask'].shape),image_feats_shape=list(d['image_feats'].shape),text_feats_shape=list(d['text_feats'].shape),detections_sha256=digest(v['detections'])))
 out=dict(passed=True,views=len(rows),new_rear_views=sum(not x['reused'] for x in rows),reused_front_views=sum(x['reused'] for x in rows),detections=sum(x['detections'] for x in rows),new_rear_detections=sum(x['detections'] for x in rows if not x['reused']),rgb_hash_equal_all=True,actual_CLIP_and_masks_present_all=True,GT_read=False,new_model_calls=0,manifest_sha256=digest(p/'manifest.json'),source_sha256=digest(__file__),records=rows)
 dump(root/'outputs/reports/multiview_viewcount_20261123/sensors/features_independent_audit.json',out);print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
