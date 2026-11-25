"""过去与当前已补偿到世界坐标的对象点云，平移ICP运动估计；证据弱时保持原稳健质心估计。"""
import copy
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
PARAMS=dict(max_correspondence_m=.6,minimum_points=8,iterations=15,trim_fraction=.8,minimum_source_fitness=.5,maximum_rmse_m=.25,minimum_xy_covariance_eigenvalue_m2=.015,maximum_points=512,velocity_history=3)
COUNTERS=Counter()
def translation(source,target):
 source=np.asarray(source);target=np.asarray(target)
 if min(len(source),len(target))<PARAMS['minimum_points']:return None,dict(reason='too_few_points')
 source=source[np.linspace(0,len(source)-1,min(len(source),PARAMS['maximum_points']),dtype=int)];target=target[np.linspace(0,len(target)-1,min(len(target),PARAMS['maximum_points']),dtype=int)]
 if min(np.linalg.eigvalsh(np.cov(source[:,:2].T)))<PARAMS['minimum_xy_covariance_eigenvalue_m2']:return None,dict(reason='XY_geometry_aperture_ambiguity')
 tree=cKDTree(target);offset=np.zeros(3)
 for _ in range(PARAMS['iterations']):
  transformed=source+offset;dist,idx=tree.query(transformed);valid=np.flatnonzero(dist<=PARAMS['max_correspondence_m'])
  if len(valid)<PARAMS['minimum_points']:return None,dict(reason='too_few_correspondences')
  valid=valid[np.argsort(dist[valid])[:max(PARAMS['minimum_points'],round(len(valid)*PARAMS['trim_fraction']))]]
  correction=np.median(target[idx[valid]]-transformed[valid],axis=0);correction[2]=0;offset+=correction
  if np.linalg.norm(correction)<1e-4:break
 dist,idx=tree.query(source+offset);mask=dist<=PARAMS['max_correspondence_m'];fitness=float(mask.mean());rmse=float(np.sqrt(np.mean(dist[mask]**2))) if mask.any() else float('inf');passed=fitness>=PARAMS['minimum_source_fitness'] and rmse<=PARAMS['maximum_rmse_m']
 return (offset if passed else None),dict(reason='translation_icp' if passed else 'poor_fitness_or_residual',fitness=fitness,rmse_m=rmse,xy_translation_to_past_m=offset[:2].tolist())
def install(isolated_mapping):
 import conceptgraph.slam.utils as utility
 original=utility.estimate_observation_velocity
 def estimate(cfg,obj,det):
  key=(str(obj.get('stable_object_id',obj.get('inst_color'))),obj.get('timestamp_s'),det.get('timestamp_s'))
  cached=det.get('_translation_icp_cached')
  # merge内部再次调用时复用“合并前”的配准，严禁拿含当前帧自身的融合云算零运动。
  if cached is not None and cached['key']==key:COUNTERS['merge_repeat_cached']+=1;return cached['value']
  baseline=original(cfg,obj,det);dt=(det.get('timestamp_s') or 0)-(obj.get('timestamp_s') or 0);offset=None;audit=dict(reason='invalid_or_duplicate_time')
  if dt>0:offset,audit=translation(np.asarray(det['pcd'].points),np.asarray(obj['pcd'].points))
  COUNTERS[audit['reason']]+=1
  if offset is not None:
   history=obj.setdefault('_translation_icp_velocity_history',[]);history.append((-offset[:2]/dt).tolist());history[:]=history[-PARAMS['velocity_history']:];speed=float(np.linalg.norm(np.median(history,axis=0)));value=(speed,baseline[1],baseline[2])
  else:value=baseline
  obj['_translation_icp_last_audit']=dict(audit,dt_s=dt,velocity_mps=float(value[0]),fallback_robust_xy=offset is None)
  det['_translation_icp_cached']=dict(key=key,value=value)
  return value
 utility.estimate_observation_velocity=estimate
 isolated_mapping.estimate_observation_velocity=estimate
 return COUNTERS
