"""官方CARLA闭环几何消融：开关定义固定，不读取目标actor/终区真值。"""
from types import SimpleNamespace
from shapely.geometry import Point
import numpy as np
import carla_agent_isolated_002 as nav
from carla_agent_isolated_002 import PerceptionVehicleAgent,mapping_config
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper

CONDITIONS=['full_rule_geometry','current_frame_only','without_relations','without_history_expiry','without_evidence_check']
class NoRelations:
 def update_snapshot(self,*a,**kw):return SimpleNamespace(edges=[])

class Agent(PerceptionVehicleAgent):
 def __init__(self,query,prior,classes,dest,condition):
  super().__init__(query,prior,classes,dest);self.condition=condition
  if condition not in CONDITIONS:raise ValueError(condition)
  if condition=='without_relations':self.graph=NoRelations()
  if condition=='without_history_expiry':
   cfg=mapping_config();cfg.dynamic_ttl_s=1.e9;self.mapper=StreamlinedMapper(cfg,classes)
  if condition=='without_evidence_check':self.evidence_policy['enabled']=False
 def perceive(self,index,*a,**kw):
  if self.condition=='current_frame_only':self.mapper=StreamlinedMapper(mapping_config(),self.classes);self.attributes={}
  result=super().perceive(index,*a,**kw)
  result['ablation_condition']=self.condition
  result['semantic_coverage']={'clear_category_rule':True,'complex_Qwen_called':False,'scope':'仅清晰类别规则闭环；Qwen复杂语义另测，不能称完整语义系统消融'}
  return result
 def approach_goal(self,node_id):
  obj=next(o for o in self.mapper.objects if o.get('stable_object_id')==self.nodes[node_id].metadata['object_identity'])
  progress=np.array([self.line.project(Point(p[:2])) for p in np.asarray(obj['pcd'].points)])
  goal_s=max(0.,float(np.quantile(progress,.05))-nav.SPEC.front_from_rear_axle_m-2.)
  return nav.pose_on(self.line,goal_s)
