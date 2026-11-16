"""真实感知图的编号提议与控制授权分开；当前为离线合作函数，闭环需另验。"""
import re,time
from real_graph_contract_003 import reliable
C={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
CONDITIONS=['full','without_rule_query_resolution','without_attribute_check','without_candidate_reranking','without_evidence_check']
def categories(query):
 # 关系指令的参照类别不作为目标类别；多目标“在A或B中...”保留两类。
 text=query.split('选择')[-1] if query.startswith('以') else query
 return {c for c,w in C.items() if w in text}
def choose(query,nodes,relations,prediction,condition='full'):
 if condition not in CONDITIONS:raise ValueError('unknown condition')
 requested=categories(query);allowed=list(nodes);explicit_authorization='授权' in query or '证据可靠' in query
 if condition!='without_attribute_check' and requested:allowed=[n for n in allowed if n['category'] in requested]
 if explicit_authorization and condition!='without_evidence_check':allowed=[n for n in allowed if reliable(n)]
 clear=not any(x in query for x in ['参照','第二','不要'])
 proposal=None;source='no_candidate';reason='no_compatible_observed_candidate';backend_used=False
 if allowed:
  if clear and condition!='without_rule_query_resolution':proposal=min(allowed,key=lambda n:(n['distance_m'],n['node_id']))['node_id'];source='clear_rule';reason='clear_condition_resolved_by_rule'
  elif condition=='without_candidate_reranking':proposal=allowed[0]['node_id'];source='fixed_input_order';reason='reranking_removed'
  elif prediction and prediction.get('schema_valid'):
   backend_used=True;proposal=prediction['normalized_output']['selected_node_id'];source='actual_saved_Qwen_output';reason='complex_or_rule_replacement'
   if proposal not in {n['node_id'] for n in allowed}:proposal=None;reason='backend_abstained_or_attribute_rejected'
  else:reason='invalid_backend_rejected';source='backend_invalid';backend_used=True
 chosen=next((n for n in nodes if n['node_id']==proposal),None);passed=chosen is not None and (condition=='without_evidence_check' or reliable(chosen));authorized=proposal if passed else None
 return dict(proposed_node_id=proposal,authorized_node_id=authorized,valid=True,source=source,reason=reason,backend_output_used=backend_used,explicit_authorization_query=explicit_authorization,evidence_check_enabled=condition!='without_evidence_check',current_evidence_passed=bool(chosen is not None and reliable(chosen)),ordinary_proposal_does_not_authorize_actuation=True)

def complete_geometry_rule(query,nodes,relations,*,evidence_check_enabled=True):
 """受限句法的强规则对照；全部基于预测图，未读取官方答案。"""
 requested=categories(query);allowed=[n for n in nodes if not requested or n['category'] in requested];explicit='授权' in query or '证据可靠' in query
 if explicit and evidence_check_enabled:allowed=[n for n in allowed if reliable(n)]
 if query.startswith('以'):
  anchor_text=query.split('为参照')[0][1:];anchor_categories={c for c,w in C.items() if w in anchor_text};anchors=[n for n in nodes if n['category'] in anchor_categories];axis=1 if '左侧' in query or '右侧' in query else 0;positive='左侧' in query or '前方' in query
  allowed=[n for n in allowed if any(a['node_id']!=n['node_id'] and abs(n['center_xy'][axis]-a['center_xy'][axis])>1 and (n['center_xy'][axis]>a['center_xy'][axis])==positive and sum((n['center_xy'][i]-a['center_xy'][i])**2 for i in [0,1])<=144 for a in anchors)]
 allowed=sorted(allowed,key=lambda n:(n['distance_m'],n['node_id']))
 if '第二' in query:allowed=allowed[1:]
 proposal=allowed[0]['node_id'] if allowed else None;chosen=allowed[0] if allowed else None
 return dict(proposed_node_id=proposal,authorized_node_id=proposal if chosen is not None and (not evidence_check_enabled or reliable(chosen)) else None,valid=True,source='restricted_complete_geometry_rule',current_evidence_passed=chosen is not None and reliable(chosen),backend_output_used=False,explicit_authorization_query=explicit,ordinary_proposal_does_not_authorize_actuation=True)
