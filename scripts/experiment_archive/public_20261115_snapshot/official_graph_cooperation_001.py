"""受控标注图诊断中的可独立开关合作模块。与车载实时桥接另行集成验收。"""
import math,re
C={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
CONDITIONS=['full','without_rule_query_resolution','without_attribute_check','without_candidate_reranking','without_evidence_check']
def reliable(n):
 e=n.get('evidence',{});return e.get('source')=='official_annotation_geometry' and e.get('observations',0)>=3 and e.get('lidar_pixels',0)>=5 and e.get('version')==n.get('version') and 0<=n.get('now_s',float('inf'))-n.get('last_seen_s',float('-inf'))<=2 and not e.get('conflict',True)
def select(query,nodes,relations,prediction,condition='full'):
 if condition not in CONDITIONS:raise ValueError('unknown condition')
 allowed=[n for n in nodes if condition=='without_evidence_check' or reliable(n)];requested={c for c,word in C.items() if word in query}
 if condition!='without_attribute_check' and requested:allowed=[n for n in allowed if n['category'] in requested]
 if not allowed:return dict(selected_node_id=None,no_match=True,valid=True,source='deterministic_rejection',reason='no_current_supported_matching_candidate')
 clear=not any(x in query for x in ['编号','第二','不要'])
 if clear and condition!='without_rule_query_resolution':selected=min(allowed,key=lambda n:(n['distance_m'],n['node_id']))['node_id'];return dict(selected_node_id=selected,no_match=False,valid=True,source='clear_rule_resolution')
 if condition=='without_candidate_reranking':return dict(selected_node_id=allowed[0]['node_id'],no_match=False,valid=True,source='fixed_current_input_order')
 if not prediction or not prediction.get('schema_valid'):return dict(selected_node_id=None,no_match=True,valid=True,source='invalid_backend_rejection')
 p=prediction['normalized_output'];selected=p.get('selected_node_id');ids={n['node_id'] for n in allowed}
 if selected is None or selected not in ids:return dict(selected_node_id=None,no_match=True,valid=True,source='backend_abstention_or_outside_allowed')
 return dict(selected_node_id=selected,no_match=False,valid=True,source='Qwen_complex_resolution')
