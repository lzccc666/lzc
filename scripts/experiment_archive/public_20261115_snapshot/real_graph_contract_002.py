"""真实感知候选图的语义契约；协议上下文版本不冒充人工核验或真实全局实例正确。"""
import json
from conceptgraph.completion.semantic_contract import validate
PROFILE='real_perception_graph_proposal_v2'
def build_completion_prompt(task,row,profile=PROFILE):
 if task!='candidate_rerank':raise ValueError('only candidate rerank')
 ids=row['candidate_ids'];example=dict(schema_version='1.0.0',selected_node_id=ids[0] if ids else None,ranking=ids,no_match=not bool(ids),explanation='按观测证据判断',confidence=.8)
 payload={k:row[k] for k in ['query_text','candidate_nodes','relation_metadata']}
 return ('只输出一个JSON对象。你在车辆真实感知生成的候选图中选目标；位置x为自车前方、y为左方，米。输入先后顺序不代表答案顺序。'
 '目标必须满足全部指令；同类未规定先后时选距离最近；第二近和否定最近、参照物左右前后条件必须检查。'
 '普通选择任务只提议目标编号，不是控制授权；可按类别、距离、关系选择当前观测到的对象。只有指令明确要求证据可靠或授权时才检查下列门限：motion_state=static，至少3次observations，至少5个lidar_pixels，support_fraction>=0.4，'
 'source=current_projected_lidar，evidence.version等于version，0<=now_s-last_seen_s<=2，conflict=false。'
 '授权检查只验证观测一致性，不保证真实身份或路线安全；明确要求授权的指令中任一条件不满足就拒绝该对象。实际驾驶永远须由另一个控制授权门闸再次检查，普通目标提议不能绕过它。'
 '没有匹配、目标缺失或无法确定时selected_node_id=null且no_match=true；其他时候选ID须在ranking首位，no_match=false。'
 'ranking必须包含给定全部候选ID且各一次，不生成新ID。不要编造颜色和细属性。\n'
 '输出字段示例：'+json.dumps(example,ensure_ascii=False,separators=(',',':'))+'\n输入：'+json.dumps(payload,ensure_ascii=False,separators=(',',':')))

def reliable(n):
 e=n.get('evidence',{});age=n['now_s']-n['last_seen_s']
 return n.get('motion_state')=='static' and e.get('source')=='current_projected_lidar' and e.get('observations',0)>=3 and e.get('lidar_pixels',0)>=5 and e.get('support_fraction',0)>=.4 and e.get('version')==n.get('version') and 0<=age<=2 and e.get('conflict') is False
