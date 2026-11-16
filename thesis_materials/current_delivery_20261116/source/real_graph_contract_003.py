"""真实感知候选图的语义契约；协议上下文版本不冒充人工核验或真实全局实例正确。"""
import json
from conceptgraph.completion.semantic_contract import validate as legacy_validate, CompletionRanking
from conceptgraph.vlm.schemas import SchemaValidationResult,extract_json_object
from pydantic import Field,ValidationError
from typing import Literal
from collections import Counter,defaultdict
PROFILE='real_perception_graph_proposal_v3_empty_supported'
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


class EmptyRanking(CompletionRanking):
 selected_node_id: None = None
 ranking: list[int] = Field(min_length=0,max_length=0)
 no_match: Literal[True]

def validate(task,raw,*,allowed_candidate_ids=None):
 if task!='candidate_rerank' or allowed_candidate_ids!=[]:return legacy_validate(task,raw,allowed_candidate_ids=allowed_candidate_ids)
 parsed,errors=extract_json_object(raw)
 if parsed is None:return SchemaValidationResult(task,raw,None,None,errors)
 try:output=EmptyRanking.model_validate(parsed).model_dump()
 except (ValueError,TypeError,ValidationError) as exc:return SchemaValidationResult(task,raw,parsed,None,[dict(code='empty_candidate_schema_invalid',loc=[],message=str(exc))])
 return SchemaValidationResult(task,raw,parsed,output,[])

def audit_rows(rows):
 ids=[r['sample_id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate sample IDs')
 logs=defaultdict(set)
 for r in rows:
  if r['task']!='candidate_rerank':raise ValueError('unsupported training task')
  logs[r['split']].add(r['log_token']);v=validate(r['task'],r['response'],allowed_candidate_ids=r['candidate_ids'])
  if not v.valid:raise ValueError(f"invalid label {r['sample_id']}: {v.validation_errors}")
 if logs['train']&logs['validation']:raise ValueError('train validation log leakage')
 return dict(counts=dict(Counter(r['split']+'/'+r['task'] for r in rows)),independent_logs={k:sorted(v) for k,v in logs.items()},empty_candidates_retained=sum(not r['candidate_ids'] for r in rows))
