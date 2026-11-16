"""公开标注几何图上的受控语义诊断契约；不给图像、感知、路径或人工标注信用。"""
import json
from conceptgraph.completion.semantic_contract import build_completion_prompt as old_prompt,validate
PROFILE='official_graph_evidence_v1'
def build_completion_prompt(task,row,profile=PROFILE):
 if task!='candidate_rerank':return old_prompt(task,row,'completion_v3')
 ids=[c['node_id'] for c in row['candidate_nodes']]
 template=dict(schema_version='1.0.0',selected_node_id=ids[0],ranking=ids,no_match=False,explanation='根据当前证据选择',confidence=.9)
 payload={k:row[k] for k in ['query_text','candidate_nodes','relation_metadata']}
 return ('只返回一个JSON对象。任务是语义目标选择，不是路径安全证明。候选位置x为自车前方、y为自车左侧，单位米。'
 '编号是对象身份标识，输入先后不是答案顺序。只允许给定候选，ranking必须包含全部ID且各一次；选择的ID须在首位。'
 '没有匹配或无法确定时selected_node_id=null，no_match=true。不得猜测缺少的证据。'
 '检查属性和证据：至少3次观测、至少5个LiDAR像素、来源official_annotation_geometry，'
 'evidence.version等于version，now_s-last_seen_s<=2，conflict=false。任一不满足就不能授权选择该对象。'
 '所有位置和类别均是本诊断给定的观测；不从文本以外编造颜色、运动或可达性。'
 '多个匹配且未规定顺序时优先距离最近；有第二近、否定最近、左右前后限制时严格按指令。\n'
 '输出字段：'+json.dumps(template,ensure_ascii=False,separators=(',',':'))+'\n输入：'+json.dumps(payload,ensure_ascii=False,separators=(',',':')))
