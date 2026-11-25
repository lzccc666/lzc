"""从冻结节点/真实trace导出方向旁路与分类覆盖统计；不改变运动判定。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,gzip,hashlib,json,math
from collections import Counter,defaultdict
from itertools import combinations
import numpy as np
from conceptgraph.project_paths import PROJECT_ROOT as ROOT

RUN=ROOT/'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2'
REPORT=ROOT/'outputs/reports/node_motion_unified_20261125'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readlines(p):
 with gzip.open(p,'rt') as f:
  for l in f:yield json.loads(l)
def vector(trace):
 times=[r['timestamp_s'] for r in trace]
 if len(times)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(r['raw_center_world_m'] is None for r in trace):return None
 centers=np.asarray([r['raw_center_world_m'][:2] for r in trace])
 pairs=[(centers[j]-centers[i])/(times[j]-times[i]) for i,j in combinations(range(len(times)),2) if times[j]-times[i]>=.75]
 return np.median(pairs,axis=0) if pairs else (centers[-1]-centers[-2])/(times[-1]-times[-2])

def main(scope):
 source=RUN/(scope+'_frames.jsonl.gz');target=RUN/(scope+'_task_nodes.jsonl.gz');assert not target.exists()
 fingerprint=sha(source);stats=defaultdict(Counter);first=defaultdict(Counter);all_reasons=defaultdict(Counter);total=Counter()
 with gzip.open(target,'wt',compresslevel=6) as out:
  for frame in readlines(source):
   states={s['state_uid']:s for s in frame['evidence_states']};nodes=copy.deepcopy(frame['nodes'])
   for node in nodes:
    state=states.get(node['measurement']['state_uid']);v=vector(state['raw_trace']) if state else None
    raw_speed=node['measurement']['measured_raw_speed_mps']
    if v is not None and raw_speed is not None:assert abs(float(np.linalg.norm(v))-raw_speed)<1e-8
    valid=node['motion_state']!='unknown';heading=math.atan2(v[1],v[0]) if v is not None and np.linalg.norm(v)>1e-9 else None
    node.update(raw_velocity_world_xy_mps=v.tolist() if v is not None else None,
       raw_motion_heading_world_rad=heading,raw_velocity_valid_for_semantics=bool(valid and v is not None),
       motion_velocity_world_xy_mps=v.tolist() if valid and v is not None else None,
       motion_heading_available=bool(node['motion_state']=='dynamic' and node['motion_heading_world_rad'] is not None),
       body_heading_available=False)
    assert node['body_heading_world_rad'] is None
    group=node['category']+'|shape='+str(node['shape_available']);stats[group]['nodes']+=1;stats[group][node['motion_state']]+=1
    stats[group]['current_update']+=int(node['motion_time']['current_update']);stats[group]['raw_velocity_available']+=int(v is not None)
    total['nodes']+=1;total['raw_velocity_available']+=int(v is not None);total[node['motion_state']]+=1
    reasons=node['measurement']['unknown_reasons']
    if reasons:
     first[group][reasons[0]]+=1
     for r in reasons:all_reasons[group][r]+=1
   result={k:frame[k] for k in ['schema_version','scene','log','frame_idx','render_timestamp_s','ego_pose_world','ego_heading_world_xy','frame_source','predictions','task_predictions','GT_in_method','vehicle_safety_authorized']}
   result.update(nodes=nodes,evidence_source_file=str(source),evidence_source_sha256=fingerprint,
       direction_scope='raw displacement/velocity direction is not vehicle/pedestrian body orientation')
   out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
 summary=dict(passed=True,scope=scope,counts=dict(total),by_category_shape={k:dict(v) for k,v in stats.items()},
    unknown_first_reason={k:dict(v) for k,v in first.items()},unknown_all_reasons_overlapping={k:dict(v) for k,v in all_reasons.items()},
    method_motion_or_geometry_changed=False,body_heading_available=False,source_sha256=fingerprint,output_sha256=sha(target),extension_source_sha256=sha(__file__))
 with (RUN/(scope+'_node_coverage.json')).open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(scope=scope,counts=dict(total),output=str(target))),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('scope',choices=['R1','S2']);main(p.parse_args().scope)
