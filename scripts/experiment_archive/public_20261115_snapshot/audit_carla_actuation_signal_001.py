"""保存记录中的真实加速度超限与控制信号路径核查；不生成反事实到达。"""
import ast
import json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest

R=Path(__file__).resolve().parents[1]
O=R/'carla_official/actuation_signal_audit_001'
SOURCE=R/'source/carla_ablation_episode_003.py'

def names(node):
 return sorted({n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)})

def main():
 O.mkdir(exist_ok=False)
 tree=ast.parse(SOURCE.read_text())
 signal=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign):
   targets={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
   if targets & {'desired','desired_throttle','throttle','brake','ctrl'}:
    signal.append(dict(line=node.lineno,targets=sorted(targets),inputs=names(node.value),source=ast.get_source_segment(SOURCE.read_text(),node)))
 # This checks direct AST references only, not a formal transitive information-flow proof.
 throttle_signals=[s for s in signal if set(s['targets']) & {'desired_throttle','throttle','brake','ctrl'}]
 direct_acceleration_reference=any('acceleration' in s['inputs'] for s in throttle_signals)
 assert not direct_acceleration_reference
 reports=[];spikes=[];sources={str(SOURCE):digest(SOURCE),str(Path(__file__)):digest(__file__)}
 for p in sorted((R/'carla_official/ablation_batch_003').glob('*/*/execution.jsonl')):
  E=p.parent;release=json.loads((E/'release.json').read_text());spec=release['vehicle_spec'];rows=list(map(json.loads,p.open()));assert len(rows)>=2
  summary=json.loads((E/'summary.json').read_text()) if (E/'summary.json').exists() else None
  t=np.array([r['state']['time_s'] for r in rows]);v=np.array([r['state']['speed_mps'] for r in rows]);a=np.diff(v)/np.diff(t);limit=spec['max_acceleration_mps2'];bad=np.flatnonzero(a>limit+1e-5)+1
  for i in bad:
   before,after=rows[i-1],rows[i]
   spikes.append(dict(episode=E.parent.name,condition=E.name,tick=after['tick'],time_s=after['state']['time_s'],actual_dt_s=float(t[i]-t[i-1]),measured_acceleration_mps2=float(a[i-1]),acceptance_limit_mps2=limit,previous_measured_speed_mps=float(v[i-1]),measured_speed_mps=float(v[i]),applied_control=after['applied_control'],previous_next_control=before['next_control'],next_control=after['next_control'],actual_front_wheel_deg=after['actual_front_wheel_deg'],reported_desired_speed_mps=after['tracking'].get('desired_speed_mps'),actual_low_speed_desired_cap_mps=.55,official_vehicle_z_m=after['ego_transform_carla'][2][3]))
  audit=E/'independent_audit_003/report.json';independent=json.loads(audit.read_text()) if audit.exists() else None
  reports.append(dict(episode=E.parent.name,condition=E.name,observed_ticks=len(rows),completed=summary is not None,reported_status=summary['status'] if summary else 'INCOMPLETE_STORAGE_STOP',geometric_arrival=independent['geometric_identity_stop_arrival'] if independent else None,strict_safe_arrival=independent['strict_safe_arrival'] if independent else None,maximum_acceleration_mps2=float(a.max()),minimum_acceleration_mps2=float(a.min()),acceleration_violations=int(len(bad)),maximum_speed_mps=float(v.max()),negative_query=not release['episode']['positive']))
  for f in [p,E/'release.json']+([audit] if audit.exists() else []):sources[str(f)]=digest(f)
 dump(O/'report.json',dict(episodes_attempted=len(reports),episodes_complete=sum(r['completed'] for r in reports),actual_new_physical_episodes=0,total_over_limit_ticks=len(spikes),episode_statistics=reports,over_limit_tick_evidence=spikes,control_signal_paths=sorted(signal,key=lambda s:s['line']),tracker_acceleration_directly_referenced_by_throttle_or_brake=False,interpretation='车辆约束跟踪器输出acceleration，但实际油门/刹车采用独立目标速度与slew/coast规则。其余跟踪字段仍影响控制，不能说整个规划器无效。当前源码没有将跟踪器加速度请求直接用于执行器映射。',causal_limitation='超限是真实官方速度差分；信号核查不能单独证明超限成因，未运行替代执行器，不能据此声称已经修复/新增到达。路面、扭矩、离散物理和低速特性可能共同影响，均未作为已证实解释。',repair_contract='未来执行器应显式使用纵向加速度请求，结合实测速率闭环与饱和/抗积分饱和；先在官方CARLA独立标定多姿态启停，再冻结并重跑同一传感器闭环清单。不得事后放宽1m/s2门限，也不得用离线重放替代真实物理。',inputs_sha256=sources,whole_project_sealed_test=False,goal_complete=False))
 print(json.dumps(dict(episodes=len(reports),complete=sum(r['completed'] for r in reports),over_limit_ticks=len(spikes),new_physical_episodes=0),ensure_ascii=False))

if __name__=='__main__':main()
