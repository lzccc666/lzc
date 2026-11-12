"""独立用排序/线性分位数重算计时统计，逐项绑定真实输入和已有执行来源。"""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import gzip,json,math,pickle
from conceptgraph.completion.stage9 import OUTPUT,digest,dump

root=OUTPUT/'performance/runtime_profile_cpu_smoke_001/audit_001';root.mkdir(exist_ok=False)
planpath=OUTPUT/'performance/runtime_profile_protocol_001/plan.json';plan=json.loads(planpath.read_text())
checks=[]
def quantile(values,p):
 values=sorted(values);x=p*(len(values)-1);a=math.floor(x);b=math.ceil(x)
 return values[a]+(x-a)*(values[b]-values[a])
for run in ['runtime_profile_cpu_frontend_001','runtime_profile_cpu_recorded_001']:
 folder=OUTPUT/'performance'/run;report=json.loads((folder/'report.json').read_text())
 assert report['device']=='cpu' and not report['unique_CUDA_process_verified']
 assert report['CPU_shared_host_no_exclusive_performance_credit'] and report['end_to_end_throughput_credit'] is False
 rows=list(map(json.loads,(folder/'timings.jsonl').open()));groups=defaultdict(list)
 for r in rows:
  assert math.isfinite(r['duration_s']) and r['duration_s']>0 and isinstance(r['succeeded'],bool)
  groups[r['stage']+'/'+r['phase']].append(r)
 assert set(groups)==set(report['stages'])
 for key,items in groups.items():
  values=[r['duration_s'] for r in items];total=math.fsum(values);saved=report['stages'][key]
  assert saved['calls']==len(items) and saved['failures']==sum(not r['succeeded'] for r in items)
  for name,expected in [('median_s',quantile(values,.5)),('p95_s',quantile(values,.95)),('summed_s',total),('successful_calls_per_measured_s',sum(r['succeeded'] for r in items)/total)]:
   assert math.isclose(saved[name],expected,rel_tol=1e-12,abs_tol=1e-12)
 checks.append(dict(run=run,timing_rows=len(rows),groups=len(groups),report_sha256=digest(folder/'report.json'),timings_sha256=digest(folder/'timings.jsonl')))
front=OUTPUT/'performance/runtime_profile_cpu_frontend_001'
records=json.loads((front/'frontend_inputs_outputs.json').read_text())['records'];assert len(records)==12
for i,(r,source) in enumerate(zip(records,plan['frontend_inputs'])):
 assert all(r[k]==v for k,v in source.items()) and digest(source['rgb'])==source['rgb_sha256']
 path=front/f'{i:03d}_detections.pkl.gz';assert digest(path)==r['output_sha256']
 with gzip.open(path) as h:gobs=pickle.load(h)
 assert gobs['frame_provenance']['rgb_sha256']==source['rgb_sha256']
 count=len(gobs['xyxy']);assert count==r['detections']
 assert all(len(gobs[k])==count for k in ['confidence','class_id','mask','image_crops','image_feats','text_feats'])
 assert len(gobs['classes'])==len(plan['classes'])
recorded=OUTPUT/'performance/runtime_profile_cpu_recorded_001'
frames=[json.loads(p.read_text()) for p in sorted(recorded.glob('frame_*.json'))]
assert len(frames)==len(plan['recorded_frames'])==10
assert all(not r['gt_sent_to_agent'] for r in frames)
assert [r['frame_idx'] for r in frames]==[r['frame_idx'] for r in plan['recorded_frames']]
case=Path(plan['recorded_case']);assert digest(case/'execution.jsonl')==plan['recorded_execution_sha256']
controls=list(map(json.loads,(case/'execution.jsonl').open()));assert len(controls)==391
# 本次控制数值重算已在实际runner内逐字段校验；此处核对来源/分母，不伪称第二套车辆实现。
recomputed=json.loads((recorded/'report.json').read_text())['actual']
assert recomputed['recomputed_saved_control_states']==391 and recomputed['new_closed_loop_executions']==0
hardware={}
for name in ['hardware_queued.json']:
 path=OUTPUT/'performance/runtime_profile_after_faults_001'/name
 if path.exists():hardware[name]=dict(path=str(path),sha256=digest(path))
report=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
 CPU_actual_frontend_calls=12,frontend_source_chains=12,recorded_frames=10,recomputed_control_states=391,
 original_plan_sha256=digest(planpath),CUDA_model_calls=0,Qwen_model_calls=0,
 hardware_snapshots=hardware,
 limitations=['CPU共享负载，不是独占性能','计时值来源于perf_counter_ns；本次独立重算统计，不是第二个外部时钟校准',
 '391状态使用既有车辆函数重算，不是第二套独立物理实现或新执行','嵌套阶段不相加；成功仅表示函数正常返回，模型/格式/准确率另列',
 '规划只有2次，控制仅10次；小分母P95不可泛化','固定4开发场景12RGB与一个合成任务的录制链，不是完整闭环吞吐'],
 formal_GPU_performance_complete=False,goal_complete=False)
dump(root/'report.json',report)
lines=['# CPU分阶段计时试跑与独立统计复核','','12个真实RGB的YOLO/MobileSAM/CLIP重新执行；10帧真实保存观测重算建图/规划，391个保存控制状态逐字段重算。没有新的车辆执行或到达信用，Qwen/GPU调用0。','','|阶段/条件|调用|中位秒|P95秒|函数异常|','|---|---:|---:|---:|---:|']
for run in ['runtime_profile_cpu_frontend_001','runtime_profile_cpu_recorded_001']:
 r=json.loads((OUTPUT/'performance'/run/'report.json').read_text())
 for stage,v in r['stages'].items():lines.append(f"|{stage}|{v['calls']}|{v['median_s']:.6f}|{v['p95_s']:.6f}|{v['failures']}|")
lines += ['','CPU共享主机，以上不能作为GPU实时性能或与后台GPU主矩阵混成一张性能表。规划只有2调用、控制10调用，不能将P95解释为总体上界。外部阶段包含内部阶段；CLIP含预处理、图像与文本编码，不是纯编码核函数。模型加载和首次调用分开，未清OS缓存，不宣称冷机。','',
 '正式GPU队列等待既有三个作业真实退出，原Qwen/旧Adapter/新Adapter使用相同72条不同缓存键请求，各执行72无缓存+72命中复跑；新调用共216。GPU前端12输入独立子进程，采样监测其它CUDA PID，污染则取消独占信用。当前全部GPU性能调用0；queue存在不是执行证据。','',
 '原协议001有一组重复排序输入，语义未执行；002去重后216个真实模型配置缓存键预检通过，在cache.get处截获并抛专用异常，未加载模型、未写语义响应。不能计入模型准确率或缓存推理复跑。']
(root/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(passed=True,actual_CPU_frontend_calls=12,recorded_frames=10,recomputed_control_states=391,goal_complete=False)))
