"""只读逐帧评价记录，核三/六共同GT的相邻原始ID变化；非人工身份真值。"""
import gzip,json,shutil
from collections import Counter
from pathlib import Path
from conceptgraph.project_paths import PROJECT_ROOT,experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump

def main():
 root=PROJECT_ROOT;source=root/'outputs/results/nuscenes/validation/iterations/mv136_20261123_main_v1';counts={s:Counter() for s in ['front','front3','union6']};inputs={}
 for scene in ['scene-0012','scene-0013','scene-0019','scene-0035','scene-0042','scene-0047']:
  path=source/scene/'mv136_20261123_evaluation.jsonl.gz';inputs[str(path)]=digest(path);previous={scope:{} for scope in counts};previousframe=-2
  for row in map(json.loads,gzip.open(path,'rt')):
   idx=row['frame_idx']
   for scope,c in counts.items():
    truth=row['evaluation_only_official'][scope];maps={}
    for method in ['T3_3','T3_6']:
     data=row['result'][method][scope];maps[method]={truth[m['truth_index']]['instance_token']:data['predictions'][m['prediction_index']]['node_id'] for m in data['metrics']['matches']}
    common=maps['T3_3'].keys()&maps['T3_6'].keys();c['common_GT_frames']+=len(common)
    if idx==previousframe+1:
     for token in common&previous[scope].keys():
      c['common_adjacent_GT_pairs']+=1
      for method in maps:c[method+'_raw_ID_changes']+=maps[method][token]!=previous[scope][token][method]
    previous[scope]={token:{method:maps[method][token] for method in maps} for token in common}
   previousframe=idx
 expected={'front':(142,88,4,4),'front3':(307,199,10,22),'union6':(313,205,12,20)}
 for scope,c in counts.items():assert tuple(c[k] for k in ['common_GT_frames','common_adjacent_GT_pairs','T3_3_raw_ID_changes','T3_6_raw_ID_changes'])==expected[scope]
 run=experiment_directory('nuscenes','validation','mv136_20261123_common_ID_audit_v1',create=True);(run/'source_snapshot').mkdir();shutil.copy2(__file__,run/'source_snapshot'/Path(__file__).name)
 out=dict(passed=True,scope_counts={k:dict(v) for k,v in counts.items()},input_sha256=inputs,source_sha256=digest(__file__),source_snapshot=str(run/'source_snapshot'/Path(__file__).name),new_model_calls=0,new_mapping_calls=0,interpretation='same GT object-frame intersection and consecutive frames per original scene, raw predicted node ID inequality; diagnostic uses automatic 3m evaluator assignments, not human identity truth or primary full-denominator IDF1')
 target=root/'outputs/reports/multiview_viewcount_20261123/independent/common_3vs6_identity_audit.json';dump(target,out);dump(run/'audit.json',out)
 report=root/'outputs/reports/multiview_viewcount_20261123/independent/六视角独立结果与主方案建议.md';text=report.read_text();text=text.replace('strictfront142帧/88对两者完全相同。另1/3/6', 'strictfront142帧/88对两者完全相同。同一共同集相邻原始ID变化也已只读独立核：front3为10→22，union6为12→20，strictfront为4→4；六路身份连续性风险增加。这是自动3m配对后的原始ID变化，不是人工串ID真值，也不替代全分母IDF1。另1/3/6');text+=f'\n补充身份核验：[共同三/六ID变化独立证据]({target})；新增模型/建图调用均0。\n';report.write_text(text)
 print(json.dumps(dict(passed=True,counts={k:dict(v) for k,v in counts.items()},report=str(target)),ensure_ascii=False))
if __name__=='__main__':main()
