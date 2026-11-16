"""校验代码包并在独立展开目录重算真实帧与语义决策；复用已装依赖，不冒称全新环境安装。"""
import json,os,subprocess,tarfile
from pathlib import Path
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,dump,digest
R=Path(__file__).resolve().parents[1];S=R/'source_bundle/reproduction_002';O=R/'source_bundle/reproduction_smoke_001'
def main():
 O.mkdir(exist_ok=False);checkout=O/'checkout';checkout.mkdir();receipt=json.loads((S/'receipt.json').read_text());archive=S/'actual_source_snapshot.tar.gz';assert digest(archive)==receipt['archive_sha256'];manifest=json.loads((S/'source_manifest.json').read_text())
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   p=Path(member.name);assert not p.is_absolute() and '..' not in p.parts and member.isfile()
  tar.extractall(checkout)
 for row in manifest:assert digest(checkout/row['archive_path'])==row['sha256']
 frozen=json.loads((R/'source_bundle/reproduction_001/source_manifest.json').read_text());changed=[r['path'] for r in frozen if not Path(r['path']).exists() or digest(r['path'])!=r['sha256']]
 sources=[OLD/'mapping_evaluation/causal_mapping_full_001'/s/'frames.jsonl' for s in ['scene-0012','scene-0013','scene-0014']]
 semantic=R/'qwen_adapter_fair_comparison/real_graph_dataset_003/development_holdout.jsonl'
 code='''import json,sys
from pathlib import Path
import conceptgraph.completion.quality as quality
from real_graph_cooperation_001 import choose
from real_graph_contract_003 import validate
assert Path(quality.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
checks=[]
for source in map(Path,sys.argv[3:]):
 rows=list(map(json.loads,source.open()));selected=next(r for r in rows if r['predictions']['one_to_one_depth_robust_motion'] and r['evaluation_only_official']);actual=quality.object_frame_metrics(selected['predictions']['one_to_one_depth_robust_motion'],selected['evaluation_only_official']);assert actual==selected['metrics']['one_to_one_depth_robust_motion'];checks.append(dict(source=str(source),frame_idx=selected['frame_idx'],true_positive=actual['true_positive']))
rows=list(map(json.loads,Path(sys.argv[2]).open()));decisions=[]
for row in rows[:9]:
 v=validate(row['task'],row['response'],allowed_candidate_ids=row['candidate_ids']);assert v.valid
 result=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],None);assert result['authorized_node_id'] is None or result['authorized_node_id'] in row['candidate_ids'];decisions.append(dict(sample_id=row['sample_id'],result=result))
print(json.dumps(dict(mapping_checks=checks,semantic_contract_checks=len(decisions),source_module=str(quality.__file__),no_model_loading=True)))
'''
 isolated_source=checkout/R.relative_to(PROJECT)/'source';env=os.environ.copy();env.update(PYTHONPATH=str(checkout)+':'+str(isolated_source),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_PRELOAD='/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6');p=subprocess.run(['/root/miniconda3/envs/conceptgraph/bin/python','-c',code,str(checkout),str(semantic),*map(str,sources)],cwd=checkout,env=env,capture_output=True,text=True);(O/'stdout.log').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
 dump(O/'report.json',dict(returncode=p.returncode,archive_verified=True,source_files_verified=len(manifest),first_source_snapshot_files_checked=len(frozen),changed_since_first_source_snapshot=changed,isolated_source_import_and_real_frame_rescore_passed=p.returncode==0,clean_environment_install_passed=False,existing_dependencies_reused=True,data_or_weights_copied=False,archive_sha256=digest(archive),manifest_sha256=digest(S/'source_manifest.json'),test_inputs_sha256={str(p):digest(p) for p in sources+[semantic]},test_source_sha256=digest(__file__),limitation='这是独立代码展开与真实保存帧的小规模重计分检查；不是新Conda/OS从零安装，也没有重新执行完整GPU前端/导航',goal_complete=False))
 print(json.dumps(dict(returncode=p.returncode,verified=len(manifest),old_source_changes=len(changed),clean_environment_install_passed=False)),flush=True)
 if p.returncode:raise RuntimeError(p.stderr)
if __name__=='__main__':main()
