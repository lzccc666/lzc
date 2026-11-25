"""从源码包展开并冻结三帧/权重/查询，启动实际新环境CPU链路，禁止借用旧Python依赖。"""
import datetime,hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import install_clean_cpu_environment_001 as base
R=base.R;PROJECT=R.parents[3];O=R/'reproduction/clean_cpu_chain_002'
def main():
 O.mkdir(exist_ok=False);envreport=R/'reproduction/clean_cpu_install_001/dependencies_report_003.json';assert json.loads(envreport.read_text())['clean_venv_dependency_install_and_imports_passed'];S=R/'source_bundle/reproduction_003';archive=S/'actual_source_snapshot.tar.gz';manifest=json.loads((S/'source_manifest.json').read_text());receipt=json.loads((S/'receipt.json').read_text());assert base.sha(archive)==receipt['archive_sha256'];checkout=O/'checkout';checkout.mkdir()
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  tar.extractall(checkout)
 for row in manifest:assert base.sha(checkout/row['archive_path'])==row['sha256']
 checkpoint=PROJECT/'checkpoint';assert checkpoint.is_dir();(checkout/'checkpoint').symlink_to(checkpoint,target_is_directory=True)
 src=PROJECT/'outputs/project_v2/thesis_completion_v1/mapping_evaluation/causal_mapping_full_001/scene-0012/frames.jsonl';allrows=list(map(json.loads,src.open()));frames=[{k:x[k] for k in ['scene','frame_idx','camera_timestamp_s','rgb','depth','camera_pose','intrinsic']} for x in allrows[:3]];assert [x['frame_idx'] for x in frames]==[0,1,2]
 training=R/'qwen_adapter_fair_comparison/training/real_graph_seed17_001/training_report.json';adapter=Path(json.loads(training.read_text())['best_checkpoint']['path']);weights=[checkpoint/'yolov8l-world.pt',checkpoint/'mobile_sam.pt',checkpoint/'hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors',Path('/root/.cache/clip/ViT-B-32.pt'),adapter/'adapter_model.safetensors']+sorted((checkpoint/'Qwen2-VL-2B-Instruct').glob('*'))
 inputfiles=sorted({Path(row[key]) for row in frames for key in ['rgb','depth']});runtime=R/'source/clean_cpu_perception_semantics_001.py';config=json.loads((R/'nuscenes_ablation/graph_near6_correction_001/release.json').read_text())['new_actual_graph_config'];sources=[Path(__file__),runtime,src,envreport,S/'source_manifest.json',training,R/'reproduction/clean_cpu_install_001/installed_freeze_004.txt',R/'reproduction/clean_cpu_install_001/explicit_runtime_dependency_addendum_004.json']+inputfiles+[p for p in weights if p.is_file()];freeze={str(p):base.sha(p) for p in sources}
 release=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkout=str(checkout),source_files_verified=len(manifest),source_archive_sha256=base.sha(archive),embedded_source_relative=str(R.relative_to(PROJECT)/'source'),base_model=str(checkpoint/'Qwen2-VL-2B-Instruct'),adapter=str(adapter),frames=frames,source_and_weight_sha256=freeze,classes=(checkout/'conceptgraph/vehicle_classes.txt').read_text().splitlines(),queries=['选择最近的小汽车','不要最近的小汽车，选择第二近的小汽车','以公交车为参照，选择其左侧最近的小汽车'],resolved_graph_config=config,max_input_tokens=2048,max_output_tokens=256,device='cpu',new_venv=True,runtime_auto_install_allowed=False,installed_dependency_freeze='reproduction/clean_cpu_install_001/installed_freeze_004.txt',source_reconstructed_from_archive=True,GT_in_agent_inputs=False,old_frame_GT_used_only_to_locate_fixed_indices=False,scope='开发前三帧独立安装功能复现；无找对、封存或实际到达信用',thesis_prose_written=False,goal_complete=False)
 base.guard(64*2**20)
 with (O/'release.json').open('x') as f:json.dump(release,f,ensure_ascii=False,indent=2)
 env=os.environ.copy();env.pop('PYTHONHOME',None);env.update(YOLO_AUTOINSTALL='false',PIP_CONFIG_FILE='/dev/null',PIP_INDEX_URL='https://pypi.org/simple',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONPATH=str(checkout),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',XDG_CACHE_HOME=str(O/'cache'),YOLO_CONFIG_DIR=str(O/'yolo_config'),HF_HOME=str(O/'hf_cache'))
 for name in ['cache','yolo_config','hf_cache']:(O/name).mkdir()
 py=str(R/'reproduction/clean_cpu_install_001/venv/bin/python');begun=time.monotonic()
 with (O/'runtime.log').open('x') as f:
  p=subprocess.run([py,str(runtime),'--release',str(O/'release.json')],cwd=checkout,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=False)
 base.save('clean_chain_process_receipt_002.json',dict(exit_code=p.returncode,wall_s=time.monotonic()-begun,runtime_log_sha256=base.sha(O/'runtime.log'),output_directory=str(O),GPU_reproduction_passed=False,goal_complete=False));print(json.dumps(dict(exit_code=p.returncode,wall_s=time.monotonic()-begun),ensure_ascii=False),flush=True)
 if p.returncode:raise RuntimeError('真实新环境CPU链路失败，现场保留')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if O.exists():
   with (O/'launcher_failure.json').open('x') as f:json.dump(dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False),f,ensure_ascii=False,indent=2)
  raise
