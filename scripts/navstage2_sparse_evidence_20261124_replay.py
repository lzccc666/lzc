"""固定两个验证场景前12帧，真实重放R1并验证溯源旁路不改变预测。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,gzip,json,pickle,shutil,time
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from conceptgraph.completion.stage9 import predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.navigation.navstage2_repair_diagnosis_20261124 import ROOT,BASE,sha,read,lines,dump
from conceptgraph.navigation.navstage2_sparse_evidence_20261124 import RecordingR1Mapper
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.streamlined_mapping import MappingFrame

DATA=ROOT/'dataset/processed/navstage2_sparse_evidence_20261124_v1'
R1=ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
RUN='navstage2_sparse_evidence_20261124_v1'
SCENES=['scene-0012','scene-0013']

def prepare():
    DATA.mkdir(exist_ok=False)
    tasks=[];inputs={str(R1/'release.json'):sha(R1/'release.json')}
    for scene in SCENES:
        inputs[str(R1/scene/'frames.jsonl.gz')]=sha(R1/scene/'frames.jsonl.gz')
        oldpath=BASE/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl'
        inputs[str(oldpath)]=sha(oldpath)
        expected={r['frame_idx']:r for r in lines(R1/scene/'frames.jsonl.gz')}
        for row in lines(oldpath):
            if row['frame_idx']>=12:break
            idx=row['frame_idx'];det=expected[idx]['input']['detections']
            for p in [row['rgb'],row['depth'],det]:inputs[str(p)]=sha(p)
            tasks.append(dict(scene=scene,frame_idx=idx,timestamp_s=row['camera_timestamp_s'],rgb=row['rgb'],
                depth=row['depth'],detections=det,intrinsic=row['intrinsic'],camera_pose=row['camera_pose']))
    inputpath=DATA/'method_frames.jsonl';inputpath.write_text(''.join(json.dumps(r)+'\n' for r in tasks));inputs[str(inputpath)]=sha(inputpath)
    sources=[Path(__file__),ROOT/'conceptgraph/navigation/navstage2_sparse_evidence_20261124.py',
        ROOT/'conceptgraph/navigation/navstage2_repair_diagnosis_20261124.py']+[
        ROOT/'conceptgraph/slam'/n for n in ['dense_refinement.py','quality_mapper.py','sparse_observations.py',
        'audited_mapper.py','evidence_mapper.py','streamlined_mapping.py','utils.py','mapping.py']]+[
        ROOT/'conceptgraph/completion'/n for n in ['motion.py','association.py','depth.py','stage9.py']]
    release=read(R1/'release.json')
    dump(DATA/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=17,
        scenes=SCENES,frames_per_scene=12,total_frames=len(tasks),role='validation instrumentation only; not performance optimization',
        source_sha256={str(p):sha(p) for p in sources},input_sha256=inputs,config=release['config'],
        budget=dict(wall_s=180,new_bytes=16*1024**2,GPU_calls=0,minimum_free_bytes=20*1024**3),
        expected='逐帧所有预测字节语义精确一致；仅旁路导出真实测量证据，不改运动/关联/尺寸/深度门限',
        no_history81_used_for_method_choice=True))
    print(str(DATA/'manifest.json'),sha(DATA/'manifest.json'))

def run():
    manifest=read(DATA/'manifest.json')
    for group in ['source_sha256','input_sha256']:
        for p,h in manifest[group].items():assert sha(p)==h
    dest=experiment_directory('nuscenes','validation',RUN,create=True);shutil.copy2(DATA/'manifest.json',dest/'manifest.json')
    for p in manifest['source_sha256']:
        target=dest/'source_snapshot'/Path(p).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    begun=time.monotonic();tasks=list(lines(DATA/'method_frames.jsonl'));counts=Counter();per_scene={}
    try:
        with gzip.open(dest/'raw_navigation_evidence.jsonl.gz','wt') as out:
            for scene in SCENES:
                mapper=RecordingR1Mapper(OmegaConf.create(manifest['config']),CLASSES)
                expected={r['frame_idx']:r for r in lines(R1/scene/'frames.jsonl.gz')}
                for row in [r for r in tasks if r['scene']==scene]:
                    assert time.monotonic()-begun<180
                    assert shutil.disk_usage(ROOT).free>=20*1024**3
                    with gzip.open(row['detections'],'rb') as f:gobs=pickle.load(f)
                    idx=row['frame_idx'];np.random.seed(17+idx)
                    frame=MappingFrame(idx,np.asarray(Image.open(row['rgb']).convert('RGB')),
                        np.asarray(Image.open(row['depth']),float)/1000.,np.asarray(row['intrinsic']),
                        np.asarray(row['camera_pose']),copy.deepcopy(gobs),row['rgb'],timestamp_s=row['timestamp_s'])
                    mapper.update(frame)
                    predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
                    assert json.loads(json.dumps(predictions,allow_nan=False))==expected[idx]['predictions']['R1'],f'prediction_changed:{scene}:{idx}'
                    evidence=mapper.navigation_trace(row['timestamp_s'])
                    counts['frames_exactly_reproduced']+=1;counts['sparse_track_frame_records']+=len(evidence)
                    for e in evidence:
                        counts['raw_trace_records']+=len(e['raw_trace'])
                        counts['raw_motion_'+e['diagnostic_motion_from_raw_observations']]+=1
                        counts['current_depth_passed']+=int(e['current_depth_passed_original_gate'])
                        counts['complete_original_navigation_evidence']+=int(e['complete_original_navigation_evidence'])
                    out.write(json.dumps(dict(scene=scene,frame_idx=idx,predictions_exact=True,
                        predictions=predictions,sparse_navigation_evidence=evidence),ensure_ascii=False,allow_nan=False)+'\n')
                    out.flush()
                del mapper
        for group in ['source_sha256','input_sha256']:
            for p,h in manifest[group].items():assert sha(p)==h
        dump(dest/'summary.json',dict(counts=dict(counts),elapsed_s=time.monotonic()-begun,
            no_prediction_changed=True,configuration_changed=False,GT_in_inference=False,GPU_calls=0,
            navigation_candidate_authorization_added=0,
            result='真实稀疏观测证据已导出；原完整导航门通过数如为0则仍unknown/不可授权，不作涨分声明'))
        print(json.dumps(dict(counts),ensure_ascii=False))
    except BaseException as error:
        dump(dest/'interrupted.json',dict(error=repr(error),elapsed_s=time.monotonic()-begun,counts=dict(counts)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    {'prepare':prepare,'run':run}[a.action]()
