"""隔离保存修正后的真实YOLO/MobileSAM/CLIP输出，历史数据只读。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from hydra import compose, initialize_config_dir
import torch

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, available_development, digest, dump
from conceptgraph.scripts.streamlined_detections import load_detection_models, run_detections


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--scene',action='append')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('需要真实CUDA前端')
    rows=available_development()
    if args.scene:
        rows=[r for r in rows if r['scene'] in args.scene]
        if len(rows)!=len(set(args.scene)):
            raise ValueError('只允许开发场景')
    run=OUTPUT/'mapping_evaluation'/args.run_id
    frozen=dict(scenes=rows,frontend='completion-v1-preserve-boxes',
        code_sha256=digest(PROJECT/'conceptgraph/scripts/streamlined_detections.py'),
        crop_code_sha256=digest(PROJECT/'conceptgraph/utils/model_utils.py'),
        purpose='纠正实际GPU验证的SAM输入原地缩放；不改写历史',
        models=['yolov8l-world.pt','mobile_sam.pt','CLIP-ViT-H-14-laion2B-s32B-b79K'],
        ground_truth_in_inference=False)
    if run.exists():
        if not args.resume or json.loads((run/'release.json').read_text())!=frozen:
            raise ValueError('已有run或指纹变更')
    else:
        run.mkdir(parents=True)
        dump(run/'release.json',frozen)
        shutil.copy2(PROJECT/'conceptgraph/scripts/streamlined_detections.py',run/'executed_frontend.py')
    models=None
    reports=[]
    for scene in rows:
        name=scene['scene']
        directory=run/'inputs'/name
        directory.mkdir(parents=True,exist_ok=True)
        source=Path(scene['root'])/'converted'/name
        for part in ['color','depth','poses.txt','intrinsics.txt','frame_alignment.json']:
            link=directory/part
            if not link.exists():
                link.symlink_to(source/part,target_is_directory=(source/part).is_dir())
        completed=directory/'frontend_complete.json'
        if completed.exists():
            record=json.loads(completed.read_text())
            for p,h in record['output_sha256'].items():
                if digest(p)!=h:
                    raise ValueError('输出指纹不一致')
            reports.append(record)
            continue
        with initialize_config_dir(version_base=None,config_dir=str(PROJECT/'conceptgraph/hydra_configs')):
            cfg=compose(config_name='streamlined_detections',overrides=[
                f'dataset_root={run / "inputs"}', f'scene_id={name}',
                f'dataset_config={PROJECT / "conceptgraph/nuscenes/nuscenes.yaml"}',
                f'classes_file={PROJECT / "conceptgraph/vehicle_classes.txt"}',
                'exp_suffix=streamlined_yolo_outdoor_objects_v1','save_video=false','device=cuda:0'])
        started=time.monotonic()
        if models is None:
            models=load_detection_models(cfg)
        try:
            run_detections(cfg,models,skip_existing=args.resume)
        except Exception as e:
            failure=directory/f'failed_{time.time_ns()}.json'
            dump(failure,dict(error=repr(e),goal_complete=False))
            raise
        torch.cuda.synchronize()
        outputs=sorted((directory/'exp_streamlined_yolo_outdoor_objects_v1/detections').glob('*.pkl.gz'))
        record=dict(scene=name,frame_count=len(outputs),runtime_s=time.monotonic()-started,
            output_sha256={str(p):digest(p) for p in outputs},
            gpu=torch.cuda.get_device_name(),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            real_detection=True,real_segmentation=True,real_clip=True,
            source_rgb_root=str(source/'color'),ground_truth_used=False)
        dump(completed,record)
        reports.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='output_sha256'},ensure_ascii=False),flush=True)
    if not (run/'complete.json').exists():
        dump(run/'complete.json',dict(scenes=len(reports),frames=sum(r['frame_count'] for r in reports),
            complete=True,goal_complete=False,reports=[str(run/'inputs'/r['scene']/'frontend_complete.json') for r in reports]))


if __name__=='__main__':
    main()
