"""在固定开发观测上实际重建两种关联门限；评价距离门限始终保持3m。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import time

from omegaconf import OmegaConf

from conceptgraph.completion.fault_robustness import mapping_config, run_case
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--scene',action='append')
    parser.add_argument('--wall-budget-s',type=float,default=1800)
    args=parser.parse_args()
    begun=time.monotonic()
    known={r['scene']:r for r in available_development()}
    scenes=args.scene or sorted(known)
    if not set(scenes)<=known.keys() or len(scenes)!=len(set(scenes)):
        raise ValueError('only fixed unique development scenes allowed')
    dest=OUTPUT/'performance'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    baseline=OUTPUT/'faults/recorded_fault_full_001'
    sources=[OUTPUT/'mapping_evaluation/robust_motion_full_001'/s/'frames.jsonl' for s in scenes]
    modules=[Path(__file__),PROJECT/'conceptgraph/completion/fault_robustness.py',
        PROJECT/'conceptgraph/slam/streamlined_mapping.py',PROJECT/'conceptgraph/slam/mapping.py',
        PROJECT/'conceptgraph/slam/utils.py',PROJECT/'conceptgraph/completion/depth.py',
        PROJECT/'conceptgraph/completion/association.py',PROJECT/'conceptgraph/completion/motion.py',
        PROJECT/'conceptgraph/completion/quality.py',PROJECT/'conceptgraph/completion/stage9.py']
    release=dict(thresholds=[.25,.35,.45],new_inference_thresholds=[.25,.45],seed=17,
        baseline_reuse='已完成录制故障无扰动条件seed17，原配置sim_threshold=0.35；逐场景SHA冻结，不重复算新数据',
        baseline_reports={s:digest(baseline/s/'seed17/unperturbed/report.json') for s in scenes},
        source_frames_sha256={str(p):digest(p) for p in sources},
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in modules},
        fixed_config=OmegaConf.to_container(mapping_config()),evaluation_center_gate_m=3.,
        scene_logs={s:known[s]['log_token'] for s in scenes},wall_budget_s=args.wall_budget_s,
        storage_budget_bytes=300000000,device='cpu',other_gpu_experiment='realtime_three_model_full_001',
        exclusive_performance_credit=False,parameter_applied_to_running_navigation=False,
        independent_test=False,goal_complete=False)
    dump(dest/'release.json',release)
    with tarfile.open(dest/'source_snapshot.tar.gz','x:gz') as archive:
        for source in modules:
            archive.add(source,arcname=str(source.relative_to(PROJECT)))
    rows=[]
    with (dest/'cases.jsonl').open('x') as handle:
        for path in sources:
            if digest(path)!=release['source_frames_sha256'][str(path)]:
                raise ValueError('development source changed')
            frames=[json.loads(line) for line in path.open()]
            classes_path=Path(frames[0]['detections']).parents[2]/'gsa_classes_streamlined_yolo.json'
            frontend=json.loads((classes_path.parent/'frontend_complete.json').read_text())
            for filename,sha in frontend['output_sha256'].items():
                if digest(filename)!=sha:
                    raise ValueError('saved frontend changed')
            classes=json.loads(classes_path.read_text())
            for threshold in [.25,.35,.45]:
                cfg=mapping_config()
                cfg.sim_threshold=threshold
                if threshold==.35:
                    report_path=baseline/path.parent.name/'seed17/unperturbed/report.json'
                    report=json.loads(report_path.read_text())
                    assert digest(report_path)==release['baseline_reports'][path.parent.name]
                    assert digest(report_path.parent/'frames.jsonl')==report['records_sha256']
                    reused=True
                else:
                    case=dest/path.parent.name/f'threshold_{threshold}'
                    case.mkdir(parents=True)
                    report=run_case(frames,'unperturbed',17,case,classes,begun+args.wall_budget_s,inference_config=cfg)
                    report_path=case/'report.json'
                    reused=False
                row=dict(scene=path.parent.name,log_token=known[path.parent.name]['log_token'],
                    threshold=threshold,report_path=str(report_path),report_sha256=digest(report_path),
                    reused_baseline=reused,inference_config=OmegaConf.to_container(cfg),metrics=report['metrics'])
                rows.append(row)
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                handle.flush()
                print(json.dumps(dict(scene=row['scene'],threshold=threshold,reused_baseline=reused),ensure_ascii=False),flush=True)
                size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
                if size>release['storage_budget_bytes']:
                    raise RuntimeError('frozen storage budget exceeded; keep all raw prefix')
    dump(dest/'summary.json',dict(cases=len(rows),expected=len(scenes)*3,new_rebuild_cases=len(scenes)*2,
        source_frames=len(scenes),unique_frames=sum(len(p.read_text().splitlines()) for p in sources),
        records_sha256=digest(dest/'cases.jsonl'),elapsed_s=time.monotonic()-begun,
        independent_test=False,parameter_applied_to_running_navigation=False,goal_complete=False))


if __name__=='__main__':
    main()
