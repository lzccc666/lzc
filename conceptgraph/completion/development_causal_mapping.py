"""审计通过后绑定相同RGB真实检测和过去LiDAR，调用既有CPU建图链。"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from conceptgraph.completion.development_causal_inputs import exact_link
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


METHODS = ['original_fusion', 'timestamp_only', 'one_to_one_no_expiry',
    'one_to_one_timestamp', 'one_to_one_depth_quality', 'one_to_one_depth_robust_motion']


def prepare(input_run, frontend_run, audit_id='audit_001'):
    source = OUTPUT/'data_manifest'/input_run
    complete = json.loads((source/'complete.json').read_text())
    report_path = source/audit_id/'report.json'
    report = json.loads(report_path.read_text())
    if (not report['passed'] or report['complete_sha256'] != digest(source/'complete.json')
            or report['independent_projection_mismatched_pixels'] != 0
            or report['future_lidar_pairs'] != 0 or report['actual_new_frontend_calls'] != 0):
        raise ValueError('缺少有效独立因果输入审计')
    for relative, sha in complete['files_sha256'].items():
        if digest(source/relative) != sha:
            raise ValueError('审计后的转换输入变化')
    plans = json.loads((source/'alignment_manifest.json').read_text())['plans']
    scene_names = sorted({p['camera']['scene'] for p in plans})
    counts = {s: sum(p['camera']['scene'] == s for p in plans) for s in scene_names}
    destination = OUTPUT/'mapping_evaluation'/frontend_run
    destination.mkdir(exist_ok=False)
    shutil.copy2(__file__, destination/'binding_source.py')
    dump(destination/'release.json', dict(version='audited-development-causal-binding-v1',
        input_run=input_run, input_complete_sha256=digest(source/'complete.json'),
        independent_audit=str(report_path), independent_audit_sha256=digest(report_path),
        source_sha256=digest(__file__), scenes=scene_names, cameras=complete['cameras'],
        actual_new_frontend_calls=0, reused_same_RGB_real_frontend=True,
        new_test_opened=False, ground_truth_in_inference=False, performance_credit=False,
        evaluation_limit=report['evaluation_limit'], goal_complete=False))
    for name in scene_names:
        original = source/'converted'/name
        folder = destination/'inputs'/name
        folder.mkdir(parents=True)
        # 整个目录入口只是转换产物的只读引用，不改写旧传感器数据。
        for path in original.iterdir():
            exact_link(folder/path.name, path)
        outputs = {str(folder/path.relative_to(original)): digest(path)
                   for path in original.rglob('*') if path.is_file()}
        dump(folder/'frontend_complete.json', dict(scene=name, frame_count=counts[name],
            output_sha256=outputs, real_detection=True, real_segmentation=True, real_clip=True,
            actual_new_frontend_calls=0, reused_real_frontend_frames=counts[name],
            input_complete_sha256=digest(source/'complete.json'),
            past_lidar_independent_audit_passed=True, ground_truth_used=False))
    dump(destination/'complete.json', dict(scenes=len(scene_names), frames=complete['cameras'],
        actual_new_frontend_calls=0, existing_real_frontend_frames_reused=complete['cameras'],
        no_model_inference_executed=True, new_test_opened=False, goal_complete=False))
    return destination, scene_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--frontend-run', required=True)
    parser.add_argument('--mapping-run', required=True)
    parser.add_argument('--audit-id', default='audit_001')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    frontend, scenes = prepare(args.input_run, args.frontend_run, args.audit_id)
    command = [sys.executable, '-m', 'conceptgraph.completion.stage9', '--run-id', args.mapping_run,
        '--frontend-run', args.frontend_run, '--methods', *METHODS]
    for scene in scenes:
        command.extend(['--scene', scene])
    dump(frontend/'mapping_command.json', dict(command=command, execute_requested=args.execute,
        mapping_source_sha256=digest(PROJECT/'conceptgraph/completion/stage9.py'),
        GPU_inference=False, new_test_opened=False, goal_complete=False))
    print(json.dumps(dict(frontend_binding=str(frontend), mapping_command=command), ensure_ascii=False), flush=True)
    if args.execute:
        result = subprocess.run(command, cwd=PROJECT, check=False)
        dump(frontend/'mapping_exit.json', dict(returncode=result.returncode, goal_complete=False))
        result.check_returncode()


if __name__ == '__main__':
    main()
