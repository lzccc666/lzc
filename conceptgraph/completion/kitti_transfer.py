"""KITTI Object 跨数据集离线诊断。prepare / infer 严格分离评价标签。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image

from conceptgraph.completion.depth import depth_support
from conceptgraph.completion.kitti_geometry import calibration, labels, project_lidar
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

DEFAULT_ARCHIVES = Path('/root/autodl-pub/KITTI_Object/raw')
METHODS = ('mask_points_mean', 'mask_points_robust_mean')


def source_files():
    return [Path(__file__), PROJECT/'conceptgraph/completion/kitti_geometry.py',
            PROJECT/'conceptgraph/completion/depth.py',
            PROJECT/'conceptgraph/scripts/streamlined_detections.py',
            PROJECT/'conceptgraph/vehicle_classes.txt',
            PROJECT/'conceptgraph/utils/model_utils.py']


def source_hashes():
    return {str(p): digest(p) for p in source_files()}


def budget_guard(start, args, destination):
    if time.monotonic() - start > args.wall_budget_s:
        raise TimeoutError('达到事前声明的墙钟预算')
    if shutil.disk_usage(OUTPUT).free < args.minimum_free_gb * 1024**3:
        raise OSError('可用磁盘低于事前保护预算')
    size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    if size > args.storage_budget_gb * 1024**3:
        raise OSError('达到事前声明的新增存储预算')


def prepare(args):
    start = time.monotonic()
    dest = OUTPUT/'data_manifest'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    # 在读取任何样本标签或运行模型前，将均匀选样规则和ID写入不可覆盖文件。
    ids = np.linspace(0, 7480, args.frames).round().astype(int).tolist()
    if len(set(ids)) != args.frames:
        raise ValueError('重复选样')
    selection = dict(dataset='KITTI Object training', frame_ids=ids,
                     selection='round(linspace(0,7480,N)); 未按类别或成败筛选',
                     independent_test=False, role='external_cross_dataset_development_diagnostic',
                     source_sha256=source_hashes(), methods=list(METHODS),
                     classes=(PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines(),
                     geometry='rectified camera0 meters; P2 full 3x4; nearest measured pixel',
                     minimum_depth_m=1, maximum_depth_m=65, minimum_depth_points=5,
                     evaluation_iou=.5, evaluation_center_gate_m=3,
                     primary_classes=['car', 'truck', 'pedestrian'],
                     cyclist_scope='Cyclist->bicycle 仅单列代理诊断，不并入主要三类别',
                     actual_navigation_credit=False, official_kitti_ap=False,
                     inference_uses_gt=False, pretrained_dataset_overlap='unknown',
                     license='CC BY-NC-SA 3.0; 不将原始图像/标签/点云提交Git',
                     budgets=vars(args) | {'archives': str(args.archives)},
                     official_sources=['https://www.cvlibs.net/datasets/kitti/',
                                       'https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d',
                                       'https://semantic-kitti.org/dataset'])
    dump(dest/'selection.json', selection)
    paths = {'image': 'data_object_image_2.zip', 'lidar': 'data_object_velodyne.zip',
             'calib': 'data_object_calib.zip', 'labels': 'data_object_label_2.zip',
             'devkit': 'devkit_object.zip'}
    archives = {k: ZipFile(args.archives/v) for k, v in paths.items()}
    try:
        mapping_bytes = archives['devkit'].read('mapping/train_mapping.txt')
        random_bytes = archives['devkit'].read('mapping/train_rand.txt')
        mapping = mapping_bytes.decode().splitlines()
        permutation = [int(x) for x in random_bytes.decode().replace('\n', '').split(',') if x]
        if len(mapping) != 7481 or sorted(permutation) != list(range(1, 7482)):
            raise ValueError('官方采集drive映射不完整')
        agent_rows, truth_rows, member_rows = [], [], []
        for number in ids:
            budget_guard(start, args, dest)
            name = f'{number:06d}'
            member = {'image': f'training/image_2/{name}.png',
                      'lidar': f'training/velodyne/{name}.bin',
                      'calib': f'training/calib/{name}.txt',
                      'labels': f'training/label_2/{name}.txt'}
            raw = {k: archives[k].read(v) for k, v in member.items()}
            date, drive, original_frame = mapping[permutation[number] - 1].split()
            inp = dest/'agent_inputs'/name
            inp.mkdir(parents=True)
            (inp/'rgb.png').write_bytes(raw['image'])
            (inp/'calib.txt').write_bytes(raw['calib'])
            points = np.frombuffer(raw['lidar'], dtype='<f4').reshape(-1, 4)
            with Image.open(io.BytesIO(raw['image'])) as image:
                wh = image.size
            calib = calibration(raw['calib'].decode())
            depth, xyz, info = project_lidar(points, calib, wh)
            np.savez_compressed(inp/'projected_lidar.npz', depth=depth, xyz_camera0=xyz)
            row = dict(frame_id=name, drive=drive, date=date, raw_frame=original_frame,
                       rgb=str(inp/'rgb.png'), calib=str(inp/'calib.txt'),
                       projected_lidar=str(inp/'projected_lidar.npz'), image_wh=list(wh),
                       projection=info, input_sha256={str(p): digest(p) for p in inp.iterdir()})
            agent_rows.append(row)
            # GT 只存在于 evaluation_only，infer 不读取其路径及其文件。
            truth_rows.append(dict(frame_id=name, drive=drive, objects=labels(raw['labels'].decode())))
            member_rows.append(dict(frame_id=name, members={k: dict(archive=paths[k], member=v,
                sha256=hashlib.sha256(raw[k]).hexdigest(), bytes=len(raw[k])) for k, v in member.items()}))
            print(json.dumps(dict(prepared=name, occupied_pixels=info['occupied_pixels'])), flush=True)
        dump(dest/'agent_manifest.json', dict(selection_sha256=digest(dest/'selection.json'),
                                             classes=selection['classes'], frames=agent_rows))
        dump(dest/'evaluation_only/truth.json', dict(frames=truth_rows))
        dump(dest/'evaluation_only/raw_member_provenance.json', member_rows)
        dump(dest/'complete.json', dict(frames=len(ids), drives=len({r['drive'] for r in agent_rows}),
            selection_sha256=digest(dest/'selection.json'),
            agent_manifest_sha256=digest(dest/'agent_manifest.json'),
            truth_sha256=digest(dest/'evaluation_only/truth.json'),
            archive_inventory={k: dict(path=str(args.archives/v), size=(args.archives/v).stat().st_size,
                mtime_ns=(args.archives/v).stat().st_mtime_ns, entries=len(archives[k].infolist())) for k, v in paths.items()},
            mapping_sha256=hashlib.sha256(mapping_bytes).hexdigest(),
            permutation_sha256=hashlib.sha256(random_bytes).hexdigest(),
            duration_s=time.monotonic()-start, source_sha256=source_hashes()))
    except BaseException as exc:
        dump(dest/'interrupted.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        for archive in archives.values():
            archive.close()


def observations(result, data):
    rows = []
    for i, (bbox, class_id, confidence, mask) in enumerate(zip(
            result['xyxy'], result['class_id'], result['confidence'], result['mask'])):
        methods = {}
        for method in METHODS:
            valid, evidence = depth_support(data['depth'], mask, robust='robust' in method)
            center = data['xyz_camera0'][valid].mean(axis=0).astype(float).tolist() if evidence['sufficient_support'] else None
            methods[method] = dict(center_camera0=center, depth_evidence=evidence)
        rows.append(dict(prediction_index=i, category=result['classes'][int(class_id)],
                         bbox=np.asarray(bbox, dtype=float).tolist(),
                         confidence=float(confidence), methods=methods))
    return rows


def infer(args):
    import os
    # 在导入torch及任何模型前禁用GPU，避免干扰仍在进行的冻结主矩阵。
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU旁路实验必须显式设置 CUDA_VISIBLE_DEVICES=""')
    import torch
    from conceptgraph.scripts.streamlined_detections import detect_rgb_file, load_detection_models

    start = time.monotonic()
    source = Path(args.agent_manifest).resolve()
    if source.name != 'agent_manifest.json':
        raise ValueError('infer只接受agent_manifest.json，不读取评价标签')
    agent = json.loads(source.read_text())
    dest = OUTPUT/'mapping_evaluation'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    frames = agent['frames'][:args.limit] if args.limit else agent['frames']
    checkpoint = PROJECT/'checkpoint'
    weights = [checkpoint/'yolov8l-world.pt', checkpoint/'mobile_sam.pt',
        checkpoint/'hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']
    release = dict(agent_manifest=str(source), agent_manifest_sha256=digest(source),
                   source_sha256=source_hashes(), model_sha256={str(p): digest(p) for p in weights},
                   frame_ids=[r['frame_id'] for r in frames], classes=agent['classes'],
                   device='cpu', cuda_visible_devices='', threads=args.threads,
                   methods=list(METHODS), budgets=vars(args), gt_read=False,
                   performance_credit=False, concurrent_gpu_matrix=True,
                   scope='单帧真实YOLO/MobileSAM/CLIP + 实测稀疏LiDAR定位；非全建图/导航')
    dump(dest/'release.json', release)
    event_path = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def resource_event(kind):
        if event_path.parent.exists():
            with event_path.open('a') as handle:
                handle.write(json.dumps(dict(event=kind, run_id=args.run_id,
                    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), device='cpu',
                    threads=args.threads, performance_credit=False))+'\n')
    resource_event('kitti_cpu_frontend_start')
    try:
        torch.set_num_threads(args.threads)
        cfg = SimpleNamespace(device='cpu')
        models = load_detection_models(cfg)
        for model in models[:2]:
            model.to('cpu')
            model.overrides['device'] = 'cpu'
        cold_s = time.monotonic()-start
        rows = []
        for row in frames:
            budget_guard(start, args, dest)
            for path, expected in row['input_sha256'].items():
                if digest(path) != expected:
                    raise ValueError('当前帧输入SHA不一致')
            tick = time.monotonic()
            result, _ = detect_rgb_file(cfg, models, row['rgb'], agent['classes'])
            if torch.cuda.is_initialized() or any(str(next(m.model.parameters()).device) != 'cpu' for m in models[:2]):
                raise RuntimeError('CPU隔离失败')
            if result['frame_provenance']['rgb_sha256'] != row['input_sha256'][row['rgb']]:
                raise ValueError('前端图像溯源不一致')
            inp = np.load(row['projected_lidar'])
            record = dict(frame_id=row['frame_id'], drive=row['drive'],
                          predictions=observations(result, inp), elapsed_s=time.monotonic()-tick,
                          frame_provenance=result['frame_provenance'], cuda_initialized=False)
            frame_dir = dest/'frames'/row['frame_id']
            dump(frame_dir/'observations.json', record)
            with gzip.open(frame_dir/'detections.pkl.gz', 'xb') as handle:
                pickle.dump(result, handle)
            rows.append(dict(frame_id=row['frame_id'], observation=str(frame_dir/'observations.json'),
                             observation_sha256=digest(frame_dir/'observations.json'),
                             detection_sha256=digest(frame_dir/'detections.pkl.gz'),
                             elapsed_s=record['elapsed_s'], predictions=len(record['predictions'])))
            dump(frame_dir/'complete.json', rows[-1])
            print(json.dumps(rows[-1]), flush=True)
        if source_hashes() != release['source_sha256']:
            raise ValueError('实验中源代码发生变化')
        dump(dest/'summary.json', dict(frames=rows, frame_count=len(rows),
            duration_s=time.monotonic()-start, cold_start_s=cold_s, gt_read=False,
            cuda_initialized=torch.cuda.is_initialized(), release_sha256=digest(dest/'release.json'),
            actual_frontend_calls=len(rows), actual_navigation_arrivals=0,
            actual_scenegraph_fusions=0, performance_credit=False))
    except BaseException as exc:
        dump(dest/'interrupted.json', dict(type=type(exc).__name__, message=str(exc),
                                         duration_s=time.monotonic()-start))
        raise
    finally:
        resource_event('kitti_cpu_frontend_end')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run-id', required=True)
    p.add_argument('--archives', type=Path, default=DEFAULT_ARCHIVES)
    p.add_argument('--frames', type=int, default=64)
    p.add_argument('--wall-budget-s', type=float, default=1800)
    p.add_argument('--storage-budget-gb', type=float, default=1)
    p.add_argument('--minimum-free-gb', type=float, default=6)
    p = sub.add_parser('infer')
    p.add_argument('--run-id', required=True)
    p.add_argument('--agent-manifest', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--wall-budget-s', type=float, default=10800)
    p.add_argument('--storage-budget-gb', type=float, default=2)
    p.add_argument('--minimum-free-gb', type=float, default=6)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
