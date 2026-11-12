"""在推理前检查所有可用候选与历史轨迹的空间邻近，不读对象标签。"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, HISTORY, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform


def main():
    candidates=json.loads((OUTPUT/'data_manifest/new_scene_candidates_03_04.json').read_text())['candidates']
    sources=json.loads((OUTPUT/'data_manifest/new_scene_sources_03_04.json').read_text())
    old=defaultdict(list)
    for root in [HISTORY,HISTORY/'prospective_supplement_v2']:
        for r in json.loads((root/'scene_selection_manifest.json').read_text())['scenes']:
            pose=np.loadtxt(root/'converted'/r['scene']/'poses.txt').reshape(-1,4,4)
            old[r['strata']['location']].extend(pose[:,:2,3])
    trees={loc:cKDTree(points) for loc,points in old.items()}
    meta=PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    cal={r['token']:r for r in iter_records(meta/'calibrated_sensor.json')}
    required={f['sensors']['CAM_FRONT']['ego_pose_token'] for r in candidates for f in sources[r['scene']]['sequence']}
    poses={r['token']:r for r in iter_records(meta/'ego_pose.json') if r['token'] in required}
    rows=[]
    for r in candidates:
        camera=[]
        for f in sources[r['scene']]['sequence']:
            c=f['sensors']['CAM_FRONT']
            p=poses[c['ego_pose_token']]
            k=cal[c['calibrated_sensor_token']]
            t=build_transform(p['translation'],p['rotation'])@build_transform(k['translation'],k['rotation'])
            camera.append(t[:2,3])
        distances,_=trees[r['strata']['location']].query(camera)
        rows.append(dict(r,min_historical_camera_distance_m=float(min(distances)),
            median_historical_camera_distance_m=float(np.median(distances)),
            camera_trajectory_xy=np.asarray(camera).tolist()))
    summary={}
    for margin in [20,50,100]:
        selected=[r for r in rows if r['min_historical_camera_distance_m']>=margin]
        summary[str(margin)]=dict(scenes=len(selected),logs=len({r['log_token'] for r in selected}))
    dump(OUTPUT/'splits/geographic_candidate_audit_001.json',dict(candidates=rows,threshold_summary_m=summary,
        source='官方ego_pose与标定；CAM_FRONT世界位置；与全部历史69scene相机轨迹同城比较',
        inference_executed=False,test_opened=False,policy='选场前的空间隔离审计，不按方法成绩筛场'))
    print(json.dumps(summary))


if __name__=='__main__':
    main()
