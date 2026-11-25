"""All22-sequence applicability ledger; partial RGB certification stays partial."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from conceptgraph.completion.kitti_local_all_20261125 import ROOT,REPORT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pairpath=REPORT/'semantic_RGB_LiDAR_pose_pair_inventory_v2.json';partial=json.loads(pairpath.read_text())
    pairmatrix={r['sequence']:r for r in partial['matrix']}
    archive=Path('/root/autodl-pub/SemanticKITTI');archives={key:zipfile.ZipFile(archive/name) for key,name in
        [('lidar','data_odometry_velodyne.zip'),('calib','data_odometry_calib.zip'),('labels','data_odometry_labels.zip')]}
    counts={};index_SHA={}
    for key,z in archives.items():
        c=Counter();h=hashlib.sha256()
        for entry in z.infolist():
            if entry.is_dir():continue
            h.update(json.dumps([entry.filename,entry.CRC,entry.file_size],separators=(',',':')).encode())
            parts=entry.filename.split('/')
            if len(parts)>3 and parts[1]=='sequences':
                if key=='lidar' and '/velodyne/' in entry.filename and entry.filename.endswith('.bin'):c[parts[2]]+=1
                if key=='labels' and '/labels/' in entry.filename and entry.filename.endswith('.label'):c[parts[2]]+=1
        counts[key]=c;index_SHA[key]=h.hexdigest()
    matrix=[]
    for seq in [f'{i:02d}' for i in range(22)]:
        record=dict(sequence=seq,LiDAR_frames=counts['lidar'][seq],point_label_frames=counts['labels'][seq],
            object_3D_box_GT=False,current_Object_center_protocol_applicable=False,
            strict_causal_pose_certified=False,pose_source='provider SuMa SLAM estimate, not raw OXTS',
            new_GPU_model_calls=0,new_metric_protocol_required=True)
        for kind,zkey in [('poses','labels'),('times','calib'),('calib','calib')]:
            name=f'dataset/sequences/{seq}/{kind}.txt'
            if name not in archives[zkey].namelist():record[kind+'_available']=False;continue
            content=archives[zkey].read(name);record[kind+'_available']=True
            record[kind+'_sha256']=hashlib.sha256(content).hexdigest()
            if kind in ('poses','times'):
                numbers=np.atleast_2d(np.loadtxt(content.decode().splitlines())) if kind=='poses' else np.atleast_1d(np.loadtxt(content.decode().splitlines()))
                record[kind+'_rows']=len(numbers);record[kind+'_finite']=bool(np.isfinite(numbers).all())
                if kind=='times':record['times_strictly_increasing']=bool(np.all(np.diff(numbers)>0))
        if seq in pairmatrix:
            p=pairmatrix[seq];n=p['common_RGB_LiDAR_SuMa_pose_time_frames']
            record.update(official_raw_mapping_available=True,raw_drive=p['raw_drive'],raw_start=p['raw_start'],raw_end=p['raw_end'],
                RGB_certification_status=p['status'],verified_RGB_LiDAR_pose_time_pairs=n,
                RGB_directory_fully_certified=seq in partial['completed_RGB_sequences'],
                candidate_RGB_upper_bound=p['expected_odometry_frames'] if p['status']!='no_matching_local_raw_RGB' else 0,
                unverified_frames_must_not_be_called_absent=p['status']=='RGB_header_certification_pending')
        else:record.update(official_raw_mapping_available=False,
            RGB_certification_status='no_official_raw_correspondence_in_available_devkit; no global RGB absence claim',
            verified_RGB_LiDAR_pose_time_pairs=0,RGB_directory_fully_certified=False,candidate_RGB_upper_bound=None)
        matrix.append(record)
    for z in archives.values():z.close()
    object_index=ROOT/'dataset/processed/kitti_object_all_20261125_v1/method_frames.jsonl'
    object_drives={json.loads(l)['drive'].removesuffix('_sync') for l in object_index.open()}
    raw_drives={p['raw_drive'] for p in pairmatrix.values()}
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),sequence_count=22,
        complete_applicability_ledger=True,complete_RGB_frame_certification=partial['complete'],
        verified_pair_lower_bound=partial['total_common_frames'],
        certified_complete_RGB_sequences=partial['completed_RGB_sequences'],
        explicitly_partial_RGB_sequences=[r['sequence'] for r in matrix if r.get('unverified_frames_must_not_be_called_absent')],
        LiDAR_total_frames=sum(counts['lidar'].values()),public_point_label_total_frames=sum(counts['labels'].values()),
        matrix=matrix,archive_central_directory_index_sha256=index_SHA,
        Object_training_RGB_raw_drive_intersection_with_official_odometry_mapping=sorted(object_drives&raw_drives),
        source_sha256={str(p):sha(p) for p in [pairpath,object_index,Path(__file__),ROOT/'scripts/kitti_semantic_scope_20261125.py']},
        claims=['10883 is a certified lower bound, not all locally matchable RGB frames',
                'three complete RGB directories and partial07 retained; no continued broad scan after600s',
                'point semantics/instances are not3D boxes; no point-surface center substituted for Object box GT',
                'no new inference/training; no body heading or raw measured-motion truth inferred from SuMa poses'])
    with (REPORT/'SemanticKITTI_22seq_applicability_v1.json').open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(sequences=22,LiDAR_total=value['LiDAR_total_frames'],point_label_total=value['public_point_label_total_frames'],verified_pair_lower_bound=value['verified_pair_lower_bound'],RGB_full_certification=False)))


if __name__=='__main__':main()
