"""Apply the closed RGB source certificate to the immutable22-sequence ledger."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path

from conceptgraph.completion.kitti_local_all_20261125 import ROOT,REPORT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    initial=REPORT/'SemanticKITTI_22seq_applicability_v1.json'
    final=REPORT/'semantic_RGB_LiDAR_pose_pair_inventory_v3.json'
    ledger=copy.deepcopy(json.loads(initial.read_text()));source=json.loads(final.read_text())
    assert source['complete'] and source['total_common_frames']==22400
    keys={(p['sequence'],p['frame_idx']) for p in source['pairs']};assert len(keys)==22400
    assert all(p['point_instance_label_available'] for p in source['pairs'])
    matrix={r['sequence']:r for r in source['matrix']}
    for row in ledger['matrix']:
        seq=row['sequence']
        if seq not in matrix:continue
        record=matrix[seq];count=record['common_RGB_LiDAR_SuMa_pose_time_frames']
        row.update(RGB_certification_status=record['status'],verified_RGB_LiDAR_pose_time_pairs=count,
            RGB_directory_fully_certified=seq in source['completed_RGB_sequences'],
            unverified_frames_must_not_be_called_absent=False,
            input_pairing_only_not_completed_mapping=True,current_Object_center_protocol_applicable=False)
        if count:
            assert count==row['LiDAR_frames']==row['point_label_frames']
            assert {frame for s,frame in keys if s==seq}==set(range(count))
    ledger.update(created_utc=datetime.now(timezone.utc).isoformat(),complete_RGB_frame_certification=True,
        certified_input_pair_count=22400,verified_pair_lower_bound=22400,
        certified_complete_RGB_sequences=source['completed_RGB_sequences'],explicitly_partial_RGB_sequences=[],
        mapped_sequence_with_confirmed_missing_Depth_raw_RGB=['03'],
        interpretation='22400 input pairs with pointGT, not completed mapping/inference and not valid Object3D-box center evaluation',
        new_inference_or_training_calls=0,new_GPU_calls=0,
        claims=['all known official00..10 raw mappings checked;03 lacks corresponding Depth raw RGB',
            '11..21 have LiDAR/provider poses/times but no public pointGT and no raw mapping in the available official devkit; no claim of global RGB absence',
            'pointGT is not3D boxGT; no new surface-center truth, body heading or dynamic ground truth constructed',
            'SuMa poses are provider offline SLAM estimates; no measured-OXTS or strict-causality claim'],
        source_sha256={str(p):sha(p) for p in [initial,final,Path(__file__),ROOT/'scripts/kitti_semantic_scope_final_20261125.py']})
    with (REPORT/'SemanticKITTI_22seq_applicability_final_v2.json').open('x') as f:json.dump(ledger,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(sequences=22,input_pairs=22400,new_inference=0,GT_box_protocol_applicable=False)))


if __name__=='__main__':main()
