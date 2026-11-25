"""新八方向实验命名空间；原输入和历史实验实体不变。"""
from pathlib import Path
import json
from conceptgraph.completion import nuscenes_full_20261125 as source
SHARED_DATA=source.DATA
DATA=source.ROOT/'dataset/processed/nuscenes_full_20261125_single8_v1'
PILOT_DATA=source.ROOT/'dataset/processed/nuscenes_full_20261125_single8_pilot_v1'
RUN_ID='nuscenes_full_20261125_single8_main_v1'
RUN=source.ROOT/'outputs/results/nuscenes/validation/iterations'/RUN_ID
PILOT_RUN=RUN.parent/'nuscenes_full_20261125_single8_pilot_v1'
REPORT=source.REPORT
CONTRACT='single8_v1_source_relative_to_target'

def configure():
    from conceptgraph.completion import nuscenes_full_20261125_run as base
    from conceptgraph.completion import nuscenes_full_20261125_resource_pilot as pilot
    base.DATA=DATA;base.RUN=RUN;base.RUN_ID=RUN_ID
    pilot.FULL_DATA=DATA;pilot.FULL_RUN=RUN;pilot.DATA=PILOT_DATA;pilot.RUN=PILOT_RUN

def prepare_inputs():
    DATA.mkdir(exist_ok=True)
    for name in ['archives','member_binding.jsonl.gz','source_certification.json','phaseB_source_preseal_v1.json','scene_role_access_overlay_v2.json','resource_pilot_preregistration.json','effective_resource_policy_v8.json','overall_execution_clock_v2.json','formal_relation_contract_hold_v1.json','formal_relation_contract_release_v1.json','KITTI_GPU_released_notice.json']:
        target=DATA/name;origin=SHARED_DATA/name
        if target.is_symlink():assert target.resolve()==origin.resolve()
        elif target.exists():raise FileExistsError(target)
        else:target.symlink_to(origin,target_is_directory=origin.is_dir())
    target=DATA/'manifest.json'
    if not target.exists():
        old=json.loads((SHARED_DATA/'manifest.json').read_text())
        source.dump(target,dict(old,phase='single8_fixed_method_inputs_pending_full_source_certification',run=str(RUN),source_index_run=old['run'],source_index_manifest_sha256=source.sha(SHARED_DATA/'manifest.json'),relation_contract=CONTRACT,relation_protocol_path=str(PILOT_DATA/'relation_protocol_v1.json'),relation_protocol_sha256=source.sha(PILOT_DATA/'relation_protocol_v1.json')))
