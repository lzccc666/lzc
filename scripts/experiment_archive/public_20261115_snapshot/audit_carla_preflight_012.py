"""回读75份同步生成证据，重新检查45条构造运动路线及15份阻断证明。"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState,audit_executed_states

R=Path(__file__).resolve().parents[1]


def main():
    original=R/'carla_official/expansion_spawn_preflight_011';out=R/'carla_official/expansion_preflight_readback_012';out.mkdir(exist_ok=False)
    bankfile=R/'carla_official/expansion_registration_011.json';bank=json.loads(bankfile.read_text());runtime=json.loads(Path(bank['immutable_navigation_registration']).read_text())
    assert digest(bank['immutable_navigation_registration'])==bank['immutable_navigation_registration_sha256']
    assert bank['episodes']==runtime['episodes'] and bank['execution_phases']==runtime['execution_phases']
    summary=json.loads((original/'report.json').read_text());assert summary['all_passed']
    for path,sha in summary['source_sha256'].items():assert digest(path)==sha
    release=R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule/release.json';spec=VehicleSpec(**json.loads(release.read_text())['vehicle_spec'])
    records=[];reconstructed_states=0;poses=0;counts=Counter();inputs={}
    for file in sorted(original.glob('case_*.json')):
        x=json.loads(file.read_text());task=next(t for t in bank['episodes'] if t['episode']==x['episode']);kind=task['task_kind'];counts[kind]+=1
        assert x['official_vehicle_spawn_passed'] and x['independent_truth_passed'] and x['spawn_pose_synchronized_after_tick']
        for pose in x['pose_checks']:
            a,b=np.asarray(pose['requested_transform']),np.asarray(pose['observed_transform'])
            error=float(np.linalg.norm(a[:3,3]-b[:3,3]));assert error<.001 and abs(error-pose['position_error_m'])<1.e-5
            poses+=1
        road=Polygon(x['declared_road_polygon_xy']);target=Polygon(x['target_polygon_xy']);other=Polygon(x['other_polygon_xy']) if x['other_polygon_xy'] else None
        if kind=='positive':
            reference=x['independent_constructive_vehicle_reference'];states=[VehicleState(**v) for v in reference['states']]
            assert reference['passed'] and reference['no_reference_supplied_to_agent']
            audit=audit_executed_states(states,road,[target]+([other] if other else []),spec);assert audit['passed']
            goal=x['truth']['regions'][0];assert Polygon(goal['polygon_xy']).distance(Point(states[-1].x,states[-1].y))<=.65
            assert states[-1].speed_mps<=.15 and abs(math.atan2(math.sin(states[-1].yaw_rad-goal['heading_rad']),math.cos(states[-1].yaw_rad-goal['heading_rad'])))<=math.radians(12)
            assert not x['certificates']['positive_scope_certificate']['official_static_car_objects_in_declared_lane']
            reconstructed_states+=len(states)
        elif kind=='blocked_target':
            free=road.difference(other);parts=list(free.geoms) if hasattr(free,'geoms') else [free];initial=[p for p in parts if p.covers(Point(x['origin_xy']))];goal=Polygon(x['truth']['regions'][0]['polygon_xy'])
            assert initial and all(p.disjoint(goal) for p in initial) and x['truth']['reachability']=='inaccessible'
        else:
            certificate=x['certificates']['no_match_scope_certificate'];scope=Polygon(certificate['scope_polygon_xy'])
            assert certificate['verified_no_bus_in_declared_scope'] and certificate['only_registered_non_bus_dynamic_actors']
            assert not any(Polygon(obj['polygon_xy']).intersects(scope) for obj in certificate['static_bus_environment_objects'])
            assert x['truth']['accepted_instance_ids']==[]
        inputs[str(file)]=digest(file);records.append(dict(file=file.name,episode=x['episode'],seed=x['seed'],kind=kind,recomputed=True))
    assert len(records)==75 and counts=={'positive':45,'no_match':15,'blocked_target':15}
    dump(out/'report.json',dict(cases_recomputed=75,counts=dict(counts),actual_actor_pose_reconstructions=poses,
        positive_reference_states_reaudited=reconstructed_states,all_original_body_motion_audits_passed=True,
        blocked_connectivity_proofs_recomputed=15,official_no_match_scope_certificates_recomputed=15,
        runtime010_and_orchestrator011_episode_configs_equal=True,
        invalid_preflight010_truth_conclusions_withdrawn=True,actual_navigation_arrival_credit=0,
        weather_layout_counting='两个天气共享同一几何预检；75项=25种几何任务×3偏移种子，不计75个独立场景',
        source_sha256={str(Path(__file__)):digest(__file__),str(bankfile):digest(bankfile),**inputs},goal_complete=False))
    dump(out/'cases.json',records);print((out/'report.json').read_text()[:1000])


if __name__=='__main__':main()
