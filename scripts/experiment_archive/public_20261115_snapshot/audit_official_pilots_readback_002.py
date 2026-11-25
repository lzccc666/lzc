"""独立重算10条官方标定轨迹，并核对冻结控制类实际生成的每个命令。"""
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
O=R/'carla_official/all_pilot_readback_002'
RUNS=[('stock_vehicle_feedback_pilot_004','carla_stock_vehicle_feedback_pilot_004.py'),
      ('prius_braking_pilot_005','carla_prius_braking_pilot_005.py'),
      ('prius_braking_pilot_006','carla_prius_braking_pilot_006.py'),
      ('prius_braking_pilot_007','carla_prius_braking_pilot_007.py')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    O.mkdir(exist_ok=False);records=[];sources={}
    for run,filename in RUNS:
        folder=R/'carla_official'/run;source=R/'source'/filename
        report=json.loads((folder/'report.json').read_text());registration=json.loads((folder/'registration.json').read_text())
        assert all(sha(p)==h for p,h in registration['source_sha256'].items())
        tree=ast.parse(source.read_text());definition=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AccelerationFeedback')
        assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='apply_physics_control' for x in ast.walk(tree))
        scope={'np':np};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),scope)
        for item in report['results']:
            path=folder/f"{item['case']}_{item['policy']}.jsonl.gz"
            assert sha(path)==item['raw_sha256']
            with gzip.open(path,'rt') as stream:values=np.array(list(map(json.loads,stream)))
            assert len(values)==510 and np.all(values[:,1]==values[:,2])
            assert np.allclose(np.diff(values[:,3]),.1,atol=1e-7)
            assert np.allclose(np.linalg.norm(values[:,4:6],axis=1),values[:,17],atol=1e-10)
            assert np.allclose(values[:,13],values[:,19],atol=1e-7) and np.allclose(values[:,14],values[:,20],atol=1e-7)
            controller=scope['AccelerationFeedback']();command=(0.,1.);maximum_command_error=0.
            for row in values:
                maximum_command_error=max(maximum_command_error,abs(command[0]-row[13]),abs(command[1]-row[14]))
                driving=10<=row[0]<410
                command=controller.step(row[16],row[18],row[17],.55 if driving else 0.,not driving)
                if row[0]<9:command=(0.,1.)
            assert maximum_command_error<1e-6,(run,item['case'],maximum_command_error)
            acc=np.diff(values[:,17])/np.diff(values[:,3]);wheel=-np.arctan(np.mean(np.tan(np.radians(values[:,11:13])),axis=1));rate=np.abs(np.diff(wheel)/np.diff(values[:,3]));distance=float(np.linalg.norm(np.diff(values[:,7:9],axis=0),axis=1).sum())
            violations=dict(acceleration=int((acc>1+1e-5).sum()),braking=int((acc< -2-1e-5).sum()),speed=int((values[:,17]>2+1e-5).sum()),steering=int((np.abs(wheel)>.5585053606381855+1e-5).sum()),steering_rate=int((rate>.4363323129985824+1e-5).sum()))
            safe=not any(violations.values()) and not item['collisions'];functional=bool(values[:,17].max()>=.25 and distance>=2 and values[-1,17]<=.08)
            assert safe==item['actuation_constraints_satisfied'] and functional==item['functional_drive_and_stop']
            assert bool(safe and functional)==item['combined_calibration_pass']
            assert abs(acc.max()-item['maximum_acceleration_mps2'])<1e-10 and abs(acc.min()-item['minimum_acceleration_mps2'])<1e-10
            release=json.loads((folder/f"{item['case']}_{item['policy']}_release.json").read_text());assert release['actual_torque_curve']==release['original_torque_curve']
            records.append(dict(run=run,case=item['case'],policy=item['policy'],actual_recorded_ticks=510,raw_sha256=sha(path),maximum_controller_command_error=maximum_command_error,actual_velocity_matches_report=True,maximum_acceleration_mps2=float(acc.max()),minimum_acceleration_mps2=float(acc.min()),final_speed_mps=float(values[-1,17]),path_distance_m=distance,violations=violations,safe=safe,functional=functional,combined_pass=bool(safe and functional),actual_navigation_or_arrival_credit=False))
            sources[str(path)]=sha(path)
        sources[str(source)]=sha(source);sources[str(folder/'report.json')]=sha(folder/'report.json')
    assert len(records)==10 and sum(x['combined_pass'] for x in records)==4
    with (O/'report.json').open('x') as stream:json.dump(dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_calibration_trajectories=10,
        actual_recorded_ticks=5100,additional_pre_sensor_warmup_ticks=100,records=records,
        control_commands_recomputed_from_frozen_AST_and_saved_observations=True,
        maximum_command_tolerance=1e-6,old_failures_retained=True,source_sha256=sources,
        evaluator_sha256=sha(__file__),full_navigation_rerun_proved=False,goal_complete=False,
    ),stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(trajectories=10,actual_recorded_ticks=5100,combined_pass=4,latest007_pass='2/2',actual_arrival_credit=False,goal_complete=False)),flush=True)


if __name__=='__main__':main()
