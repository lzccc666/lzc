"""复用已经验证的官方进程所有权/就绪/关闭机制，新路径不覆盖旧日志。"""
import ast
from pathlib import Path
R = Path(__file__).resolve().parents[1]


def main():
    old = R/'source/run_official_prius_cutoff_pilot_015.py'
    new = R/'source/run_official_guarded_complex_016.py'
    text = old.read_text()
    changes = {
        'official_prius_cutoff_execution_015': 'official_guarded_complex_execution_016',
        "PILOT=R/'source/carla_prius_cutoff_pilot_008.py'": "PILOT=R/'source/run_guarded_complex_batch_008.py'",
        "DONE=R/'json/official_prius_braking_execution_009/report.json'":
            "DONE=R/'json/official_prius_cutoff_execution_015/report.json'",
        "R/'carla_official/prius_cutoff_pilot_008/registration.json'":
            "R/'carla_official/guarded_complex_registration_008.json'",
        '320*1024': '3456*1024**2',
        'stdout_graphics_prius_cutoff_017.log': 'stdout_graphics_guarded_complex_018.log',
        'carla_graphics_prius_cutoff_017.log': 'carla_graphics_guarded_complex_018.log',
        "[sys.executable,str(PILOT),'run']": "[sys.executable,str(PILOT)]",
        'timeout=1800': 'timeout=5400',
        "report=R/'carla_official/prius_cutoff_pilot_008/report.json'":
            "report=R/'json/guarded_complex_jobs_008/report.json'",
        "assert actual['actual_new_physical_calibrations']==2":
            "assert actual['actual_started_navigation_episodes']==3 and actual['subprocesses_and_audits_complete']",
        'actual_new_physical_calibrations=2': 'actual_new_physical_calibrations=0',
        'actual_new_navigation_episodes=0,actual_arrival_credit=False':
            'actual_new_navigation_episodes=3,strict_arrival_must_read_independent_reports=True',
        'two actual official Prius braking calibrations complete':
            'three actual guarded complex semantic episodes and audits complete',
        '官方执行器标定，不是全传感器导航': '官方完整传感器复杂语义开发导航，三方法共用新保护及持续身份核验',
    }
    for before, after in changes.items():
        assert before in text, before
        text = text.replace(before, after)
    ast.parse(text)
    with new.open('x') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
