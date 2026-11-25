"""冻结P1/T2/T3v2与公平A，在封存两log一次性评价；不依据新分数调参。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_validation import digest


def main():
    root = PROJECT_ROOT
    base = root/'outputs/results/nuscenes/validation/iterations'
    report = root/'outputs/reports/multiview_identity_repair_20261123/validation'
    sealed = json.loads((report/'nextlogs/sealed_inputs.json').read_text())
    assert sealed['no_method_mapping_or_scoring_executed_on_this_batch']
    manifest = Path(sealed['manifest'])
    truth = Path(sealed['GT'])
    assert digest(manifest) == sealed['manifest_sha256'] and digest(truth) == sealed['GT_sha256']
    methods = [('P1','mvrepair_20261123_P1_pilot_v1','mvrepair_20261123_P1_crosslog_v1',
        'conceptgraph.slam.mvrepair_20261123_conflict_position:ConflictOnlyFrontMapper'),
        ('T2','mvrepair_20261123_T2_pilot_v1','mvrepair_20261123_T2_crosslog_v1',
        'conceptgraph.slam.mvrepair_20261123_prototype:PrototypeContinuityMapper'),
        ('T3','mvrepair_20261123_T3_pilot_v2','mvrepair_20261123_T3_crosslog_v2',
        'conceptgraph.slam.mvrepair_20261123_native_representative_v2:NativeRepresentativeMapper')]
    fixed = {}
    for tag, pilot, regression, mapper in methods:
        for old in [pilot,regression]:
            assert json.loads((base/old/'mvrepair_20261123_report.json').read_text())['complete']
        release = json.loads((base/pilot/'mvrepair_20261123_release.json').read_text())
        assert release['mapper'] == mapper
        for p, meta in release['source_sha256'].items():
            assert digest(p) == meta['sha256'] == digest(meta['snapshot'])
            if p in fixed:
                assert fixed[p] == meta['sha256']
            fixed[p] = meta['sha256']
    assert fixed[str(root/'conceptgraph/slam/mvrepair_20261123_native_representative_v2.py')] == '16f16aad31f235081183e9f4f18843b7e103d244bcf8571dc326c5b5c3582c92'
    wave = experiment_directory('nuscenes','validation','mvrepair_final_nextlogs_wave_20261123_v1',create=True)
    destination = report/'final_nextlogs_wave_v1'
    destination.mkdir()
    for path in [Path(__file__).resolve(), root/'scripts/multiview_identity_repair_20261123_fair_front.py']:
        fixed[str(path)] = digest(path)
    for name in fixed:
        path = Path(name)
        snapshot = wave/'source_snapshot'/path.relative_to(root)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,snapshot)
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=17, methods=methods,
        scope='scene0042 and0047, first20 groups each; original38 development/debug; frozen one-wave confirmation, no unseen-test claim',
        cameras=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'],main_method_list=['A_front','C_three_dedup','A_front_physical_only','P1_three','T2_three','T3_three'],
        manifest_sha256=digest(manifest),ground_truth_sha256=digest(truth),source_sha256=fixed, GT_counts=sealed['GT_counts'],
        max_wall_s=900,max_new_bytes=128*2**20,minimum_free_bytes=20*2**30,model_calls=0,GPU_calls=0,
        no_score_based_parameter_changes=True,no_new_training_or_sensor_inputs=True,
        generic_runner_note='Historical generic release primary string mentions105/77; actual evaluator uses this frozen GT225/137 and reports full denominators.')
    (wave/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
    started = time.monotonic()
    env = dict(os.environ,PYTHONPATH=str(root),LD_PRELOAD='/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    runs = {}
    for tag,pilot,regression,mapper in methods:
        assert shutil.disk_usage(root).free > 20*2**30
        run_id = 'mvrepair_'+tag+'_nextlogs_confirmation_20261123_v1'
        command = [sys.executable,'-m','scripts.mvrepair_20261123_run_v3','--manifest',str(manifest),'--ground-truth',str(truth),
            '--run-id',run_id,'--report-dir',str(destination/tag),'--tag',tag,'--mapper',mapper]
        with (destination/(tag+'.log')).open('x') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,900-(time.monotonic()-started)))
        runs[tag] = str(base/run_id)
        print(tag,'predictions/evaluation complete; no method changes',flush=True)
    run_id = 'mvrepair_Aphysical_nextlogs_confirmation_20261123_v1'
    command = [sys.executable,'-m','scripts.multiview_identity_repair_20261123_fair_front','--manifest',str(manifest),'--ground-truth',str(truth),
        '--baseline-run',runs['P1'],'--run-id',run_id,'--report-dir',str(destination/'Aphysical')]
    with (destination/'Aphysical.log').open('x') as log:
        subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,900-(time.monotonic()-started)))
    runs['Afair'] = str(base/run_id)
    assert all(digest(p) == sha for p,sha in fixed.items())
    assert digest(manifest) == sealed['manifest_sha256'] and digest(truth) == sealed['GT_sha256']
    complete = dict(complete=True,runs=runs,wall_s=time.monotonic()-started,source_unchanged=True,input_manifest_GT_unchanged=True,
        no_score_based_method_changes=True,model_GPU_training_calls=0)
    (wave/'complete.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2)+'\n')
    (destination/'complete.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(complete,ensure_ascii=False))


if __name__ == '__main__':
    main()
