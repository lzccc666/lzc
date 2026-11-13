"""原GPU队列终止后安装新CPU环境，实际导航复演、独立审核并重算450主表。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
SOURCE = OUTPUT / 'source_bundle'
ORIGINALS = {} # 串行恢复监督器等待GPU队列退出后调用；保留GPU检查


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def live(pid, tick):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == tick
    except FileNotFoundError:
        return False


def main():
    destination = OUTPUT / 'reproduction/isolated_environment_after_GPU_005'
    destination.mkdir(exist_ok=False)
    environment = OUTPUT / 'reproduction/isolated_cpu_environment_004'
    navigation = OUTPUT / 'reproduction/isolated_cpu_executed_002'
    plan = json.loads((environment / 'plan.json').read_text())
    assert not plan['missing_metadata_requirements']
    assert plan['runtime_package_override']['version'] == '3.1.7'
    helpers = [
        SOURCE / 'isolated_cpu_environment_pack_004.py',
        SOURCE / 'standalone_cpu_runner_005.py',
        SOURCE / 'prepare_isolated_environment_navigation_003.py',
        SOURCE / 'isolated_environment_navigation_audit_003.py',
        Path(__file__).resolve(), environment / 'plan.json',
        environment / 'installed_with_clip_001.json', environment / 'dependency_patch_clip_001/report.json',
        OUTPUT / 'reproduction/standalone_portable_release_001/release.json',
    ]
    frozen = {str(path): sha(path) for path in helpers}
    save(destination / 'release.json', dict(
        prepared_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        queue_process_pid=os.getpid(),
        queue_process_start_ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
        original_process_start_ticks=ORIGINALS, maximum_wait_s=176400,
        maximum_execution_wall_s=7200, minimum_free_bytes_before_install=11_000_000_000,
        minimum_free_bytes_during_execution=6_000_000_000,
        new_environment_storage_budget_bytes=7_000_000_000,
        navigation_storage_budget_bytes=500_000_000,
        model_policy='新环境CPU前端；不启动Qwen，不初始化CUDA；原GPU队列全部终止且GPU空闲后启动',
        earlier_failures_preserved_and_not_assumed_successful=True,
        frozen_main_table_credit=False, new_independent_scene_credit=0,
        shared_base_Python_standard_library_and_native_libraries_declared=True,
        queued_work_is_not_completed=True, new_test_opened=False, goal_complete=False,
    ))
    begun = time.monotonic()
    try:
        while any(live(pid, tick) for pid, tick in ORIGINALS.items()):
            if time.monotonic() - begun > 176400:
                raise TimeoutError('原GPU队列在冻结等待预算内未终止；未安装新环境')
            time.sleep(5)
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True, timeout=15)
        assert not gpu.stdout.strip(), '原队列终止但GPU仍占用；不并行启动复演'
        for path, expected in frozen.items():
            assert sha(path) == expected, path
        assert shutil.disk_usage(OUTPUT).free >= 11_000_000_000
        preceding = {}
        for relative in [
            'ablations/semantic_navigation_chain_004/complete.json',
            'faults/realtime_fault_after_semantics_002/complete.json',
            'performance/runtime_profile_after_faults_002/complete.json',
        ]:
            path = OUTPUT / relative
            preceding[relative] = dict(exists=path.is_file(), sha256=sha(path) if path.is_file() else None)
        save(destination / 'preceding_terminal.json', dict(
            all_original_processes_exited=True, GPU_compute_processes=[],
            preceding=preceding, preceding_success_not_assumed=True, goal_complete=False))
        execution_begun = time.monotonic()
        child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                         PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                         LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
                         PYTHONPATH='/root/autodl-tmp/lizichong')
        jobs = []

        def run(label, command, cwd, timeout, env=None):
            for path, expected in frozen.items():
                assert sha(path) == expected, path
            assert shutil.disk_usage(OUTPUT).free >= 6_000_000_000
            remaining = 7200 - (time.monotonic() - execution_begun)
            assert remaining > 1
            with (destination / (label + '.log')).open('x') as log:
                result = subprocess.run(command, cwd=cwd, env=env or child_env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(timeout, remaining), check=False)
            record = dict(job=label, command=command, cwd=str(cwd), returncode=result.returncode)
            jobs.append(record)
            save(destination / (label + '_exit.json'), record)
            return result.returncode

        assert (environment / 'installed_with_clip_001.json').is_file()
        patch = json.loads((environment / 'dependency_patch_clip_001/report.json').read_text())
        for row in patch['files']:
            assert sha(row['destination']) == row['sha256']
        installed = json.loads((environment / 'installed_with_clip_001.json').read_text())
        python = installed['python']
        navigation_code = run('01_navigation', [sys.executable,
            str(SOURCE / 'prepare_isolated_environment_navigation_003.py'),
            '--environment', str(environment), '--destination', str(navigation), '--execute'], OUTPUT, 1080)
        audit_code = None
        if (navigation / 'manifest.json').is_file():
            audit_code = run('02_independent_navigation_audit', [sys.executable,
                str(SOURCE / 'isolated_environment_navigation_audit_003.py')], SOURCE.parents[3], 900)
        audit_path = navigation / 'independent_audit_001/report.json'
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
        navigation_passed = bool(navigation_code == 0 and audit_code == 0 and audit
                                 and audit['successful_task_reproduction_accepted'])
        statistics_equal = False
        statistics_code = None
        if (navigation / 'manifest.json').is_file():
            manifest = json.loads((navigation / 'manifest.json').read_text())
            checkout = Path(manifest['isolated_root'])
            new_output = checkout / 'outputs/project_v2/thesis_completion_v1'
            original_input = OUTPUT / 'interactive_simulation/realtime_three_model_full_001'
            target = new_output / 'interactive_simulation/realtime_three_model_full_001'
            assert not target.exists()
            target.symlink_to(original_input, target_is_directory=True)
            save(destination / 'statistics_inputs.json', dict(
                dataset_read_only_reference=str(original_input), new_checkout_link=str(target),
                input_sha256={name: sha(original_input / name)
                              for name in ['release.json', 'summary.json', 'trial_records.jsonl']},
                inference_finished_before_statistics_reference_added=True,
                raw_frames_copied=False, new_model_calls=0, goal_complete=False))
            stats_env = dict(child_env, PYTHONPATH=str(checkout), TMPDIR=str(navigation / 'tmp'),
                             MPLCONFIGDIR=str(navigation / 'matplotlib'),
                             YOLO_CONFIG_DIR=str(navigation / 'ultralytics'))
            statistics_code = run('03_main_table_recompute', [python, '-m',
                'conceptgraph.completion.realtime_matrix_statistics',
                '--input-run', 'realtime_three_model_full_001',
                '--run-id', 'isolated_environment_main450_recompute_001'], checkout, 1800, stats_env)
            if statistics_code == 0:
                reproduced = new_output / 'statistics/isolated_environment_main450_recompute_001'
                original = OUTPUT / 'statistics/realtime_three_model_full_001'
                comparisons = {}
                for name in ['report.json', 'audited_trials.jsonl', 'trials.csv', 'generator_source.py']:
                    comparisons[name] = dict(original_sha256=sha(original / name),
                                             reproduced_sha256=sha(reproduced / name))
                statistics_equal = all(row['original_sha256'] == row['reproduced_sha256']
                                       for row in comparisons.values())
                save(destination / 'statistics_comparison.json', dict(
                    byte_identical=statistics_equal, files=comparisons,
                    input_trials=450, original_environment_not_used_for_recompute=True,
                    recompute_python=python, new_model_calls=0, goal_complete=False))
        complete = navigation_passed and statistics_code == 0 and statistics_equal
        save(destination / ('complete.json' if complete else 'incomplete.json'), dict(
            complete=complete, jobs=jobs, new_environment_installed=True,
            successful_small_task_reproduction=navigation_passed,
            independent_navigation_audit_sha256=sha(audit_path) if audit_path.is_file() else None,
            main_450_recomputed_in_new_environment=statistics_code == 0,
            main_statistics_byte_identical=statistics_equal,
            environment_installed_manifest_sha256=sha(environment / 'installed_with_clip_001.json'),
            actual_new_Python_environment=True, actual_new_operating_system=False,
            manuscript_text_written=False, new_test_opened=False,
            new_independent_scene_credit=0, frozen_main_table_credit=False,
            execution_elapsed_s=time.monotonic() - execution_begun,
            elapsed_s_including_wait=time.monotonic() - begun, goal_complete=False,
        ))
        if not complete:
            raise RuntimeError('新环境复演或主表核验未通过；失败保留')
    except BaseException as error:
        save(destination / 'interrupted.json', dict(error=repr(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
