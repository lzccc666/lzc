"""只汇总两项独立审计闭合的真实回合；保留未完成分母和全部普通失败。"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

R = Path(__file__).resolve().parents[1]
METRICS = ['target_route_generated', 'correct_first_proposal', 'correct_lock',
           'wrong_lock', 'strict_safe_arrival', 'collision', 'lane_violation']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, inputs):
    raw = Path(path).read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def key(job):
    return tuple(str(job[k]) for k in ['episode', 'seed', 'mode', 'condition'])


def classify(dest, jobfile, inputs):
    """审计失败不能进入成功率；超时但审计通过必须作为正式负结果进入。"""
    if not jobfile.exists():
        if (dest / 'failure.json').exists():
            return 'execution_error_pending_job', None
        return ('executed_pending_audits' if (dest / 'summary.json').exists()
                else 'started_unclosed' if dest.exists() else 'not_started'), None
    job = read(jobfile, inputs)
    if (job['exit_code'] or len(job['audits']) != 2
            or {a['source'] for a in job['audits']} != {
                'audit_carla_expansion_009.py', 'audit_carla_guarded_execution_002.py'}
            or any(a['exit_code'] for a in job['audits'])):
        return 'execution_or_audit_error', None
    return 'closed_independently_audited', job


def validate_closed(summary, arrival, guard):
    assert summary['frames'] == arrival['physical_ticks'] == guard['physical_ticks']
    assert summary['status'] == arrival['status']
    assert not summary['GT_in_agent'] and arrival['GT_not_sent_to_agent']
    assert guard['every_tick_guard_executed']
    assert guard['every_tick_obstacles_reconstructed']
    assert guard['raw_LiDAR_frames_verified'] == summary['frames']
    assert guard['next_controls_match_next_actual_applied']
    assert not arrival['strict_safe_arrival'] or (
        summary['actually_arrived'] and arrival['correct_lock']
        and not arrival['wrong_lock'] and not summary['collisions']
        and not arrival['lane_violation_ticks']
        and not any(arrival['actuation_limits'].values()))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['completion_state'] == 'closed_independently_audited':
            groups[(row['phase'], row['mode'], row['condition'], row['task_kind'])].append(row)
    output = []
    for (phase, mode, condition, kind), part in sorted(groups.items()):
        item = dict(phase=phase, mode=mode, condition=condition, task_kind=kind,
                    closed_episode_denominator=len(part),
                    distinct_tasks=len({r['episode'] for r in part}),
                    distinct_configs=len({r['config_id'] for r in part}),
                    seeds=sorted({r['seed'] for r in part}),
                    statuses=dict(Counter(r['status'] for r in part)))
        for metric in METRICS:
            item[metric + '_count'] = sum(bool(r[metric]) for r in part)
            item[metric + '_fraction_of_closed'] = item[metric + '_count'] / len(part)
        item['arrival_task_denominator'] = len(part) if kind == 'positive' else 0
        item['positive_arrival_fraction_of_closed'] = (
            item['strict_safe_arrival_fraction_of_closed'] if kind == 'positive' else None)
        item['wall_s_mean'] = statistics.mean(r['wall_s'] for r in part)
        item['wall_s_sample_sd'] = (statistics.stdev(r['wall_s'] for r in part)
                                    if len(part) > 1 else None)
        item['CI'] = None
        item['statistics_note'] = '固定队列未闭合，当前比例仅描述已闭合前缀；帧和种子不作为独立场景，不作总体置信区间或效果结论。'
        output.append(item)
    return output


def profile_snapshot(inputs, out):
    path = R / 'performance/carla_expansion_live_profile_001/samples.jsonl'
    if not path.exists():
        return {}
    raw = path.read_bytes()
    inputs[str(path) + '#snapshot_prefix'] = hashlib.sha256(raw).hexdigest()
    with (out / 'performance_samples_snapshot.jsonl').open('xb') as stream:
        stream.write(raw)
    grouped = {}
    # 最后一行若写入未完成，仅保留完整换行前缀；不干扰采样器。
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            continue
        sample = json.loads(line)
        k = key(sample)
        v = grouped.setdefault(k, dict(samples=0, CPU_VmHWM_MiB=0.,
                                      GPU_sampled_peak_MiB=None,
                                      first_sample_elapsed_process_s=sample['elapsed_process_s']))
        v['samples'] += 1
        v['CPU_VmHWM_MiB'] = max(v['CPU_VmHWM_MiB'], sample['VmHWM_MiB'])
        if sample['GPU_sampled_used_MiB'] is not None:
            v['GPU_sampled_peak_MiB'] = max(v['GPU_sampled_peak_MiB'] or 0,
                                          sample['GPU_sampled_used_MiB'])
        v['last_sample_UTC'] = sample['UTC']
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id必须为单个新目录名')
    out = R / 'carla_official' / args.run_id
    out.mkdir(exist_ok=False)
    inputs = {}
    bankpath = R / 'carla_official/expansion_registration_011.json'
    bank = read(bankpath, inputs)
    task_lookup = {e['episode']: e for e in bank['episodes']}
    profiles = profile_snapshot(inputs, out)
    rows = []
    for phase, jobs in bank['execution_phases'].items():
        for number, job in enumerate(jobs, 1):
            task = task_lookup[job['episode']]
            dest = (R / 'carla_official/expansion_navigation_010' / job['episode']
                    / ('seed' + str(job['seed'])) / job['mode'] / job['condition'])
            jobfile = R / ('json/expansion_jobs_011_' + phase) / f'job_{number:03d}.json'
            state, receipt = classify(dest, jobfile, inputs)
            row = dict(phase=phase, planned_job_number=number, **job,
                       config_id=task['config_id'], task_kind=task['task_kind'],
                       completion_state=state, episode_directory=str(dest))
            if receipt is not None:
                assert key(receipt) == key(job)
                assert Path(receipt['episode_directory']) == dest
                summary = read(dest / 'summary.json', inputs)
                arrival = read(dest / 'independent_expansion_audit_009/report.json', inputs)
                guard = read(dest / 'independent_motion_guard_audit_002/report.json', inputs)
                validate_closed(summary, arrival, guard)
                # 核对审计引用内容，防止汇总已被改写的陈旧报告。
                for filename, expected in arrival['inputs_sha256'].items():
                    assert digest(filename) == expected, filename
                row.update({m: arrival[m] for m in METRICS if m in arrival})
                row.update(collision=bool(summary['collisions']),
                           lane_violation=bool(arrival['lane_violation_ticks']),
                           status=summary['status'], physics_ticks=summary['frames'],
                           frontend_calls=summary['perception_calls'], wall_s=summary['wall_s'],
                           sim_s=summary['sim_s'],
                           actual_Qwen_calls=arrival['actual_final_GPU_model_calls'],
                           strict_schema_valid_outputs=arrival['schema_valid_model_outputs'],
                           maximum_contiguous_hold_s=arrival['maximum_contiguous_hold_s'],
                           motion_guard_stop_requests=guard['actual_stop_commands_verified'],
                           no_match_any_selection=summary['no_match_any_selection'],
                           semantic_mode_alone_does_not_prove_Qwen_used=True,
                           profiler=json.dumps(profiles.get(key(job)), ensure_ascii=False))
                positive = task['task_kind'] == 'positive'
                row['failure_stage'] = (
                    'arrived' if arrival['strict_safe_arrival'] else
                    'not_an_arrival_task' if not positive else
                    'wrong_lock' if arrival['wrong_lock'] else
                    'no_correct_proposal' if not arrival['correct_first_proposal'] else
                    'proposal_without_correct_lock' if not arrival['correct_lock'] else
                    'lock_without_target_route' if not arrival['target_route_generated'] else
                    'selected_locked_planned_but_not_arrived')
            rows.append(row)
    closed = [r for r in rows if r['completion_state'] == 'closed_independently_audited']
    positive = [r for r in closed if r['phase'] == 'primary' and r['task_kind'] == 'positive']
    winners = [r for r in positive if r['strict_safe_arrival']]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / 'episodes.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  registered_episodes=len(rows), closed_episodes=len(closed),
                  completion_counts=dict(Counter(r['completion_state'] for r in rows)),
                  phase_planned=dict(Counter(r['phase'] for r in rows)),
                  phase_closed=dict(Counter(r['phase'] for r in closed)),
                  distinct_actual_closed_tasks=len({r['episode'] for r in closed}),
                  primary_positive_closed=len(positive),
                  primary_positive_safe_arrivals=len(winners),
                  distinct_primary_positive_safe_tasks=len({r['episode'] for r in winners}),
                  distinct_primary_positive_safe_configs=len({r['config_id'] for r in winners}),
                  physical_ticks=sum(r['physics_ticks'] for r in closed),
                  frontend_calls=sum(r['frontend_calls'] for r in closed),
                  actual_GPU_Qwen_calls=sum(r['actual_Qwen_calls'] for r in closed),
                  per_group=aggregate(rows),
                  paired_method_comparisons_pending=True,
                  full_history_maintenance_ablation_pending=True,
                  no_collision_alone_is_not_task_success=True,
                  nonarrival_tasks_not_in_positive_arrival_denominator=True,
                  no_realtime_or_Qwen_improvement_claim=True,
                  performance_note='CPU是已采样时刻进程VmHWM；GPU为3秒采样最大值下界，首回合未采样；异步Qwen延迟不能从前端耗时反推。',
                  inputs_sha256=inputs, source_sha256=digest(__file__),
                  whole_project_sealed_test=False, goal_complete=False)
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in [
        'registered_episodes', 'closed_episodes', 'completion_counts',
        'primary_positive_safe_arrivals', 'physical_ticks', 'actual_GPU_Qwen_calls']},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
