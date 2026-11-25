"""用户要求缩至10回合后的只读汇总；保留中止回合，不改历史数据。"""
from collections import Counter
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics

R = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = R / 'carla_official/carla_ten_041'
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(path):
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    progress = R / 'carla_official/expansion_progress_040'
    read(progress / 'report.json')
    inputs[str(progress / 'episodes.csv')] = digest(progress / 'episodes.csv')
    with (progress / 'episodes.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['phase'] == 'primary'
                and int(r['planned_job_number']) <= 10]
    assert len(rows) == 10
    for number, row in enumerate(rows, 1):
        assert int(row['planned_job_number']) == number
        assert row['completion_state'] == 'closed_independently_audited'
        job = read(R / f'json/expansion_jobs_011_primary/job_{number:03d}.json')
        assert job['exit_code'] == 0 and len(job['audits']) == 2
        assert all(a['exit_code'] == 0 for a in job['audits'])
        d = Path(job['episode_directory'])
        read(d / 'summary.json')
        read(d / 'independent_expansion_audit_009/report.json')
        read(d / 'independent_motion_guard_audit_002/report.json')

    def count(part, name):
        return sum(r[name] == 'True' for r in part)

    positive = [r for r in rows if r['task_kind'] == 'positive']
    relation = Path(rows[6]['episode_directory'])
    graph = Counter()
    candidate = Counter()
    relation_count = 0
    perception = relation / 'perception.jsonl.gz'
    inputs[str(perception)] = digest(perception)
    with gzip.open(perception, 'rt') as f:
        for line in f:
            p = json.loads(line)
            graph.update(n['category'] for n in p['nodes'])
            semantic = p['semantic_audit']['input_row']
            candidate.update(n['category'] for n in semantic['candidate_nodes'])
            relation_count += len(semantic['relation_metadata'])
    callsfile = relation / 'semantic_backend/calls.jsonl'
    inputs[str(callsfile)] = digest(callsfile)
    calls = [json.loads(line) for line in callsfile.open()]
    outputs = [json.loads(c['raw_output']) for c in calls]
    assert len(calls) == int(rows[6]['actual_Qwen_calls']) == 65
    assert all(c['schema_valid'] for c in calls)
    queue = read(relation / 'semantic_queue_final.json')
    events = Counter((e['kind'], e.get('reason')) for e in queue['events'])
    blocked = Path(rows[2]['episode_directory'])
    with gzip.open(blocked / 'perception.jsonl.gz', 'rt') as f:
        for line in f:
            last = json.loads(line)
    blocked_audit = last['semantic_audit']
    inputs[str(blocked / 'perception.jsonl.gz')] = digest(blocked / 'perception.jsonl.gz')
    guard = read(R / 'carla_official/expansion_failure_trace_014/report.json')
    interrupted = R / ('carla_official/expansion_navigation_010/'
        'town10_spawn000_CloudyNoon_car20/seed17/adapter/full_rule_geometry')
    failure = interrupted / 'failure.json'
    if failure.exists():
        read(failure)
    report = dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        user_requested_closed_episode_limit=10, closed_independently_audited=10,
        task_counts=dict(Counter(r['task_kind'] for r in rows)),
        distinct_configs=len({r['config_id'] for r in rows}),
        positive_task_denominator=len(positive),
        correct_first_proposal=count(positive, 'correct_first_proposal'),
        ever_correct_lock=count(positive, 'correct_lock'),
        ever_wrong_lock=count(positive, 'wrong_lock'),
        target_route_generated=count(positive, 'target_route_generated'),
        strict_safe_arrival=count(positive, 'strict_safe_arrival'),
        strict_safe_arrival_fraction=count(positive, 'strict_safe_arrival') / len(positive),
        collisions=count(rows, 'collision'), lane_violations=count(rows, 'lane_violation'),
        physical_ticks=sum(int(r['physics_ticks']) for r in rows),
        frontend_calls=sum(int(r['frontend_calls']) for r in rows),
        actual_Qwen_calls=sum(int(r['actual_Qwen_calls']) for r in rows),
        relation_task=dict(job=7, graph_category_appearances=dict(graph),
            candidate_category_appearances=dict(candidate), semantic_relation_appearances=relation_count,
            model_calls=len(calls), schema_valid_outputs=sum(c['schema_valid'] for c in calls),
            no_match_outputs=sum(o.get('no_match') is True for o in outputs),
            mean_model_latency_wall_s=statistics.mean(c['GPU_latency_s'] for c in calls),
            queue_events=[dict(kind=k[0], reason=k[1], count=v) for k, v in events.items()],
            interpretation='输入中缺失指令要求的卡车参照与对象关系；不能据此估计Qwen本身的正确率'),
        blocked_task_example=dict(job=3, current_evidence_passed=blocked_audit['current_evidence_passed'],
            semantic_decision=blocked_audit['decision'], plan_check=blocked_audit['plan_check'],
            interpretation='已观察并提议卡车；车辆可行路线复核不通过，最终候选为空'),
        vehicle_stall_example=dict(job=2, motion_guard_stop_ticks=guard['motion_guard_stop_ticks'],
            plan_failure_ticks=guard['plan_failure_ticks'],
            interpretation='已有正确提议、锁定、路线后出现保护停车与初始车身复核失败；是逐帧关联证据，尚未用干预实验确认单一根因'),
        scope_stop=dict(official_CARLA_and_batch_stopped=True,
            interrupted_job_number=11, interrupted_directory=str(interrupted),
            interrupted_record_retained=True, interrupted_excluded_from_denominator=True,
            explanation='自动批处理在停止前已启动11；按用户要求中止，未删除传感器或失败记录',
            followup_210_started=False, do_not_restart_CARLA_batch=True),
        cautions=['仅固定队列前10回合，来自同一Town10，不是10张独立地图',
            'correct_lock是曾经正确锁定；可与曾经错误锁定同时为真',
            '2个无目标与2个被阻挡任务不进入应当到达任务的分母',
            '未证明实时性，也未完成base/Adapter同输入配对对比，不能声称微调提升',
            '无碰撞不等于任务完成；总体目标仍未完成'],
        inputs_sha256=inputs, source_sha256=digest(__file__), goal_complete=False)
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with (out / 'episodes.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = '''# CARLA 首批10回合及当前卡点

用户将当前批次缩至10回合。10回合均已完成两项独立审计，官方CARLA、批处理和剖析器已停止；不继续原150主回合或后210回合。第11回合在停止前被自动启动，现已中止，其记录保留且不计入成绩。

10回合来自同一Town10，共6个应当到达任务、2个无目标任务、2个被阻挡任务。6个应当到达任务：5个首次提议正确，5个曾正确锁定，5个生成目标路线，1个严格安全到达（1/6，16.67%）；其中1个曾发生错误锁定。全部10回合无碰撞或独立核验的越界，但这不代表导航成功。

当前主要卡点是“有路线以后能否让车稳定开到终点”。第2回合已选对、锁定并生成路线，但614个物理帧受到停车保护，434个前端帧出现INVALID_INITIAL_BODY，最终超时。需核查规划、车身安全余量、实际转向/制动和重规划一致性，不能靠放宽到达标准改善成绩。

另一个卡点是“复杂指令需要的参照物没有进入图”。第7回合要求以卡车为参照，但450次建图和模型输入均只有小汽车，关系为空。真实Qwen调用65次，65次格式正确且均拒绝选择。它没有足够信息作答，不能直接解释为Qwen识别错误；也不能由此声称微调提升。队列另有13次上下文变化丢弃、2次过期丢弃，仍需后续排查延迟影响。

被阻挡任务不能与上述控制失败混为一谈。第3回合已观察到卡车且当前证据通过，但车辆路线复核返回NO_FEASIBLE_PRIMITIVE_ROUTE；没有最终锁定是规划门闸拦截，而非未识别到卡车。无公交车的两个任务也不应计入“应当到达”的成功率分母。

当前建议顺序：先只读定位第2/5/6/10回合的停车、偏航、规划失效与身份切换时间线；再查第7回合卡车未进入图的原因（视野/遮挡、检测、深度支持或筛选）；修复后在固定配置和相同传感器输入上核验，不扩大CARLA批量。所有旧结果和普通失败保留，goal_complete=false。

同目录episodes.csv为逐回合表；report.json为指标、原始结果位置、文件哈希和诊断证据。更大的原计划库存仍在expansion_progress_040中，其330为历史登记数，不是完成数，也不是当前执行范围。
'''
    (out / '说明.md').open('x').write(text)
    handoff = dict(UTC=report['UTC'], latest_scope='CARLA先做10个，已经完成并停机',
        CARLA_restart_allowed=False, CARLA_limit=10, report=str(out / 'report.json'),
        prior_handoff=str(R / 'json/continuation_handoff_016.json'),
        inherited_other_project_tasks_remain_pending=True, goal_complete=False)
    with (R / 'json/continuation_handoff_017.json').open('x') as f:
        json.dump(handoff, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ['closed_independently_audited',
        'positive_task_denominator', 'strict_safe_arrival', 'actual_Qwen_calls', 'goal_complete']},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
