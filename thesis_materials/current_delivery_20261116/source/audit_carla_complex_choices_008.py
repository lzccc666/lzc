"""独立回读复杂语义实际提议/授权/锁定/到达；双受控对象一对一身份绑定。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def audit(episode):
    dest = episode/'independent_choice_audit_008'
    dest.mkdir(exist_ok=False)
    release = json.loads((episode/'release.json').read_text())
    truth = json.loads((episode/'evaluation_only/truth.json').read_text())
    summary = json.loads((episode/'summary.json').read_text())
    arrival = json.loads((episode/'independent_audit_003/report.json').read_text())
    observations = list(map(json.loads, (episode/'perception.jsonl').open()))
    target = truth['target_actor_id']
    targets = [dict(instance_id=target, center_xy=truth['center_xy'], category=truth['category'])]
    targets += truth['other_controlled_actors']
    frames = []
    submitted = {}
    serialization_changed = 0
    from real_graph_contract_003 import build_completion_prompt
    for obs in observations:
        nodes = obs['nodes']
        matches = {}
        if nodes:
            costs = np.array([[np.linalg.norm(np.asarray(n['center'][:2])-t['center_xy'])
                if n['category'] == t['category'] else 1e6 for t in targets] for n in nodes])
            ri, ci = linear_sum_assignment(costs)
            matches = {nodes[i]['node_id']: targets[j]['instance_id'] for i, j in zip(ri, ci)
                       if costs[i, j] <= 2.5}
        semantic = obs['semantic_audit']
        proposal = semantic['proposed_node_id']
        authorized = semantic['authorized_node_id']
        first = obs['candidates'][0]['node_id'] if obs['candidates'] else None
        assert first == authorized
        locked = obs['selected_node_id']
        correct = lambda i: i is not None and matches.get(i) == target
        stable = obs['lock']['state_after'] == 'LOCKED'
        frames.append(dict(frame_idx=obs['frame_idx'], time_s=obs['sim_time_s'],
            semantic_proposed_node_id=proposal, semantic_proposal_correct=correct(proposal),
            control_authorized_node_id=authorized, authorized_first_proposal_correct=correct(first),
            correct_stable_lock=bool(stable and correct(locked)),
            wrong_stable_lock=bool(stable and locked is not None and not correct(locked)),
            candidate_coverage=any(correct(n['node_id']) for n in obs['candidates']),
            observed_semantic_pool_coverage=any(correct(int(i)) for i in semantic['pool']['id_translation'].values()),
            target_route_generated=bool(obs['path_world'] and obs['path_purpose'] == 'target_approach'),
            actual_backend_result_used=semantic['background_result_used'],
            current_evidence_passed=semantic['current_evidence_passed'],
            identity_binding=matches))
        for event in semantic['events']:
            if event['kind'] == 'submitted' and event['request_id'] is not None:
                row = semantic['input_row']
                before = hashlib.sha256(build_completion_prompt('candidate_rerank', row).encode()).hexdigest()
                assert before == event['prompt_sha256'], '原始提交事件与本帧观测不一致'
                # 原队列submit实际以sort_keys=True序列化再恢复，字典顺序改变但字段和值未变。
                canonical = json.loads(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                assert canonical == row
                actual = hashlib.sha256(build_completion_prompt('candidate_rerank', canonical).encode()).hexdigest()
                serialization_changed += actual != before
                submitted[actual] = canonical
    backend_log = episode/'semantic_backend/calls.jsonl'
    calls = list(map(json.loads, backend_log.open())) if backend_log.exists() else []
    from carla_real_graph_bridge_002 import fingerprint
    for call in calls:
        assert call['prompt_sha256'] in submitted, '模型调用没有对应的真实观测提交'
        assert call['payload'] == submitted[call['prompt_sha256']]
        assert fingerprint(call['payload']) == call['payload_sha256']
        assert set(call['payload']) == {'task','sample_id','query_text','candidate_nodes',
                                       'candidate_ids','relation_metadata'}
        assert call['actual_CUDA'] and not call['fallback_used']
        assert call['input_tokens'] <= 2048 and call['output_tokens'] <= 256
    queue_final = json.loads((episode/'semantic_queue_final.json').read_text())
    # summary在finally关闭后台线程前写出；排队中的最后一次推理可能在cleanup期间完成。
    final_model_calls = queue_final['actual_model_calls'] or 0
    assert len(calls) == final_model_calls
    assert 0 <= summary['actual_Qwen_calls'] <= len(calls)
    assert queue_final['worker'] is None or queue_final['worker']['worker_stopped']
    assert sum(e['kind']=='backend_completed' for e in queue_final['events']) == len(calls)
    assert summary['summary_metric_schema'] == 2
    assert summary['correct_object_selected'] == any(f['authorized_first_proposal_correct'] for f in frames)
    assert summary['correctly_locked'] == any(f['correct_stable_lock'] for f in frames)
    assert summary['candidate_target_coverage'] == any(f['candidate_coverage'] for f in frames)
    with (dest/'frames.jsonl').open('x') as stream:
        for f in frames:
            stream.write(json.dumps(f, ensure_ascii=False)+'\n')
    files = [episode/'release.json', episode/'summary.json', episode/'perception.jsonl',
        episode/'execution.jsonl', episode/'evaluation_only/truth.json',
        episode/'semantic_queue_final.json', episode/'independent_audit_003/report.json']
    if backend_log.exists():
        files.append(backend_log)
    report = dict(mode=release['semantic_mode'], episode=release['episode']['episode'],
        observations=len(frames), semantic_proposal_correct_ever=any(f['semantic_proposal_correct'] for f in frames),
        authorized_first_proposal_correct_ever=any(f['authorized_first_proposal_correct'] for f in frames),
        correct_stable_lock_ever=any(f['correct_stable_lock'] for f in frames),
        wrong_stable_lock_ever=any(f['wrong_stable_lock'] for f in frames),
        target_route_generated=any(f['target_route_generated'] for f in frames),
        observed_semantic_pool_coverage_ever=any(f['observed_semantic_pool_coverage'] for f in frames),
        actual_GPU_model_calls=len(calls), valid_model_outputs=sum(c['schema_valid'] for c in calls),
        calls_completed_at_summary_snapshot=summary['actual_Qwen_calls'],
        calls_completed_during_shutdown=len(calls)-summary['actual_Qwen_calls'],
        final_queue_and_model_log_counters_match=True,
        actual_queue_serialization_alignment_verified=True, original_submitted_event_hashes_verified=True,
        observed_submission_serialization_order_changes=serialization_changed,
        queue_source_sha256=digest(R.parents[3]/'conceptgraph/completion/realtime_semantics.py'),
        supplemental_audit_reason='原006审计未重建队列的排序序列化；本008按真实队列重建并逐条比对模型输入，最后一次后台调用按关闭后日志计数，原日志/错误/模型输出均未修改',
        backend_application_frames=sum(f['actual_backend_result_used'] for f in frames),
        strict_safe_arrival=arrival['strict_safe_arrival'], actuation_limits=arrival['actuation_limits'],
        maximum_acceleration_mps2=arrival['maximum_acceleration_mps2'],
        minimum_acceleration_mps2=arrival['minimum_acceleration_mps2'], final_status=summary['status'],
        source_sha256={str(p): digest(p) for p in files}, evaluator_sha256=digest(__file__),
        identity_scope='两个受控官方车辆，以同类中心2.5m门限一对一匹配；未向智能体提供此匹配',
        scope='固定开发场景真实物理闭环；不是封存集、不是大规模统计或实时吞吐',
        goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    args = parser.parse_args()
    audit(R/'carla_official'/args.episode)
