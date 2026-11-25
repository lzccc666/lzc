"""对全部真实候选快照作只读CPU重放；不把它当成新的闭环或模型性能实验。"""
from collections import Counter
import gzip
import json
import time
from pathlib import Path

from carla_reference_readiness_047 import reference_readiness
from diagnose_carla_ten_bottlenecks_043 import sha

R = Path(__file__).resolve().parents[1]


def main():
    job = json.loads((R / 'json/expansion_jobs_011_primary/job_007.json').read_text())
    episode = Path(job['episode_directory'])
    pp = episode / 'perception.jsonl.gz'
    cp = episode / 'semantic_backend/calls.jsonl'
    gate = R / 'source/carla_reference_readiness_047.py'
    protocol = dict(frozen_before_replay=True, inputs_sha256={str(p): sha(p) for p in [pp, cp, gate]},
        samples='all450_saved_frontend_snapshots_and_all65_previous_GPU_call_payloads',
        unit='one previously recorded episode; frames are not independent scenes',
        actual_new_GPU_budget=0, actual_new_CARLA_budget=0, output_budget_bytes=32 * 1024**2,
        applies_identically_to=['rule', 'base', 'adapter'],
        claim='only observation-readiness decisions; no navigation or Qwen improvement credit',
        source_sha256=sha(__file__), goal_complete=False)
    with (R / 'json/reference_readiness_protocol_047.json').open('x') as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)
    out = R / 'carla_official/reference_readiness_replay_047'
    out.mkdir(exist_ok=False)
    counts = Counter()
    durations = []
    with gzip.open(pp, 'rt') as f, (out / 'frames.jsonl').open('x') as writer:
        for line in f:
            p = json.loads(line)
            row = p['semantic_audit']['input_row']
            begun = time.perf_counter()
            decision = reference_readiness(row['query_text'], row)
            durations.append(time.perf_counter() - begun)
            counts[decision['reason']] += 1
            writer.write(json.dumps(dict(frame_idx=p['frame_idx'], decision=decision), ensure_ascii=False) + '\n')
    old_calls = [json.loads(line) for line in cp.open()]
    insufficient_calls = sum(not reference_readiness(c['payload']['query_text'], c['payload'])['ready'] for c in old_calls)
    assert len(durations) == 450 and len(old_calls) == 65
    assert insufficient_calls == 65
    report = dict(recorded_frontend_snapshots=len(durations), readiness_counts=dict(counts),
        previous_actual_GPU_calls=len(old_calls), previous_call_payloads_missing_required_context=insufficient_calls,
        actual_new_GPU_calls=0, actual_new_CARLA_calls=0,
        mean_CPU_gate_s=sum(durations) / len(durations),
        frames_not_independent_scenes=True, old_results_unchanged=True,
        closed_loop_validation_pending=True, no_accuracy_or_arrival_improvement_claim=True,
        protocol_sha256=sha(R / 'json/reference_readiness_protocol_047.json'),
        inputs_sha256=protocol['inputs_sha256'], source_sha256=sha(__file__), goal_complete=False)
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    assert sum(p.stat().st_size for p in out.iterdir()) < protocol['output_budget_bytes']
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
