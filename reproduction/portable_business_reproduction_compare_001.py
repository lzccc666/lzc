"""比较便携入口与原独立入口的完整逻辑业务；不把不同轨迹强制变成一致。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).open()]


def locate(base):
    manifest = read(base/'manifest.json')
    root = Path(manifest['isolated_root'])
    case = root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/live_navigation_sessions_001'/manifest['session']
    return manifest, root, case


# 只剔除无法决定逻辑时钟业务的实测计时字段。文件位置以相同角色前缀替换；
# 坐标、候选、原文、地图、身份、路径、事件、控制、逻辑时间和图像SHA均保留。
TIMING_FIELDS = {'completed_wall_elapsed_s', 'duration_s', 'frontend_mapping_duration_s'}


def canonical(value, root, case):
    if isinstance(value, dict):
        return {key:canonical(item, root, case) for key, item in value.items() if key not in TIMING_FIELDS}
    if isinstance(value, list):
        return [canonical(item, root, case) for item in value]
    if isinstance(value, str):
        if value.startswith(str(case)):
            return '<CASE>'+value[len(str(case)):]
        if value.startswith(str(root)):
            return '<SOURCE>'+value[len(str(root)):]
    return value


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def main():
    first = OUTPUT/'reproduction/standalone_cpu_delivery_002'
    second = OUTPUT/'reproduction/portable_executed_001'
    a, root_a, case_a = locate(first)
    b, root_b, case_b = locate(second)
    assert a['source_sha256'] == b['source_sha256'] and a['source_modes'] == b['source_modes']
    assert a['task_release_sha256'] == b['task_release_sha256']
    assert a['extension_sha256'] == b['extension_sha256'] and a['runner_sha256'] == b['runner_sha256']
    assert a['task_id'] == b['task_id'] == 'config00-positive0' and a['seed'] == b['seed'] == 17
    assert read(first/'independent_audit_001/report.json')['passed']
    assert read(second/'independent_audit_001/report.json')['passed']
    result_a, result_b = read(case_a/'result.json'), read(case_b/'result.json')
    inputs, conditions, differences = {}, {}, []
    for name in ['initial_state.json', 'perception.jsonl', 'execution.jsonl', 'thread_observations.jsonl']:
        va = records(case_a/name) if name.endswith('.jsonl') else read(case_a/name)
        vb = records(case_b/name) if name.endswith('.jsonl') else read(case_b/name)
        va, vb = canonical(va, root_a, case_a), canonical(vb, root_b, case_b)
        same = va == vb
        conditions[name] = dict(equal=same, original_sha256=signature(va), portable_sha256=signature(vb),
            original_records=len(va) if isinstance(va, list) else 1,
            portable_records=len(vb) if isinstance(vb, list) else 1)
        if isinstance(va, list) and isinstance(vb, list):
            for i, (left, right) in enumerate(zip(va, vb)):
                if left != right:
                    differences.append(dict(file=name, index=i, original=left, portable=right))
                    if len(differences) >= 10:
                        break
        inputs[str(case_a/name)] = digest(case_a/name)
        inputs[str(case_b/name)] = digest(case_b/name)
    four = ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival']
    conditions['four_outcomes'] = dict(equal=all(result_a[key] == result_b[key] for key in four),
        original={key:result_a[key] for key in four}, portable={key:result_b[key] for key in four})
    conditions['terminal_state'] = dict(equal=result_a['final_state'] == result_b['final_state'])
    destination = OUTPUT/'reproduction/portable_business_comparison_001'
    destination.mkdir(exist_ok=False)
    dump(destination/'differences.json', differences)
    dump(destination/'files.json', inputs)
    dump(destination/'report.json', dict(passed=all(value['equal'] for value in conditions.values()),
        updated_utc=datetime.now(timezone.utc).isoformat(), comparisons=conditions,
        ignored_timing_fields=sorted(TIMING_FIELDS), ignored_business_fields=[],
        path_prefix_normalization=['<CASE>', '<SOURCE>'], numerical_tolerance_used=False,
        original_actual_frontend_calls=result_a['actual_frontend_calls'],
        portable_actual_frontend_calls=result_b['actual_frontend_calls'],
        Qwen_backend_calls_both_runs=0, full_three_task_Qwen_cache_gate_closed=False,
        actual_sensor_and_frontend_executed_in_both=True,
        new_independent_scenes=0, frozen_main_table_credit=False,
        source_sha256=digest(__file__), source_files_sha256=digest(destination/'files.json'),
        goal_complete=False))
    print(json.dumps(read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
