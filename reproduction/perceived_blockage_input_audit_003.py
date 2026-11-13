"""核验唯一真实排序请求对应当时安全候选、无GT字段和精确模型提示原文。"""
from datetime import datetime, timezone
import hashlib
import json

from conceptgraph.completion.semantic_contract import build_completion_prompt
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def read(path):
    return json.loads(path.read_text())


def forbidden(value):
    if isinstance(value,dict):
        return [k for k in value if k in {'truth','goal_truth','instance_id','accepted_instance_ids','regions',
            'evaluation_only_matches','evaluation_only_identity','evaluation_only_dynamic_obstacle'}]+[
            key for v in value.values() for key in forbidden(v)]
    if isinstance(value,list):
        return [key for v in value for key in forbidden(v)]
    return []


def main():
    case=OUTPUT/'faults/perceived_blockage_cpu_002/delayed_actual_rank'
    dest=case/'input_audit_003'
    dest.mkdir(exist_ok=False)
    try:
        request=read(case/'submitted_request.json')
        payload=request['payload']
        events=read(case/'navigation/semantic_events.json')
        submitted=[e for e in events['events'] if e['kind']=='submitted']
        assert len(submitted)==1 and request['task']=='candidate_rerank'
        signature=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        assert signature==submitted[0]['payload_sha256']
        assert not forbidden(payload)
        fixture=read(case/'config_and_task.json')
        assert payload['query_text']==fixture['task']['query_text']
        started=read(case/'navigation/driver_clock.json')['started_wall_s']
        frames=[json.loads(s) for s in (case/'navigation/perception.jsonl').read_text().splitlines()]
        relevant=[f for f in frames if started+f['sensed_wall_elapsed_s'] <= submitted[0]['submitted_wall_s'] <= started+f['completed_wall_elapsed_s']]
        assert len(relevant)==1
        frame=relevant[0]
        # 原异步调度按键排序冻结JSON；反序列化后键顺序属于实际提示原文。
        frozen_payload=json.loads(json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False))
        candidates=frozen_payload['candidate_nodes']
        ids=[c['node_id'] for c in candidates]
        assert ids==frame['semantic_audit']['safety_verified_order']
        checks={c['node_id']:c for c in frame['semantic_audit']['candidate_checks']}
        nodes={n['node_id']:n for n in frame['nodes']}
        for candidate in candidates:
            i=candidate['node_id']
            assert candidate['path_validated'] and checks[i]['passed'] and checks[i]['independent_sweep_audit']['passed']
            assert candidate['center_xy']==nodes[i]['center'][:2] and candidate['category']==nodes[i]['category']
        expected=build_completion_prompt('candidate_rerank',dict(query_text=payload['query_text'],
            candidate_ids=ids,candidate_nodes=candidates,relation_metadata=payload['relation_metadata']),'completion_v3')
        actual=read(case/'actual_model_text.json')
        assert expected==actual['prompt']
        model_audits=[read(p)['result'] for p in (case/'navigation/qwen_audit').glob('*.json')]
        assert len(model_audits)==1 and model_audits[0]['raw_output']==actual['raw_output']
        assert model_audits[0]['output_schema_valid'] and not model_audits[0]['fallback_used']
        assert not model_audits[0]['cache']['enabled']
        physical=read(case/'evaluation_only_dynamic_obstacle.json')
        assert submitted[0]['submitted_wall_s']<physical['birth_wall_s']
        dump(dest/'report.json',dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),
            request_id=request['request_id'],immutable_payload_sha256=signature,current_observation_frame=frame['frame_idx'],
            current_safety_verified_candidates=ids,request_precedes_physical_barrier=True,
            exact_completion_v3_prompt_verified=True,no_GT_fields_in_payload=True,
            actual_raw_model_output_verified=True,actual_CPU_model_calls=1,cache_enabled=False,
            model_source_devices=read(case/'actual_model.json')['devices'],actual_new_model_calls_added_by_audit=0,
            request_sha256=digest(case/'submitted_request.json'),model_text_sha256=digest(case/'actual_model_text.json'),
            source_sha256=digest(__file__),goal_complete=False))
    except BaseException as error:
        dump(dest/'interrupted.json',dict(error=repr(error),goal_complete=False))
        raise


if __name__=='__main__':
    main()
