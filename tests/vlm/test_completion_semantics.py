from __future__ import annotations

import json

import numpy as np

from conceptgraph.completion.semantic_contract import build_prompt, validate
from conceptgraph.completion.semantic_dataset import make_queries, audit_rows, query_response
from conceptgraph.navigation.semantic import apply_qwen_query_parse, parse_navigation_query, rank_candidate_targets
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from tests.navigation.conftest import make_node


def test_new_ranking_explicit_no_match_and_strict_candidate_set():
    output=dict(schema_version='1.0.0',selected_node_id=None,ranking=[9,2],no_match=True,
        explanation='没有匹配的卡车',confidence=.9)
    assert validate('candidate_rerank',output,allowed_candidate_ids=[2,9]).valid
    assert not validate('candidate_rerank',output,allowed_candidate_ids=[2,9,3]).valid
    output['selected_node_id']=9
    assert not validate('candidate_rerank',output,allowed_candidate_ids=[2,9]).valid


def test_independent_expressions_have_distinct_families_and_texts():
    train=make_queries('train',1400,[dict(scene='a',log_token='train-log')])
    val=make_queries('validation',180,[dict(scene='b',log_token='val-log')])
    assert audit_rows(train+val)['counts']['train/query_parse']==1400
    assert not {r['query_text'] for r in train} & {r['query_text'] for r in val}
    assert any(r['response']['excluded_categories'] for r in train)
    assert any(r['response']['relation']=='left_of' for r in train)


def test_completion_delegate_uses_full_prompt_and_preserves_anchor_exclusions(tmp_path):
    response=query_response('car',anchor='bus',relation='left_of',excluded='truck',static=True)
    prompts=[]
    def backend(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(response,ensure_ascii=False)
    config=QwenDelegateConfig(enabled=True,use_qwen_query_parse=True,prompt_profile='completion_v1',
        cache_dir=tmp_path/'cache')
    delegate=QwenSemanticDelegate(config,backend=backend)
    result=delegate.parse_query('去公交车左侧停着的小汽车旁边，不要卡车')
    assert result['used_qwen'] and result['relation']=='left_of'
    assert result['anchor_categories']==['bus'] and result['excluded_categories']==['truck']
    assert 'excluded_categories' in prompts[0]
    assert delegate.parse_query('去公交车左侧停着的小汽车旁边，不要卡车')['cache']['hit']
    assert delegate.backend_call_count==1
    delegate.close()


def test_negation_and_static_requirement_are_deterministic_filters():
    rule=parse_navigation_query('靠近车辆',8)
    response=query_response('car',excluded='truck',static=True)
    parsed=apply_qwen_query_parse(rule,response,8)
    nodes={0:make_node(0,tag='car',motion_state='dynamic'),
           1:make_node(1,tag='truck',motion_state='static'),
           2:make_node(2,tag='car',motion_state='static')}
    ranked=rank_candidate_targets(parsed,nodes,[],np.array([0,0]),.5,.3,.2)
    assert [r.node_id for r in ranked]==[2]


def test_unknown_color_does_not_prove_negated_color():
    parsed=parse_navigation_query('靠近小汽车',8)
    parsed.excluded_colors=['red']
    nodes={0:make_node(0,metadata={'attributes':{'colors':['unknown']}}),
           1:make_node(1,metadata={'attributes':{'colors':['white']}}),
           2:make_node(2,metadata={'attributes':{'colors':['red']}})}
    ranked=rank_candidate_targets(parsed,nodes,[],np.array([0,0]),.5,.3,.2)
    assert [r.node_id for r in ranked]==[1]


def test_visual_prompt_has_no_caption_color_copy_contract():
    prompt=build_prompt('object_attributes',dict(metadata={'caption':'red car','detection_label':'car'}))
    assert 'red car' not in prompt and 'colors_required_exact' not in prompt
    assert '裁剪图' in prompt and 'unknown' in prompt
