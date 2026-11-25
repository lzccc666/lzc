"""独立结构化安全候选监督；不将合成符号记录冒充真实传感器或人工数据。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from conceptgraph.completion.kinematic_benchmark import rectangle
from conceptgraph.completion.semantic_dataset import audit_rows, CATEGORIES, RELATIONS
from conceptgraph.completion.stage9 import OUTPUT, dump, digest
from conceptgraph.navigation.vehicle_geometry import VehicleState, VehicleSpec, step_vehicle, audit_executed_states

COLORS = {'red':'红色','blue':'蓝色','white':'白色','black':'黑色','yellow':'黄色'}


def safe_bank(role,index,rng):
    """非正式仿真配置：独立生成直道、符号对象，并实际积分和审计参考执行。"""
    yaw = float(rng.uniform(-math.pi,math.pi))
    origin = rng.uniform(-200,200,2)
    tangent = np.array([math.cos(yaw),math.sin(yaw)])
    left = np.array([-tangent[1],tangent[0]])
    line = LineString([origin-6*tangent,origin+45*tangent])
    road = line.buffer(1.8,cap_style=2)
    nodes = []
    ids = rng.choice(np.arange(10,900),size=4,replace=False)
    # Multiple objects share a category; matching cannot reduce to a unique class lookup.
    categories = list(rng.choice(list(CATEGORIES),size=3))
    categories.insert(1,categories[0])
    colors = list(rng.choice(list(COLORS),size=4,replace=False))
    for j,(node_id,category,color) in enumerate(zip(ids,categories,colors)):
        progress = 12+j*6+float(rng.uniform(-.5,.5))
        center = origin+progress*tangent+(3.2 if j%2==0 else -3.2)*left
        nodes.append(dict(node_id=int(node_id),category=str(category),colors=[str(color)],
            center_xy=center.tolist(),motion_state='static',distance_m=float(np.linalg.norm(center-origin)),
            path_validated=True,goal_xy=(origin+(progress-2)*tangent).tolist()))
    obstacles = [rectangle(np.asarray(n['center_xy']),yaw,4.6,1.85) for n in nodes]
    audits = []
    for node in nodes:
        state = VehicleState(*origin,yaw)
        states = [state]
        goal = np.asarray(node['goal_xy'])
        for _ in range(500):
            distance = float(np.dot(goal-[state.x,state.y],tangent))
            desired = min(1.,math.sqrt(max(0,2*.8*(distance-.2)))) if distance>.25 else 0.
            state = step_vehicle(state,1.5*(desired-state.speed_mps),0)
            states.append(state)
            if desired==0 and state.speed_mps<.05:
                break
        audit = audit_executed_states(states,road,obstacles)
        if not audit['passed'] or np.linalg.norm([state.x-goal[0],state.y-goal[1]])>1:
            raise ValueError('independent reference execution did not validate candidate safety')
        node['safety_evidence'] = 'rectangular swept-body audit of actual forward bicycle reference rollout'
        audits.append(dict(node_id=node['node_id'],states=[asdict(s) for s in states],audit=audit))
    relations = []
    for a in nodes:
        for b in nodes:
            if a is b:
                continue
            delta = np.array(a['center_xy'])-b['center_xy']
            for value,positive,negative in [(np.dot(delta,tangent),'in_front_of','behind'),
                                            (np.dot(delta,left),'left_of','right_of')]:
                if abs(value)>1:
                    relations.append(dict(source=a['node_id'],target=b['node_id'],
                                          relation=positive if value>0 else negative,
                                          coordinate_frame='ego_heading',confidence=1.0))
    return dict(bank_id=f'symbolic-{role}-{index:02d}',nodes=nodes,relations=relations,
                origin_xy=origin.tolist(),heading_xy=tangent.tolist(),road_polygon=list(road.exterior.coords),
                independent_reference_rollouts=audits,
                source='独立解析几何生成的符号监督；没有真实RGB、人工颜色或现实数据独立场景信用')


def make_row(bank,role,index,rng):
    nodes = bank['nodes']
    target = nodes[int(rng.integers(len(nodes)))]
    choice = index%4
    category = target['category']
    color = target['colors'][0]
    acceptable = []
    if choice==0:
        query = f"去{COLORS[color]}{CATEGORIES[category]}附近"
        acceptable = [n['node_id'] for n in nodes if n['category']==category and color in n['colors']]
        query_kind = 'category_color_positive'
    elif choice==1:
        query = f"去最近的{CATEGORIES[category]}附近"
        options = [n for n in nodes if n['category']==category]
        nearest = min(n['distance_m'] for n in options)
        acceptable = [n['node_id'] for n in options if n['distance_m']<=nearest+1e-6]
        query_kind = 'same_category_nearest'
    elif choice==2:
        # A color absent within this category, while matching category still exists.
        forbidden = {c for n in nodes if n['category']==category for c in n['colors']}
        absent = next(c for c in COLORS if c not in forbidden)
        query = f"去{COLORS[absent]}{CATEGORIES[category]}附近"
        query_kind = 'same_category_no_color_match'
    else:
        edge = bank['relations'][int(rng.integers(len(bank['relations'])))]
        by_id = {n['node_id']:n for n in nodes}
        a,b = by_id[edge['source']],by_id[edge['target']]
        relation = edge['relation']
        query = f"去{COLORS[b['colors'][0]]}{CATEGORIES[b['category']]}{RELATIONS[relation]}的{CATEGORIES[a['category']]}附近"
        anchors = {n['node_id'] for n in nodes if n['category']==b['category'] and n['colors']==b['colors']}
        acceptable = sorted({e['source'] for e in bank['relations'] if e['relation']==relation and
            e['target'] in anchors and by_id[e['source']]['category']==a['category']})
        query_kind = 'relation_anchor_positive'
    if role=='validation':
        query = query.replace('去','请将',1).replace('附近','作为靠近目标')
    shuffled = list(rng.permutation(len(nodes)))
    candidates = [{k:v for k,v in nodes[i].items() if k not in {'goal_xy'}} for i in shuffled]
    ranked = sorted(nodes,key=lambda n:(n['node_id'] not in acceptable,n['distance_m'],n['node_id']))
    selected = ranked[0]['node_id'] if acceptable else None
    response = dict(schema_version='1.0.0',selected_node_id=selected,
        ranking=[n['node_id'] for n in ranked],no_match=not bool(acceptable),
        explanation='controlled structured semantic constraints; safety independently fixed',confidence=1.)
    return dict(task='candidate_rerank',sample_id=f"rank-{role}-{index:04d}",split=role,
        scene_id=bank['bank_id'],log_token=bank['bank_id'],query_text=query,query_kind=query_kind,
        candidate_ids=[n['node_id'] for n in candidates],candidate_nodes=candidates,
        relation_metadata=bank['relations'],acceptable_node_ids=acceptable or [None],response=response,
        source_bank_id=bank['bank_id'],source_kind='controlled_symbolic_supervision',
        independent_real_scene_credit=0,manual_reviewed=False,
        label_provenance='精确符号场景的类别/颜色/关系与实际车辆参考轨迹审计；不读取旧排序或联合效用')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    dest = OUTPUT/'semantic_dataset'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    source = OUTPUT/'semantic_dataset/structured_supervision_001'
    rows = []
    banks = []
    for role,count,bank_count,seed in [('train',720,14,179),('validation',180,6,293)]:
        rng = np.random.default_rng(seed)
        subset = [safe_bank(role,i,rng) for i in range(bank_count)]
        banks.extend(subset)
        for i in range(count//2):
            a = make_row(subset[i%bank_count],role,i,rng)
            a['sample_id'] = f'rank-{role}-{2*i:04d}'
            # Two permutations of exactly the same semantic evidence and label.
            b = json.loads(json.dumps(a))
            b['sample_id'] = f'rank-{role}-{2*i+1:04d}'
            b['permutation_pair'] = a['sample_id']
            permutation = rng.permutation(len(b['candidate_nodes']))
            b['candidate_nodes'] = [b['candidate_nodes'][int(j)] for j in permutation]
            b['candidate_ids'] = [n['node_id'] for n in b['candidate_nodes']]
            rows.extend([a,b])
        rows.extend(json.loads(s) for s in (source/f'{role}.jsonl').read_text().splitlines())
    audit = audit_rows(rows)
    dump(dest/'symbolic_banks.json',dict(banks=banks,vehicle=asdict(VehicleSpec()),
         independent_real_scene_credit=0,formal_simulation_tasks_used=False,nuScenes_test_used=False))
    for role in ['train','validation']:
        with (dest/f'{role}.jsonl').open('x') as f:
            for r in rows:
                if r['split']==role:
                    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    rank = [r for r in rows if r['task']=='candidate_rerank']
    distributions = {}
    for role in ['train','validation']:
        subset = [r for r in rank if r['split']==role]
        positions = Counter('no_match' if r['response']['selected_node_id'] is None else
            str(r['candidate_ids'].index(r['response']['selected_node_id'])) for r in subset)
        distributions[role] = dict(queries=dict(Counter(r['query_kind'] for r in subset)),selected_positions=dict(positions))
        distributions[role]['unique_semantic_evidence_before_permutation'] = len({json.dumps(dict(
            query_text=r['query_text'],candidate_nodes=sorted(r['candidate_nodes'],key=lambda n:n['node_id']),
            relation_metadata=r['relation_metadata']),sort_keys=True) for r in subset})
    dump(dest/'manifest.json',dict(audit=audit,files={p.name:digest(p) for p in dest.glob('*.jsonl')},
        symbolic_banks_sha256=digest(dest/'symbolic_banks.json'),source_sha256=digest(Path(__file__)),
        previous_dataset_manifest_sha256=digest(source/'manifest.json'),ranking_distribution=distributions,
        train_row_count=sum(r['split']=='train' for r in rows),
        validation_row_count=sum(r['split']=='validation' for r in rows),
        unique_supervision_payloads=len({hashlib.sha256(json.dumps({k:r[k] for k in
            ['query_text','candidate_nodes','relation_metadata']},sort_keys=True).encode()).hexdigest() for r in rank}),
        manual_review_count=0,independent_new_test_used=False,
        limitations=['排序为受控符号监督，无真实感知排序标注或真实道路增益信用',
            '候选置换对不增加场景或独立样本计数',
            '图像属性仍沿用官方类别，真实颜色、细属性人工记录仍缺失',
            '结构化数据验证分数不等于实际导航改善']))
    print(json.dumps(dict(audit=audit,ranking_distribution=distributions),ensure_ascii=False))


if __name__=='__main__':
    main()
