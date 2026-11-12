"""过去LiDAR目标接近策略：全查询同池、独立身份打分及log配对区间复算。"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path('/root/autodl-tmp/lizichong')
OUT = ROOT/'outputs/project_v2/thesis_completion_v1'
SOURCE = OUT/'ablations/causal_goals_full_001'
DEST = SOURCE/'audit_001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    DEST.mkdir(exist_ok=False)
    release, report = read(SOURCE/'release.json'), read(SOURCE/'report.json')
    assert sha(SOURCE/'queries.jsonl') == report['raw_sha256']
    sources = {Path(k): v for k,v in release['source_sha256'].items()}
    sources.update({Path(k): v for k,v in release['source_frames_sha256'].items()})
    retrieval = OUT/'main_experiments'/release['retrieval_run']
    sources[retrieval/'queries.jsonl'] = release['source_query_sha256']
    for k,v in release['map_sha256'].items():
        sources[ROOT/'dataset/nuscenes_map_expansion/expansion'/f'{k}.json'] = v
    camera = OUT/'mapping_evaluation/camera_time_gt_dev_full_001'
    camera_audit = read(camera/'audit_001/report.json')
    assert camera_audit['passed'] and camera_audit['all_original_frame_metrics_exact']
    assert camera_audit['source_report_sha256']==sha(camera/'report.json')
    frames = {}
    for path in release['source_frames_sha256']:
        for frame in map(json.loads, Path(path).open()):
            frames[frame['scene'],frame['frame_idx']] = frame
    queries = [q for q in map(json.loads,(retrieval/'queries.jsonl').open())
        if q['method']=='existing_object_joint_rule']
    frozen = {(q['scene'],q['frame_idx'],q['query_index']):q for q in queries}
    rows = list(map(json.loads,(SOURCE/'queries.jsonl').open()))
    methods = sorted(release['conditions'])
    lookup = {(r['scene'],r['frame_idx'],r['query_index'],r['method']):r for r in rows}
    assert len(lookup)==len(rows)==3345
    assert set(lookup)=={(*k,m) for k in frozen for m in methods}
    contexts, differences = {}, []
    full_differences = []
    for row in rows:
        key = row['scene'],row['frame_idx'],row['query_index']
        q,frame = frozen[key],frames[key[:2]]
        assert all(row[k]==q[k] for k in ['query_text','query_type','positive','acceptable_instances'])
        assert row['stable_correct_lock'] is None and row['actual_arrival'] is None
        assert not row['vehicle_executed'] and not row['gt_in_inference']
        selected = row['inference']['selected']
        assert row['selected'] == row['route_generated'] == (selected is not None)
        identities = {v['prediction_index']:frame['evaluation_only_official'][v['truth_index']]['instance_token']
            for v in frame['metrics']['one_to_one_depth_quality']['matches']}
        identity = identities.get(selected['node_id']) if selected else None
        assert identity==row['evaluation_identity']
        assert row['selected_correct']==(identity in row['acceptable_instances'] if row['positive'] else None)
        assert row['false_selection_on_no_match']==(bool(selected) if not row['positive'] else None)
        contexts[row['context']['path']] = row['context']
        if selected:
            node=frame['predictions']['one_to_one_depth_quality'][selected['node_id']]
            center,extent=np.asarray(node['center'])[:2],np.asarray(node['extent'])[:2]
            distance=float(np.linalg.norm(np.maximum(np.abs(np.asarray(selected['goal']['goal_xy'])-center)-extent/2,0)))
            assert abs(distance-selected['goal']['distance_to_object_m'])<1e-9
            assert row['endpoint_2m_from_predicted_bbox']==(distance>=2-1e-6)
            checked=selected['audit']
            assert row['static_body_pose_legal']==checked['body_pose_legal']
            assert row['full_road_legal']==(checked['full_lane_legal'] and checked['continuous_point_drivable'])
            feasible=[c for c in row['inference']['evaluated'] if c['reachable']]
            assert selected['node_id'] in [c['node_id'] for c in feasible]
            if row['method']=='first_reachable_same_safe_pool':
                first=min(feasible,key=lambda c:(c['semantic_rank'],c['node_id']))
                assert selected['node_id']==first['node_id']
        else:
            for name in ['static_body_pose_legal','full_road_legal','endpoint_2m_from_predicted_bbox']:
                assert row[name] is None
    for key in frozen:
        joint,first,removed=[lookup[(*key,m)] for m in
            ['joint_safe_approach','first_reachable_same_safe_pool','joint_without_safe_approach']]
        assert joint['context']==first['context']==removed['context']
        assert joint['inference']['evaluated']==first['inference']['evaluated']
        assert joint['inference']['raw_ranked_ids']==first['inference']['raw_ranked_ids']==removed['inference']['raw_ranked_ids']
        assert joint['acceptable_instances']==first['acceptable_instances']==removed['acceptable_instances']
        if joint['inference']['selected']!=first['inference']['selected']:
            differences.append(key)
        if joint['inference']!=first['inference']:
            full_differences.append(key)
    for path,entry in contexts.items():
        sources[Path(path)/'context.json']=entry['context_sha256']
        sources[Path(path)/'rasters.npz']=entry['rasters_sha256']
    # 场景log取官方只读元数据，而非评价生成器的内部数组。
    scenes={s['name']:s for s in read(ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval/scene.json')}
    scene_logs={s:scenes[s]['log_token'] for s in {r['scene'] for r in rows}}
    logs=sorted(set(scene_logs.values()))
    assert len(logs)==report['logs']==11 and len(scene_logs)==report['scenes']==38
    counts=np.zeros((len(logs),len(methods),4),dtype=int)
    for row in rows:
        positive=bool(row['positive'])
        counts[logs.index(scene_logs[row['scene']]),methods.index(row['method'])] += [
            bool(row['selected_correct']) if positive else 0,positive,
            bool(row['false_selection_on_no_match']) if not positive else 0,not positive]
    totals=counts.sum(axis=0)
    estimate=totals[:,[0,2]]/totals[:,[1,3]]
    random=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
    samples=counts[random].sum(axis=1)
    boot=np.divide(samples[:,:,[0,2]],samples[:,:,[1,3]],
        out=np.full((4000,len(methods),2),np.nan),where=samples[:,:,[1,3]]>0)
    for j,m in enumerate(methods):
        value=report['conditions'][m];current=[r for r in rows if r['method']==m]
        for k,name in enumerate(['positive_selection','no_match_false_selection']):
            metric=value[name];assert [metric['numerator'],metric['denominator']]==totals[j,k*2:k*2+2].tolist()
            assert abs(metric['estimate']-estimate[j,k])<1e-12
            valid=boot[:,j,k][np.isfinite(boot[:,j,k])]
            assert np.allclose(np.quantile(valid,[.025,.975]),metric['ci95'],rtol=0,atol=1e-12)
            assert len(valid)==metric['valid_resamples']
        assert value['selected']==sum(r['selected'] for r in current)
        for name in ['route_generated','full_road_legal','static_body_pose_legal','endpoint_2m_from_predicted_bbox']:
            actual=[r[name] for r in current if r[name] is not None]
            metric=value[name]
            assert [metric['count'],metric['denominator']]==[sum(actual),len(actual)]
            assert metric['rate']==(sum(actual)/len(actual) if actual else None)
        assert value['failures']==dict(Counter(r['inference']['failure'] for r in current if r['inference']['failure']))
    for c in report['paired_differences']:
        j,b,k=methods.index(c['method']),methods.index(c['baseline']),['positive_selection','no_match_false_selection'].index(c['metric'])
        assert abs(c['difference']-(estimate[j,k]-estimate[b,k]))<1e-12
        valid=boot[:,j,k]-boot[:,b,k];valid=valid[np.isfinite(valid)]
        assert np.allclose(c['ci95'],np.quantile(valid,[.025,.975]),rtol=0,atol=1e-12)
        assert len(valid)==c['valid_resamples']
    for path,digest in sources.items():
        assert sha(path)==digest,str(path)
    result=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),records=len(rows),paired_queries=len(frozen),
        actual_pool_equal_pairs=len(frozen),first_vs_joint_changed_selections=differences,
        first_vs_joint_changed_inference_records=full_differences,contexts_sha_checked=len(contexts),
        source_files_checked=len(sources),source_files={str(k):v for k,v in sources.items()},
        positive_and_negative_identity_scoring_independently_recomputed=True,
        saved_source_frame_identity_matches_previously_independently_audited=True,
        six_CI_and_four_paired_CI_recomputed=True,route_and_endpoint_counters_recomputed=True,
        static_body_geometry_recomputed=False,vehicle_executed=False,actual_arrival_credit=False,
        source_report_sha256=sha(SOURCE/'report.json'),source_raw_sha256=sha(SOURCE/'queries.jsonl'),
        audit_source_sha256=sha(__file__),new_test_opened=False,goal_complete=False)
    (DEST/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 过去LiDAR目标接近策略完整复核','',
        '1115查询×三条件=3345条，38开发场景11log；同池与输入SHA、独立身份打分、六项区间和四项配对区间通过。',
        f'首个可达与联合改变选择{len(differences)}条，改变完整推理记录{len(full_differences)}条。',
        '车身及车道使用保存的既有审计结果，本次不声称独立复算这批车身几何。没有车辆执行、正确锁定或实际到达信用。','',
        '|条件|选对/733|负误选/382|路线/1115|车道且道路合法/路线|静态矩形通过/路线|预测框外2m/路线|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        s=report['conditions'][m];n=s['selected']
        lines.append(f"|{m}|{s['positive_selection']['numerator']}|{s['no_match_false_selection']['numerator']}|{n}|{s['full_road_legal']['count']}/{n}|{s['static_body_pose_legal']['count']}/{n}|{s['endpoint_2m_from_predicted_bbox']['count']}/{n}|")
    lines+=['','联合与首达均34/733、负误选6/382，差0；去接近保护26/733，正差−1.091pp、95%区间[−2.116,0]pp，不能声称确定优势。去保护32路线终点均不在预测框外2m；此项是预测几何代理，不是独立安全终点真值。']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},ensure_ascii=False))


if __name__=='__main__':
    main()
