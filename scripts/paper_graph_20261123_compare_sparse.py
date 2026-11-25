"""冻结S与R1全范围配对比较，完整保存收益、损失及11log区间。"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence


P='paper_graph_20261123_'
BASE=PROJECT/'outputs/results/nuscenes/validation/iterations'
R1=BASE/'recall_refinement_R1_full_20261120_v2'
S=BASE/'paper_graph_20261123_S_full_v1'
DEST=PROJECT/'outputs/reports/paper_campaign_20261123/graph'
METHOD='R1_S_image_continuity'
KEYS=('TP','GT','pred','duplicates','error_sum','IDTP','IDGT','IDpred','common_n','common_error_sum')


def rows(path):
    return list(map(json.loads,gzip.open(path,'rt')))


def values(array):
    x={k:array[...,i] for i,k in enumerate(KEYS)}
    return dict(recall=x['TP']/x['GT'],precision=x['TP']/x['pred'],F1=2*x['TP']/(x['GT']+x['pred']),
        IDF1=2*x['IDTP']/(x['IDGT']+x['IDpred']),center_error_m=x['error_sum']/x['TP'],
        duplicate_nodes=x['duplicates'],common_matched_center_error_m=x['common_error_sum']/x['common_n'])


def main():
    release=json.loads((S/(P+'release.json')).read_text())
    actual=json.loads((S/(P+'report.json')).read_text())
    if not actual['full_validation_scope'] or actual['frames']!=1508:
        raise AssertionError('候选范围不全')
    grouped=defaultdict(Counter)
    subgroups=defaultdict(Counter)
    scene_rows=[]
    gains,losses=[],[]
    paired=[]
    inputs={}
    for scene in release['scenes']:
        name,log=scene['scene'],scene['log_token']
        old=rows(R1/name/'frames.jsonl.gz')
        new=rows(S/name/(P+'frames.jsonl.gz'))
        source=OUTPUT/'mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
        raw=list(map(json.loads,source.open()))
        if len(old)!=len(new) or len(old)!=len(raw):
            raise AssertionError('范围不同')
        inputs.update({str(p):digest(p) for p in [R1/name/'frames.jsonl.gz',S/name/(P+'frames.jsonl.gz'),source]})
        scene_counts=defaultdict(Counter)
        for a,b,frame in zip(old,new,raw):
            if a['evaluation_only_official']!=b['evaluation_only_official'] or a['input']!=b['input'] or a['frame_idx']!=b['frame_idx']:
                raise AssertionError('GT或实际输入变更')
            matches={m:{r['truth_index']:r for r in f['metrics'][m]['matches']} for m,f in [('R1',a),(METHOD,b)]}
            common=set(matches['R1'])&set(matches[METHOD])
            for m,f in [('R1',a),(METHOD,b)]:
                metric=f['metrics'][m]
                c=scene_counts[m]
                c.update(TP=metric['true_positive'],GT=metric['truth'],pred=metric['predictions'],duplicates=metric['duplicate_nodes'],
                    error_sum=sum(x['distance_m'] for x in metric['matches']),common_n=len(common),
                    common_error_sum=sum(matches[m][i]['distance_m'] for i in common))
                for pred in f['predictions'][m]:
                    subgroups[m,'category',pred['category']]['pred']+=1
                    distance=float(np.linalg.norm(np.asarray(pred['center'][:2])-np.asarray(frame['camera_pose'])[:2,3]))
                    bucket='0_20m' if distance<20 else '20_40m' if distance<40 else '40m_plus'
                    subgroups[m,'predicted_range',bucket]['pred']+=1
            for i,obj in enumerate(a['evaluation_only_official']):
                distance=float(np.linalg.norm(np.asarray(obj['center'][:2])-np.asarray(frame['camera_pose'])[:2,3]))
                bucket='0_20m' if distance<20 else '20_40m' if distance<40 else '40m_plus'
                for m in ['R1',METHOD]:
                    for kind,group in [('category',obj['category']),('GT_range',bucket)]:
                        subgroups[m,kind,group]['GT']+=1
                        subgroups[m,kind,group]['TP']+=i in matches[m]
                had,has=i in matches['R1'],i in matches[METHOD]
                if had!=has:
                    item=dict(scene=name,log_token=log,frame_idx=a['frame_idx'],truth_index=i,
                        instance_token=obj['instance_token'],category=obj['category'],GT_range_m=distance,
                        baseline_match=matches['R1'].get(i),candidate_match=matches[METHOD].get(i),
                        baseline_prediction=a['predictions']['R1'][matches['R1'][i]['prediction_index']] if had else None,
                        candidate_prediction=b['predictions'][METHOD][matches[METHOD][i]['prediction_index']] if has else None,
                        source_R1=str(R1/name/'frames.jsonl.gz'),source_S=str(S/name/(P+'frames.jsonl.gz')))
                    (gains if has else losses).append(item)
            paired.append(dict(scene=name,frame_idx=a['frame_idx'],TP_R1=a['metrics']['R1']['true_positive'],
                TP_S=b['metrics'][METHOD]['true_positive'],pred_R1=a['metrics']['R1']['predictions'],pred_S=b['metrics'][METHOD]['predictions'],
                gained=len(set(matches[METHOD])-set(matches['R1'])),lost=len(set(matches['R1'])-set(matches[METHOD]))))
        for m,frames in [('R1',old),(METHOD,new)]:
            identity=identity_counts(prepare_sequence(frames,m),'all_gated')['counts']
            c=scene_counts[m]
            c.update(IDTP=identity['IDTP'],IDGT=identity['GT_observations'],IDpred=identity['prediction_observations'])
            grouped[log,m].update(c)
            scene_rows.append(dict(scene=name,log_token=log,method=m,**c,**{k:float(v) for k,v in values(np.asarray([c[k] for k in KEYS],dtype=float)).items()}))
    logs=sorted({k[0] for k in grouped})
    draws=np.random.default_rng(20261123).integers(len(logs),size=(10000,len(logs)))
    estimates,boot={},{}
    counts={}
    for m in ['R1',METHOD]:
        array=np.array([[grouped[log,m][k] for k in KEYS] for log in logs],dtype=float)
        counts[m]={k:float(v) for k,v in zip(KEYS,array.sum(axis=0))}
        estimates[m]={k:float(v) for k,v in values(array.sum(axis=0)).items()}
        boot[m]=values(array[draws].sum(axis=1))
    differences={}
    for metric in estimates['R1']:
        sample=boot[METHOD][metric]-boot['R1'][metric]
        differences[metric]=dict(candidate_minus_R1=estimates[METHOD][metric]-estimates['R1'][metric],
            paired_log_CI95=np.quantile(sample,[.025,.975]).tolist(),primary=metric=='IDF1',
            multiple_comparison_family_size=1 if metric=='IDF1' else None,
            secondary_intervals_descriptive=metric!='IDF1')
    if counts['R1']['GT']!=5723 or counts[METHOD]['GT']!=5723 or counts[METHOD]['TP']-counts['R1']['TP']!=len(gains)-len(losses):
        raise AssertionError('全分母/收益损失不闭合')
    if abs(estimates[METHOD]['IDF1']-actual['result']['metrics']['IDF1'])>1e-12:
        raise AssertionError('身份复算不一致')
    files={}
    for name,data in [('scene_comparison',scene_rows),('frame_comparison',paired),
        ('group_comparison',[dict(method=m,group_kind=kind,group=value,**c) for (m,kind,value),c in subgroups.items()]),
        ('main_comparison',[dict(method=m,**counts[m],**estimates[m]) for m in ['R1',METHOD]])]:
        p=DEST/(P+'S_'+name+'.csv')
        keys=list(dict.fromkeys(k for row in data for k in row))
        with p.open('x',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        files[str(p)]=digest(p)
    dump(DEST/(P+'S_gain_loss_cases.json'),dict(gained=len(gains),lost=len(losses),gains=gains,losses=losses))
    dump(DEST/(P+'S_comparison.json'),dict(complete=True,scenes=38,logs=11,frames=1508,GT=5723,
        method_counts=counts,metrics=estimates,differences=differences,gained=len(gains),lost=len(losses),
        common_matched_GT_frames=counts['R1']['common_n'],bootstrap_seed=20261123,repetitions=10000,
        primary_hypothesis='IDF1 improvement',primary_comparisons=1,
        model_selection_limit='same repeatedly used validation/development scenes; pilot overlaps;not independent test',
        output_sha256=files,input_sha256=inputs,source_sha256=digest(__file__)))
    print(json.dumps(dict(metrics=estimates,differences=differences,gained=len(gains),lost=len(losses)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
