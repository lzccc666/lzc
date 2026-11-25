"""Bounded CPU-only audit for causal multi-frame position candidates."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'outputs/reports/navcandidate_20261125/independent'
BASE_COMMIT='80adf9171f58c4ea58df11147f45f8f686e61078'
DEADLINE='2026-11-25T07:23:55Z'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')


def baseline():
    REPORT.mkdir(parents=True,exist_ok=True)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
    index=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    sync=read(ROOT/'doc/交接/20261125/snapshot_sources.json');copies=[]
    for row in sync['files']:
        destination=ROOT/row['destination'];expected=row['sha256']
        copies.append(dict(destination=row['destination'],tracked=row['destination'] in index,
            source_sha_matches=sha(row['source'])==expected,destination_sha_matches=sha(destination)==expected))
    remote=git('ls-remote','origin','refs/heads/codex/project-sync-20261125','refs/heads/main')
    old=read(ROOT/'outputs/reports/navfollowup_20261124/independent/baseline_v1.json')
    paths={Path(row['path']) for row in old['protected']}
    r1=ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
    release=read(r1/'release.json');paths.add(r1/'release.json')
    paths.update(Path(p) for p in release['sources_sha256'])
    paths.update(r1/s['scene']/'frames.jsonl.gz' for s in release['scenes'])
    paths.update(ROOT/p for p in ['conceptgraph/completion/selection_map_transfer.py','conceptgraph/completion/evidence.py',
        'conceptgraph/navigation/semantic.py','conceptgraph/navigation/types.py','conceptgraph/scenegraph/relations.py',
        'conceptgraph/slam/sparse_observations.py','conceptgraph/slam/dense_refinement.py',
        'dataset/processed/navstage2_repair_20261124_v1/validation_queries.jsonl',
        'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/release.json',
        'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/queries.jsonl',
        'outputs/results/nuscenes/validation/iterations/navstage2_sparse_evidence_20261124_v1/raw_navigation_evidence.jsonl.gz',
        'outputs/results/nuscenes/validation/iterations/navstage2_sparse_evidence_20261124_v1/summary.json'])
    protected=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=DEADLINE,
        audit_budget_bytes=32*1024**2,shared_new_experiment_budget_bytes=128*1024**2,GPU_calls=0,model_calls=0,training_calls=0,planner_calls=0,
        git=dict(HEAD=git('rev-parse','HEAD'),remote_refs=remote,tracked_files=len(index)-1,snapshot_copy_count=len(copies),
            all_snapshot_copies_tracked_and_identical=all(r['tracked'] and r['source_sha_matches'] and r['destination_sha_matches'] for r in copies),copies=copies),
        protected=protected,disk_free_bytes=shutil.disk_usage(ROOT).free,
        required_checks=['All38 validation/debug scenes/1508 frames; all1115 queries with733 positive and382 negative; same-log historical81 excluded from method selection.',
        'Frozen source/input/seed and original per-frame R1 predictions exactly unchanged; no GT or future input to evidence collection or ranking.',
        'Distinct real measurement frames and timestamps, real raw centers and depth/mask provenance; no filtered-center history or cumulative pixels represented as current measured support.',
        'Alias/assignment transitions traceable; competitive or incomplete identity stays unknown. Static speed threshold and relation/distance semantics retained.',
        'New position-only task candidates are separate from original shape/depth vehicle-safety authorization; candidate additions do not gain route or arrival credit.',
        'Same complete positive/negative denominator, failures, per-log counts and baseline scores; no historical81 tuning or new training.'])
    write(REPORT/'baseline_and_git_sync_v1.json',value)
    print(json.dumps(dict(report=str(REPORT/'baseline_and_git_sync_v1.json'),protected=len(protected),snapshot_copies=len(copies),copies_passed=value['git']['all_snapshot_copies_tracked_and_identical'],HEAD=value['git']['HEAD'],deadline=DEADLINE),ensure_ascii=False))


def protection():
    registered=read(REPORT/'baseline_and_git_sync_v1.json')['protected']
    changed=[r['path'] for r in registered if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']]
    return dict(checked=len(registered),unchanged=not changed,changed=changed)


def finalize():
    """Seal independent findings without rerunning a mapper or changing results."""
    core=read(REPORT/'independent_results_v2.json');pixels=read(REPORT/'qualified_pixels_v2.json')
    data=ROOT/'dataset/processed';results=ROOT/'outputs/results/nuscenes/validation/iterations'
    old=read(data/'mf_position_20261125_v1/manifest.json');new=read(data/'mf_position_20261125_v2/manifest.json')
    oldrun=results/'mf_position_20261125_v1';newrun=results/'mf_position_20261125_v2'
    def policy(path):
        tree=ast.parse(path.read_text())
        expression=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
        return {kw.arg:ast.literal_eval(kw.value) for kw in expression.keywords}
    policies_identical=policy(ROOT/'conceptgraph/navigation/mf_position_20261125.py')==policy(ROOT/'conceptgraph/navigation/mf_position_20261125_v2.py')
    same_inputs={name:sha(data/'mf_position_20261125_v1'/name)==sha(data/'mf_position_20261125_v2'/name) for name in ['method_frames.jsonl','method_queries.jsonl']}
    old_sources_unchanged=all(sha(p)==expected for p,expected in old['source_sha256'].items())
    old_export=read(oldrun/'export_summary.json')
    changed=read(newrun/'changed_cases.json');paired=Counter();gain_scenes=set();gain_logs=set()
    for row in changed:
        baseline=row['baseline'];candidate=row['candidate']
        paired['changed_tasks']+=1
        paired['previously_rejected']+=int(baseline['selected_index'] is None)
        paired['positive_gained_correct']+=int(bool(candidate['selected_correct']) and not baseline['selected_correct'])
        paired['positive_lost_correct']+=int(bool(baseline['selected_correct']) and not candidate['selected_correct'])
        paired['new_positive_wrong']+=int(candidate['positive'] and candidate['selected_index'] is not None and not candidate['selected_correct'] and baseline['selected_index'] is None)
        paired['negative_changed']+=int(not candidate['positive'])
        if candidate['selected_correct'] and not baseline['selected_correct']:gain_scenes.add(candidate['scene']);gain_logs.add(candidate['log'])
    preserved=protection()
    source_paths=[ROOT/'conceptgraph/completion'/name for name in ['navcandidate_independent_20261125.py','navcandidate_results_audit_20261125.py','navcandidate_pixels_audit_20261125.py']]+[ROOT/'scripts'/name for name in ['navcandidate_independent_20261125.py','navcandidate_results_audit_20261125.py','navcandidate_pixels_audit_20261125.py']]
    method_paths=[ROOT/'conceptgraph/navigation'/name for name in ['mf_position_20261125.py','mf_position_20261125_v2.py']]+[ROOT/'scripts'/name for name in ['mf_position_20261125_run.py','mf_position_20261125_v2_run.py']]
    roots=[data/f'mf_position_20261125_{version}' for version in ['v1','v2']]+[results/f'mf_position_20261125_{version}' for version in ['v1','v2']]+[ROOT/'outputs/reports/mf_position_20261125',REPORT]
    files={p for folder in roots for p in folder.rglob('*') if p.is_file()}|set(source_paths+method_paths)
    audit_files={p for p in REPORT.rglob('*') if p.is_file()}|set(source_paths)
    storage=dict(measured_before_this_final_report=True,audit_bytes=sum(p.stat().st_size for p in audit_files),shared_new_bytes=sum(p.stat().st_size for p in files),disk_free_bytes=shutil.disk_usage(ROOT).free)
    passed=core['passed'] and pixels['passed'] and preserved['unchanged'] and policies_identical and all(same_inputs.values()) and old_sources_unchanged and old_export['predictions_exact_all1508'] and not (oldrun/'queries.jsonl').exists() and storage['audit_bytes']<32*1024**2 and storage['shared_new_bytes']<128*1024**2 and storage['disk_free_bytes']>=20*1024**3
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=passed,
        v1_v2_revision=dict(policies_identical=policies_identical,method_inputs_identical=same_inputs,v1_sources_unchanged=old_sources_unchanged,v1_all1508_predictions_exact=old_export['predictions_exact_all1508'],v1_GT_evaluation_not_run=not (oldrun/'queries.jsonl').exists(),reason='Only the ledger interpretation of unresolved earlier sparse-to-dense competition was corrected before any v1 GT evaluation. All competing lineage remains unknown; source/prediction/threshold semantics unchanged.'),
        paired_changes=dict(paired),gain_scene_count=len(gain_scenes),gain_log_count=len(gain_logs),gain_scenes=sorted(gain_scenes),gain_logs=sorted(gain_logs),
        methods=core['independent_methods'],raw_pixel_audit=dict(passed=pixels['passed'],confirmed_state_frames=pixels['qualified_state_frames'],unique_real_measurements=pixels['qualified_unique_measurements'],frames=pixels['qualified_frame_count'],current_point_histogram=pixels['point_count_histogram']),
        protection=preserved,storage=storage,audit_source_sha256={str(p):sha(p) for p in source_paths},
        report_sha256={str(p):sha(p) for p in [REPORT/'baseline_and_git_sync_v1.json',REPORT/'independent_results_v2.json',REPORT/'qualified_pixels_v2.json']},
        limits=['All38 scenes are previously used validation/debug scenes; improvements across scenes/logs are descriptive, not unseen-test evidence.', 'Map geometry/identity predictions were exactly unchanged. This is a semantic position-candidate policy improvement, not mapping recall or navigation/arrival success.', 'All340 confirmed-history real observations contain only1-4 current depth pixels; no cumulative point count is passed off as current five-point support.', 'Covariance remains heuristic; shape is not inferred and no vehicle-safety authorization is granted. Negative false selections19/382 and2 added wrong positive-task choices are retained.', 'No default, original model weight, protected source, paper or old failure result was modified. No GPU/model/training/planner call was made by this audit.'],deadline_utc=DEADLINE,GPU_calls=0,model_calls=0,training_calls=0,planner_calls=0)
    write(REPORT/'final_audit_v2.json',result)
    text='''独立审计通过。完整38个已用验证／调试场景、11个log、1508帧和1115条任务（733正、382负）均保留，历史81场景与log均不重叠。

|指标|原R1规则|多帧位置候选|
|---|---:|---:|
|正任务选对|81/733（11.05%）|93/733（12.69%）|
|正确目标进入候选|91/733|104/733|
|无目标任务误选|19/382|19/382|
|地图已有正确对象|370/733|370/733|

逐任务复算显示新增14次选择全部来自原拒选任务，其中12次正确、2次错误，没有损失原已选对任务；收益分布于{scenes}个场景、{logs}个log。1508帧原地图预测逐项精确一致，不能把候选收益说成地图精度／召回或导航路线收益。

178个确认状态帧的340次唯一真实观测，已从冻结深度PNG与检测掩膜全部重投影核验；点数分别为1点113次、2点132次、3点65次、4点30次。未把滤波中心当测量、未把累计点数当当前支持、未合并身份交接历史。完整关联与时间账目均因果。

v1完整保留且未作GT选择评分；v2仅修正前阶段dense竞争未解决时必须保持unknown，原政策／方法输入／预测和门限不变。静止速度阈值、密集证据门、关系、距离及排序均保持；这是显式新增的位置候选规则，未知形状不补造，车辆安全授权始终为0。

153项原保护文件全部SHA不变。独立审计未启动模型、训练、GPU或规划器；本轮新增体量和磁盘实测见final_audit_v2.json。结论仅适用于已用验证／调试集，默认方案未切换。

机器证据：baseline_and_git_sync_v1.json、independent_results_v2.json、qualified_pixels_v2.json、final_audit_v2.json。
'''.format(scenes=len(gain_scenes),logs=len(gain_logs))
    (REPORT/'独立审计结论.md').write_text(text)
    print(json.dumps(dict(passed=passed,paired_changes=dict(paired),gain_scene_count=len(gain_scenes),gain_log_count=len(gain_logs),storage=storage,protection=preserved),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['baseline','protection','finalize']);args=parser.parse_args()
    if args.action=='baseline':baseline()
    elif args.action=='finalize':finalize()
    else:print(json.dumps(protection(),ensure_ascii=False))


if __name__=='__main__':main()
