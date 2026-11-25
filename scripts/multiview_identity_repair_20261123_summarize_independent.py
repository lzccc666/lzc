"""汇总已有独立复核与封存下批输入，不触发模型或建图。"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_independent_scores import write_csv


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root = PROJECT_ROOT
    report = root/'outputs/reports/multiview_identity_repair_20261123/validation'
    run = experiment_directory('nuscenes', 'validation', 'mvrepair_validation_delivery_20261123_v2', create=True)
    registration = json.loads((report/'registration.json').read_text())
    protected = {name: digest(name) == sha for name, sha in registration['protected_sha256'].items()}
    assert all(protected.values())
    snapshots = {}
    for path in sorted((root/'scripts').glob('multiview_identity_repair_20261123_*.py')):
        destination = run/'source_snapshot'/path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshots[str(path)] = dict(sha256=digest(path), snapshot=str(destination))
    table = []
    sources = [('L1_pilot_v1', 'pilot', [('L1:A_front','A'),('L1:C_three_dedup','C'),('L1:L1_lifecycle','L1')]),
        ('L4_pilot_v1', 'pilot', [('L4:L4_three','L4'),('Aphys:A_front_physical_only','A_physical'),('Aphys:L3_physical_hypotheses','L3')]),
        ('L2_pilot_v1', 'pilot', [('L2:L2_three','L2')]),
        ('P1_pilot_v1', 'pilot', [('P1:P1_three','P1')]),
        ('L5_pilot_v1', 'pilot', [('L5:L5_sameview_three','L5')]),
        ('L1_crosslog_v1', 'regression', [('crosslog:A_front','A'),('crosslog:C_three_dedup','C'),('crosslog:L1_lifecycle','L1')]),
        ('L2_crosslog_v1', 'regression', [('L2:L2_three','L2')]),
        ('L4_crosslog_v1', 'regression', [('L4:L4_three','L4')]),
        ('A_physical_crosslog_v1', 'regression', [('Afair:A_front_physical_only','A_physical')]),
        ('P1_crosslog_v1', 'regression', [('P1:P1_three','P1')]),
        ('L5_crosslog_v1', 'regression', [('L5:L5_sameview_three','L5')])]
    for prefix, scope, selected in sources:
        rows = list(csv.DictReader((report/(prefix+'_main.csv')).open(encoding='utf-8-sig')))
        for key, label in selected:
            found = [r for r in rows if r['method'] == key]
            assert len(found) == 2, (prefix, key)
            for row in found:
                table.append(dict(dataset=scope, label=label, **row, evidence=str(report/(prefix+'_main.csv'))))
    write_csv(report/'独立主表_阶段v1.csv', table)
    positions = []
    for prefix in ['L4_crosslog_v1', 'P1_crosslog_v1']:
        data = json.loads((root/'outputs/results/nuscenes/validation/iterations'/('mvrepair_independent_'+prefix)/'paired_position_samples.json').read_text())
        groups = defaultdict(lambda: dict(errors=[], steps=[]))
        for row in data['samples']:
            groups[row['scope'],row['scene'],row['method']]['errors'].append(row['center_error_xy_m'])
        for row in data['adjacent_pairs']:
            groups[row['scope'],row['scene'],row['method']]['steps'].append(row['delta_error_norm_m'])
        for (scope, scene, method), values in groups.items():
            positions.append(dict(comparison=prefix, scope=scope, scene=scene, method=method,
                common_GT_frames=len(values['errors']), common_adjacent_pairs=len(values['steps']),
                mean_error_xy_m=float(np.mean(values['errors'])),
                mean_error_change_m=float(np.mean(values['steps'])) if values['steps'] else None))
    write_csv(report/'共同对象位置_逐log_阶段v1.csv', positions)
    folder = root/'dataset/processed/multiview_identity_repair_20261123_nextlogs_v1'
    manifest = json.loads((folder/'manifest.json').read_text())
    front_run = Path(manifest['frontend_run'])
    frontend = json.loads((front_run/'frontend_complete.json').read_text())
    truth = json.loads((folder/'multiview_identity_repair_20261123_ground_truth_release.json').read_text())
    audit = json.loads((report/'nextlogs/independent_input_audit.json').read_text())
    source_audit = json.loads((report/'nextlogs/official_source_audit.json').read_text())
    assert frontend['views'] == audit['views'] == 120 and audit['passed'] and source_audit['passed']
    sealed = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='inputs_only_sealed_pending_final_candidate_freeze',
        role='original38 development/debug, new unused logs; not unseen test',
        selection_rule=manifest['selection_rule'], scenes=[dict(scene=s['scene'], log_token=s['log_token']) for s in manifest['scenes']],
        manifest=str(folder/'manifest.json'), manifest_sha256=digest(folder/'manifest.json'),
        GT=str(folder/'multiview_identity_repair_20261123_evaluation_ground_truth.jsonl'), GT_sha256=truth['output_sha256'], GT_counts=truth['counts'],
        frontend_run=str(front_run), actual_new_frontend_calls=120, gpu_wall_s=frontend['gpu_wall_s'],
        no_method_mapping_or_scoring_executed_on_this_batch=True, no_candidate_selection_from_batch=True,
        audits=dict(input=str(report/'nextlogs/independent_input_audit.json'), official=str(report/'nextlogs/official_source_audit.json')),
        future_lidar_views=audit['future_lidar_views'], missing_past_lidar_views=audit['missing_past_lidar_views'])
    (report/'nextlogs/sealed_inputs.json').write_text(json.dumps(sealed, ensure_ascii=False, indent=2)+'\n')
    new_roots = [root/'dataset/raw'/name for name in ['multiview_identity_repair_20261123_validation_v1','multiview_identity_repair_20261123_nextlogs_v1']]
    new_roots += [root/'dataset/processed'/name for name in ['multiview_identity_repair_20261123_validation_v1','multiview_identity_repair_20261123_nextlogs_v1']]
    results = root/'outputs/results/nuscenes/validation/iterations'
    prefixes = ['mvrepair_independent_', 'mvrepair_identity_', 'multiview_identity_repair_20261123_frontend_', 'mvrepair_L5_point_audit_']
    names = {'mvrepair_L1_crosslog_validation_20261123_v1','mvrepair_L2_crosslog_regression_20261123_v1','mvrepair_L4_crosslog_regression_20261123_v1','mvrepair_A_physical_crosslog_regression_20261123_v1','mvrepair_validation_delivery_20261123_v1','mvrepair_validation_delivery_20261123_v2'}
    new_roots += [p for p in results.iterdir() if p.is_dir() and (p.name in names or any(p.name.startswith(k) for k in prefixes))]
    new_roots += [report]
    files = {p.resolve() for base in new_roots for p in base.rglob('*') if p.is_file()}
    size = sum(p.stat().st_size for p in files)
    verification = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), source_snapshots=snapshots,
        registered_protected_unchanged=protected, protected_count=len(protected), scoped_new_bytes=size,
        scoped_new_MiB=size/2**20, free_GiB=shutil.disk_usage(root).free/2**30,
        budget_new_MiB=768, total_real_frontend_calls=240,
        total_gpu_wall_s=frontend['gpu_wall_s']+json.loads((results/'multiview_identity_repair_20261123_frontend_validation_v1/frontend_complete.json').read_text())['gpu_wall_s'],
        training_calls=0, new_test_opened=False, next_batch_method_scores_opened=False,
        boundary='Resource count covers this agent-owned raw/processed inputs, audits, replay confirmations and independent-score runs; main/lifecycle candidate runs are separate. No repeated frontend on old inputs.')
    assert size < 768*2**20 and verification['free_GiB'] > 20
    (report/'阶段交付核验_v1.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 独立复核阶段结果 v1', '',
        '当前保留原3m同类别XY匹配、全GT与预测分母。pilot为2scene同1log、105 union/77 front/28 side对象帧；跨log为scene0019与0035两个log、357 union/171 front/186 side。跨log现已诊断，后续方法再跑称回归。', '',
        '|数据|方法|口径|TP/GT|预测|重复|P %|R %|F1 %|IDF1 %|', '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f'|{row["dataset"]}|{row["label"]}|{row["scope"]}|{row["TP"]}/{row["GT"]}|{row["predictions"]}|{row["duplicates"]}|{100*float(row["precision"]):.2f}|{100*float(row["recall"]):.2f}|{100*float(row["F1"]):.2f}|{100*float(row["IDF1"]):.2f}|')
    lines += ['', '主表保留L2/L4/P1/L5全部结果，未择优省略。L4/P1在两个log保持C的177匹配和107/186侧向匹配；P1修复L4旧front覆盖造成的共同前视位置退步，但共同前视位置只回到C、未超A。union共同位置仍略劣C。L5降低一条重复却IDTP152降150，不能作为身份改善。', '',
        '公平A_physical独立CPU40组已跑完，audit-only旁路与原A逐条完全相同。P1_front与公平A_physical在两组40时刻的类别+中心多重集合精确一致；metadata/ID字符串非全同，不能写完整JSON精确复现。', '',
        '共同前视身份诊断固定59对象帧、37连续对：A/L4链IDTP53/52，IDFN及IDFP6/7，ID变化1/2。此固定几何配对链指标不替代全分母all-gated IDF1。最大车链0035 f12–14为L4 sparse24→34→24；另有0019拥挤行人一个ID匹配多个GT。3m自动配对不是人审身份真值。', '',
        '原图、框和mask查看v2入口；registry见证只能证明曾参与关联，未证明当帧输出实际来源。f12–14待main深拷贝状态/真实关联矩阵交叉核；旧v1标题过强，保留历史并以v2明确限制。', '',
        'L5原点索引检查1163观测：431 dense/732 sparse，均为真实npz zbuffer赢家且位于原mask；sparse rint及无扰动深度点精确复核。dense来源代码含voxel contributor，但原jitter输入/contributor列表未持久化，不能声称独立精确重放dense trace。', '',
        '下一批scene0042/log d7fd2bb9696d43af901326664e42340b、scene0047/log f38ef5a1e9c941aabb2155768670b92a各20时刻，已真实120前端、过去LiDAR/官方pose/K/投影核过；GT225 union/137 front/88 side。封存只准备输入，无方法建图或评分，等最终组合冻结后开启一次评价。', '',
        '证据入口：', '',
        f'- [完整主表]({report}/独立主表_阶段v1.csv)',
        f'- [L4逐scene/log主表]({report}/L4_crosslog_v1_scene.csv)',
        f'- [P1逐scene/log主表]({report}/P1_crosslog_v1_scene.csv)',
        f'- [共同位置逐log]({report}/共同对象位置_逐log_阶段v1.csv)',
        f'- [共同前视身份完整链]({report}/L4_crosslog_identity_v2_identity_diagnostics.json)',
        f'- [身份实际图源v2]({root}/outputs/results/nuscenes/validation/iterations/mvrepair_identity_visual_sources_20261123_v2/index.html)',
        f'- [原L1三条重复双侧图源]({report}/L1_crosslog_duplicate_visual_sources.csv)',
        f'- [L5原索引审计]({report}/L5_original_point_index_audit.json)',
        f'- [下一批封存]({report}/nextlogs/sealed_inputs.json)',
        f'- [保护、资源与源码核验]({report}/阶段交付核验_v1.json)', '',
        f'本线新增约{size/2**20:.2f}MiB，240次实际前端GPU共{verification["total_gpu_wall_s"]:.2f}s，训练0、新测试0、注册保护{len(protected)}项均未变。图是离线公开数据回放证据，不是车辆闭环。']
    (report/'独立复核_阶段结果_v1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,report=str(report/'独立复核_阶段结果_v1.md'),table_rows=len(table),new_MiB=size/2**20,protected=len(protected))))


if __name__ == '__main__':
    main()
