"""准备已完成实验的Git交付副本；不提交、不推送，不导出传感器数据或权重。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
SOURCE = OUT/'source_bundle'
GIT = SOURCE/'github_sync_001'
CHECKOUT = GIT/'checkout'
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    now = datetime.now(timezone.utc).isoformat()
    checkpoint = read(SOURCE/'source_checkpoint_037/checkpoint.json')
    assert len(checkpoint['sources']) == 390
    assert all(sha(PROJECT/p) == sha(CHECKOUT/p) == h for p, h in checkpoint['sources'].items())
    assert sha(PROJECT/'.git/index') == checkpoint['original_index_sha256']
    previous_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
    assert previous_head == 'f75725f8fd422ec619775bfa0407dc6733312488'
    assert read(SOURCE/'perceived_blockage_final_verification_001.json')['passed']
    assert read(SOURCE/'development_result_tables_final_verification_002.json')['passed']
    assert read(OUT/'faults/perceived_blockage_cpu_002/recovery_001/complete.json')['all_required_exposures_passed']
    exports = {}
    table_root = OUT/'experiment_tables/development_evidence_003'
    for path in sorted(table_root.rglob('*')):
        if path.is_file() and path.suffix in {'.md', '.json', '.csv', '.docx', '.xlsx', '.pdf', '.html'}:
            exports['experiments/development_tables/'+str(path.relative_to(table_root))] = path
    helpers = ['perceived_blockage_cpu_worker_001.py', 'perceived_blockage_cpu_worker_002.py',
        'perceived_blockage_cpu_run_001.py', 'perceived_blockage_cpu_run_002.py',
        'perceived_blockage_cpu_resume_003.py', 'perceived_blockage_cpu_audit_001.py',
        'perceived_blockage_cpu_audit_002.py', 'perceived_blockage_cpu_audit_003.py',
        'perceived_blockage_cpu_audit_004.py', 'perceived_blockage_geometry_witness_001.py',
        'perceived_blockage_geometry_witness_002.py', 'perceived_blockage_input_audit_003.py',
        'perceived_blockage_viewer_002.py', 'perceived_blockage_browser_qa_002.py',
        'perceived_blockage_finalize_002.py', 'development_result_tables_003.py',
        'development_result_tables_audit_002.py', 'refresh_delivery_aliases_020.py', Path(__file__).name]
    for name in helpers:
        exports['reproduction/'+name] = SOURCE/name
    for path in (OUT/'task_protocol').glob('perceived_blockage*.md'):
        exports['experiments/perceived_blockage/protocol/'+path.name] = path
    reports = ['faults/perceived_blockage_cpu_001/complete.json',
        'faults/perceived_blockage_cpu_002/recovery_001/complete.json',
        'faults/perceived_blockage_cpu_002/delayed_actual_rank/input_audit_003/report.json',
        'visualizations/perceived_blockage_browser_qa_002/report.json',
        'source_bundle/perceived_blockage_final_verification_001.json',
        'source_bundle/perceived_blockage_structured_validation_001.json',
        'source_bundle/perceived_blockage_protection_001.json']
    for run in ['perceived_blockage_cpu_001', 'perceived_blockage_cpu_002']:
        for condition in ['rule_blockage', 'delayed_actual_rank']:
            version = '003' if run.endswith('002') else '001'
            reports += [f'faults/{run}/{condition}/independent_audit_{version}/report.json']
            if run.endswith('002'):
                reports += [f'faults/{run}/{condition}/geometry_witness_002/report.json']
    for relative in reports:
        exports['experiments/perceived_blockage/evidence/'+relative.replace('/', '__')] = OUT/relative
    for target, relative in {
        'doc/thesis_completion/experiment_delivery_status_020.md': '实验完成度与交付清单_020.md',
        'doc/thesis_completion/experiment_delivery_status_020.json': '实验完成度与交付清单_020.json',
        'doc/thesis_completion/validation_snapshot_020.json': 'validation_report.json',
        'doc/thesis_completion/perceived_blockage_CPU_handoff.md': '真实屏障与迟到排序_CPU交接_001.md',
        'doc/thesis_completion/reproducible_commands_020.md': 'reproducible_commands.md',
    }.items():
        exports[target] = OUT/relative
    assert sum(path.stat().st_size for path in exports.values()) < 15_000_000
    assert not any((CHECKOUT/name).exists() for name in exports), '新增版本目标已存在，不覆盖'
    inputs = {str(path.relative_to(OUT)):sha(path) for path in exports.values()}
    assert not any(SECRET.search(path.read_bytes()) for path in exports.values())
    # 预检所有来源后再写入；失败保留，不以重试覆盖交付副本。
    copied = {}
    for name, path in sorted(exports.items()):
        target = CHECKOUT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(target) == inputs[str(path.relative_to(OUT))] == sha(path)
        copied[name] = dict(source=str(path.relative_to(OUT)), sha256=sha(target), bytes=target.stat().st_size)
    note = CHECKOUT/'doc/thesis_completion/prepared_increment_024.md'
    with note.open('x') as stream:
        stream.write('''# 已完成实验增量024：准备中，尚未推送

本增量只整理已完成的开发实验、复现代码和可编辑实验表，没有新增推理或导航样本。正式GPU消融、故障、独占性能、人工审核和新的封存评价仍待完成；Goal为false。当前不写论文正文。

- 开发结果表位于 `experiments/development_tables/`，含15表114行、Word/Excel/PDF、逐表CSV和来源快照；是开发证据，不是封存主表。
- 屏障/迟到排序证据位于 `experiments/perceived_blockage/`，白目标两次未暴露保留，红目标两次暴露通过但均未到达。原始输入和控制不随Git复制；完整归档保留在原服务器Goal目录。
- 关键原始输入、模型、数据和资产路径属于服务器证据链接。快照JSON中的SHA不能被解释为这些大文件已包含于Git。
- `reproduction/`中新脚本是原服务器复现辅助，部分固定run-id不能原地覆盖；执行前选新版本目录并核对协议。便携完整小任务入口仍为 `reproduction/standalone_navigation/` 的README。
- 新增交付快照020与原快照017同时保留，数字及时间戳以各自快照为准；实时进度查询原服务器账目。

根项目390业务文件、原索引与已有Git历史未更改。本副本未提交或推送；最终实验核验后按用户授权统一同步GitHub。
''')
    assert not SECRET.search(note.read_bytes())
    check = subprocess.run(['git', 'diff', '--check'], cwd=CHECKOUT, capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stdout+check.stderr
    syntax = subprocess.run([sys.executable, '-m', 'ruff', 'check', *[str(CHECKOUT/'reproduction'/name) for name in helpers]],
        capture_output=True, text=True, check=False)
    with (GIT/'ruff_increment_024.log').open('x') as stream:
        stream.write(syntax.stdout+syntax.stderr)
    assert syntax.returncode == 0, syntax.stdout+syntax.stderr
    assert all(sha(PROJECT/p) == sha(CHECKOUT/p) == h for p, h in checkpoint['sources'].items())
    assert sha(PROJECT/'.git/index') == checkpoint['original_index_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == previous_head
    receipt = dict(created_utc=now, previous_manifest='incremental_export_manifest_023.json',
        business_source_files=390, business_and_original_Git_index_unchanged=True,
        copied_files=copied, extra_handoff=dict(path=str(note.relative_to(CHECKOUT)), sha256=sha(note)),
        copied_bytes=sum(v['bytes'] for v in copied.values()), secret_pattern_scan_passed=True,
        source_and_copy_SHA_verified=True, Ruff_passed=True, git_diff_check_passed=True,
        code_committed=False, code_pushed=False, previous_public_commit=previous_head,
        model_weights_exported=False, raw_sensor_data_exported=False, model_calls_added=0,
        manuscript_text_written=False, goal_complete=False)
    save(GIT/'incremental_export_manifest_024.json', receipt)
    print(json.dumps({k:v for k, v in receipt.items() if k not in ['copied_files', 'extra_handoff']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
