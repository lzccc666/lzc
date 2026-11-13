"""准备标签审核与独立环境增量；不提交、不推送，不导出原始数据或人工记录。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT / 'outputs/project_v2/thesis_completion_v1'
SOURCE = OUTPUT / 'source_bundle'
GIT = SOURCE / 'github_sync_001'
CHECKOUT = GIT / 'checkout'
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint = read(SOURCE / 'source_checkpoint_037/checkpoint.json')
    assert len(checkpoint['sources']) == 390
    assert all(sha(PROJECT/name) == sha(CHECKOUT/name) == expected
               for name, expected in checkpoint['sources'].items())
    assert sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
    previous_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
    assert previous_head == 'f75725f8fd422ec619775bfa0407dc6733312488'
    assert read(GIT / 'incremental_export_verification_024.json')['passed']
    assert read(SOURCE / 'human_adjudication_final_verification_002.json')['passed']
    assert read(SOURCE / 'isolated_environment_preparation_verification_001.json')['passed']
    assert not (OUTPUT / 'reproduction/isolated_cpu_environment_002/installed.json').exists()
    helpers = [
        'isolated_environment_import_preflight_001.py', 'isolated_cpu_environment_pack_001.py',
        'isolated_cpu_environment_pack_002.py', 'standalone_cpu_runner_003.py',
        'prepare_isolated_environment_navigation_001.py', 'isolated_environment_navigation_audit_001.py',
        'isolated_environment_after_GPU_001.py', 'human_query_adjudication_001.py',
        'test_human_query_adjudication_001.py', 'human_adjudication_browser_qa_001.py',
        'human_adjudication_browser_qa_002.py', 'human_adjudication_browser_qa_003.py',
        'human_adjudication_fixture_browser_001.py', 'human_adjudication_fixture_browser_002.py',
        Path(__file__).name,
    ]
    exports = {'reproduction/' + name: SOURCE / name for name in helpers}
    adjudication = OUTPUT / 'annotation/human_natural_adjudication_001'
    for path in sorted(adjudication.rglob('*')):
        if path.is_file() and path.suffix in {'.md', '.html', '.json', '.sha256'}:
            exports['experiments/human_adjudication/' + str(path.relative_to(adjudication))] = path
    evidence = [
        'source_bundle/human_adjudication_final_verification_001.json',
        'source_bundle/human_adjudication_final_verification_002.json',
        'source_bundle/isolated_environment_preparation_verification_001.json',
        'visualizations/human_adjudication_browser_qa_001/diagnostic.json',
        'visualizations/human_adjudication_browser_qa_002/report.json',
        'visualizations/human_adjudication_browser_qa_003/report.json',
        'reproduction/human_adjudication_browser_fixture_001/report.json',
        'reproduction/human_adjudication_browser_fixture_002/report.json',
        'reproduction/human_adjudication_fixture_validation_001/exit.json',
        'reproduction/human_adjudication_fixture_validation_001/worker.log',
        'source_bundle/human_adjudication_browser_qa_001.log',
        'reproduction/isolated_environment_import_preflight_001/release.json',
        'reproduction/isolated_environment_import_preflight_001/exit.json',
        'reproduction/isolated_environment_import_preflight_001/worker.log',
        'reproduction/isolated_environment_after_GPU_001/release.json',
        'reproduction/isolated_environment_after_GPU_001/README.md',
        'reproduction/isolated_cpu_environment_001/plan.json',
        'reproduction/isolated_cpu_environment_002/plan.json',
    ]
    for relative in evidence:
        exports['experiments/preparation_evidence_025/' + relative.replace('/', '__')] = OUTPUT / relative
    exports['doc/thesis_completion/experiment_status_021.md'] = OUTPUT / '当前Goal进度与收尾安排_021.md'
    exports['doc/thesis_completion/experiment_status_021.json'] = OUTPUT / '当前Goal进度与收尾安排_021.json'
    assert sum(path.stat().st_size for path in exports.values()) < 15_000_000
    assert not any((CHECKOUT / name).exists() for name in exports), '不覆盖已有导出版本'
    inputs = {str(path.relative_to(OUTPUT)): sha(path) for path in exports.values()}
    assert not any(SECRET.search(path.read_bytes()) for path in exports.values())
    copied = {}
    for name, path in sorted(exports.items()):
        target = CHECKOUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(target) == inputs[str(path.relative_to(OUTPUT))] == sha(path)
        copied[name] = dict(source=str(path.relative_to(OUTPUT)), sha256=sha(target), bytes=target.stat().st_size)
    note = CHECKOUT / 'doc/thesis_completion/prepared_increment_025.md'
    with note.open('x') as stream:
        stream.write('''# 增量025：人工标签审核与独立环境复演准备

本增量尚未提交或推送。Goal为false，新封存测试未开启，论文正文暂缓。

真实自然表达的后续标签审核已在服务器8770运行。10项隔离后端/HTTP测试、桌面/手机正式页面只读检查和独立合成表单流程通过；所有正式页面请求均GET，真实人工表达和审核仍0。第3版页面切换源时清空旧标签和确认勾选。合成夹具记录不算人工标签。官方真值、传感器图像、crop和任何真实人工提交不随此增量导出。

独立新Python环境仍未安装或实际导航。已锁定29494个复制文件约1.52GB、两份官方CPU轮子SHA；原GPU队列终止且存储门槛满足后才安装，随后实际导航、独立几何/身份/停车保持审核，并重算450主统计。脚本和排队清单不是完成证据。环境的基础标准库和原生系统库共享，不能称新操作系统。

环境清单001和首轮浏览器失败作为历史保留；环境清单002和浏览器003为当前版本。官方轮子未包含在Git，现有依赖和声明的包元数据冲突须以实际新环境执行验收。辅助脚本的服务器绝对路径、固定run-id和原始SHA引用仍属于服务器复现入口；重新执行须另立版本，不能覆盖原结果。便携任务入口仍见reproduction/standalone_navigation/。

主开发450已结束；原缺一消融、正式GPU故障、性能和新封存验收仍未完成。实验快照021有自己的时间戳，实时进度以原服务器账目为准。390业务源码与原项目索引保持不变，实验核验后按用户授权统一同步GitHub。
''')
    assert not SECRET.search(note.read_bytes())
    check = subprocess.run(['git', 'diff', '--check'], cwd=CHECKOUT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    lint = subprocess.run([sys.executable, '-m', 'ruff', 'check']
        + [str(CHECKOUT / 'reproduction' / name) for name in helpers], capture_output=True, text=True)
    with (GIT / 'ruff_increment_025.log').open('x') as stream:
        stream.write(lint.stdout + lint.stderr)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    assert all(sha(PROJECT/name) == sha(CHECKOUT/name) == expected
               for name, expected in checkpoint['sources'].items())
    assert sha(PROJECT / '.git/index') == checkpoint['original_index_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == previous_head
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest='incremental_export_manifest_024.json', copied_files=copied,
        copied_bytes=sum(row['bytes'] for row in copied.values()), business_files_unchanged=390,
        original_Git_index_unchanged=True, source_and_copy_SHA_verified=True,
        secret_pattern_scan_passed=True, Ruff_passed=True, git_diff_check_passed=True,
        extra_handoff=dict(path=str(note.relative_to(CHECKOUT)), sha256=sha(note)),
        real_human_records_exported=0, raw_sensor_data_exported=False,
        model_weights_exported=False, new_environment_installed=False,
        actual_model_calls_added=0, manuscript_text_written=False,
        code_committed=False, code_pushed=False, previous_public_commit=previous_head,
        new_test_opened=False, goal_complete=False)
    with (GIT / 'incremental_export_manifest_025.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {'copied_files', 'extra_handoff'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
