"""只读检查代码同步范围；不暂存、不提交、不推送、不输出任何凭据内容。"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PROJECT = Path('/root/autodl-tmp/lizichong')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'source_bundle/git_preflight_077'
ALLOWED_SUFFIXES = {'.py', '.yaml', '.yml', '.txt', '.md', '.json', '.html',
                    '.js', '.css', '.toml', '.ini', '.cfg', '.sh', '.ps1'}
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.git', 'outputs', 'wandb',
              'results', 'tmp', 'node_modules'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(path, base):
    if path.is_symlink() or not path.is_file():
        return False
    relative = path.relative_to(base)
    if set(relative.parts) & SKIP_PARTS or path.name.startswith('.env'):
        return False
    return path.suffix in ALLOWED_SUFFIXES or path.name in {'LICENSE', 'NOTICE', '.gitignore'}


def git(*args, timeout=30):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    return subprocess.run(['git', *args], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def write(name, data):
    with (OUT/name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    selected = set()
    for base in [PROJECT/'conceptgraph', PROJECT/'scripts', PROJECT/'tests', PROJECT/'doc']:
        if base.is_dir():
            selected.update(p for p in base.rglob('*') if eligible(p, base))
    # outputs被整体忽略：最终只能显式添加本批源代码，不能git add -f整个outputs。
    source = ROOT/'source'
    selected.update(p for p in source.rglob('*') if eligible(p, source))
    for name in ['README.md', 'README_nuscenes_minimal.md', 'README_streamlined_detection.md',
                 'setup.py', 'environment.yml', 'LICENSE', '.gitignore']:
        path = PROJECT/name
        if path.is_file():
            selected.add(path)
    before = git('status', '--porcelain=v1', '-z')
    assert before.returncode == 0
    fingerprints = {}
    sizes = Counter()
    parse_errors, credential_hits, oversized = [], [], []
    # 只记路径/行号/类型；匹配到的值和包含它的行绝不进入报告。
    patterns = {
        'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
        'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'credential_in_github_url': re.compile(r'https?://[^\s/]+:[^\s/]+@github\.com'),
    }
    for path in sorted(selected):
        name = str(path.relative_to(PROJECT))
        raw = path.read_bytes()
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
        sizes[path.suffix or path.name] += len(raw)
        if len(raw) > 5*1024**2:
            oversized.append(dict(path=name, bytes=len(raw)))
        text = raw.decode('utf-8', errors='replace')
        if path.suffix == '.py':
            try:
                ast.parse(raw, filename=name)
            except (SyntaxError, UnicodeError) as exc:
                parse_errors.append(dict(path=name, line=getattr(exc, 'lineno', None),
                                         error_type=type(exc).__name__))
        for number, line in enumerate(text.splitlines(), 1):
            for kind, expression in patterns.items():
                if expression.search(line):
                    credential_hits.append(dict(path=name, line=number, kind=kind))
    head = git('rev-parse', 'HEAD')
    branch = git('branch', '--show-current')
    remote_sha, remote_state = None, None
    try:
        remote = git('-c', 'credential.helper=', 'ls-remote',
                     'https://github.com/lzccc666/lzc.git', 'HEAD', timeout=30)
        remote_state = 'public_read_ok' if remote.returncode == 0 else 'public_read_failed'
        if remote.returncode == 0 and remote.stdout.strip():
            remote_sha = remote.stdout.split()[0]
            assert re.fullmatch(r'[0-9a-f]{40,64}', remote_sha)
    except subprocess.TimeoutExpired:
        remote_state = 'public_read_timeout'
    # 远程错误文本可能含环境信息；故不复制stdout/stderr或任何Git配置。
    after = git('status', '--porcelain=v1', '-z')
    assert after.returncode == 0 and before.stdout == after.stdout
    assert all(sha(PROJECT/p) == expected for p, expected in fingerprints.items())
    write('source_files.json', fingerprints)
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), files=len(selected),
        total_source_bytes=sum(sizes.values()), bytes_by_suffix=dict(sizes),
        local_HEAD=head.stdout.strip() if head.returncode == 0 else None,
        local_branch=branch.stdout.strip() if branch.returncode == 0 else None,
        public_remote=remote_state, public_remote_HEAD=remote_sha,
        source_AST_errors=parse_errors, credential_pattern_hits=credential_hits,
        files_over_5MiB=oversized, source_files_sha256=sha(OUT/'source_files.json'),
        repository_status_unchanged=True, sources_unchanged=True,
        dataset_copied=False, model_weights_copied=False, git_index_changed=False,
        git_commit_created=False, git_push_performed=False,
        write_authentication_verified=False,
        ready_for_later_code_only_snapshot=not (parse_errors or credential_hits or oversized),
        experiments_complete=False, goal_complete=False)
    write('report.json', report)
    with (OUT/'同步准备说明.md').open('x') as f:
        f.write('# 代码同步准备\n\n这里只读检查了源码范围、语法、文件大小及常见凭据模式，未更改Git暂存区、历史或远端。source_files.json按原项目相对路径列全部源码哈希，新增实验源码保留原outputs/.../source布局；最终须逐项显式纳入，禁止强制添加整个outputs。\n\n')
        f.write('代码清单包含HTML/JS/CSS和shell/PowerShell脚本，避免只归档Python时漏掉交互页面或入口。未复制原始数据、传感器文件、权重或训练产物。没有把MEMORY/PAPER_MEMORY等会话记忆作为代码交付。常见模式扫描不是完整安全审计。\n\n')
        f.write('GitHub公开只读访问不等于已验证写权限。实验结束并核验后，重新核对清单和远端分支变化，再以不覆盖远端历史的方式提交同步；不使用聊天记录中的令牌，不强制推送。实验尚未完成，因此此目录不声称已交付远端仓库。\n')
    print(json.dumps({k: report[k] for k in ['files', 'total_source_bytes', 'public_remote',
        'source_AST_errors', 'credential_pattern_hits', 'files_over_5MiB',
        'repository_status_unchanged', 'ready_for_later_code_only_snapshot', 'goal_complete']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
