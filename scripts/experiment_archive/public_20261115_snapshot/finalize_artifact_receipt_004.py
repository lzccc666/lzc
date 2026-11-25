"""25GiB策略下的新关闭清单；重算旧清单文件，不覆盖旧回执、不授予Goal完成。"""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

R=Path(__file__).resolve().parents[1]
O=R/'storage_audit'
M=O/'artifact_manifest_004.jsonl.gz'
RECEIPT=O/'artifact_manifest_receipt_004.json'
EXCLUDED=[
    'annotation/server_002.log', 'annotation/human_submissions/',
    'annotation/human_natural_submissions/', 'tables/completed_office_001/runtime/',
    'json/near6_export_completion_001/libreoffice_profile/',
    'storage_audit/artifact_manifest_004.jsonl.gz',
    'storage_audit/artifact_manifest_receipt_004.json',
    'storage_audit/artifact_manifest_failure_004.json',
]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(2**20):h.update(block)
    return h.hexdigest()


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    began=time.monotonic();assert not M.exists() and not RECEIPT.exists()
    for path in ['json/official_stock_navigation_execution_006/server_shutdown.json',
                 'json/official_prius_braking_execution_007/server_shutdown.json',
                 'json/official_prius_braking_execution_008/server_shutdown.json',
                 'json/official_prius_braking_execution_009/server_shutdown.json',
                 'source_bundle/closing_supplement_003/receipt.json']:
        assert (R/path).exists(),path
    policy=json.loads((R/'json/resource_policy_002.json').read_text())
    used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    assert used+32*2**20<policy['new_project_artifact_budget_bytes']
    assert shutil.disk_usage(R).free>=policy['minimum_physical_free_GiB']*2**30
    old_manifest=O/'artifact_manifest_001.jsonl.gz'
    old_receipt=json.loads((O/'artifact_manifest_receipt_001.json').read_text())
    assert digest(old_manifest)==old_receipt['manifest_sha256']
    with gzip.open(old_manifest,'rt') as stream:old={x['path']:x for x in map(json.loads,stream)}
    assert len(old)==47957
    count=0;total=0;seen=set();links=[];differences=[]
    with gzip.open(M,'xt',compresslevel=6) as stream:
        for p in sorted(R.rglob('*')):
            relative=str(p.relative_to(R))
            if p.is_symlink():
                links.append(dict(path=relative,target=os.readlink(p)));continue
            if not p.is_file() or any(relative==q or relative.startswith(q) for q in EXCLUDED):continue
            before=p.stat();value=digest(p);after=p.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),relative
            entry=dict(path=relative,bytes=after.st_size,sha256=value)
            if relative in old:
                expected=old[relative]
                if expected['bytes']!=entry['bytes'] or expected['sha256']!=value:
                    differences.append(dict(path=relative,old_sha256=expected['sha256'],now_sha256=value))
            stream.write(json.dumps(entry,separators=(',',':'),ensure_ascii=False)+'\n')
            count+=1;total+=after.st_size;seen.add(relative)
            if count%10000==0:print(json.dumps(dict(files_hashed=count,source_GiB=total/2**30)),flush=True)
    assert set(old)<=seen,sorted(set(old)-seen)[:20]
    assert not differences, differences
    # Reopen the actual compressed list and check its complete shape, not just gzip success.
    with gzip.open(M,'rt') as stream:
        reopened=list(map(json.loads,stream))
    assert len(reopened)==count and sum(x['bytes'] for x in reopened)==total
    protected=json.loads((R/'source_bundle/reproduction_001/source_manifest.json').read_text())
    assert all(digest(x['path'])==x['sha256'] for x in protected)
    git=subprocess.run(['git','status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True)
    baseline=(O/'git_status_at_continuation_006.txt').read_text();assert git.stdout==baseline
    now=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    free=shutil.disk_usage(R).free
    assert now+128*1024<policy['new_project_artifact_budget_bytes'] and free>=20*2**30
    save(RECEIPT,dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_hashed=count,
        source_bytes=total,manifest_sha256=digest(M),manifest_bytes=M.stat().st_size,
        compressed_manifest_reopened_records=len(reopened),all_included_files_stable_during_hash=True,
        prior_manifest_entries_rehashed=len(old),prior_manifest_changed_files=differences,
        protected_original_source_files_rehashed=len(protected),tracked_git_status_unchanged=True,
        tracked_dirty_files=len(git.stdout.splitlines()),project_bytes_before_receipt=now,
        project_GiB=now/2**30,project_budget_bytes=policy['new_project_artifact_budget_bytes'],
        remaining_project_GiB=(policy['new_project_artifact_budget_bytes']-now)/2**30,
        physical_free_bytes=free,physical_free_GiB=free/2**30,minimum_physical_free_GiB=20,
        symlinks=links,excluded_prefixes=EXCLUDED,
        exclusions_reason='活动人工服务日志及未来人工提交、LibreOffice运行配置/缓存、清单自身和回执不自包含。无删除。',
        external_bulk_scope='符号链接只记录目标，不追入共享CARLA安装、原始数据或模型；外部文件按各release与安装清单追溯。',
        full_historical_data_outside_this_new_root_rehashed=False,
        source_supplement_receipt_sha256=digest(R/'source_bundle/closing_supplement_003/receipt.json'),
        source_sha256=digest(__file__),elapsed_s=time.monotonic()-began,deleted_files=[],
        whole_project_accepted=False,goal_complete=False,
    ))
    print(json.dumps(dict(files_hashed=count,old_entries_unchanged=len(old),
                          project_GiB=now/2**30,free_GiB=free/2**30,goal_complete=False)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        fail=O/'artifact_manifest_failure_004.json'
        if not fail.exists():save(fail,dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False))
        raise
