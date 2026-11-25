"""关闭已完成证据并封装新增源码；正在运行的导航只保存实时句柄，不冒充完成。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from conceptgraph.completion.stage9 import PROJECT, digest, dump

R=Path(__file__).resolve().parents[1]


def live_jobs():
    names={'run_official_clean_cpu_navigation_014.py','carla_clean_cpu_navigation_episode_002.py',
        'run_official_prius_cutoff_pilot_015.py','carla_prius_cutoff_pilot_008.py'}
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if len(args)<2 or Path(args[1]).name not in names:continue
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]!='Z':jobs.append(dict(PID=int(p.name),start_tick=stat[19],command=args))
        except (OSError,UnicodeDecodeError):pass
    return jobs


def main():
    out=R/'source_bundle/closing_supplement_004'
    out.mkdir(exist_ok=False)
    old={x['path']:x['sha256'] for x in json.loads((R/'source_bundle/reproduction_003/source_manifest.json').read_text())}
    old.update({x['path']:x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_001/receipt.json').read_text())['supplemental_files']})
    old.update({str(PROJECT/p):h for p,h in json.loads((R/'source_bundle/closing_supplement_002/files.json').read_text()).items()})
    old.update({x['path']:x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_003/files.json').read_text())})
    assert all(digest(p)==h for p,h in old.items()), '旧源码/交付版本发生变化'
    qa=R/'json/software_validation_continuation_007'
    qa.mkdir(exist_ok=False)
    sources=sorted((R/'source').glob('*.py'))
    for p in sources:ast.parse(p.read_text())
    checks=[]
    for name,command in [('ruff_functional',['/root/miniconda3/envs/conceptgraph/bin/ruff','check','--no-cache','--select','E9,F63,F7,F82',str(R/'source')]),
        ('git_diff_check',['git','diff','--check'])]:
        result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
        path=qa/(name+'.log')
        path.write_text(result.stdout+result.stderr)
        checks.append(dict(check=name,exit_code=result.returncode,log_sha256=digest(path)))
        assert result.returncode==0, name
    dump(qa/'report.json',dict(syntax_checked_source_files=len(sources),checks=checks,
        old_source_and_delivery_files_rehashed_unchanged=len(old),
        injected_backend_checks=['json/carla_real_graph_bridge_validation_001/report.json',
            'json/carla_real_graph_bridge_validation_002/report.json'],
        original_452_business_tests_not_repeated=True,reason='主体源码未变，新增实验源码、实际运行与逐条审计单独核查',goal_complete=False))
    idx=R/'source_bundle/module_index_006'
    idx.mkdir(exist_ok=False)
    previous=list(csv.DictReader((R/'source_bundle/module_index_005/modules.csv').open()))
    additions=[
        ('原厂Prius全传感器规则导航','carla_prius_navigation_episode_005.py','main','carla_official/prius_navigation_summary_005/report.json','两个有目标任务严格安全到达1/2；无匹配正确拒绝1/1；未调用Qwen'),
        ('当前真实图与模型后台选择','carla_real_graph_bridge_002.py','RealGraphNavigationBridge','carla_official/complex_semantic_diagnosis_008/report.json','三方法同一复杂任务已实际跑完，安全到达均失败；旧001接入错误保留'),
        ('图节点输入翻译','carla_real_graph_bridge_002.py','observed_row','carla_official/complex_semantic_diagnosis_008/actual_model_calls.csv','只用预测图和自车位姿；包含未获驾驶授权的干扰对象'),
        ('真实CUDA模型后台推理','carla_real_graph_bridge_002.py','ActualQwenBackend','carla_official/complex_semantic_diagnosis_008/report.json','基础38次、Adapter25次，逐条输入和原始输出可追溯；非实时吞吐'),
        ('复杂语义官方物理回合','carla_complex_semantic_episode_007.py','main','carla_official/complex_semantic_diagnosis_008/report.json','三方法均保存碰撞/越线失败，不以有效JSON代替找对或到达'),
        ('复杂语义独立输入与计数审计','audit_carla_complex_choices_008.py','audit','carla_official/complex_semantic_navigation_007/complex_second_nearest20/adapter/independent_choice_audit_008/report.json','按真实队列序列化回读，并单独计关闭期间最后一次调用；006/007审计失败保留'),
        ('复杂回合错误分层诊断','diagnose_complex_navigation_008.py','main','carla_official/complex_semantic_diagnosis_008/report.json','原始模型提议、结果作废、驾驶授权、锁定、物理到达分别统计'),
        ('新CPU依赖完整导航','carla_clean_cpu_navigation_episode_002.py','main','carla_official/clean_cpu_navigation_002/car20/full_rule_geometry/fresh_environment_modules.json','本次正在运行，待真实终止和独立到达核验；首次缺shapely的失败保留'),
        ('新依赖隔离补充','install_clean_navigation_overlay_002.py','main','reproduction/clean_cpu_navigation_dependencies_002/receipt.json','官方PyPI固定shapely2.1.2安装至新overlay，不修改旧venv'),
        ('无损传感器存储实验','probe_temporal_lossless_storage_002.py','main','storage_audit/temporal_lossless_probe_002/report.json','80个已有帧逐字节一致；移动深度增大，不在当前冻结实验中启用'),
        ('停车立即切油的物理标定','carla_prius_cutoff_pilot_008.py','run','carla_official/prius_cutoff_pilot_008/report.json','已冻结并排队，未获得标定或导航成功信用；不改变原厂物理参数'),
    ]
    for label,name,entry,evidence,boundary in additions:
        path=R/'source'/name
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==entry)
        previous.append(dict(module=label,source=str(path),entry=entry,line=node.lineno,
            meaning=boundary,evidence=str(R/evidence),evidence_exists_at_index_generation=(R/evidence).exists(),
            boundary=boundary,source_sha256=digest(path)))
    with (idx/'modules.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(previous[0]));w.writeheader();w.writerows(previous)
    (idx/'README.md').open('x').write('模块索引006：叠加旧索引；evidence存在不等于指标通过，运行中项目必须读取独立终止报告。\n')
    nav=json.loads((R/'carla_official/prius_navigation_summary_005/report.json').read_text())
    complex_report=json.loads((R/'carla_official/complex_semantic_diagnosis_008/report.json').read_text())
    jobs=live_jobs()
    used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    handoff=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        continuation_classification='progress: actual 3 rule-navigation + 3 complex-semantic episodes; real GPU calls and independent readback; new CPU physical run live',
        closed_original_Prius_episodes=3,actual_rule_navigation_ticks=nav['actual_execution_ticks'],
        actual_rule_frontend_calls=nav['actual_frontend_mapping_calls'],strict_positive_rule_arrival='1/2',
        no_match_rule_refusal='1/1',complex_semantic_physical_episodes=3,
        actual_new_complex_GPU_calls=complex_report['actual_GPU_calls'],complex_safe_arrivals='0/3',
        actual_base_calls=38,base_valid_outputs=6,actual_adapter_calls=25,adapter_valid_outputs=17,
        base_raw_correct_proposals=0,adapter_raw_correct_proposals=1,
        bound_query_or_geometry_discarded_base=29,bound_query_or_geometry_discarded_adapter=23,
        current_live_jobs=jobs,project_artifact_GiB=used/2**30,physical_free_GiB=shutil.disk_usage(R).free/2**30,
        authority_reports=['carla_official/prius_navigation_summary_005/report.json',
            'carla_official/complex_semantic_diagnosis_008/report.json'],
        control_issue='失去路径时control先返回等待，未执行其后扫掠保护；episode把刹停请求转换为慢速减油，当前可见障碍前仍发生碰撞。下一步需要不依赖有效路径的当前LiDAR停车保护及实际标定，不能只调语义分数。',
        next_actions=['先续读当前CPU导航PID与同一会话；不能因超时观察而重复启动',
            'CPU导航终止后读取独立到达与依赖隔离报告；再续读排队的立即切油双标定',
            '按真实失效原因改进语义身份持续锁定/后台有效性与始终执行的车辆安全保护，三方法同一新协议重新比较',
            '继续原完整配置/任务/种子范围、CARLA消融、新封存≥400有效查询与交付，不把本次开发小样本代替'],
        human_reviews_received=0,human_review_status='已给出20张离线包入口及逐步说明，待真人导出JSON；禁止代填',
        whole_project_sealed_test_opened=False,stage9_to12_complete=False,goal_complete=False,
        historic_sources_unchanged=True,thesis_prose_written=False)
    handoff_path=R/'json/continuation_handoff_008.json'
    dump(handoff_path,handoff)
    doc=R/'交付补充说明006.md'
    doc.open('x').write('''# 实验续跑交付补充006

本轮完成原厂Prius三项完整传感器规则导航：1761个实际物理tick、881次检测/分割/建图调用。两个有目标任务中严格安全到达1/2；无匹配任务正确拒绝1/1。car30选对并锁定仍超时，不能记为到达。独立证据见carla_official/prius_navigation_summary_005/。

复杂语义三方法已真实执行同一两车开发任务。基础Qwen38次真实GPU调用，有效6次；Adapter25次，有效17次。原始有效提议正确分别0次与1次；当前主链授权正确、锁定正确和严格安全到达均不能据此推定。规则回合越线、两个模型回合碰撞，安全到达0/3。见carla_official/complex_semantic_diagnosis_008/和各回合independent_choice_audit_008/。

第一次复杂接入006有搜索路线误清和规划预算问题，保留44帧/88tick前缀后停止，未继续运行两个模型；不是三方法比较结果。007修正并冻结后才执行三方法。旧006/007审计未考虑队列序列化顺序及关闭期间最后一次模型调用，失败日志保留；008按真实队列和最终计数逐条回读，不修改原数据。

独立CPU环境完整导航正在执行，实际依赖为已独立安装venv加新Shapely官方overlay。首次导入缺库发生在世界/传感器运行前，实际物理episode为0；旧作业脚本的“started=1”只表示启动过Python子进程。002已通过导入预检并开始官方传感器/建图/锁定/控制，不提前授予到达信用。最新PID与输出路径见json/continuation_handoff_008.json。其结束后会顺序运行已冻结的立即切油双标定，不并发占用官方服务器。

80个已有帧做了无损存储试验，所有RGB/深度逐字节一致。RGB初始段约省42.7%、移动段约省15.8%；移动深度反而增大约9.1%。旧数据未压缩、删除或移动，当前冻结回合仍保持原存储格式。

模块入口与行号见source_bundle/module_index_006/modules.csv；新增源码包见source_bundle/closing_supplement_004/。仍缺真实人工审核、稳定复杂语义安全到达、原任务范围完整对照/消融、最终新封存和完整复现验收，goal_complete=false。没有撰写论文正文，也没有推送Git。
''')
    paths=[p for p in sources if str(p) not in old]
    paths += [doc,handoff_path,idx/'modules.csv',idx/'README.md',qa/'report.json']
    for pattern in ['carla_official/*registration_00[567].json','carla_official/clean_cpu_navigation_registration_00[12].json',
        'carla_official/prius_cutoff_pilot_008/registration.json','reproduction/clean_cpu_navigation_dependencies_002/receipt.json',
        'storage_audit/*lossless_probe_00[12]/report.json']:
        paths += list(R.glob(pattern))
    rows=[]
    for p in sorted(set(paths)):
        assert p.is_file() and p.resolve().is_relative_to(R)
        rows.append(dict(path=str(p),archive_path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=digest(p)))
    archive=out/'source_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in rows:tar.add(row['path'],arcname=row['archive_path'],recursive=False)
    with tarfile.open(archive) as tar:
        for row in rows:assert hashlib.sha256(tar.extractfile(row['archive_path']).read()).hexdigest()==row['sha256']
    dump(out/'files.json',rows)
    dump(out/'receipt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        old_source_and_delivery_files_rehashed_unchanged=len(old),new_archived_files=len(rows),
        archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
        all_archive_members_readback_SHA_verified=True,live_jobs_excluded_from_closed_results=True,
        all_ongoing_runtime_logs_and_sensor_data_excluded=True,goal_complete=False))
    print(json.dumps(dict(old_unchanged=len(old),new_archive_files=len(rows),
        live_jobs=jobs,project_GiB=used/2**30,goal_complete=False),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
