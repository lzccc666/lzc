"""第一项新起点严格到达的闭合回执；其余回合保持未完成，不报总体成功率。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from conceptgraph.completion.stage9 import PROJECT,digest,dump
from close_continuation_delivery_011 import old_sources,live_jobs

R=Path(__file__).resolve().parents[1]


def main():
    old=old_sources();old.update({x['path']:x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_008/files.json').read_text())})
    assert all(digest(p)==h for p,h in old.items())
    episode=R/'carla_official/expansion_navigation_010/town10_spawn001_ClearNoon_car20/seed17/adapter/full_rule_geometry'
    summary=json.loads((episode/'summary.json').read_text());arrival=json.loads((episode/'independent_expansion_audit_009/report.json').read_text());guard=json.loads((episode/'independent_motion_guard_audit_002/report.json').read_text())
    assert arrival['strict_safe_arrival'] and arrival['GT_not_sent_to_agent'] and not any(arrival['actuation_limits'].values())
    for p,h in arrival['inputs_sha256'].items():assert digest(p)==h
    for p,h in guard['source_sha256'].items():assert digest(p)==h
    assert guard['physical_ticks']==guard['raw_LiDAR_frames_verified']==531 and guard['actual_stop_commands_verified']==10
    qa=R/'json/software_validation_continuation_011';qa.mkdir(exist_ok=False)
    files=[Path(__file__),R/'source/profile_carla_expansion_live_001.py']
    for p in files:ast.parse(p.read_text())
    result=subprocess.run(['/root/miniconda3/envs/conceptgraph/bin/ruff','check','--no-cache','--select','E9,F63,F7,F82',*map(str,files)],capture_output=True,text=True)
    assert result.returncode==0
    (qa/'ruff.log').open('x').write(result.stdout+result.stderr)
    dump(qa/'report.json',dict(new_sources_ast_checked=2,ruff_exit_code=0,old_source_and_delivery_rehashed_unchanged=len(old),original_business_sources_unchanged=True,original_business_tests_not_repeated=True,goal_complete=False))
    jobs=live_jobs()
    profiler=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if len(argv)>1 and argv[1].decode(errors='replace').endswith('/source/profile_carla_expansion_live_001.py'):
                state=(p/'stat').read_text().rsplit(')',1)[1].split()
                if state[0]!='Z':profiler.append(dict(PID=int(p.name),start_tick=state[19],command=' '.join(x.decode(errors='replace') for x in argv)))
        except OSError:pass
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn_classification='progress：真实20图人工审核及用户更正入库',
        this_goal_turn_classification='progress：008三方法闭合、75同步布局与5445参考状态重算、首个新起点实际安全到达完成两项独立审计',
        prior_snapshot='json/continuation_handoff_013.json',
        first_new_task=dict(episode='town10_spawn001_ClearNoon_car20',seed=17,semantic_mode='adapter',
            query='靠近本车道内最近的小汽车',physical_ticks=531,frontend_calls=266,
            correct_first_proposal=arrival['correct_first_proposal'],correct_lock=arrival['correct_lock'],
            strict_safe_arrival=True,hold_s=arrival['maximum_contiguous_hold_s'],
            max_acceleration_mps2=arrival['maximum_acceleration_mps2'],min_acceleration_mps2=arrival['minimum_acceleration_mps2'],
            wall_s=summary['wall_s'],sim_s=summary['sim_s'],actual_Qwen_calls=arrival['actual_final_GPU_model_calls'],
            clear_rule_branch=True,Qwen_improvement_credit=False,raw_LiDAR_frames_recomputed=531,
            nominal_guard_stop_commands_verified=10,arrival_report=str(episode/'independent_expansion_audit_009/report.json'),
            motion_report=str(episode/'independent_motion_guard_audit_002/report.json')),
        current_primary_closed_episodes=1,planned_primary_episodes=150,remaining_primary_at_snapshot=149,
        no_overall_success_rate_claim=True,minimum_5_config_10_positive_task_goal_not_yet_met=True,
        live_jobs=jobs,live_profiler=profiler,official_launcher_session=43408,profile_session=54577,
        additional_method_and_ablation_phases_not_started=True,
        performance_note='进程VmHWM与GPU占用3s只读采样已启动，首回合结束后才启动，首回合不虚构峰值；GPU采样峰值为下界，不能冒充CUDA精确分配峰值。',
        source_archive='source_bundle/closing_supplement_009',latest_module_index='source_bundle/module_index_010/modules.csv',
        human_review_images=20,human_3D_identity_reviews=0,human_natural_instructions=0,
        project_artifact_GiB=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())/2**30,
        physical_free_GiB=shutil.disk_usage(R).free/2**30,project_budget_GiB=25,
        new_sealed_test_opened=False,thesis_prose_written=False,git_committed_or_pushed=False,goal_complete=False)
    status=R/'json/continuation_handoff_014.json';dump(status,report)
    doc=R/'交付补充说明010.md'
    doc.open('x').write('''# 实验续跑交付补充010

新起点town10_spawn001_ClearNoon_car20已完成真实感知—建图—锁定—规划—执行—到达，共531物理时刻、266次实际前端。独立评价重新核对正确对象、目标区域、朝向、最终速度≤0.08m/s、连续保持2秒、碰撞/车道/实际轮角和控制限值，严格安全到达通过。最大加速0.71368m/s²、最小加速度-0.75608m/s²，各限值违规为0。另一独立审计从531份原始整圈LiDAR重建障碍及两种名义保护预测，10次保护停车命令全部落实到下一实际控制。名义模型本身不是通用物理安全证明。

这是本轮扩展第1个闭合回合，不报总体100%成功；还需继续其余149个主系统回合及60方法、120模块消融。此项为简单类别指令，实际由规则分支完成，Qwen调用0，不能归因Adapter。444.24秒墙钟推进53秒仿真，不能称实时吞吐。当前至少5配置10不同正任务实际到达门槛尚未满足。

逐帧结果：carla_official/expansion_navigation_010/town10_spawn001_ClearNoon_car20/seed17/adapter/full_rule_geometry/。项目内execution.jsonl.gz、perception.jsonl.gz、release.json、evaluation_only/truth.json、sensor_files_manifest.json及两份independent审计完整；frames和guard_lidar为已授权共享卷逐项入口。没有移动或删除旧数据。

当前队列已经继续到town10_spawn002_ClearNoon_car18，轮询同一官方launcher112329、batch113411和实际episode PID；session43408。新只读性能采样器记录进程CPU内存高水位和3秒GPU占用，首回合结束后才启动，首回合峰值保留未采集，GPU样本峰值明确为下界。session54577，performance/carla_expansion_live_profile_001/。

人工审核20图、008三方法全部失败、75个同步布局独立真值证书以及被撤销的010预检详见交付补充说明009.md。当前状态json/continuation_handoff_014.json，最新模块索引source_bundle/module_index_010/modules.csv，首项闭合证据和新增源码归档source_bundle/closing_supplement_009/。完整封存、配对消融、性能统计与交付尚未完成；未撰写论文正文或提交推送Git，goal_complete=false。
''')
    index=R/'source_bundle/module_index_010';index.mkdir(exist_ok=False)
    rows=list(csv.DictReader((R/'source_bundle/module_index_009/modules.csv').open()))
    p=R/'source/profile_carla_expansion_live_001.py';entry=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    rows.append(dict(module='扩展真实进程只读内存采样',source=str(p),entry='main',line=entry.lineno,meaning='每3s采样VmHWM与GPU占用；GPU峰值为采样下界，首回合缺采不补造',evidence=str(R/'performance/carla_expansion_live_profile_001/registration.json'),evidence_exists_at_index_generation=True,boundary='不增加GPU模型/物理调用，不宣称精确CUDA分配峰值',source_sha256=digest(p)))
    with (index/'modules.csv').open('x') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (index/'README.md').open('x').write('010包含新扩展第一项真实安全到达回执和性能采样入口；完整主实验尚在进行。\n')
    paths=[*files,status,doc,index/'modules.csv',index/'README.md',*qa.glob('*'),R/'performance/carla_expansion_live_profile_001/registration.json',R/'json/expansion_jobs_011_primary/job_001.json']
    paths += [episode/name for name in ['summary.json','release.json','sensor_files_manifest.json','evaluation_only/truth.json','semantic_queue_final.json','independent_expansion_audit_009/report.json','independent_expansion_audit_009/frames.jsonl','independent_expansion_audit_009/choices.jsonl','independent_motion_guard_audit_002/report.json','independent_motion_guard_audit_002/ticks.jsonl']]
    out=R/'source_bundle/closing_supplement_009';out.mkdir(exist_ok=False)
    members=[dict(path=str(p),archive_path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    archive=out/'source_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for x in members:tar.add(x['path'],arcname=x['archive_path'],recursive=False)
    with tarfile.open(archive) as tar:
        for x in members:assert hashlib.sha256(tar.extractfile(x['archive_path']).read()).hexdigest()==x['sha256']
    dump(out/'files.json',members);dump(out/'receipt.json',dict(old_source_and_delivery_files_rehashed_unchanged=len(old),archived_new_files=len(members),archive_sha256=digest(archive),all_members_reopened_SHA_verified=True,raw_sensor_files_not_duplicated_or_moved=True,goal_complete=False))
    print(json.dumps(dict(strict_safe_arrival=True,old_unchanged=len(old),archived_new_files=len(members),live_jobs=jobs,profiler=profiler,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
