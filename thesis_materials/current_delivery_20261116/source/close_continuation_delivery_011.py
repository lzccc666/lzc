"""封存真实人工记录、008全量终止证据和新扩展源码；不封存活日志为完整回合。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zlib

from conceptgraph.completion.stage9 import PROJECT,digest,dump

R=Path(__file__).resolve().parents[1]


def old_sources():
    old={x['path']:x['sha256'] for x in json.loads((R/'source_bundle/reproduction_003/source_manifest.json').read_text())}
    old.update({x['path']:x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_001/receipt.json').read_text())['supplemental_files']})
    old.update({str(PROJECT/p):h for p,h in json.loads((R/'source_bundle/closing_supplement_002/files.json').read_text()).items()})
    for version in ['003','004','005','006','007']:
        old.update({x['path']:x['sha256'] for x in json.loads((R/f'source_bundle/closing_supplement_{version}/files.json').read_text())})
    return old


def live_jobs():
    rows=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if len(argv)<2 or 'python' not in Path(argv[0].decode(errors='replace')).name:
                continue
            command=' '.join(x.decode(errors='replace') for x in argv)
            if not any(x in command for x in ['run_official_expansion_019.py','run_carla_expansion_batch_011.py','carla_expansion_episode_010.py']):continue
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':continue
            rows.append(dict(PID=int(p.name),start_tick=fields[19],state=fields[0],command=command))
        except (OSError,ValueError):pass
    return rows


def main():
    old=old_sources();assert all(digest(p)==h for p,h in old.items())
    out=R/'source_bundle/closing_supplement_008';out.mkdir(exist_ok=False)
    qa=R/'json/software_validation_continuation_010';qa.mkdir(exist_ok=False)
    sources=sorted((R/'source').glob('*.py'))
    for p in sources:ast.parse(p.read_text())
    commands=[('new_tests',['/root/miniconda3/envs/conceptgraph/bin/python','-m','unittest','discover','-s',str(R/'source'),'-p','test_carla_expansion*.py','-v']),
        ('ruff_functional',['/root/miniconda3/envs/conceptgraph/bin/ruff','check','--no-cache','--select','E9,F63,F7,F82',str(R/'source')]),('git_diff_check',['git','diff','--check'])]
    checks=[]
    for name,command in commands:
        result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
        log=qa/(name+'.log');log.open('x').write(result.stdout+result.stderr)
        checks.append(dict(check=name,exit_code=result.returncode,log_sha256=digest(log)))
        assert result.returncode==0,(name,result.stdout,result.stderr)
    dump(qa/'report.json',dict(syntax_checked_files=len(sources),checks=checks,new_tests_passed=7,
        old_source_and_delivery_files_rehashed_unchanged=len(old),original_business_sources_unchanged=True,
        original_452_business_tests_not_repeated=True,goal_complete=False))
    human=json.loads((R/'annotation/human_attribute_reference_003/report.json').read_text())
    comparison=json.loads((R/'carla_official/guarded_comparison_summary_010/report.json').read_text())
    preflight=json.loads((R/'carla_official/expansion_preflight_readback_012/report.json').read_text())
    jobs=live_jobs();episode=R/'carla_official/expansion_navigation_010/town10_spawn001_ClearNoon_car20/seed17/adapter/full_rule_geometry'
    status=dict(complete=(episode/'summary.json').exists())
    execution=episode/'execution.jsonl.gz'
    if execution.exists():
        raw=execution.read_bytes();decoded=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(raw)
        rows=[json.loads(line) for line in decoded.splitlines(keepends=True) if line.endswith(b'\n')]
        if rows:status.update(observed_prefix_ticks=len(rows),last_sim_s=rows[-1]['state']['time_s'],last_evaluation=rows[-1]['evaluation'],live_prefix_is_not_complete_episode=not status['complete'])
    size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    handoff=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn_classification='progress：导入真实用户38次记录/20张，原文件保留；用户明确更正公交车1项',
        this_goal_turn_classification='progress：008三方法完整回读、真实75布局重检及5445参考运动状态复核、新起点实际感知执行已启动',
        human_review=dict(unique_images=20,raw_records=38,known_single_color_images=16,multiple_color_images=2,unknown_color_images=2,fine_body_type_images=5,
            human_3D_identity_records=0,human_natural_instructions=0,reference='annotation/human_attribute_reference_003/report.json'),
        comparison008=dict(actual_physics_ticks=comparison['actual_physics_ticks'],actual_frontend_calls=comparison['actual_frontend_calls'],actual_GPU_calls=86,actual_episodes=3,strict_arrivals=0,scope='一个开发配置/任务/种子，不能视为大规模泛化',report='carla_official/guarded_comparison_summary_010/report.json'),
        future_sensor_bulk_authorized=True,storage_authorization='json/sensor_bulk_authorization_003.json',
        failed_setup009=dict(actual_perception_frames=0,actual_Qwen_calls=0,reason='无关路外干扰车生成碰撞；错误全部保留'),
        invalid_preflight010=dict(spawn_API_passed=75,independent_truth_credit=0,reason='spawn后未tick，Actor.get_transform读到缓存零坐标；原42项真值通过全部撤销'),
        corrected_preflight011=dict(cases=75,positive=45,no_match=15,blocked=15,synchronized_actor_poses=180,vehicle_reference_states_recomputed=preflight['positive_reference_states_reaudited'],actual_navigation_arrival_credit=0,report='carla_official/expansion_preflight_readback_012/report.json'),
        expansion_registration='carla_official/expansion_registration_011.json',immutable_navigation_registration='carla_official/expansion_registration_010.json',
        expanded_tasks=dict(configurations=10,tasks=50,seeds=[17,29,43],primary_episodes=150,additional_method_episodes=60,additional_ablation_episodes=120,
            primary_queued=True,additional_phases_not_launched=True,short_term_expiry_ablation_not_whole_history_removal=True),
        current_live_jobs=jobs,official_launcher_session=43408,first_new_episode=status,
        project_artifact_GiB=size/2**30,project_budget_GiB=25,physical_free_GiB=shutil.disk_usage(R).free/2**30,
        source_archive='source_bundle/closing_supplement_008',module_index='source_bundle/module_index_009/modules.csv',
        next_actions=['继续轮询同一112329/113411/实际episode PID及session43408；不因观察超时重启',
            '每回合结束会实际运行expansion009与motion002独立审计；普通超时/碰撞等负结果保留后继续，执行/审计错误停下诊断',
            '逐项核验最少5配置10个不同正任务安全到达；不能把几何预检或已选对/锁定算到达',
            '主系统150完成后还需配对方法60与模块消融120；当前未自动启动额外阶段',
            '新封存至少20场景400有效查询、真人歧义实例核验、完整历史维护消融、总表统计及最终交付仍未完成'],
        whole_project_sealed_test_opened=False,thesis_prose_written=False,git_committed_or_pushed=False,goal_complete=False)
    handoffpath=R/'json/continuation_handoff_013.json';dump(handoffpath,handoff)
    doc=R/'交付补充说明009.md'
    doc.open('x').write('''# 实验续跑交付补充009

真实用户审核已入库：38次保存覆盖20张/20场景，保留18次修订。用户在聊天明确将scene-0014的卡车更正为公交车，更正单独保存，原文件字节未改。16张有明确单一颜色、2张多色、2张未知；5张有细分车型。17张主类别可比较且与检测类别一致，另1张厢式货车保留other等待分类口径统一、2张看不清，不把这20张偏置开发样本写成整体准确率。源证据color=unknown是待标注字段，不是模型颜色预测，不计算伪颜色准确率。本批不提供三维跨帧身份或自然指令标注。原始文件及逐图参考在annotation/offline_imports/user_lzc_20261116_001/和annotation/human_attribute_reference_003/。

008强规则、基础Qwen、Adapter三方法全部终止，且每物理时刻LiDAR、名义保护与下一真实命令独立回读完成。共1519物理时刻、761次前端、86次真实Qwen调用。三项均超时、实际到达0；规则选对/锁定过目标，Qwen两项未正确锁定。基础格式0/45，Adapter为38/41，格式改善不等于目标选择或实际到达改善。完整表carla_official/guarded_comparison_summary_010/methods.csv，不能用原地无碰撞声称安全成功。

用户已授权未来大体积传感器文件放共享卷。新RGB/原始深度/LiDAR/掩膜进入/autodl-fs/data/thesis_completion_v1_bulk/carla_sensor_expansion_009/runtime010/，项目保留逐项目录入口、SHA256和结果；旧文件未移动或删除。执行/感知日志采用逐记录flush的gzip，逐字节无损读回测试通过，仍计入25GiB项目预算。

扩展预注册10配置、50任务：30正目标、10无匹配、10独立阻断，每任务3个固定偏移种子。配置为5个官方起点×2种天气，地图和相近道路共享，种子及天气不冒充独立城市。主系统完整合作模式150回合，配对额外方法60回合和额外消融120回合。simple任务走规则分支，只有复杂关系任务调用真实Adapter；不能把简单成功归因微调。无关系消融保留节点坐标，方法仍可能推断几何；短期过期消融只禁用TTL，不能冒充去掉全部历史维护。详见carla_official/expansion_registration_011.json及其明确绑定的导航配置010。

009首先因无关路外干扰车在生成时碰撞，尚无图像/Qwen，错误保留。010预检虽75项官方生成成功，但在spawn后未推进tick，Actor.get_transform返回缓存零坐标；42项真值通过全部撤销。011先推进tick并核验同步snapshot与请求姿态，再全量重新执行75布局。45正目标都有构造的车辆运动参考，15无匹配核对官方静态Bus标签与受控actor，15阻断有车道连通分量断开证明。再次从保存记录独立复核180个actor姿态及5445个参考运动状态，全部通过。这只是有效任务设置与可达/无匹配/阻断证明，不给任何实际到达信用。见carla_official/expansion_preflight_readback_012/。

新起点town10_spawn001_ClearNoon_car20的真实感知导航已启动，正在原厂Prius上执行；以实际summary及两份独立审计为准。不要重启已确认运行的句柄。源码入口run_official_expansion_019.py→run_carla_expansion_batch_011.py→carla_expansion_episode_010.py；当前只启动主系统150，额外方法与消融阶段尚未启动。新的共同开发修复含道路形状车辆积分、障碍并集复核、停稳后距感知目标仍超过1m可继续接近、场景生成/车道任务范围修正与1800s墙钟预算；不能把相对008的多项变化归因于单个模块。所有新方法对照均使用同一冻结设置。

继续读取json/continuation_handoff_013.json，源码/闭合证据归档source_bundle/closing_supplement_008/，模块位置索引source_bundle/module_index_009/modules.csv。新封存至少20场景400有效查询、歧义实例人工核验、完整配对闭环对照与消融、至少5配置10不同正任务实际安全到达、最终图表统计及交付仍未完成。未写论文正文、未提交/推送Git，goal_complete=false。
''')
    index=R/'source_bundle/module_index_009';index.mkdir(exist_ok=False)
    modules=list(csv.DictReader((R/'source_bundle/module_index_008/modules.csv').open()))
    additions=[('真实人工属性评价参考','integrate_human_attribute_review_003.py','main','annotation/human_attribute_reference_003/report.json','20样本真实属性，非跨帧身份或新封存集'),
        ('008三方法真实终止对照','summarize_guarded_comparison_010.py','main','carla_official/guarded_comparison_summary_010/report.json','3回合均未到达；格式率与任务率分开'),
        ('可配置的证据授权门闸','carla_real_graph_bridge_003.py','RealGraphNavigationBridge','json/software_validation_continuation_010/report.json','证据消融真实改变执行逻辑；不移除整车保护'),
        ('当前状态车道候选积分','carla_lane_seed_bridge_006.py','LaneSeedBridge','json/software_validation_continuation_010/report.json','不把道路折线当车辆路线；先积分及原逐障碍扫掠，失败转原Hybrid'),
        ('官方布局同步位姿预检','preflight_carla_expansion_011.py','main','carla_official/expansion_preflight_readback_012/report.json','75布局/180姿态/45参考路线；无实际到达信用'),
        ('官方参考路线再次独立回读','audit_carla_preflight_012.py','main','carla_official/expansion_preflight_readback_012/report.json','重算5445参考运动状态与阻断/无匹配证书'),
        ('新扩展实际传感器与控制执行','carla_expansion_episode_010.py','main','json/continuation_handoff_013.json','实际运行中的完整链；全部330规划回合尚未完成'),
        ('新扩展主系统任务调度','run_carla_expansion_batch_011.py','main','json/continuation_handoff_013.json','主系统150；额外方法/消融阶段未启动'),
        ('新回合独立到达与一对一身份','audit_carla_expansion_009.py','audit','json/software_validation_continuation_010/report.json','读取实际轮角、位姿、终区、保持与单一身份匹配；闭合回合后运行'),
        ('共享卷LiDAR逐物理时刻复核','audit_carla_guarded_execution_002.py','audit','json/software_validation_continuation_010/report.json','校验授权共享路径及每帧原始LiDAR，不把名义模型当物理安全证明')]
    for title,filename,entry,evidence,boundary in additions:
        p=R/'source'/filename;n=next(n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==entry)
        modules.append(dict(module=title,source=str(p),entry=entry,line=n.lineno,meaning=boundary,evidence=str(R/evidence),evidence_exists_at_index_generation=(R/evidence).exists(),boundary=boundary,source_sha256=digest(p)))
    with (index/'modules.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(modules[0]));w.writeheader();w.writerows(modules)
    (index/'README.md').open('x').write('009最新模块入口；软件通过、任务几何证明与实际导航完成分开。冻结的失败及被撤销预检都保留。\n')
    paths=[p for p in sources if str(p) not in old]+[doc,handoffpath,index/'modules.csv',index/'README.md']+list(qa.glob('*'))
    for root in ['annotation/offline_imports/user_lzc_20261116_001','annotation/human_attribute_reference_003','carla_official/guarded_comparison_summary_010','carla_official/expansion_spawn_preflight_010','carla_official/expansion_spawn_preflight_011','carla_official/expansion_preflight_readback_012']:
        paths.extend(p for p in (R/root).rglob('*') if p.is_file())
    for relative in ['carla_official/expansion_registration_009.json','carla_official/expansion_registration_010.json','carla_official/expansion_registration_011.json','json/sensor_bulk_authorization_003.json','storage_audit/full_carla_storage_projection_004.json','json/runtime_derivation_009.json','json/expansion_software_validation_009/report.json','json/guarded_motion_audit_jobs_001/report.json']:
        paths.append(R/relative)
    for mode in ['rule','base','adapter']:
        folder=R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20'/mode
        paths += [folder/name for name in ['independent_motion_guard_audit_001/report.json','independent_choice_audit_008/report.json','independent_audit_003/report.json','summary.json','semantic_queue_final.json']]
    members=[dict(path=str(p),archive_path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths)) if str(p) not in old]
    archive=out/'source_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for x in members:tar.add(x['path'],arcname=x['archive_path'],recursive=False)
    with tarfile.open(archive) as tar:
        for x in members:assert hashlib.sha256(tar.extractfile(x['archive_path']).read()).hexdigest()==x['sha256']
    dump(out/'files.json',members)
    dump(out/'receipt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_source_and_delivery_files_rehashed_unchanged=len(old),archived_new_files=len(members),archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,all_members_reopened_SHA_verified=True,live_episode_logs_not_archived_as_complete=True,goal_complete=False))
    print(json.dumps(dict(old_unchanged=len(old),new_archived=len(members),live_jobs=jobs,first_new_episode=status,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
