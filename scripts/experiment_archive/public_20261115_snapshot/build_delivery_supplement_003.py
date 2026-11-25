"""关闭本轮实际作业后，新增交付说明、索引和只含源码/配置的补充包。"""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import shutil

R = Path(__file__).resolve().parents[1]
PROJECT = R.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert PROJECT.name == 'lizichong'
    policy=json.loads((R/'json/resource_policy_002.json').read_text())
    used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    assert used+64*2**20<policy['new_project_artifact_budget_bytes'] and shutil.disk_usage(R).free>=20*2**30
    shutdown = R/'json/official_stock_navigation_execution_006/server_shutdown.json'
    assert shutdown.exists(), '先让当前正式物理作业关闭，不封存活动记录'
    assert (R/'json/official_prius_braking_execution_007/server_shutdown.json').exists(), '停车修正标定仍在运行'
    assert (R/'json/official_prius_braking_execution_008/server_shutdown.json').exists(), '缓变停车修正标定仍在运行'
    assert (R/'json/official_prius_braking_execution_009/server_shutdown.json').exists(), '低速停车门槛官方标定仍在运行'
    gatedfile=R/'carla_official/prius_braking_pilot_007/report.json'
    gated=json.loads(gatedfile.read_text()) if gatedfile.exists() else {}
    gated_summary='；'.join(f"{x['case']}：通过={x['combined_calibration_pass']}，最大加速度{x['maximum_acceleration_mps2']:.3f}，最小加速度{x['minimum_acceleration_mps2']:.3f}，末速度{x['final_speed_mps']:.4f}" for x in gated.get('results',[])) or '低速门槛标定未完整完成，见失败回执'
    smoothfile=R/'carla_official/prius_braking_pilot_006/report.json'
    smooth=json.loads(smoothfile.read_text()) if smoothfile.exists() else {}
    smooth_summary='；'.join(f"{x['case']}：通过={x['combined_calibration_pass']}，最大加速度{x['maximum_acceleration_mps2']:.3f}，最小加速度{x['minimum_acceleration_mps2']:.3f}，末速度{x['final_speed_mps']:.4f}" for x in smooth.get('results',[])) or '缓变制动标定未完整完成，见失败回执'
    brakingfile=R/'carla_official/prius_braking_pilot_005/report.json'
    braking=json.loads(brakingfile.read_text()) if brakingfile.exists() else {}
    braking_summary='；'.join(f"{x['case']}：通过={x['combined_calibration_pass']}，最大加速度{x['maximum_acceleration_mps2']:.3f}，末速度{x['final_speed_mps']:.4f}" for x in braking.get('results',[])) or '标定未完整完成，见启动/失败回执，不授予通过'
    semantic = json.loads((R/'nuscenes_ablation/full_semantics_near6_002/run_report.json').read_text())
    assert semantic['actual_CPU_cooperation_decisions'] == 9048
    pilot = json.loads((R/'carla_official/stock_vehicle_feedback_pilot_004/report.json').read_text())
    cpu = json.loads((R/'reproduction/clean_cpu_chain_002/repeatability_audit_001.json').read_text())
    navfile = R/'json/stock_navigation_jobs_004/report.json'
    navigation = json.loads(navfile.read_text()) if navfile.exists() else {}
    semrows = list(csv.DictReader((R/'nuscenes_ablation/full_semantics_near6_002/metrics.csv').open()))
    idx = {(x['condition'],x['role'],x['scope'],x['metric']):x for x in semrows}
    semtable = []
    labels = dict(full='完整',without_depth_quality='去深度质量',without_one_to_one='去一对一关联',
                  without_dynamic_maintenance='去动态维护',without_short_term_expiry='去短期过期',without_relations='去关系边')
    for condition, label in labels.items():
        values = [idx[condition,'all',scope,key] for scope,key in [
            ('ordinary_positive','correct'),('ordinary_positive','strong_rule_correct'),
            ('ordinary_positive','covered'),('ordinary_no_match','no_match_false_proposal')]]
        semtable.append('|'+label+'|'+'|'.join(f"{x['numerator']}/{x['denominator']}" for x in values)+'|')
    navlines = []
    for job in navigation.get('jobs',[]):
        path = R/'carla_official/stock_controller_navigation_004'/job['episode']/'full_rule_geometry'
        summary = json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else {}
        audit = json.loads((path/'independent_audit_003/report.json').read_text()) if (path/'independent_audit_003/report.json').exists() else {}
        navlines.append('|'+job['episode']+'|'+str(summary.get('status','无完整summary'))+'|'+str(summary.get('correctly_locked','待核验'))+'|'+str(audit.get('strict_safe_arrival','无独立完整审计'))+'|')
    doc = f'''# 实验完善交付补充005

本文件更新交付说明.md及补充004的状态，保留旧文件以便追溯。整体仍为 **goal_complete=false**。本轮没有撰写论文正文、提交Git或推送仓库，也没有删除或移动历史文件。

## 本轮新增的实际工作

### 1. 修正后的真实图与全量语义

`nuscenes_ablation/graph_near6_correction_001/`将实际near半径从8米改为协议中的6米。对象、候选、任务和标签不变。全量关系F1由旧0.183844变为0.185721；这是开发集配置修正的效果，不能当作新的独立封存结论。

`nuscenes_ablation/full_semantics_near6_002/`在新关系输入上实际重新运行：6条件×1508条=9048次合作判断，实际Qwen调用{semantic['actual_new_GPU_requests']}次，运行{semantic['wall_s']:.1f}秒。不是缓存复打分、不是每帧都调用模型，也不是完整实时系统测速。原near8结果完整保留。

|条件|合作选对正目标|同输入强几何规则选对|候选包含正确目标|无匹配误选|
|---|---:|---:|---:|---:|
{chr(10).join(semtable)}

全体38开发场景/11日志包含训练、验证日志。日志分离的开发留出仅2日志、79请求，其中34正目标；也已用于开发，不能称最终未使用测试。单个事前固定Adapter seed17。缺失正确候选的正查询仍算失败。任务来自官方对象/几何和受控语言、故障，不是人工自然语言标签。无匹配和故障拒答不能计为“找对目标”。完整合作普通正目标110/627，仍低于强几何规则133/627；去一对一111/627略好于完整。负结果和零收益均保留。

`raw_readback_audit_001.json`逐条核对新旧9048原始记录，任务字段、候选覆盖及调用路由保持一致；同一prompt的实际模型原文差异为0。`metrics.csv`、`paired_differences.csv`、`detail_statistics_001/`分别提供分层分母、日志聚类CI、配对差异和真实调用延迟。受控复杂表达的歧义问题见旧任务范围audit002，未按结果重新改标签。

### 2. 官方原厂车辆控制与感知导航

`carla_official/stock_vehicle_feedback_pilot_004/`实际4组标定、2040记录tick，另有40预热tick。Lincoln两组均通过；Prius两组虽不超加减速限值，但10秒停车阶段结束速度仍约0.14米/秒，功能未通过。两车原厂扭矩均未修改。`stock_navigation_preparation_004/independent_pilot_readback_001.json`独立重算4条原始轨迹及轮角，确认结论。

`carla_official/stock_controller_navigation_004/`固定Lincoln和pilot004反馈器，重跑旧的两个有目标任务及一个无匹配任务。检测、分割、CLIP、LiDAR投影、对象关联、关系、规则选择、锁定、Hybrid A*与官方控制实际相连，RGB/深度/LiDAR/位姿/碰撞原始证据逐帧保存。仍使用相同90秒/30秒任务上限及原来的身份、终区、轮角、速度、加减速和停车保持标准，没有改扭矩或放宽到达判定。

|任务|本次终止状态|是否曾正确锁定|独立严格安全到达|
|---|---|---|---|
{chr(10).join(navlines) if navlines else '|运行未全部关闭|见启动失败回执|不授予|不授予|'}

启动和关闭见`json/official_stock_navigation_execution_006/`，逐例审计在各episode的`independent_audit_003/report.json`。本组仍是同Town/起点的开发功能验证，Qwen没有激活，不是复杂语言闭环、未新增独立道路、未满足原课题10配置/50任务×3种子。车辆与控制器同时变化，因此不能把旧Tesla和新Lincoln分数差归因于单个控制模块。

随后针对Prius停车失败，`prius_braking_diagnosis_001/report.json`核对末20帧：实际油门/刹车均0，但速度仍约0.14，低于0.08才刹车的门槛未触发。没有据此虚构底层动力学原因。新`prius_braking_pilot_005/`仅在停车、油门归零时增加小幅速度比例制动，不改物理和验收限值；两条相同外生转向真实标定结果：{braking_summary}。上述005虽停稳，但两组各出现一次减速超限，原始负结果全部保留。`prius_braking_diagnosis_002/report.json`记录超限点及实际刹车0.08；另立006将比例制动降为0.02*speed、上限0.012并限制增量0.0004/tick，仍按同一硬限制标定。006实际结果：{smooth_summary}。006的很小制动仍出现相同骤停；`prius_braking_diagnosis_003/report.json`对比原始命令与速度，不虚构内部物理原因。007恢复原滑行控制，仅将停车刹车触发从速度0.08改到0.15（最终停车验收仍0.08、减速上限仍2）；实际结果：{gated_summary}。`all_pilot_readback_002/report.json`重新读取4个版本共10条轨迹、5100记录tick，还用冻结控制类AST及每条实际观测重新计算全部油门/刹车命令，误差低于1e-6，实际命令、速度、轮角、扭矩和限值全部核对。共4/10组通过（Lincoln两组、最新Prius007两组），失败全留。这些只是后续控制开发证据，还没有用新停车控制完整重跑目标感知导航，不能把它算到达。

旧Tesla反馈标定负结果也保留：原控制2/2、反馈1/2。旧15次CARLA启动仅14完整，完整条件严格安全到达1/2；最后一个无匹配条件被旧10GiB预算中断，不能当成功或从分母消失。旧标定与新标定都不能当导航到达。

### 3. 新环境实际安装与真实CPU复现

`reproduction/clean_cpu_install_001/venv/`是实际新建、没有继承旧site-packages的CPU环境。依赖安装、pip check及模块路径检查真实通过；gradslam、CLIP、SAM由可追溯纯源码构建，CARLA API来自固定官方wheel。gradslam仅恢复了已有纯Python源码，不能声称重新校验了上游Git提交；底层Linux和Python解释器仍共享，CPU包版本也与GPU环境不同。

`clean_cpu_chain_001/`、`clean_cpu_chain_002/`实际两次运行YOLO、分割、CLIP、3次建图、3次关系更新和3次CPU Qwen生成。每次检测数3/2/0，空检测保留。两次共6次前端、6次CPU Qwen调用，独立数据只有1场景3帧，不得算6个新帧或新封存指令。逐元素比较检测框、mask、置信度、CLIP特征和模型原文完全一致，第二次依赖freeze未变，0运行时自动安装。

首次运行缺dill发生动态补装，原日志及来源保留；随后从官方PyPI明确安装dill0.4.1并固定`installed_freeze_004.txt`，第二次关闭自动补装。首次pip23构建失败也保留，不声称安装过程无失败。这补齐了独立依赖中的真实感知到语义功能复现，**尚未完成整个GPU导航/CARLA新环境复现**。

### 4. 图表与人工核验

用户选定Python。正式图以`figures/public_evidence_004/`四张图及`figures/corrected_semantics_001/`一张图为准，提供源CSV、SVG可编辑文字、TrueType PDF、600dpi TIFF、PNG。实际用Python渲染并查看导出PDF，中文、数值和负结果均可见。早先001—003导出/字体问题和失败记录保留，不能拿003的异常PDF作为最终图。

全量新表见`tables/full_semantics_near6_office_002/`，Word两表、Excel六页签和2页PDF，实际检查过全部PDF页面。原14张表、旧near8两张表、新near6两张表按版本分开，不混成18个独立实验。

人工核验入口：`annotation/quick_human_review_002/人工审核_车辆行人20张_打开即用.zip`。下载解压，双击“打开这个文件开始审核.html”，逐张填写类别、颜色、能否看清、观察依据，保存后导出JSON交回。001选到道路红框的问题保留，新002只预先按预测类别选择车辆/行人，不按正确率筛样。这20张仅入门属性核验，不等于全体人工标注、三维身份或自然语言指令核验。实际浏览器检查20题、原图/crop、离线/在线模式和空表拦截通过，0代理POST、0代理代填。真实用户记录必须由`import_quick_human_reviews_002.py import --help`所示接口导入；在收到并核验前相关人工指标仍待核验。

## 主要结果与复现入口

KITTI仍为Tracking 10连续序列×100帧，共1000帧：0000、0002、0003、0004、0005、0006、0007、0008、0009、0010。结果在`mapping_evaluation/kitti_mapping_full_001/`与`statistics/kitti_expanded_full_001/`。完整融合召回0.840958、IDF1约0.503511、中心误差0.947443米，重复节点307。AB3DMOT适配IDF1约0.55785，高于本方法；重复节点与身份方面不能宣称全面优胜。仅离线连续建图/检索诊断，不是KITTI完整实时车辆导航。

nuScenes 38场景1508帧六条件对象统计在`nuscenes_ablation/statistics_001/`；外部AB3DMOT适配统计正确目录是`method_comparison/ab3dmot_statistics_002/`。固定共享输入的简化适配不是作者官方榜单结果。三个新Adapter训练、108条公平GPU对照仍在`qwen_adapter_fair_comparison/`，不能把格式/拒答提升写成正目标提升。

新依赖恢复顺序：阅读`reproduction/clean_cpu_install_001/registration.json`、`vendor_source_manifest_001.json`、`explicit_runtime_dependency_addendum_004.json`、`installed_freeze_004.txt`；入口是`source/install_clean_cpu_environment_001.py`→`resume_clean_cpu_install_002.py`→`complete_clean_cpu_dependencies_003.py`，保留原失败和真实恢复顺序。新CPU链路见`source/clean_cpu_perception_semantics_001.py`和`launch_clean_cpu_chain_002.py`，实际命令记录在安装/运行receipt和log中。

所有正式入口采用固定run-id并拒绝覆盖，**不要直接重复运行现有已完成目录**。在新工作目录恢复434文件主体源码包后叠加closing_supplement_001、002、003；数据和模型按release只读挂接。跨机器需按清单映射绝对输入路径，尚未声称一条命令即可从空机器恢复全部官方数据与GPU闭环。

`source_bundle/module_index_004/`和新增005提供各模块源码、入口行号、作用、证据位置与结论边界。补充源码包不含venv、模型、原始传感器，不替代单独的数据/模型清单。

## 容量、保护与未完成项

旧source/finalize_artifact_receipt_003.py是未执行的历史预备版本，含旧预算/进程引用，不应用于本轮收尾。

用户明确将本轮新项目产物预算从10提高至25GiB，物理空闲底线仍20GiB，见`json/resource_policy_002.json`。旧10GiB守卫与中断记录不改写。官方CARLA安装/缓存仍在已授权共享卷，项目内存结果与入口；本地新CPU环境也计入25GiB。没有删除历史模型、失败记录、数据或论文。

人工颜色/细属性/歧义记录、最终冻结后的新20—30场景≥400有效指令封存测试、复杂Qwen官方CARLA闭环、原课题更广道路/任务/种子到达覆盖、整个新环境GPU导航复现仍未验收。Qwen正目标收益没有证明；不能降低基线、改标签或只挑成功结果。完整系统与这些必要证据未齐，Goal继续保持false。论文正文按用户要求暂不写。
'''
    docpath = R/'交付补充说明005.md'
    with docpath.open('x') as stream: stream.write(doc)
    # Extend the prior index while refreshing evidence existence at closing time.
    indexdir = R/'source_bundle/module_index_005'; indexdir.mkdir(exist_ok=False)
    rows = list(csv.DictReader((R/'source_bundle/module_index_004/modules.csv').open()))
    for filename, name, entry, evidence, boundary in [
        ('carla_stock_navigation_episode_004.py','原厂Lincoln全传感器导航','main','carla_official/stock_controller_navigation_004','仅三项开发任务；Qwen未启用，不算最终封存'),
        ('run_stock_navigation_batch_004.py','三项导航及独立审计','main','json/stock_navigation_jobs_004/report.json','子进程退出0不等于到达；必须读取独立审计'),
        ('run_official_stock_navigation_006.py','本次成功/失败官方启动入口','main','json/official_stock_navigation_execution_006','只关闭自己服务器，失败前缀完整保留'),
        ('audit_full_semantics_near6_readback_001.py','9048新旧语义记录复核','main','nuscenes_ablation/full_semantics_near6_002/raw_readback_audit_001.json','记录分母核验，不是新官方真值或人工标注'),
        ('build_delivery_supplement_003.py','交付与源码补充包','main','source_bundle/closing_supplement_003/receipt.json','补充包不含venv/模型/原始数据；不能代替最终验收'),
        ('carla_prius_braking_pilot_005.py','Prius停车制动首个修正','run','carla_official/prius_braking_pilot_005/report.json','005可停车但2/2各一次减速超限；不得算安全通过'),
        ('carla_prius_braking_pilot_006.py','Prius缓变小制动开发标定','run','carla_official/prius_braking_pilot_006/report.json','006结果单独保留；不是完整感知导航或实际到达'),
        ('run_official_prius_braking_pilot_008.py','缓变制动官方标定启动','main','json/official_prius_braking_execution_008/report.json','只使用自己的官方进程；实际通过仍看物理标定报告'),
        ('carla_prius_braking_pilot_007.py','Prius低速刹车门槛标定','run','carla_official/prius_braking_pilot_007/report.json','标定不授予目标/路线/锁定/实际到达信用'),
        ('run_official_prius_braking_pilot_009.py','低速门槛官方标定启动','main','json/official_prius_braking_execution_009/report.json','保持旧失败；仅结束自己进程组'),
        ('audit_official_pilots_readback_002.py','十条标定逐命令独立复算','main','carla_official/all_pilot_readback_002/report.json','5100记录tick、10条标定，仅4条通过；最新007两条通过，仍无导航到达信用'),
    ]:
        path=R/'source'/filename; line=next(x.lineno for x in ast.walk(ast.parse(path.read_text())) if isinstance(x,ast.FunctionDef) and x.name==entry)
        rows.append(dict(module=name,source=str(path),entry=entry,line=line,meaning=name,evidence=str(R/evidence),evidence_exists_at_index_generation=(R/evidence).exists(),boundary=boundary,source_sha256=sha(path)))
    for row in rows: row['evidence_exists_at_index_generation']=Path(row['evidence']).exists()
    with (indexdir/'modules.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (indexdir/'README.md').write_text('# 交付补充源码索引005\n\n模块位置和入口见modules.csv。与001—004叠加，以新版本为准。实验完成和严格到达须读取evidence，不以源码存在或子进程退出代替。\n')
    known={x['archive_path']:x['sha256'] for x in json.loads((R/'source_bundle/reproduction_003/source_manifest.json').read_text())}
    known.update({x['archive_path']:x['sha256'] for x in json.loads((R/'source_bundle/closing_supplement_001/receipt.json').read_text())['supplemental_files']})
    known.update(json.loads((R/'source_bundle/closing_supplement_002/files.json').read_text()))
    for path, expected in known.items(): assert sha(PROJECT/path)==expected, path
    candidates=set((R/'source').glob('*.py'))|{docpath,R/'json/resource_policy_002.json',R/'carla_official/stock_navigation_registration_004.json'}
    candidates.update(indexdir.iterdir());candidates.update((R/'source_bundle/module_index_004').iterdir())
    for parent in ['reproduction/clean_cpu_install_001','figures/font_runtime_001','figures/public_evidence_004','figures/corrected_semantics_001']:
        folder=R/parent
        candidates.update(folder.glob('*.json'));candidates.update(folder.glob('*.txt'));candidates.update(folder.glob('LICENSE'))
    candidates.add(R/'figures/font_runtime_001/NotoSansSC-Regular.ttf')
    candidates.add(R/'carla_official/prius_braking_pilot_005/registration.json')
    candidates.add(R/'carla_official/prius_braking_pilot_006/registration.json')
    candidates.add(R/'carla_official/prius_braking_pilot_007/registration.json')
    candidates.add(R/'json/goal_acceptance_005.json')
    vendor=R/'reproduction/clean_cpu_install_001/vendor_sources'
    candidates.update(p for p in vendor.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.parts and not any(v.endswith('.egg-info') for v in p.parts) and p.suffix!='.pyc')
    chosen=[]
    for path in sorted(candidates):
        relative=str(path.relative_to(PROJECT)); value=sha(path)
        if relative in known:
            assert known[relative]==value;continue
        chosen.append(dict(path=str(path),archive_path=relative,bytes=path.stat().st_size,sha256=value))
    folder=R/'source_bundle/closing_supplement_003';folder.mkdir(exist_ok=False);archive=folder/'source_delta.tar.gz'
    with tarfile.open(archive,'x:gz') as target:
        for row in chosen:target.add(row['path'],arcname=row['archive_path'],recursive=False)
    with tarfile.open(archive,'r:gz') as check:
        assert len(check.getmembers())==len(chosen)
        for row in chosen:
            with check.extractfile(row['archive_path']) as stream: assert hashlib.sha256(stream.read()).hexdigest()==row['sha256']
    save(folder/'files.json',chosen)
    save(folder/'receipt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_snapshot_files_and_prior_supplements_rechecked=len(known),supplement_files=len(chosen),source_py_files=sum(Path(x['path']).suffix=='.py' for x in chosen),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,all_archive_members_reopened_and_hash_matched=True,excluded=['venv','模型权重','原始传感器和官方数据','可再生pyc/build/egg-info'],navigation_job_report=str(navfile),full_goal_accepted=False,goal_complete=False))
    print(json.dumps(dict(source_supplement_files=len(chosen),archive_MiB=archive.stat().st_size/2**20,old_snapshot_files_verified=len(known),goal_complete=False),ensure_ascii=False))


if __name__=='__main__':
    main()
