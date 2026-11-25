"""交接索引：说明实际执行入口及证据边界，不把源码存在当作实验完成。"""
import ast
import csv
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
O = R / 'source_bundle/module_index_004'
ITEMS = [
    ('关系距离修正', 'nuscenes_graph_near6_correction_001.py', 'main', '重放真实对象图，仅将实际near距离8米修正为6米', 'nuscenes_ablation/graph_near6_correction_001/report.json', '开发对象不变；没有新增检测或导航执行'),
    ('修正后的全量真实语义', 'nuscenes_full_semantics_near6_002.py', 'run', '6条件×1508条，按冻结合作路由实际调用Qwen', 'nuscenes_ablation/full_semantics_near6_002/run_report.json', '38开发场景包含训练验证日志；没有封存、锁定或到达信用'),
    ('全量语义分层统计', 'summarize_full_semantics_near6_details_002.py', 'main', '分开正目标、无匹配、故障拒答、路由和真实生成耗时', 'nuscenes_ablation/full_semantics_near6_002/detail_statistics_001/report.json', '模型格式率分母只用实际调用，不能混成合作总分'),
    ('全量可编辑表', 'export_full_semantics_near6_office_002.py', 'main', '全开发/开发留出两张Word表及六页签Excel', 'tables/full_semantics_near6_office_002/receipt.json', '实际结果表，不是论文正文'),
    ('开发证据科研图', 'plot_public_evidence_004.py', 'main', '4张图，SVG/PDF/600dpiTIFF/PNG与源CSV', 'figures/public_evidence_004/visual_qa_001.json', '001—003是保留的导出失败或修复版本，以004为准'),
    ('修正后语义科研图', 'plot_corrected_semantics_001.py', 'main', '分别画合作选对、几何规则、候选覆盖和无匹配误选', 'json/near6_export_completion_001/visual_qa_001.json', '开发诊断，正目标率与故障拒答不同，不预设完整方法胜出'),
    ('原反馈控制标定', 'carla_acceleration_feedback_pilot_003.py', 'run', '官方CARLA同Tesla比较旧控制与反馈控制', 'carla_official/acceleration_feedback_pilot_003/report.json', '反馈控制1/2通过，旧控制2/2；负结果保留，不能算导航'),
    ('旧标定的成功启动器', 'run_official_actuation_pilot_004.py', 'main', '实际启动官方服务、等待就绪并只关闭自己进程组', 'json/official_actuation_pilot_execution_004/report.json', '001—003启动失败保留；不重启其他GPU作业'),
    ('原厂车辆控制标定', 'carla_stock_vehicle_feedback_pilot_004.py', 'run', 'Lincoln/Prius原厂物理、同一反馈器和两组冻结转向', 'carla_official/stock_vehicle_feedback_pilot_004/report.json', '看report判断是否通过；只有位姿速度轮角碰撞，不是完整感知导航'),
    ('原厂标定启动器', 'run_official_stock_vehicle_pilot_005.py', 'main', '等待全量GPU和统计终止后，串行使用官方服务器', 'json/official_stock_pilot_execution_005/report.json', '如未有report则仍在运行或失败，不能预授予完成'),
    ('用户入门属性核验包', 'build_quick_human_review_002.py', 'main', '20张车辆/行人原图、红框及真实crop，离线导出人工JSON', 'annotation/quick_human_review_002/manifest.json', '代理不代填；仅属性核验，不是三维身份或自然指令审核'),
    ('核验包浏览器检查', 'qa_quick_human_review_browser_003.py', 'main', '实际浏览器检查20题导航、原图和crop、空表拦截', 'annotation/quick_human_review_002/browser_qa_001/report.json', '0条人工记录、0次POST；UI通过不等于完成标注'),
    ('真实用户记录导入', 'import_quick_human_reviews_002.py', 'main', '核验用户导出JSON的来源、保留原文和修订记录', 'annotation/quick_human_review_002/source_binding_audit_001.json', '当前仅audit-source通过；须收到真实记录才可执行import'),
    ('隔离人工服务', 'review_server_isolated_002.py', 'main', '只写新目录，修复null身份字段并保留属性审核标记', 'annotation/review_service_upgrade_002.json', '在线服务运行不等于用户已提交'),
    ('新环境首次安装', 'install_clean_cpu_environment_001.py', 'main', '新建不继承旧site-packages的CPU venv、官方来源安装', 'reproduction/clean_cpu_install_001/installation_report_002.json', '首次pip23构建失败保留；实际恢复入口002'),
    ('新环境安装恢复', 'resume_clean_cpu_install_002.py', 'main', 'pip25.3、CPU torch、冻结依赖，保留首次失败', 'reproduction/clean_cpu_install_001/installation_report_002.json', '共享底层Linux和Python解释器，不是独立OS或CUDA环境'),
    ('新环境本地源码依赖', 'complete_clean_cpu_dependencies_003.py', 'main', '源码构建gradslam/CLIP/SAM并安装官方CARLA API', 'reproduction/clean_cpu_install_001/dependencies_report_003.json', 'gradslam从纯Python已安装源码恢复，未声称验证上游commit'),
    ('新环境真实CPU链路', 'clean_cpu_perception_semantics_001.py', 'main', '实际YOLO/SAM/CLIP→建图/关系→CPU Qwen', 'reproduction/clean_cpu_chain_002/report.json', '仅1开发场景3帧3条固定指令，0新增GPU、0物理导航'),
    ('固定依赖第二次验证', 'launch_clean_cpu_chain_002.py', 'main', '关闭运行时补装、固定dill依赖后第二次真实重跑', 'reproduction/clean_cpu_chain_002/repeatability_audit_001.json', '两次输入输出相同，只算3个独立帧；不算整套导航复现'),
]


def main():
    O.mkdir(exist_ok=False)
    rows = []
    for name, filename, entry, meaning, evidence, boundary in ITEMS:
        path = R / 'source' / filename
        tree = ast.parse(path.read_text())
        found = [x.lineno for x in ast.walk(tree)
                 if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == entry]
        # Some preserved scripts expose an argument dispatcher rather than main.
        line = found[0] if found else 1
        rows.append(dict(module=name, source=str(path), entry=entry, line=line,
                         meaning=meaning, evidence=str(R/evidence),
                         evidence_exists_at_index_generation=(R/evidence).exists(),
                         boundary=boundary, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (O/'modules.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    text = '# 实验完善补充源码索引004\n\n'
    text += '供交接阅读，不是论文正文。与module_index_001/002/003叠加使用，以本索引指向的新版本为准。\n\n'
    text += '源码在，不代表实验完成；请检查每项evidence对应report。属性核验页和浏览器UI通过，不代表真实人工提交。\n\n'
    for row in rows:
        text += f"- **{row['module']}**：[{Path(row['source']).name}]({row['source']}:{row['line']})。{row['meaning']}。边界：{row['boundary']}。\n"
    (O/'README.md').write_text(text)
    print(json.dumps(dict(entries=len(rows), goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
