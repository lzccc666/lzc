"""按完整目标逐项记录验收边界；开发结果不能替代封存、人工标签或物理到达。"""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
from xml.etree import ElementTree as ET

R=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert (R/'json/official_stock_navigation_execution_006/server_shutdown.json').exists()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8877/api/review_status',timeout=10) as response:
        human=json.loads(response.read())
    tests=ET.parse(R/'json/software_validation_001/junit.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
    test_counts={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','errors','failures','skipped']}
    code_manifest=R/'source_bundle/reproduction_001/source_manifest.json'
    code=json.loads(code_manifest.read_text());assert all(sha(x['path'])==x['sha256'] for x in code)
    gates=[]

    def gate(name,status,evidence,remaining):
        files=[]
        for relative in evidence:
            p=R/relative
            files.append(dict(path=str(p),exists=p.is_file(),sha256=sha(p) if p.is_file() else None))
        if status == '开发范围已完成': assert files and all(x['exists'] for x in files), (name, files)
        gates.append(dict(requirement=name,status=status,evidence=files,remaining=remaining))

    gate('KITTI连续序列与逐帧建图','开发范围已完成',[
        'statistics/kitti_expanded_full_001/report.json',
        'mapping_evaluation/kitti_gpu_frontend_full_001/summary.json'],
        '10序列1000连续帧；不授予KITTI实时导航或实际到达。3帧按精确指纹复用须披露。')
    gate('nuScenes六条件严格模块消融','开发范围已完成',[
        'nuscenes_ablation/statistics_001/metrics.csv',
        'nuscenes_ablation/graph_near6_correction_001/report.json',
        'nuscenes_ablation/full_semantics_near6_002/run_report.json',
        'nuscenes_ablation/full_semantics_near6_002/raw_readback_audit_001.json'],
        '38场景11日志1508帧，含训练验证日志；与最终未使用测试必须分开。部分模块负收益或零收益保留。')
    gate('与外部方法及内部结构比较','开发范围已完成',[
        'method_comparison/ab3dmot_shared_full_001/summary.json',
        'method_comparison/ab3dmot_statistics_002/report.json'],
        'AB3DMOT固定共享输入适配，不是作者官方榜单；KITTI IDF1与重复节点未全面优于对照。')
    gate('独立新Adapter三种子训练和公平比较','开发范围已完成',[
        'qwen_adapter_fair_comparison/real_graph_fair_gpu_001/base/report.json',
        'qwen_adapter_fair_comparison/real_graph_fair_gpu_001/real_graph_seed17/report.json',
        'qwen_adapter_fair_comparison/real_graph_fair_gpu_001/real_graph_seed29/report.json',
        'qwen_adapter_fair_comparison/real_graph_fair_gpu_001/real_graph_seed43/report.json'],
        '相同108开发请求的实际GPU对照已做；正目标收益未证实，不通过降低基线制造提升。缺全量最终封存语义确认。')
    gate('实际语义模块消融','开发范围已完成',[
        'nuscenes_ablation/complex_semantics_development_001/report.json',
        'nuscenes_ablation/full_semantics_near6_002/detail_statistics_001/report.json'],
        '普通正目标、无匹配和故障拒答分别报告；颜色/细属性标签及复杂语言闭环尚未完成。')
    gate('人工颜色细属性和歧义核验','未完成',[
        'annotation/quick_human_review_002/source_binding_audit_001.json',
        'annotation/quick_human_review_002/browser_qa_001/report.json'],
        f'实际在线计数{human}；UI和来源检查不等于真实人工记录；入门20图也不能代替三维身份/自然指令核验。')
    gate('新封存20—30场景至少400有效指令','未完成',[],
        '保留原30场景14新日志1196帧及自然指令候选准备，最终方案/自然标签未冻结和核验，尚未开启正式测试。')
    gate('车辆约束规划与独立车身几何审计','已有开发证据，完整范围未验收',[
        'carla_official/stock_navigation_registration_004.json'],
        '原有公开快照/官方闭环可追溯；不得将生成路径等同选对对象、正确锁定和实际安全到达。')
    gate('官方CARLA实际完整传感器闭环','部分完成',[
        'json/stock_navigation_jobs_004/report.json',
        'json/official_stock_navigation_execution_006/server_shutdown.json'],
        '三项原厂Lincoln开发任务完整经过前端/建图/关系/规则/锁定/规划/物理控制；严格到达需逐例审计，不能用自报ARRIVED。')
    gate('广泛配置和任务覆盖、最低正确安全到达','未完成',[
        'carla_official/stock_controller_navigation_004/car20/full_rule_geometry/independent_audit_003/report.json',
        'carla_official/stock_controller_navigation_004/car30_curve/full_rule_geometry/independent_audit_003/report.json'],
        '原课题10配置、50冻结任务各3种子及至少5配置10正任务正确安全到达未达到；小批不能缩减此门槛。')
    gate('复杂Qwen后台语义接官方CARLA与三方法闭环比较','未完成',[],
        '目前官方完整批次为清晰类别规则，Qwen未激活；不能把离线真实GPU记录算作物理语义闭环。')
    gate('缓存、异步、故障与性能','已有开发证据，最终范围未验收',[],
        '原受保护实验和测试保留；当前单模型生成延迟非完整实时系统；还需最终官方闭环真实时钟排队和失败覆盖。')
    gate('新环境实际安装和小规模真实链路','部分完成',[
        'reproduction/clean_cpu_install_001/dependencies_report_003.json',
        'reproduction/clean_cpu_chain_002/repeatability_audit_001.json'],
        '新venv的CPU检测/分割/CLIP/建图/关系/Qwen两次真实验证；独立数据1场景3帧，未完成新GPU导航/物理全链复现。')
    gate('可编辑表、正式科学图与源码交付','已完成当前开发结果交付，最终总包待验收',[
        'figures/public_evidence_004/visual_qa_001.json',
        'json/near6_export_completion_001/visual_qa_001.json',
        'tables/full_semantics_near6_office_002/receipt.json'],
        '正式图5张及版本分离的Word/Excel/PDF表已有真实源数据；新增源码补充003和最终哈希回执另行关闭。')
    gate('软件检查及历史源代码保护','已核验当前范围',[
        'json/software_validation_001/junit.xml',
        'json/continuation_resource_protection_006.json'],
        f'保留全量测试记录{test_counts}；373原始源码此刻重新SHA检查一致；不把源码校验称整个历史大数据再次完整重哈希。')
    gate('论文正文','按用户要求暂缓',[],
        '当前只完善实验、图表、代码位置与交付说明，不撰写论文正文。')
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False,
                stage9_complete=False,stage10_complete=False,stage11_complete=False,stage12_complete=False,
                whole_project_accepted=False,new_test_opened=False,manuscript_work_deferred_by_user=True,
                human_status_snapshot=human,test_counts=test_counts,protected_source_rechecked=len(code),
                required_gates=gates,source_sha256=sha(__file__),
                audit_rule='开发实验完成不代表整课题验收；缺失、间接或范围不足的证据都不判完全达标')
    with (R/'json/goal_acceptance_005.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(gates=len(gates),goal_complete=False,human_status=human),ensure_ascii=False))


if __name__=='__main__':main()
