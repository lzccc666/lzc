"""保留历史门户，新增已完成消融、GPU故障、性能与隔离复现入口。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = OUTPUT / 'visualizations/thesis_portal_016'
    destination = OUTPUT / 'visualizations/thesis_portal_017'
    additions = [
        'ablations/semantic_navigation_chain_004/complete.json',
        'faults/realtime_fault_after_semantics_002/complete.json',
        'performance/runtime_profile_after_faults_002/complete.json',
        'reproduction/isolated_environment_after_GPU_005/complete.json',
        'visualizations/gpu_fault_evidence_full_browser_002/report.json',
        'experiment_tables/fault_performance_office_002/independent_audit_001/report.json',
    ]
    assert all((OUTPUT / p).is_file() for p in additions)
    page = (parent / 'index.html').read_text()
    page = page.replace('realtime_fault_after_semantics_001', 'realtime_fault_after_semantics_002')
    page = page.replace('runtime_profile_after_faults_001', 'runtime_profile_after_faults_002')
    page = page.replace('semantic_navigation_chain_003', 'semantic_navigation_chain_004')
    page = page.replace('后续队列与人工核验', '后续验收与人工核验')
    page = page.replace('独占性能：', '单CUDA进程分阶段性能：')
    page = page.replace('独占GPU性能', '单CUDA进程分阶段性能')
    section = '''<section id="completed-additions"><h2>已完成的消融、GPU故障与复现交付</h2>
<p>缺一语义450次新增均已记账，449完整、1中断；组合750行复用主对比300行。规则、完整新Adapter、减解析、减属性、减重排各90个正条件，到达分别50、4、48、5、6。新Adapter下游负结果保留。</p>
<p>GPU故障180次均已记账：179完整、1进程超时终态未知。29次实际断流全部记录停车及新鲜观测恢复后运动；10次报告到达均通过保持区间审核。传感器TTL1.5秒、语义TTL30秒；未触发故障不获得抗故障信用。</p>
<p>正式性能包含12次GPU前端、216次无缓存Qwen推理与216次缓存请求，缓存新增调用0。分阶段耗时不能当完整导航实时吞吐。</p>
<div class="links"><a href="/visualizations/gpu_fault_evidence_full_002/index.html">全部180例故障证据</a>
<a href="/experiment_tables/semantic_ablation_office_001/开发实验结果表.html">语义消融表T16—T18</a>
<a href="/experiment_tables/fault_performance_office_002/开发实验结果表.html">故障与性能表T19—T25</a>
<a href="/experiment_tables/fault_performance_office_002/开发实验结果表.docx">Word表格</a>
<a href="/experiment_tables/fault_performance_office_002/开发实验结果表.xlsx">Excel与逐条件结果</a>
<a href="/experiment_tables/fault_performance_office_002/开发实验结果表.pdf">PDF表格</a>
<a href="/reproduction/isolated_environment_after_GPU_005/complete.json">新Python环境实际复现</a></div>
<p class="muted">新环境共享基础解释器标准库及原生系统库；依赖复制隔离，PyTorch带CUDA支持但实际CPU执行、CUDA未初始化。不是新操作系统。人工标签与20场景×20有效指令的新封存验收仍未完成；不生成论文正文。</p></section>'''
    assert '<section id="completed-main">' in page
    page = page.replace('<section id="completed-main">', section+'\n<section id="completed-main">', 1)
    destination.mkdir(exist_ok=False)
    (destination / 'index.html').write_text(page)
    evidence = json.loads((parent / 'completed_evidence.json').read_text())
    evidence.update({p: sha(OUTPUT/p) for p in additions})
    for name, value in evidence.items():
        assert sha(OUTPUT/name) == value
    (destination / 'completed_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    manifest = dict(source_sha256=sha(Path(__file__)), source_path=str(Path(__file__).relative_to(OUTPUT)),
        parent_html_sha256=sha(parent/'index.html'), html_sha256=sha(destination/'index.html'),
        created_utc=datetime.now(timezone.utc).isoformat(), completed_main_trials=450,
        GPU_fault_trials=180, read_only=True, manuscript_text_written=False, goal_complete=False)
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
