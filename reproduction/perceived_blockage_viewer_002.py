"""复用已核验的原始传感器/控制查看组件，展示四次故障开发执行及未暴露。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from conceptgraph.completion import realtime_matrix_demo as demo
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(path.read_text())


def main():
    dest=OUTPUT/'visualizations/perceived_blockage_cpu_002'
    dest.mkdir(exist_ok=False)
    (dest/'cases').mkdir()
    assert read(OUTPUT/'faults/perceived_blockage_cpu_002/recovery_001/complete.json')['execution_complete']
    selected=[('perceived_blockage_cpu_001',c) for c in ['rule_blockage','delayed_actual_rank']]
    selected += [('perceived_blockage_cpu_002',c) for c in ['rule_blockage','delayed_actual_rank']]
    inputs,files,rows,totals={},{},[],{}
    dump(dest/'release.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in [Path(__file__),Path(demo.__file__)]},
        selection=selected,selection_success_filter=False,extra_model_calls=0,extra_navigation_runs=0,goal_complete=False))
    for index,(run,condition) in enumerate(selected):
        case=OUTPUT/'faults'/run/condition
        nav=case/'navigation'
        fixture=read(case/'config_and_task.json')
        result=read(nav/'result.json')
        identity=run+'_'+condition
        method=['白目标规则（未暴露）','白目标排序（未提交）','红目标规则','红目标迟到排序'][index]
        record=dict(raw_case=str(nav),run_id=identity,variant=method,seed=17,status=result['status'],
                    result=result,result_sha256=digest(nav/'result.json'),process_timeout=False)
        data=demo.build_case(record,fixture['task'],fixture['config'],inputs,lambda:None)
        audit=case/('independent_audit_003' if run.endswith('002') else 'independent_audit_001')/'report.json'
        report=read(audit)
        paths={'故障独立审核':audit,'故障时序':case/'events.jsonl','实际执行摘要':case/'complete.json'}
        for label,relative in [('环境屏障真值（隔离评价）','evaluation_only_dynamic_obstacle.json'),
            ('阻塞前路径','old_target_path.json'),('真实模型原文','actual_model_text.json'),
            ('独立整车不可达证明','geometry_witness_002/report.json')]:
            if (case/relative).exists():
                paths[label]=case/relative
        for label,path in paths.items():
            inputs[str(path.relative_to(OUTPUT))]=digest(path)
            data['raw_urls'][label]=demo.source_url(path)
        data['independent_evidence']=dict(initial_is_reconstructed=False,initial_state_provenance='saved_initial_state',
            arrival_hold_certificate=False,proof_urls={'故障独立审核':demo.source_url(audit)})
        data['fault_exposure']=report
        name=f'cases/{index:04d}.json'
        dump(dest/name,data)
        files[name]=digest(dest/name)
        rows.append(dict(index=index,run=identity,method=method,task_id=fixture['task']['task_id'],config_id='config00',
            kind=fixture['task']['kind'],seed=17,status=result['status'],arrived=result['actual_full_perception_arrival'],
            result_missing=False,frames=data['frames'],control_records=data['control_records'],file=name,
            fault_exposure_passed=report['passed']))
        for key,value in data['input_files_verified'].items():
            totals[key]=totals.get(key,0)+value
    page=demo.html_document(read(OUTPUT/'statistics/realtime_three_model_full_001/report.json'))
    old='<section class="card"><h2>完整开发对比，所有失败保留</h2>'
    begin=page.index(old)
    end=page.index('</section>',begin)+len('</section>')
    intro=('<section class="card"><h2>真实屏障与迟到排序：四次开发执行全部保留</h2>'
        '<p>白色目标两次未形成故障前提，不能算通过。红色目标两次按独立整车及原始输入证据核验。'
        '先实际行驶，再因观测间断按原TTL停车，随后真实渲染屏障；屏障只经LiDAR估计进入代理。'
        '迟到排序条件仅一次实际CPU基础Qwen调用，不能当GPU、模型准确率或独立任务增量。</p>'
        '<p>整场结果与当前帧分开。GT默认隐藏，时间轴只展示当时处理完成的观测。'
        '解析证明和故障暴露状态见每次执行的“故障独立审核”；全部原失败、未暴露和审核修正保留。'
        'CPU感知TTL15秒，固定240秒观察窗；非主450协议、自然指令封存或独占性能。</p></section>')
    page=page[:begin]+intro+page[end:]
    page=page.replace('三模型450次开发实验回放','真实屏障与迟到排序故障回放')
    old_links=('<p class="links"><a href="/主矩阵450_实验核验交接_001.md">完整核验与限制</a>'
               '<a href="/statistics/realtime_three_model_full_001/trials.csv">全部450逐试验结果</a></p>')
    page=page.replace(old_links,'<p class="links"><a href="/task_protocol/perceived_blockage_cpu_001.md">第一版协议</a>'
        '<a href="/task_protocol/perceived_blockage_cpu_002.md">红色目标补充协议</a>'
        '<a href="/task_protocol/perceived_blockage_audit_correction_003.md">审核修正与限制</a></p>')
    page=page.replace('</style>', '.bar label{display:flex;align-items:center;gap:6px;max-width:100%;min-width:0}.bar select{flex:1;min-width:0;max-width:100%}</style>')
    (dest/'index.html').write_text(page)
    shutil.copy2(Path(__file__),dest/'generator_source.py')
    for name in ['release.json','index.html','generator_source.py']:
        files[name]=digest(dest/name)
    dump(dest/'input_files.json',inputs)
    files['input_files.json']=digest(dest/'input_files.json')
    fixture=read(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json')
    dump(dest/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=rows,files=files,
        methods=[r['method'] for r in rows],configs=['config00'],seeds=[17],output_root=str(OUTPUT.resolve()),
        vehicle_spec=fixture['vehicle'],calibration=dict(width=640,height=360,fx=320,fy=320,cx=320,cy=180),
        task_release_sha256=digest(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'),
        sensor_input_totals=totals,perception_records=sum(r['frames'] for r in rows),
        control_records=sum(r['control_records'] for r in rows),extra_model_calls=0,new_navigation_runs=0,
        new_arrival_credit=0,new_test_opened=False,goal_complete=False))
    print(json.dumps(dict(cases=4,verified_files=len(inputs),sensor_totals=totals),ensure_ascii=False))


if __name__=='__main__':
    main()
