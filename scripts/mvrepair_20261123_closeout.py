"""基于已冻结结果补逐log汇总、失败索引和最终文档；无方法运行。"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from conceptgraph.project_paths import PROJECT_ROOT
from scripts.mvrepair_20261123_finalize_v2 import sha

ROOT = PROJECT_ROOT
DEST = ROOT/'outputs/reports/multiview_identity_repair_20261123/main'
REPORTS = DEST.parent
BASE = ROOT/'outputs/results/nuscenes/validation/iterations'
P = 'mvrepair_20261123_'


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main():
    sources = json.loads((DEST/(P+'final_sources_v3.json')).read_text())['sources']
    scene_rows = []
    for src in sources:
        batch, run, methods = src['batch'], Path(src['run']), src['methods']
        prefix = Path(src['report']).name.removesuffix('report.json')
        if batch == 'nextlogs':
            label = run.name.split('_')[1]
            directory = REPORTS/'validation/final_nextlogs_wave_v1'/label
        elif 'A_physical' in run.name:
            directory = REPORTS/('lifecycle/'+run.name if batch == 'pilot' else 'validation/A_physical_regression')
        else:
            directory = DEST/run.name.removeprefix(P)
            if not directory.is_dir():
                directory = DEST/run.name.removeprefix(P).removesuffix('_v1')
        path = directory/(prefix+'scene_metrics.csv')
        for row in csv.DictReader(path.open(encoding='utf-8-sig')):
            if row['method'] in methods:
                scene_rows.append(dict(batch=batch, variant=methods[row['method']], **row, source_scene_csv=str(path)))
    write_csv(DEST/(P+'all_scene_metrics_v3.csv'), scene_rows)
    grouped = defaultdict(list)
    for row in scene_rows:
        grouped[row['batch'],row['log_token'],row['variant'],row['scope']].append(row)
    logs = []
    for (batch, log, variant, scope), values in grouped.items():
        r = dict(batch=batch, log_token=log, variant=variant, scope=scope,
            scenes='|'.join(sorted(x['scene'] for x in values)))
        for key in ['true_positive','truth','predictions','duplicate_nodes','IDTP','IDFN','IDFP','error_sum_m']:
            r[key] = sum(float(x[key]) for x in values)
        tp, gt, pred = r['true_positive'],r['truth'],r['predictions']
        r.update(precision=tp/max(1,pred),recall=tp/max(1,gt),F1=2*tp/max(1,gt+pred),
            IDF1=2*r['IDTP']/max(1,2*r['IDTP']+r['IDFN']+r['IDFP']),
            duplicate_per_prediction=r['duplicate_nodes']/max(1,pred),
            center_error_m=r['error_sum_m']/max(1,tp))
        logs.append(r)
    write_csv(DEST/(P+'all_log_metrics_v3.csv'), logs)
    candidates = []
    for run in sorted(BASE.glob('mvrepair*')):
        reports = [p for p in run.glob('*report.json') if p.is_file()]
        if not reports:
            continue
        for path in reports:
            data = json.loads(path.read_text())
            item = dict(run=str(run), report=str(path), report_sha256=sha(path),
                complete=bool(data.get('complete',False)), interpretation='frozen raw result; see variant explanation')
            if run.name == P+'T3_pilot_v1':
                item['interpretation'] = 'INVALID implementation contrast: inherited T2 pre-deleted parts before T3 ranking. Complete pilot retained; excluded from valid T3 table.'
            elif run.name == P+'L2_v1':
                item['interpretation'] = 'PARTIAL/interrupted: NumPy bool JSON serialization after 12 groups. Preserved; L2_v2 fixes serialization only.'
            candidates.append(item)
    (DEST/(P+'all_run_index_v3.json')).write_text(json.dumps(dict(runs=candidates,
        negative_variants=['U1: FP/IDF1 regression','L2/L4: recent front location overrides caused single-front and paired-location regressions',
            'L5 same-view identity merge: IDTP loss','O1/O2: unique ownership side TP loss and single-front identity loss; O2 engineering control proved O1 scores genuinely unchanged on this batch',
            'O3 reliable ownership: no TP/identity benefit, prediction +1','T1/PT1: duplicate -1 but IDTP -2',
            'T2: cab point-count representative caused one TP and one IDTP loss','T3 v1: invalid implementation contrast; replaced by frozen v2'],
        current_valid_candidate='T3 v2; partial repair, not all front/duplicate guardrails satisfied',
        no_historical_failure_deleted=True),ensure_ascii=False,indent=2))
    pooled = list(csv.DictReader((DEST/(P+'pooled_node_main_v3.csv')).open()))
    table = ['|方法|范围|TP/GT|预测|重复|P / R / F1 (%)|IDF1 (%)|', '|---|---|---:|---:|---:|---:|---:|']
    for variant in ['A','A_physical','C','P1','T2','T3']:
        for scope in ['front','union']:
            r = next(r for r in pooled if r['variant']==variant and r['scope']==scope)
            prf=' / '.join(f"{float(r[k])*100:.2f}" for k in ['precision','recall','F1'])
            table.append(f"|{variant}|{scope}|{r['true_positive']}/{r['truth']}|{r['predictions']}|{r['duplicate_nodes']}|{prf}|{float(r['IDF1'])*100:.2f}|")
    doc = '''# 本轮三视角身份与精度修复结论（最终 v3）

本轮真实运行已完成：六场景各20时刻，共120组、5条log；687个union对象帧、385个共同front对象帧。前两批用于诊断选机制，第三批在源码冻结后仅一次评价；现在均属于已看过的验证/调试证据。不是完整38场景成绩，不是车辆闭环。评价保持六类、原3米同类水平中心一对一匹配；near6和原方向/关系标准保持。

推荐保留显式候选 **T3 v2** 继续研究，系统默认和论文原稿未切换。它在原三视角C基础上保持覆盖（308→309 TP），减少输出（566→537）、重复（22→11），F1和IDF1提高；仍未满足全部修复目标：严格front精度59.41%与IDF1 42.63%低于公平单前视的61.09%与43.23%，重复11也高于单前视3。不能称“全面修好”。

'''+ '\n'.join(table)+'''

三批要分别看：pilot T3=38/105、93预测、2重复；两条回归log=177/357、307预测、2重复；最后两条新log=94/225、137预测、7重复。最后一批严格front为48/137，低于A的49/137；union匹配中的front为49，差异来自预测中心视锥筛选和独立匹配，不能把两个口径混用。第三批P1/T2/T3几何全同、原型激活数为0，因此该批只证明整条固定入口的表现，尚不能验证时序原型机制的推广收益。

T3保留已证同源的持久canonical与有覆盖见证的旧state退役、同曝光车辆互斥语义候选清理、只在当前弱侧向冲突时使用前视辅助；再保留被退役部分观测的有限关联原型，分配前拒绝无物理连续证据的接管。同组仅一次hits，部件不叠点、不叠确认，原型不输出、不延长输出TTL。原型期限1.5秒沿原关联状态期限；当前可靠、原生canonical关联分数优先选一条真实代表。原型用实际世界点与相机mask互投、仅补偿自车位姿，没有物体运动补偿；目前适用于静态/慢速部分表面，不能据此宣称高速动态通用。

真实诊断：scene0035大卡车先前被分成拖厢和车头两个state，随后39被6.074米外小平板车接管。T1只退役旧fragment使后续新ID重生，IDTP152→150；T2原型纠正物理身份，但按21点车头替代15点拖厢，中心偏到3米外，丢1TP。T3保留同canonical原生得分最高的15点部件（.80888 对 .52825），恢复TP/IDTP，且f10–13的39 hits为3/4/5/6，没有同组重复计数。该案例仍不等同于全范围人审身份准确率。39da轿车和拥挤行人的自动GT身份链存在3米最大基数匹配歧义，未按GT强拼身份。

定位负项保留：回归两log共同front 59对象帧/37相邻对，A误差0.63817米、误差变化0.43876米；C与T3为0.65754/0.47949米，未优于A。共同union 61帧/39对，T3为0.66602/0.53323米，略差于C的0.65649/0.51882米。P1与T3全177共同匹配帧的平均位置和相邻误差变化完全相同；T3收益主要是删除有见证重复和保持物理身份，不是全局中心更准。逐log原数值和最后一批配对结果均附CSV。

可复现入口为 `scripts/mvrepair_20261123_run_v3.py`，mapper为 `conceptgraph.slam.mvrepair_20261123_native_representative_v2:NativeRepresentativeMapper`，seed17，无额外settings参数。完整配置、39份源码SHA、481项输入SHA、独立GT SHA逐run保存在release。T3源SHA：16f16aad31f235081183e9f4f18843b7e103d244bcf8571dc326c5b5c3582c92。示例（run-id必须新建）：

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib /root/miniconda3/envs/conceptgraph/bin/python -m scripts.mvrepair_20261123_run_v3 \\
  --manifest dataset/processed/multiview_20261123_pilot_v1/manifest.json \\
  --ground-truth dataset/processed/multiview_20261123_pilot_v1/multiview_20261123_evaluation_ground_truth.jsonl \\
  --run-id mvrepair_20261123_T3_new_reproduction \\
  --report-dir outputs/reports/multiview_identity_repair_20261123/main/T3_new_reproduction \\
  --tag T3 --mapper conceptgraph.slam.mvrepair_20261123_native_representative_v2:NativeRepresentativeMapper
```

mapper耗时表是CPU上累计更新墙钟，排除前端和读取；峰值RSS含A/C/候选单前/候选三视四个同时驻留及输入/导入，不能当独立方法显存或独立RSS。新40仍无GPU重跑。完整80同配置复跑的320条件预测、320审计与80评价逐字段一致；610项原数据/权重/旧结果/原稿保护SHA未变。最后新40只按预登记完整运行一次，没有用其分数改规则。旧release的primary文字沿用pilot77/105描述，实际各批分母以逐GT文件及本报告三批表为准；旧冻结元数据没有改写。

后续最有依据的是统一状态空间里更严格的关联竞争与可靠定位，不应全局缩TTL或全局吞共享点。本轮不再改候选，后续1/3/6视角收益实验冻结T3 v2，单独评价计算量、重复、位置/身份与覆盖取舍。
'''
    (DEST/(P+'conclusion_v3.md')).write_text(doc)
    index = dict(created_utc=datetime.now(timezone.utc).isoformat(), conclusion=str(DEST/(P+'conclusion_v3.md')),
        tables=[str(p) for p in sorted(DEST.glob(P+'*v3.csv'))],
        source_manifest=str(DEST/(P+'final_sources_v3.json')), candidates=str(DEST/(P+'all_run_index_v3.json')),
        deterministic_protection_audit=str(DEST/(P+'final_verification_v2.json')),
        integration=str(DEST/'T3'/(P+'representative_integration_audit.json')),
        independent_evaluation=str(REPORTS/'validation'), default_changed=False, manuscript_changed=False)
    (DEST/(P+'file_index_v3.json')).write_text(json.dumps(index,ensure_ascii=False,indent=2))
    print(json.dumps(dict(scene_rows=len(scene_rows),log_rows=len(logs),candidate_runs=len(candidates),conclusion=index['conclusion']),ensure_ascii=False))


if __name__ == '__main__':
    main()
