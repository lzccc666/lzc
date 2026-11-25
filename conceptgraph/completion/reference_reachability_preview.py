"""只读显示独立条件静态可达参考；控制见证不计实际导航试验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('base preview changed; retain partial run without inventing visualization')
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reference-run', default='reference_reachability_reserved_full_002')
    parser.add_argument('--region-preview', default='reference_regions_reserved_001')
    args = parser.parse_args()
    reference = OUTPUT/'annotation'/args.reference_run
    release = json.loads((reference/'release.json').read_text())
    audit = json.loads((reference/'audit_001/report.json').read_text())
    if not audit['passed'] or audit['generator_release_sha256'] != digest(reference/'release.json'):
        raise ValueError('requires independently accepted exact reference release')
    base = OUTPUT/'visualizations'/args.region_preview
    base_manifest = json.loads((base/'manifest.json').read_text())
    if base_manifest['source_summary_sha256'] != digest(OUTPUT/'annotation'/release['region_run']/'summary.json'):
        raise ValueError('static preview and accepted reference labels differ')
    for name, sha in base_manifest['files'].items():
        if digest(base/name) != sha:
            raise ValueError('changed source static preview')
    dest = OUTPUT/'visualizations'/args.run_id
    dest.mkdir(exist_ok=False)
    shutil.copytree(base/'data', dest/'data')
    index = json.loads((base/'index.json').read_text())
    query_refs = {q['query_id']: q for q in map(json.loads, (reference/'audit_001/queries.jsonl').open())}
    if set(query_refs) != {q['query_id'] for q in index['queries']}:
        raise ValueError('reference and static query populations differ')
    checks = json.loads((reference/'audit_001/witness_checks.json').read_text())
    accepted = {(r['scene'], r['frame_idx'], r['instance_token']) for r in checks if r['passed']}
    targets = {}
    for row in map(json.loads, (reference/'targets.jsonl').open()):
        key = row['scene'], row['frame_idx'], row['instance_token']
        if key not in accepted:
            continue
        path = reference/row['reference']['path']
        if digest(path) != row['reference']['sha256']:
            raise ValueError('changed reference sequence')
        targets[key] = row['reference']
    for row in index['queries']:
        q = query_refs[row['query_id']]
        if row['accepted_instance_tokens'] != q['accepted_instance_tokens']:
            raise ValueError('reference target identities differ')
        tokens = sorted(q['certified_instance_tokens'])
        row.update(reference_status=q['reachability'], certified_instance_tokens=tokens,
            reference_scope=q['conditional_scope'], witness_urls=[
                f'/annotation/{args.reference_run}/{targets[(row["scene"], row["frame_idx"], token)]["path"]}'
                for token in tokens])
    index.update(reference_audit=audit, independent_reachable_queries=sum(bool(r['witness_urls']) for r in index['queries']))
    dump(dest/'index.json', index)
    html = (base/'index.html').read_text()
    changes = [
        ('全体可达性仍为 unknown，未生成控制轨迹或人工记录。',
         '部分查询具有独立控制见证，其余保持 unknown。参考从已知后轴位置、静止和零转角起步，冻结障碍条件下90秒内进入区域并保持2秒；没有产生待评测方法的实际执行或人工记录。'),
        ('未绘制计划路径或实际轨迹。', '紫色为隔离标签端的控制积分参考，红框为当前参考车身；不是方法计划或实际执行轨迹。'),
        ('<div class="card"><h2>当前证据</h2>',
         '<div class="card"><h2>独立控制参考</h2><p id="referenceState">当前查询未知</p><label>参考积分步 <input id="referenceStep" type="range" min="0" max="0" value="0" style="width:100%"></label><p>位置由控制积分得到。此处仅核验标签参考，四项实际导航结果均未测量。</p></div><div class="card"><h2>当前证据</h2>'),
        ('function draw(context,targets,row){', 'function draw(context,targets,row,witness=null){'),
        ("if(!context){ctx.fillStyle=", "if(!context){$('referenceState').textContent='范围内无匹配目标：无控制参考，不提供可达结论。';ctx.fillStyle="),
        ('const camera=row.camera_xy,points=[camera];',
         'const camera=row.camera_xy,points=[camera];if(witness)for(const r of witness.rows)points.push([r.state.x,r.state.y]);'),
        (' const c=xy(camera);',
         " if(witness){ctx.beginPath();witness.rows.forEach((r,i)=>{const p=xy([r.state.x,r.state.y]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.strokeStyle='#873cc3';ctx.lineWidth=3;ctx.stroke();const s=witness.rows[Number($('referenceStep').value)].state,c=Math.cos(s.yaw_rad),t=Math.sin(s.yaw_rad),body=[[-.9,-.925],[3.7,-.925],[3.7,.925],[-.9,.925]].map(([a,b])=>[s.x+a*c-b*t,s.y+a*t+b*c]);geometry({type:'Polygon',coordinates:[[...body,body[0]]]},'#cd313122','#b92d2d',2);$('referenceState').textContent=`参考 t=${s.time_s.toFixed(1)}s；速度 ${s.speed_mps.toFixed(3)}m/s；转角 ${(s.steering_rad*180/Math.PI).toFixed(2)}°；${witness.rows.length}状态。`;}else $('referenceState').textContent='未获得独立参考：unknown，不判不可接近。';\n const c=xy(camera);"),
        ('if(serial!==state.serial)return;\n draw(context,targets,row);',
         "let witness=null;if(row.witness_urls.length)witness=await get(row.witness_urls[0]);if(serial!==state.serial)return;$('referenceStep').max=witness?witness.rows.length-1:0;$('referenceStep').value=0;state.loaded={context,targets,row,witness};\n draw(context,targets,row,witness);"),
        ("$('status').textContent=titles[row.endpoint_status];",
         "$('status').textContent=witness?'有独立条件静态可达参考；从静止/冻结障碍/90秒/保持2秒，实际到达未测':titles[row.endpoint_status];"),
        ("reachability:'unknown',route:null", "reachability:row.reference_status,reference_scope:row.reference_scope,reference_states:witness?.rows.length??0,certified_instance_tokens:row.certified_instance_tokens,reference_witness_urls:row.witness_urls,route:null"),
        ("方法调用 0；真实到达证据 0。", "条件静态可达查询 ${data.independent_reachable_queries}；方法调用 0；真实到达证据 0。"),
        ("$('scene').addEventListener('change',filter);",
         "$('referenceStep').addEventListener('input',()=>{const r=state.loaded;if(r)draw(r.context,r.targets,r.row,r.witness)});$('scene').addEventListener('change',filter);"),
    ]
    for old, new in changes:
        html = replace_once(html, old, new)
    (dest/'index.html').write_text(html)
    shutil.copy2(__file__, dest/'reference_preview_source.py')
    dump(dest/'manifest.json', dict(source_sha256=digest(Path(__file__)),
        reference_release_sha256=digest(reference/'release.json'),
        reference_audit_sha256=digest(reference/'audit_001/report.json'),
        files={str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file() and p.name != 'manifest.json'},
        queries=len(index['queries']), conditional_static_reachable_queries=index['independent_reachable_queries'],
        read_only=True, actual_vehicle_executions=0, human_records_written=0, model_calls=0,
        new_test_opened=False, goal_complete=False))


if __name__ == '__main__':
    main()
