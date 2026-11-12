"""真实六条件保存查看器的桌面/手机只读检查；不提交审核、不调用模型。"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parents[1]
root=out/'visualizations/realtime_fault_cpu_preview_002'
manifest=json.loads((root/'manifest.json').read_text())
records={json.loads(p.read_text())['run_id']:json.loads(p.read_text()) for p in (out/'faults/realtime_fault_cpu_smoke_002').glob('*_record.json')}
dest=out/'visualizations/realtime_fault_browser_qa_002';dest.mkdir(exist_ok=False)
requests=[];checks=[]
with sync_playwright() as pl:
 browser=pl.chromium.launch(headless=True,args=['--no-sandbox','--no-proxy-server'])
 for name,width,height in [('desktop',1440,1000),('mobile',390,844)]:
  page=browser.new_page(viewport=dict(width=width,height=height));errors=[]
  page.on('pageerror',lambda error:errors.append(str(error)))
  page.on('request',lambda request:requests.append(dict(method=request.method,url=request.url)))
  page.goto('http://127.0.0.1:8768/visualizations/realtime_fault_cpu_preview_002/index.html')
  page.wait_for_function("document.querySelectorAll('#case option').length===6 && document.querySelector('#status').textContent.includes('ARRIVED')")
  assert not page.locator('#gt').is_checked()
  body=page.locator('body').inner_text();assert '2270' in body and '15秒' in body and '未触发过期制动' in body
  assert page.locator('#outcomes .outcome').count()==4
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
  links=page.locator('section.card').first.locator('a').evaluate_all('as=>as.map(a=>a.getAttribute("href"))')
  for link in links:assert page.request.get('http://127.0.0.1:8768'+link).status==200
  for i,case in enumerate(manifest['cases']):
   record=records[case['run']];saved=json.loads((root/case['file']).read_text())
   page.locator('button',has_text='查看保存记录').nth(i).click()
   page.wait_for_function('(run)=>document.querySelector("#status").textContent.startsWith(run)',arg=case['run'])
   active=record['exposure']['active_frames'];frame_id=active[0] if active else saved['frames'][0]['frame_idx']
   pos=next(j for j,f in enumerate(saved['frames']) if f['frame_idx']==frame_id)
   page.locator('#frame').fill(str(pos));page.locator('#frame').dispatch_event('input')
   page.wait_for_function('()=>document.querySelector("#det").complete && document.querySelector("#det").naturalWidth>0')
   assert str(frame_id) in page.locator('#frameLabel').inner_text()
   assert '/effective_detections/'+case['run']+'/' in page.locator('#det').get_attribute('src')
   assert not page.locator('#error').inner_text() and not errors
   source_links=page.locator('#sources a').evaluate_all('as=>as.map(a=>a.getAttribute("href"))')
   for link in source_links:assert page.request.get('http://127.0.0.1:8768'+link).status==200
   assert not page.locator('#gt').is_checked()
   page.screenshot(path=str(dest/f'{name}_{record["condition"]}.png'),full_page=False)
   checks.append(dict(viewport=name,condition=record['condition'],frame=frame_id,effective_detection_visible=True,
       four_outcomes=page.locator('#outcomes').inner_text(),GT_hidden=True,source_links_checked=len(source_links),page_errors=list(errors)))
  page.locator('#gt').check();assert '评估层隐藏' not in page.locator('#execution').inner_text()
  page.locator('#gt').uncheck();assert '评估层隐藏' in page.locator('#execution').inner_text()
  page.locator('#play').click();page.wait_for_timeout(800);page.locator('#play').click()
  assert not page.locator('#error').inner_text() and not errors
  page.close()
 browser.close()
assert all(r['method']=='GET' for r in requests)
report=dict(passed=True,checks=checks,read_only_requests=True,new_model_calls=0,human_records_written=0,
    limitations=['只读保存的六条件单任务CPU日志','15秒TTL下4秒断流未触发过期制动','不新增独立任务覆盖或正式GPU故障完成信用'],goal_complete=False)
(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=True,cases=len(checks),read_only_requests=True,goal_complete=False)))
