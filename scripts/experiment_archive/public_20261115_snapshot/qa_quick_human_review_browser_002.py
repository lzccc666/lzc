"""真实浏览器核验离线/在线页面，仅看图与空表校验，不代填或提交人工记录。"""
import datetime,json,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1];D=R/'annotation/quick_human_review_001';O=D/'browser_qa_002'
def status():
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open('http://127.0.0.1:8877/api/review_status',timeout=5) as response:return json.load(response)
def main():
 O.mkdir(exist_ok=False);before=status();cases=[]
 with sync_playwright() as pl:
  browser=pl.chromium.launch(executable_path='/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--no-proxy-server','--disable-gpu']);context=browser.new_context(viewport={'width':1360,'height':1050});page=context.new_page();posts=[];errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda request:posts.append(request.url) if request.method=='POST' else None)
  for mode,url in [('offline',(D/'打开这个文件开始审核.html').as_uri()),('online','http://127.0.0.1:8877/public_experiment_completion_20261115/annotation/quick_human_review_001/index.html')]:
   page.goto(url,wait_until='load',timeout=30000);page.wait_for_function("document.getElementById('rgb').width>500 && document.getElementById('crop').naturalWidth>0");assert page.evaluate('cards.length')==20;assert page.evaluate('store.records.length')==0
   blank=page.locator('#save');blank.click();assert '请填写' in page.locator('#status').inner_text();assert page.evaluate('store.records.length')==0
   for _ in range(19):page.locator('#next').click()
   page.wait_for_function('position===19');assert '20 / 20' in page.locator('#counter').inner_text()
   page.locator('#previous').click();assert page.evaluate('position')==18
   for _ in range(18):page.locator('#previous').click()
   page.wait_for_timeout(300);page.screenshot(path=str(O/f'{mode}_first_card.png'),full_page=True)
   cases.append(dict(mode=mode,cards=20,forward_backward_navigation_pass=True,canvas_width=page.locator('#rgb').evaluate('(e)=>e.width'),crop_width=page.locator('#crop').evaluate('(e)=>e.naturalWidth'),upload_disabled=page.locator('#upload').is_disabled(),empty_form_rejected=True,local_records=page.evaluate('store.records.length'),localStorage_read_available=page.evaluate("(()=>{try{return localStorage.getItem(key)!==undefined}catch(e){return false}})()")))
  browser.close()
 after=status();assert before==after and not posts and not errors
 with (O/'report.json').open('x') as f:json.dump(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,browser_errors=errors,POST_requests=posts,before=before,after=after,human_records_created=0,no_filled_or_saved_labels=True,export_download_not_tested_without_real_human_record=True,goal_complete=False),f,ensure_ascii=False,indent=2)
 print(json.dumps(cases,ensure_ascii=False))
if __name__=='__main__':main()
