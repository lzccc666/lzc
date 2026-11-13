"""真实屏障四案例桌面/手机只读验收，不发控制或人工标签。"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT=Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def main():
    root=OUT/'visualizations/perceived_blockage_cpu_002'
    dest=OUT/'visualizations/perceived_blockage_browser_qa_002'
    dest.mkdir(exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    requests,checks=[],[]
    with sync_playwright() as runtime:
        browser=runtime.chromium.launch(headless=True,
            executable_path='/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell',
            args=['--no-sandbox','--no-proxy-server'])
        for viewport,width,height in [('desktop',1440,1000),('mobile',390,844)]:
            page=browser.new_page(viewport=dict(width=width,height=height))
            errors,responses=[],[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:requests.append(dict(method=r.method,url=r.url)))
            page.on('response',lambda r:responses.append(dict(status=r.status,url=r.url)))
            page.goto('http://127.0.0.1:8768/visualizations/perceived_blockage_cpu_002/index.html')
            page.wait_for_function("document.body.dataset.renderReady==='ready'")
            assert page.locator('#case option').count()==4
            for i,row in enumerate(manifest['cases']):
                page.select_option('#case',row['file'])
                page.wait_for_function("run=>window.MATRIX_VIEW?.snapshot.run===run && window.MATRIX_VIEW.snapshot.ready",arg=row['run'])
                assert not page.is_checked('#gt')
                case=json.loads((root/row['file']).read_text())
                raw=Path(case['raw_prefix_url'].lstrip('/'))
                barrier=OUT/raw.parent/'evaluation_only_dynamic_obstacle.json'
                frame=min(json.loads(barrier.read_text())['frame_idx']+1,row['frames']-1) if barrier.exists() else row['frames']-1
                page.locator('#frame').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}',frame)
                page.wait_for_function("v=>window.MATRIX_VIEW.snapshot.frameIndex===v && window.MATRIX_VIEW.snapshot.ready",arg=frame)
                observation=page.evaluate('window.MATRIX_VIEW.snapshot')
                assert observation['perceptionFrame']==frame
                page.check('#lidar')
                page.wait_for_function("window.MATRIX_VIEW.snapshot.ready")
                if i>=2:
                    page.screenshot(path=str(dest/f'{viewport}_{i}_barrier.png'),full_page=True)
                page.select_option('#mode','control')
                page.locator('#tick').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}',row['control_records']-1)
                page.wait_for_function("v=>window.MATRIX_VIEW.snapshot.tickIndex===v && window.MATRIX_VIEW.snapshot.ready",arg=row['control_records']-1)
                value=page.evaluate('''() => ({snapshot:window.MATRIX_VIEW.snapshot,
                    overflow:document.documentElement.scrollWidth>innerWidth,
                    error:document.querySelector('#error').textContent})''')
                with (dest/f'{viewport}_{i}_diagnostic.json').open('x') as f:
                    json.dump(dict(value=value,page_errors=errors),f,ensure_ascii=False,indent=2,allow_nan=False)
                assert not value['overflow'] and not value['error'] and not errors
                snapshot=value['snapshot']
                assert snapshot['displayedState']['speed_mps']==0 and not snapshot['GT']
                assert snapshot['perceptionCompletedWall']<=snapshot['controlWall']
                checks.append(dict(viewport=viewport,run=row['run'],observation=observation,control=value))
                page.select_option('#mode','observation')
            assert all(r['status']<400 for r in responses)
            page.close()
        browser.close()
    assert all(r['method']=='GET' for r in requests)
    report=dict(passed=True,cases=4,viewports=2,checks=checks,requests=len(requests),
        all_requests_GET=True,GT_hidden_by_default=True,actual_model_calls=0,
        new_control_commands=0,human_labels_created=0,new_arrival_credit=0,goal_complete=False)
    with (dest/'report.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,case_viewports=len(checks),requests=len(requests))))


if __name__=='__main__':
    main()
