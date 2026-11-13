"""正式审核页面仅GET验收；无真实记录时不提交任何伪造标签。"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def main():
    destination = OUTPUT / 'visualizations/human_adjudication_browser_qa_002'
    destination.mkdir(exist_ok=False)
    checks, requests = [], []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True,
            executable_path='/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell',
            args=['--no-sandbox', '--no-proxy-server'])
        for name, width, height in [('desktop',1440,1000), ('mobile',390,844)]:
            page = browser.new_page(viewport=dict(width=width, height=height))
            errors, responses = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.on('response', lambda response: responses.append(dict(status=response.status, url=response.url)))
            page.goto('http://127.0.0.1:8770/')
            page.wait_for_function("document.querySelector('#scene').options.length===30 && document.querySelector('#viewed').textContent.includes('0')")
            assert not page.is_checked('#showTruth')
            assert page.locator('#fields').evaluate('(element)=>element.disabled')
            assert page.locator('#command').inner_text().startswith('尚无真实原始提交')
            assert page.locator('#objects input').count() > 0
            page.screenshot(path=str(destination / (name + '_default.png')), full_page=True)
            page.check('#showTruth')
            page.locator('#frame').evaluate('(e)=>{e.value=2;e.dispatchEvent(new Event("input"))}')
            page.wait_for_function("document.querySelector('#frameInfo').textContent.includes('帧 2') && document.querySelector('#viewed').textContent.includes('2')")
            page.screenshot(path=str(destination / (name + '_annotation_truth.png')), full_page=True)
            page.uncheck('#showTruth')
            page.select_option('#scene', index=1)
            page.wait_for_function("document.querySelector('#frameInfo').textContent.includes(document.querySelector('#scene').value) && document.querySelector('#viewed').textContent.endsWith('0')")
            state = page.evaluate('''() => ({
                overflow:document.documentElement.scrollWidth>innerWidth,
                status:document.querySelector('#status').textContent,
                truthVisible:document.querySelector('#showTruth').checked,
                submitDisabled:document.querySelector('#submit').matches(':disabled'),
                scene:document.querySelector('#scene').value,
                evidence:document.querySelector('#viewed').textContent,
                originalText:document.querySelector('#command').textContent
            })''')
            assert not state['overflow'] and not state['status'] and not state['truthVisible']
            assert state['submitDisabled'] and not errors
            assert all(response['status'] < 400 for response in responses)
            checks.append(dict(viewport=name, state=state, page_errors=errors,
                               responses=len(responses)))
            page.close()
        browser.close()
    assert all(request['method'] == 'GET' for request in requests)
    with (destination / 'report.json').open('x') as stream:
        json.dump(dict(passed=True, checks=checks, requests=requests,
            all_requests_GET=True, actual_human_records_created=0,
            model_calls_added=0, official_truth_hidden_by_default=True,
            no_source_submit_disabled=True, goal_complete=False), stream,
            ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, viewport_checks=len(checks), requests=len(requests))))


if __name__ == '__main__':
    main()
