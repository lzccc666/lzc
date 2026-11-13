"""门户017实际桌面/手机只读验收及交互服务入口检查；不代填人工、不执行导航。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from playwright.sync_api import sync_playwright

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
BASE = 'http://127.0.0.1:8768'
VIEW = 'thesis_portal_017'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    dest = OUT/'visualizations/thesis_portal_browser_qa_021'
    dest.mkdir(exist_ok=False)
    viewer = OUT/'visualizations'/VIEW
    manifest = json.loads((viewer/'manifest.json').read_text())
    assert sha(viewer/'index.html') == manifest['html_sha256']
    assert sha(OUT/manifest['source_path']) == manifest['source_sha256']
    sources = json.loads((viewer/'completed_evidence.json').read_text())
    assert all(sha(OUT/name) == value for name, value in sources.items())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE+'/api/review_status') as response:
        before = json.load(response)
    requests, errors, checks, checked_links = [], [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(PROJECT/'outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'), headless=True, args=['--no-sandbox', '--no-proxy-server'])
        for viewport, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
            assert time.monotonic()-started < 240
            page = browser.new_page(viewport=dict(width=width, height=height), device_scale_factor=1)
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(BASE+'/visualizations/'+VIEW+'/index.html')
            page.wait_for_function("document.querySelector('#refresh-time').textContent.includes('页面读取时间')")
            assert 'Goal仍在进行' in page.locator('#goal').inner_text()
            assert '未开启' in page.locator('#updated').inner_text()
            assert not page.locator('#error').inner_text() and not errors
            assert page.locator('#paper-links a').count() == 0
            assert '暂缓论文正文和框架' in page.locator('#paper-status').inner_text()
            assert page.locator('.four strong').all_text_contents() == ['生成路线', '选对对象', '正确锁定', '实际到达']
            assert page.locator('#live-navigation').get_attribute('href') == 'http://127.0.0.1:8769/'
            assert '预留标签：独立静态终点区域' in page.locator('body').inner_text()
            assert '预留标签：独立条件静态可达参考' in page.locator('body').inner_text()
            assert '当前390源码准备' in page.locator('body').inner_text()
            assert '450 / 450' in page.locator('#main-progress').inner_text()
            assert '完整结果 449，缺完整结果 1' in page.locator('#main-progress').inner_text()
            expected = {'rule': ['83/90', '84/90', '82/90', '50/90'],
                        'base_full': ['66/90', '72/90', '66/90', '39/90'],
                        'new_full': ['59/90', '33/90', '32/90', '4/90']}
            for method, values in expected.items():
                assert page.locator(f'tr[data-method="{method}"] td').all_text_contents() == values
            assert '合格新测试指令 0 条' in page.locator('#collection').inner_text()
            assert '缺一语义新增记录' in page.locator('#minus-progress').inner_text()
            assert '180全量结果已生成' in page.locator('#fault-progress').inner_text()
            assert '216无缓存Qwen' in page.locator('#performance-progress').inner_text()
            body = page.locator('body').inner_text()
            for term in ['新Adapter下游没有提升', '54例初态', '57,460感知帧', '334,607保存控制记录',
                         '64真实帧、33个drive', '247/319', '262/319', 'truck误差未改善',
                         'Cyclist代理0/20', '固定3个序列、90帧', '全部74次主类别GT只在0005', '532个身份条件', '2,028' if '2,028' in body else '2028',
                         '39参考位姿到达', '语义到达信用0', '相机之前的实测LiDAR']:
                assert term in body, term
            assert page.locator('#main-outcomes td').evaluate_all("cells=>cells.every(c=>getComputedStyle(c).whiteSpace==='nowrap'&&(()=>{let r=document.createRange();r.selectNodeContents(c);return r.getClientRects().length===1&&r.getBoundingClientRect().width<=c.clientWidth-14})())")
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            page.screenshot(path=str(dest/f'{viewport}_main.png'))
            page.locator('#kitti').scroll_into_view_if_needed()
            page.screenshot(path=str(dest/f'{viewport}_kitti.png'))
            hrefs = page.locator('main > section a, main > .grid a').evaluate_all(
                "as=>[...new Set(as.filter(a=>!a.closest('#statuses')).map(a=>a.getAttribute('href'))) ]")
            if viewport == 'desktop':
                for path in hrefs:
                    response = page.request.get(path if path.startswith('http') else BASE+path)
                    assert response.status == 200, (path, response.status)
                    checked_links.append(dict(path=path, status=response.status))
                    response.dispose()
            page.route('**/api/natural_collection_status', lambda route: route.fulfill(status=503, body='unavailable'))
            page.locator('#refresh').click()
            page.wait_for_function("document.querySelector('#collection').textContent.includes('数量未知')")
            assert not page.locator('#refresh').is_disabled()
            page.unroute('**/api/natural_collection_status')
            page.route('**/realtime_three_model_full_001/trial_records.jsonl',
                       lambda route: route.fulfill(status=200, body='{"bad":\n', content_type='application/json'))
            page.locator('#refresh').click()
            page.wait_for_function("document.querySelector('#main-progress').textContent.includes('数量未知')")
            page.unroute('**/realtime_three_model_full_001/trial_records.jsonl')
            page.route('**/validation_report.json', lambda route: route.fulfill(status=503, body='unavailable'))
            page.locator('#refresh').click()
            page.wait_for_function("document.querySelector('#goal').textContent.includes('无法读取')")
            assert page.locator('#statuses tr').count() == 0 and page.locator('#paper-links a').count() == 0
            page.unroute('**/validation_report.json')
            page.locator('#refresh').click()
            page.wait_for_function("document.querySelector('#goal').textContent.includes('Goal仍在进行')")
            page.wait_for_function("document.querySelector('#main-progress').textContent.includes('450 / 450')")
            assert not page.locator('#error').inner_text()
            # 合成HTTP响应只检查追加行解析，不进入页面业务结果或真实记录。
            page.route('**/qa_only_partial.jsonl', lambda route: route.fulfill(
                status=200, body='{"first":1}\n{"unfinished":', content_type='application/json'))
            assert page.evaluate("async()=>await records('/qa_only_partial.jsonl')") == [{'first': 1}]
            page.unroute('**/qa_only_partial.jsonl')
            checks.append(dict(viewport=viewport, main_four_outcomes=expected, no_body_overflow=True,
                paper_links_hidden_for_deferred_manuscript=True, three_label_counts_separate=True,
                live_progress_not_process_claim=True, missing_completion_not_success=True,
                invalid_log_shows_unknown=True, failed_validation_clears_stale_rows=True,
                refresh_recovers=True, incomplete_last_line_ignored=True,
                status_rows=page.locator('#statuses tr').count(), page_errors=list(errors)))
            page.close()
        browser.close()
    with opener.open(BASE+'/api/review_status') as response:
        after = json.load(response)
    assert requests and all(request['method'] == 'GET' for request in requests) and not errors
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), viewer=VIEW,
        manifest_sha256=sha(viewer/'manifest.json'), completed_evidence_sha256=sha(viewer/'completed_evidence.json'),
        QA_source_sha256=sha(__file__), checks=checks, completed_input_files_SHA_verified=len(sources),
        static_HTTP_links=checked_links, only_GET=True, browser_requests=len(requests),
        human_before=before, human_after=after, human_records_written=0,
        actual_model_calls=0, new_executions=0, new_arrivals=0,
        CPU_shared_wall_s=time.monotonic()-started, CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        limits=['门户显示的是生成时核验的冻结开发证据及即时追加数量，不能据数量证明进程目前存活',
                '缺失/损坏降级使用浏览器拦截的合成HTTP响应，不写真实文件或人工记录',
                '未完成的消融/GPU故障/独占性能仍须真实运行和后续独立审核'], goal_complete=False)
    with (dest/'report.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(passed=True, viewport_checks=len(checks), static_links=len(checked_links),
                         only_GET=True, goal_complete=False)))


if __name__ == '__main__':
    main()
