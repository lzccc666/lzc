"""隔离合成HTTP服务完整表单测试；正式服务只读取数量，绝不POST。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import threading
from urllib.request import ProxyHandler, build_opener

from playwright.sync_api import sync_playwright


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def main():
    destination = OUTPUT / 'reproduction/human_adjudication_browser_fixture_002'
    destination.mkdir(exist_ok=False)
    specification = importlib.util.spec_from_file_location(
        'synthetic_adjudication_fixture', OUTPUT / 'source_bundle/test_human_query_adjudication_001.py')
    fixtures = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(fixtures)
    store, _payload, original_path = fixtures.case.__wrapped__(destination)
    root = store.root
    shutil.copy2(OUTPUT / 'annotation/human_natural_adjudication_001/index.html', root / 'index.html')
    (root / 'manifest.sha256').write_text(fixtures.MODULE.sha(root / 'manifest.json')+'\n')
    second = deepcopy(fixtures.MODULE.load(original_path))
    second.update(submission_id='d'*32, command='请接近这辆白色汽车并安全停下')
    fixtures.write(store.raw / ('d'*32+'.json'), second)
    opener = build_opener(ProxyHandler({}))

    def formal_status():
        with opener.open('http://127.0.0.1:8770/api/adjudication_status', timeout=10) as response:
            return json.load(response)

    before = formal_status()
    class Handler(fixtures.MODULE.Handler):
        def log_message(self, *_args):
            pass
    Handler.store = store
    server = fixtures.MODULE.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    requests, errors = [], []
    try:
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True,
                executable_path='/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell',
                args=['--no-sandbox', '--no-proxy-server'])
            page = browser.new_page(viewport=dict(width=1280, height=900))
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.goto('http://127.0.0.1:'+str(server.server_port)+'/')
            page.wait_for_function("document.querySelector('#submission').options.length===2")
            page.fill('#target_category', 'old-label-must-clear')
            page.fill('#evidence_note', '上一条合成审核，不得带入第二条')
            page.check('#attest')
            page.select_option('#submission', 'd'*32)
            page.wait_for_function("document.querySelector('#command').textContent==='请接近这辆白色汽车并安全停下'")
            assert not page.locator('#reviewer').is_disabled()
            assert page.input_value('#target_category') == 'unknown'
            assert page.input_value('#evidence_note') == ''
            assert not page.is_checked('#attest')
            for index in [0, 1, 2]:
                page.locator('#frame').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}', index)
                page.wait_for_function("v=>document.querySelector('#frameInfo').textContent.includes('帧 '+v) && document.querySelector('#viewed').textContent.includes(String(v))", arg=index)
            page.locator('#objects input').first.check()
            page.fill('#reviewer', 'SYNTHETIC_FIXTURE_BROWSER')
            page.select_option('#decision', 'accept')
            page.select_option('#task_state', 'approach_static_target')
            page.select_option('#ambiguity', 'unique')
            page.select_option('#observed_motion', 'static')
            page.select_option('#observed_color', 'white')
            page.fill('#target_category', 'car')
            for name in ['fine_attribute', 'relation', 'anchor_instances', 'negation']:
                page.fill('#'+name, 'none')
            page.fill('#expression_family', 'SYNTHETIC_FIXTURE_FAMILY')
            page.check('input[name=required][value=color]')
            page.fill('#evidence_note', '仅在隔离合成HTTP服务检验表单切换与证据绑定，不是人工标注')
            page.check('#attest')
            page.click('#submit')
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('审核已追加：')")
            assert not errors
            records = store.records()
            assert len(records) == 1
            record = records[0]
            assert record['synthetic_fixture'] and record['source_submission_id'] == 'd'*32
            assert record['viewed_frames'] == [0, 1, 2]
            assert record['accepted_instance_tokens'] == ['b'*32]
            assert record['label_eligible_for_retrieval'] and not record['qualifies_for_sealed_test']
            assert record['observed_color'] == 'white' and record['semantic_labels']['target_category'] == 'car'
            page.screenshot(path=str(destination / 'synthetic_fixture_only.png'), full_page=True)
            browser.close()
        assert sum(request['method'] == 'POST' for request in requests) == 1
        assert all(request['url'].startswith('http://127.0.0.1:'+str(server.server_port)+'/') for request in requests)
        after = formal_status()
        for field in ['original_submissions', 'review_records', 'reviewed_submissions', 'qualified_test_queries']:
            assert before[field] == after[field] == 0
        with (destination / 'report.json').open('x') as stream:
            json.dump(dict(passed=True, synthetic_fixture_only=True,
                selected_submission_changed=True, saved_source_id_matches_second_submission=True,
                bound_frames=[0,1,2], bound_instance='b'*32, synthetic_POST_requests=1,
                formal_POST_requests=0, actual_human_labels_added=0,
                formal_before=before, formal_after=after, requests=requests,
                new_test_opened=False, goal_complete=False), stream,
                ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(dict(passed=True, synthetic_fixture_POSTs=1, formal_POSTs=0,
                              actual_human_labels_added=0)))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == '__main__':
    main()
