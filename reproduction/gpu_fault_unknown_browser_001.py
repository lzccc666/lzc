"""以真实超时试验的保存记录作浏览器回归夹具，验证未知暴露不显示为未触发。"""

import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

from conceptgraph.completion.navigation_demo import build_case
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    root = OUTPUT / 'visualizations/gpu_fault_evidence_smoke_004'
    destination = OUTPUT / 'visualizations/gpu_fault_unknown_browser_001'
    destination.mkdir(exist_ok=False)
    records = OUTPUT / 'interactive_simulation/realtime_faults_gpu_full_001/trial_records.jsonl'
    record = next(json.loads(line) for line in records.read_text().splitlines()
                  if json.loads(line)['run_id'] == 'realtime_faults_gpu_full_001_lidar_dropout_80pct_config01_positive0_s29')
    assert record['result'] is None and record['process_timeout']
    assert record['fault_exposure']['exposure_unknown'] and record['fault_exposure']['actual_fault_exposed'] is None
    tasks = json.loads((OUTPUT / 'interactive_simulation/synthetic_tasks_v1.json').read_text())
    task = next(t for t in tasks['tasks'] if t['task_id'] == record['task'])
    config = next(c for c in tasks['configs'] if c['config_id'] == record['config_id'])
    case = build_case(Path(record['raw_case']), task, config)
    case.update(fault_evidence=record['fault_exposure'], fault_condition=record['fault'], process_timeout=True)
    for frame in case['frames']:
        sensor = Path(record['raw_case']) / 'sensor_evidence' / f"{frame['frame_idx']:06d}.json"
        frame['fault_active'] = json.loads(sensor.read_text())['fault_active']
    page_text = (root / 'index.html').read_text()
    page_text = re.sub(r'<section class="card" id="faultPanel">.*?</section>',
        '<section class="card" id="faultPanel"><h2>超时记录浏览器回归夹具</h2>'
        '<p>只验证缺完整结果的显示，不新增实验、模型调用或到达。</p>'
        '<p id="faultFrame" class="muted"></p></section>', page_text, count=1)
    manifest = dict(cases=[dict(run=record['run_id'], task_id=record['task'], seed=record['seed'],
        kind=record['kind'], status=case['status'], arrived=False, file='negative_case.json')])
    checks, requests, errors = [], [], []
    executable = '/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, headless=True,
            args=['--no-sandbox', '--no-proxy-server', '--disable-gpu'])
        for name, width in [('desktop', 1440), ('mobile', 390)]:
            page = browser.new_page(viewport=dict(width=width, height=1000))
            page.on('request', lambda r: requests.append(r.method))
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.route('**/gpu_fault_evidence_smoke_004/index.html', lambda route: route.fulfill(
                status=200, content_type='text/html', body=page_text))
            page.route('**/gpu_fault_evidence_smoke_004/manifest.json', lambda route: route.fulfill(json=manifest))
            page.route('**/gpu_fault_evidence_smoke_004/negative_case.json', lambda route: route.fulfill(json=case))
            page.goto('http://127.0.0.1:8768/visualizations/gpu_fault_evidence_smoke_004/index.html')
            page.wait_for_function('(run)=>document.querySelector("#status").textContent.startsWith(run)', arg=record['run_id'])
            text = page.locator('#faultFrame').inner_text()
            assert '实际触发：未知（缺少完整终态）' in text
            assert '实际触发：否' not in text and 'undefined' not in text
            assert '进程超时：是' in text and record['fault'] in text
            outcomes = page.locator('#outcomes .outcome').all_inner_texts()
            assert len(outcomes) == 4 and all(t.endswith('结果未保存') for t in outcomes)
            assert not page.locator('#gt').is_checked()
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            assert not errors
            page.locator('#faultPanel').screenshot(path=str(destination / (name + '_unknown.png')))
            checks.append(dict(viewport=name, actual_label=text, outcomes=outcomes,
                unknown_not_false=True, missing_results_not_success=True))
            page.close()
        browser.close()
    assert requests and set(requests) == {'GET'}
    dump(destination / 'report.json', dict(passed=True, checks=checks,
        actual_failed_run=record['run_id'], failed_process_wall_s=record['process_wall_s'],
        original_raw_sources=case['sources_sha256'], viewer_source_sha256=digest(root / 'generator_source.py'),
        checker_source_sha256=digest(__file__), route_injection_is_UI_fixture=True,
        production_matrix_modified=False, actual_model_calls=0, new_executions=0,
        new_arrivals=0, page_errors=errors, goal_complete=False))
    print(json.dumps(dict(passed=True, checks=len(checks), actual_model_calls=0, new_arrivals=0)))


if __name__ == '__main__':
    main()
