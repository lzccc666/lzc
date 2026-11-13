"""独立核对GPU故障查看包、浏览器交互及四类结果；只进行GET请求。"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    root = OUTPUT / 'visualizations' / args.run_id
    destination = OUTPUT / 'visualizations' / args.output_id
    destination.mkdir(exist_ok=False)
    manifest = json.loads((root / 'manifest.json').read_text())
    source = OUTPUT / 'interactive_simulation' / manifest['source_matrix'] / 'trial_records.jsonl'
    assert digest(source) == manifest['input_records_sha256']
    records = {r['run_id']: r for r in map(json.loads, source.read_text().splitlines())}
    assert set(records) == {c['run'] for c in manifest['cases']}
    for name, expected in manifest['files'].items():
        assert digest(root / name) == expected, name
    frame_count = 0
    for case in manifest['cases']:
        data = json.loads((root / case['file']).read_text())
        original = records[case['run']]
        assert data['result'] == original['result']
        assert data['fault_evidence'] == original['fault_exposure']
        for frame in data['frames']:
            assert '/effective_detections/' + case['run'] + '/' in frame['detection_vis']
        frame_count += len(data['frames'])
    assert frame_count == manifest['effective_detection_images']
    lineage = [json.loads(line) for line in (root / 'effective_detection_sources.jsonl').read_text().splitlines()]
    assert len(lineage) == frame_count
    for row in lineage:
        assert digest(row['detection_source']) == row['detection_sha256']
        assert digest(row['sensor_source']) == row['sensor_sha256']
        assert digest(root / row['output_file']) == row['output_sha256']
    checks, requests, errors = [], [], []
    executable = Path('/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True,
            args=['--no-sandbox', '--no-proxy-server', '--disable-gpu'])
        for viewport, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
            page = browser.new_page(viewport=dict(width=width, height=height))
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(request.method))
            page.goto('http://127.0.0.1:8768/visualizations/' + args.run_id + '/index.html')
            page.wait_for_function('(n)=>document.querySelectorAll("#case option").length===n', arg=len(records))
            assert not page.locator('#gt').is_checked()
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            for link in page.locator('#faultPanel a').evaluate_all('a=>a.map(x=>x.getAttribute("href"))'):
                assert page.request.get('http://127.0.0.1:8768' + link).status == 200
            seen = set()
            for case in manifest['cases']:
                record = records[case['run']]
                condition = record['fault']
                if viewport == 'mobile' and condition in seen:
                    continue
                seen.add(condition)
                data = json.loads((root / case['file']).read_text())
                page.locator('#case').select_option(case['file'])
                page.wait_for_function('(run)=>document.querySelector("#status").textContent.startsWith(run)', arg=case['run'])
                assert page.locator('#outcomes .outcome').count() == 4
                keys = ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival']
                for i, key in enumerate(keys):
                    expected = ('是' if record['result'][key] else '否') if record['result'] else '结果未保存'
                    assert page.locator('#outcomes .outcome').nth(i).inner_text().endswith('：' + expected)
                indices = sorted({0, max(0, len(data['frames']) - 1)} | {
                    next((i for i, f in enumerate(data['frames']) if f['fault_active']), 0)})
                for index in indices:
                    if not data['frames']:
                        assert '没有完整保存的感知帧' in page.locator('#error').inner_text()
                        break
                    page.locator('#frame').fill(str(index))
                    page.locator('#frame').dispatch_event('input')
                    expected_src = data['frames'][index]['detection_vis']
                    page.wait_for_function('(src)=>{let e=document.querySelector("#det");return e.getAttribute("src")===src&&e.complete&&e.naturalWidth>0}', arg=expected_src)
                    assert not page.locator('#error').inner_text()
                    state = '是' if data['frames'][index]['fault_active'] else '否'
                    assert '当帧故障生效：' + state in page.locator('#faultFrame').inner_text()
                for link in page.locator('#sources a').evaluate_all('a=>a.map(x=>x.getAttribute("href"))'):
                    assert page.request.get('http://127.0.0.1:8768' + link).status == 200
                assert not errors
                checks.append(dict(viewport=viewport, run=case['run'], checked_frame_positions=indices,
                    four_outcomes_match_raw=True, GT_hidden=not page.locator('#gt').is_checked()))
            page.screenshot(path=str(destination / (viewport + '.png')), full_page=False)
            page.locator('#gt').check()
            if data['frames']:
                assert '评估层隐藏' not in page.locator('#execution').inner_text()
            page.locator('#gt').uncheck()
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            page.close()
        browser.close()
    assert requests and set(requests) == {'GET'}
    dump(destination / 'report.json', dict(passed=True, source_manifest_sha256=digest(root / 'manifest.json'),
        source_sha256=digest(__file__), cases=len(records), all_frame_lineages_verified=frame_count,
        checks=checks, page_errors=errors, read_only_browser_requests=len(requests),
        request_methods=sorted(set(requests)), actual_model_calls=0, human_records_written=0,
        new_arrivals=0, goal_complete=False))
    print(json.dumps(dict(passed=True, cases=len(records), checks=len(checks), frames=frame_count)))


if __name__ == '__main__':
    main()
