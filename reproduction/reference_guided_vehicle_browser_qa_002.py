"""完整连接控制查看器桌面/手机核验，只读取原始状态与失败条件。"""
import json

from playwright.sync_api import sync_playwright

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    root = OUTPUT/'visualizations/HDmap_reference_guided_browser_qa_003'
    root.mkdir(exist_ok=False)
    cards = json.loads((OUTPUT/'visualizations/HDmap_reference_guided_preview_003/cards.json').read_text())
    statuses = sorted({c['status'] for c in cards}-{'REFERENCE_POSE_REACHED'})+['REFERENCE_POSE_REACHED']
    cases = [next(c for c in cards if c['status'] == status) for status in statuses]
    requests, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
        for name, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
            page = browser.new_page(viewport=dict(width=width, height=height))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.goto('http://127.0.0.1:8768/visualizations/HDmap_reference_guided_preview_003/index.html')
            page.wait_for_function('cards.length===228 && checks && current && current.audit && !document.querySelector("#status").textContent.includes("正在")')
            assert page.locator('#scene option').count() == 38
            for card in cases:
                page.select_option('#scene', card['scene'])
                page.select_option('#method', card['method'])
                page.select_option('#seed', str(card['seed']))
                page.wait_for_function('(c)=>current && current.scene===c.scene && current.method===c.method && current.seed===c.seed && !document.querySelector("#status").textContent.includes("正在")', arg=card)
                observed = page.evaluate('({scene:current.scene,method:current.method,seed:current.seed,status:current.status,states:states.length,distinct_connections:current.audit.accepted_distinct_connections,candidate_intervals:current.audit.connection_candidate_intervals||0})')
                assert observed['status'] == card['status']
                raw = json.loads((OUTPUT/card['result_path']).read_text())
                assert observed['candidate_intervals'] == raw.get('connection_candidate_intervals', 0)
                assert observed['distinct_connections'] == (raw.get('accepted_distinct_connections') or 0)
                if card['states_path']:
                    states = json.loads((OUTPUT/card['states_path']).read_text())
                    for index in sorted({0, len(states)//2, len(states)-1}):
                        page.evaluate('(index)=>{document.getElementById("time").value=index;document.getElementById("time").dispatchEvent(new Event("input"))}', index)
                        actual = page.evaluate('states[Number(document.getElementById("time").value)]')
                        assert actual == states[index]
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    assert f"全程候选连接控制区间 {observed['candidate_intervals']}" in page.locator('#details').inner_text()
                    if len(states) > 1:
                        page.evaluate('()=>{document.getElementById("time").value=0;document.getElementById("time").dispatchEvent(new Event("input"))}')
                        page.click('#play')
                        page.wait_for_function('Number(document.getElementById("time").value)>=1')
                        page.click('#play')
                        assert page.locator('#play').inner_text() == '播放保存状态'
                else:
                    assert page.locator('#play').is_disabled()
                    assert '没有实际控制状态' in page.locator('#details').inner_text()
                checks.append(dict(viewport=name, **observed))
            assert '全程完整见证过渡 2 次' in page.locator('#details').inner_text()
            local = page.locator('#map').evaluate('(canvas)=>canvas.toDataURL()')
            page.select_option('#zoom', 'corridor')
            complete = page.locator('#map').evaluate('(canvas)=>canvas.toDataURL()')
            assert local != complete
            page.select_option('#zoom', 'trajectory')
            page.evaluate('()=>{document.getElementById("time").value=Math.floor(states.length/2);document.getElementById("time").dispatchEvent(new Event("input"))}')
            page.screenshot(path=str(root/f'{name}.png'), full_page=True)
            assert not errors
            page.close()
        browser.close()
    assert all(r['method'] == 'GET' for r in requests)
    dump(root/'requests.json', requests)
    dump(root/'report.json', dict(passed=True, cases_total=228, scenes_total=38, viewports=2,
        condition_checks=len(checks), checks=checks, original_saved_states_compared=True, candidate_and_transition_counters_match_original=True, local_and_full_view_checked=True,
        playback_checked=True, missing_execution_disables_playback=True, browser_errors=0,
        horizontal_overflow=False, POST_requests=0, model_calls=0, semantic_arrival_credit=0,
        page_manifest_sha256=digest(OUTPUT/'visualizations/HDmap_reference_guided_preview_003/manifest.json'), goal_complete=False))
    print(json.dumps(dict(passed=True, condition_checks=len(checks), model_calls=0)))


if __name__ == '__main__':
    main()
