"""桌面和手机实际操作新CPU会话；操作日志不是人工标注。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

from playwright.sync_api import sync_playwright

from conceptgraph.completion.stage9 import OUTPUT, digest, dump

BASE = 'http://127.0.0.1:8769'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with OPENER.open(BASE+path, timeout=10) as response:
        return json.load(response)


def wait_state(page, expression, timeout=120000):
    page.wait_for_function(expression, timeout=timeout)
    return page.evaluate('state')


def check_page(page):
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
    assert page.locator('img[src]').count() == 3
    page.wait_for_function('Array.from(document.querySelectorAll("img[src]")).every(x=>x.complete&&x.naturalWidth===640)')
    assert not page.locator('#error').inner_text()


def main():
    begun = time.monotonic()
    destination = OUTPUT/'visualizations/live_navigation_browser_qa_001'
    destination.mkdir(exist_ok=False)
    assert get('/api/live/state')['phase'] == 'IDLE'
    errors, checkpoints = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
        desktop = browser.new_page(viewport=dict(width=1440, height=1000))
        desktop.on('pageerror', lambda error:errors.append(str(error)))
        desktop.goto(BASE, wait_until='networkidle')
        desktop.wait_for_function('tasks.length===50')
        desktop.select_option('#task', 'config00-positive0')
        desktop.select_option('#seed', '17')
        desktop.click('#start')
        initial = wait_state(desktop, 'state?.phase==="PAUSED" && state.counts.actual_frontend_calls===1')
        run_id = initial['run_id']
        checkpoints.append(dict(action='initial_observation', state=initial['state']))
        for index in range(6):
            desktop.click('#step')
            target = (index+1)*.5
            snapshot = wait_state(desktop, f'state?.phase==="PAUSED" && state.pending_ticks===0 && Math.abs(state.state.time_s-{target})<1e-8')
            checkpoints.append(dict(action='step', state=snapshot['state']))
        assert snapshot['state']['speed_mps'] > 0
        frozen = snapshot['state']
        time.sleep(1.2)
        assert get('/api/live/state')['state'] == frozen
        check_page(desktop)
        desktop.screenshot(path=str(destination/'desktop_paused.png'), full_page=True)
        desktop.click('#run')
        wait_state(desktop, 'state?.state?.time_s>=5.0')
        desktop.click('#pause')
        paused = wait_state(desktop, 'state?.phase==="PAUSED" && state.mode==="PAUSED" && state.pending_ticks===0')
        time.sleep(1.2)
        assert get('/api/live/state')['state'] == paused['state']
        checkpoints.append(dict(action='continuous_then_pause', state=paused['state']))
        mobile = browser.new_page(viewport=dict(width=390, height=844), is_mobile=True)
        mobile.on('pageerror', lambda error:errors.append(str(error)))
        mobile.goto(BASE, wait_until='networkidle')
        wait_state(mobile, 'state?.phase==="PAUSED"')
        mobile.click('#step')
        target = paused['state']['time_s']+.5
        advanced = wait_state(mobile, f'state?.phase==="PAUSED" && Math.abs(state.state.time_s-{target})<1e-8')
        checkpoints.append(dict(action='mobile_step', state=advanced['state']))
        check_page(mobile)
        mobile.screenshot(path=str(destination/'mobile_before_stop.png'), full_page=True)
        mobile.click('#stop')
        final = wait_state(mobile, 'state?.phase==="FINISHED" && state.result && !state.worker_alive')
        assert final['run_id'] == run_id
        assert final['result']['status'] == 'USER_STOPPED' and final['result']['terminal_zero_speed']
        assert not final['result']['actual_full_perception_arrival']
        assert final['result']['distinct_RGB'] > 1 and final['result']['distinct_depth'] > 1
        assert 'llvmpipe' in final['counts']['actual_renderer'].lower()
        assert final['result']['independent_execution_audit']['passed']
        check_page(mobile)
        mobile.screenshot(path=str(destination/'mobile_stopped.png'), full_page=True)
        links = mobile.locator('#links a').evaluate_all('rows=>rows.map(x=>x.getAttribute("href"))')
        for link in links:
            with OPENER.open(BASE+link, timeout=10) as response:
                assert response.status == 200
        assert not errors, errors
        browser.close()
    case = OUTPUT/'interactive_simulation/live_navigation_sessions_001'/run_id
    files = {str(p.relative_to(OUTPUT)):digest(p) for p in case.rglob('*') if p.is_file()}
    # state.json会更新至最终终态，此处worker实际已退出，所有源证据固定。
    commands = [json.loads(line) for line in (case/'commands.jsonl').read_text().splitlines()]
    processed = [json.loads(line) for line in (case/'processed_commands.jsonl').read_text().splitlines()]
    assert len(commands) == len(processed)
    assert {r['command_id'] for r in commands} == {r['command_id'] for r in processed}
    assert {'step', 'run', 'pause', 'stop'} <= {r['command'] for r in commands}
    dump(destination/'files.json', files)
    dump(destination/'report.json', dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)), session=run_id, checkpoints=checkpoints,
        actual_frontend_calls=final['counts']['actual_frontend_calls'], commands_applied=len(processed),
        RGB_and_LiDAR_changed_after_actual_controls=True, pause_freezes_state=True,
        user_stop_integrated_to_zero_speed=True, desktop_mobile_actual_operation=True,
        actual_renderer=final['counts']['actual_renderer'], checked_HTTP_links=len(links),
        final_result=final['result'], files_manifest_sha256=digest(destination/'files.json'),
        original_frozen_table_credit=False, automated_browser_operations_are_not_human_labels=True,
        actual_Qwen_calls=0, performance_credit=False, manuscript_text_written=False,
        shared_CPU_wall_s=time.monotonic()-begun, goal_complete=False))
    print(json.dumps(dict(passed=True, session=run_id, actual_frontend_calls=final['counts']['actual_frontend_calls'],
        controls_applied=len(processed), status=final['result']['status'], goal_complete=False)))


if __name__ == '__main__':
    main()
