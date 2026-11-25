"""真实浏览器离线读图、保存和导出验证；夹具不进入人工队列。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from playwright.sync_api import sync_playwright

from conceptgraph.completion.natural_collection import validate_collection

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT/'annotation/natural_offline_087/自然指令填写_离线试用.html'
OUT = ROOT/'annotation/natural_offline_087/browser_qa_088'
BROWSER = Path('/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell')


def status():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8877/api/natural_collection_status', timeout=10) as response:
        return json.load(response)


def ready(page, scene, frame_idx):
    page.wait_for_function('(a)=>scene.scene===a[0]&&frame.frame_idx===a[1]&&img!==null&&img.naturalWidth>0',
                           arg=[scene, frame_idx], timeout=10000)


def main():
    OUT.mkdir(exist_ok=False)
    before = status()
    errors, requests, images = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(BROWSER), headless=True,
                    args=['--no-sandbox', '--disable-gpu', '--no-proxy-server'])
        context = browser.new_context(viewport=dict(width=1350, height=1000), accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url)
                if request.url.startswith(('http:', 'https:')) else None)
        page.goto(HTML.as_uri(), wait_until='load')
        scenes = page.locator('#scene option').evaluate_all('(xs)=>xs.map(x=>x.value)')
        assert len(scenes) == 10
        for scene in scenes:
            page.select_option('#scene', scene)
            for frame in range(6):
                page.locator('#frame').evaluate('(el,n)=>{el.value=n;el.dispatchEvent(new Event("input"));}', frame)
                ready(page, scene, frame)
                assert not page.locator('#submit').is_disabled()
                images.append(dict(scene=scene, frame_idx=frame,
                                   width=page.evaluate('img.naturalWidth')))
        page.select_option('#scene', scenes[0])
        ready(page, scenes[0], 0)
        page.screenshot(path=str(OUT/'offline_ready.png'), full_page=True)
        # 明确的软件夹具，只在隔离浏览器和本QA目录内；从不POST/写入人工目录。
        page.fill('#author', 'QA_SOFTWARE_FIXTURE_NOT_HUMAN')
        page.fill('#command', '软件接口检查用文本，不构成人工数据')
        page.fill('#note', 'QA夹具：不描述真实观察，不作实验标注')
        page.click('#submit')
        assert page.evaluate('rows.length') == 0
        assert '未保存' in page.locator('#status').inner_text()
        page.check('#attest')
        page.locator('#rgb').click(position=dict(x=200, y=100))
        page.click('#submit')
        assert page.evaluate('rows.length') == 1
        with page.expect_download() as download:
            page.click('#export')
        fixture = OUT/'SOFTWARE_QA_ONLY_NOT_HUMAN_export.json'
        download.value.save_as(fixture)
        raw = json.loads(fixture.read_text())
        assert len(raw['records']) == 1
        checked = validate_collection(raw['records'][0], output=ROOT.parent)
        assert checked['qualifies_for_sealed_test'] is False
        assert checked['target_identity'] is None
        page.reload(wait_until='load')
        ready(page, scenes[0], 0)
        assert page.evaluate('rows.length') == 1
        page.set_input_files('#import', str(fixture))
        page.wait_for_function('document.getElementById("status").textContent.includes("恢复 0 条")')
        assert page.evaluate('rows.length') == 1
        # 新浏览器上下文验证从文件恢复，禁止把本地缓存偶然存在当作恢复成功。
        other = browser.new_context(viewport=dict(width=1200, height=900))
        fresh = other.new_page()
        fresh.goto(HTML.as_uri(), wait_until='load')
        ready(fresh, scenes[0], 0)
        assert fresh.evaluate('rows.length') == 0
        fresh.set_input_files('#import', str(fixture))
        fresh.wait_for_function('rows.length===1')
        other.close()
        context.close()
        browser.close()
    after = status()
    assert before == after
    assert not requests and not errors
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), real_file_protocol_test=True,
                  browser_offline_images_loaded=images, image_count=len(images),
                  browser_errors=errors, network_HTTP_requests=requests,
                  export_validates_against_original_campaign=True,
                  missing_attestation_rejected=True, reload_retention_verified=True,
                  duplicate_restore_ignored=True, new_context_file_restore_verified=True,
                  human_queue_before=before, human_queue_after=after,
                  software_fixture_is_not_human_record=True, software_fixture_never_imported_to_server=True,
                  screenshot_visual_inspection_pending=True, actual_new_GPU_calls=0,
                  actual_new_CARLA_calls=0, human_records_created=0, goal_complete=False)
    with (OUT/'report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(images=len(images), http_requests=len(requests), browser_errors=errors,
                         human_records_created=0, goal_complete=False)))


if __name__ == '__main__':
    main()
