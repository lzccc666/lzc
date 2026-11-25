"""用真实Chromium检查新证据页和已有KITTI全部序列，不创建标注或模型请求。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT/'visualizations/evidence_browser_082'
OUT = VIEW/'browser_qa_083'
BASE = 'http://127.0.0.1:8877/public_experiment_completion_20261115'
BROWSER = Path('/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8877/api/natural_collection_status', timeout=10) as response:
        return json.load(response)


def ready(page):
    page.wait_for_function("current !== null && ['rgb','detection','depth'].every(id=>{const e=document.getElementById(id);return e.complete&&e.naturalWidth>0;})", timeout=15000)


def main():
    OUT.mkdir(exist_ok=False)
    begun = time.monotonic()
    protected = {str(VIEW/name): sha(VIEW/name) for name in ['index.html', 'viewer.js', 'manifest.json', 'receipt.json']}
    before = status()
    cases, kitti_cases, errors, posts = [], [], [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(BROWSER), headless=True,
                args=['--no-sandbox', '--no-proxy-server', '--disable-gpu'])
            context = browser.new_context(viewport=dict(width=1440, height=1080))
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
            page.goto(BASE+'/visualizations/evidence_browser_082/index.html', wait_until='load', timeout=30000)
            ready(page)
            assert page.locator('#scene option').count() == 30
            assert page.locator('#condition option').count() == 10
            scenes = page.locator('#scene option').evaluate_all('(options)=>options.map(o=>o.value)')
            for scene in [scenes[0], scenes[14], scenes[-1]]:
                page.locator('#scene').select_option(scene)
                page.wait_for_function('(scene)=>current.scene===scene', arg=scene)
                for position in [0, -1]:
                    end = int(page.locator('#seek').get_attribute('max'))
                    index = end if position == -1 else 0
                    page.locator('#seek').evaluate('(el,n)=>{el.value=n;el.dispatchEvent(new Event("input"));}', index)
                    page.wait_for_function('(n)=>current.frame_idx===n', arg=index)
                    ready(page)
                    assert not page.locator('#status').evaluate('(el)=>el.classList.contains("bad")')
                    cases.append(page.evaluate('({scene:current.scene,frame_idx:current.frame_idx,nodes:current.objects.length,edges:current.edges.length,images_loaded:["rgb","detection","depth"].map(id=>document.getElementById(id).naturalWidth)})'))
            # 这是已知关闭关系开关的实现检查，不读取官方关系答案。
            page.locator('#condition').select_option('without_relations')
            page.wait_for_function('current.condition === "without_relations"')
            ready(page)
            assert page.evaluate('current.edges.length') == 0
            page.locator('#condition').select_option('full')
            page.wait_for_function('current.condition === "full"')
            ready(page)
            initial = page.evaluate('[yaw,zoom]')
            box = page.locator('#space').bounding_box()
            page.mouse.move(box['x']+120, box['y']+120)
            page.mouse.down()
            page.mouse.move(box['x']+200, box['y']+160, steps=5)
            page.mouse.up()
            page.mouse.wheel(0, -100)
            page.wait_for_timeout(200)
            changed = page.evaluate('[yaw,zoom]')
            assert initial[0] != changed[0] and initial[1] != changed[1]
            page.screenshot(path=str(OUT/'nuscenes_last_frame.png'), full_page=True)
            page.goto(BASE+'/kitti_tracking_expanded/index.html', wait_until='load', timeout=30000)
            page.wait_for_function('records.length===1000 && document.querySelector("#rgb").naturalWidth>0')
            sequences = page.locator('#sequence option').evaluate_all('(options)=>options.map(o=>o.value)')
            assert len(sequences) == 10
            for sequence in sequences:
                page.locator('#sequence').select_option(sequence)
                for index in [0, 99]:
                    page.locator('#frame').evaluate('(el,n)=>{el.value=n;el.dispatchEvent(new Event("input"));}', index)
                    page.wait_for_function("['rgb','detection'].every(id=>{const e=document.getElementById(id);return e.complete&&e.naturalWidth>0})")
                    assert f'{index}/99' in page.locator('#number').inner_text()
                    kitti_cases.append(dict(sequence=sequence, frame_idx=index, image_loaded=True))
            page.screenshot(path=str(OUT/'kitti_last_frame.png'), full_page=True)
            browser.close()
        after = status()
        assert not errors and not posts
        assert all(sha(p) == h for p, h in protected.items())
        report = dict(UTC=datetime.now(timezone.utc).isoformat(), scenes_available=30,
            conditions_available=10, nuscenes_browser_cases=cases, kitti_sequences=10,
            kitti_browser_cases=kitti_cases, actual_browser='existing Chromium / software rendering',
            relation_off_switch_verified=True, canvas_rotate_zoom_verified=True,
            browser_errors=errors, POST_requests=posts, human_queue_before=before,
            human_queue_after=after, source_files_unchanged=True, sources_sha256=protected,
            browser_runtime_sha256=sha(BROWSER), elapsed_s=time.monotonic()-begun,
            screenshots_visual_inspection_pending=True,
            new_official_labels_read=False, actual_new_GPU_calls=0, actual_new_CARLA_calls=0,
            goal_complete=False)
        with (OUT/'report.json').open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(json.dumps(dict(nuscenes_checked_frames=len(cases), kitti_checked_frames=len(kitti_cases),
                             browser_errors=errors, POST_requests=posts, goal_complete=False)))
    except BaseException as exc:
        with (OUT/'failure.json').open('x') as f:
            json.dump(dict(error=repr(exc), browser_errors=errors, POST_requests=posts,
                           completed_nuscenes_cases=cases, completed_kitti_cases=kitti_cases,
                           source_files_preserved=True, goal_complete=False), f, ensure_ascii=False, indent=2)
        raise


if __name__ == '__main__':
    main()
