"""真实采集页桌面/手机检查；正式目录不产生任何成功人工提交。"""
import argparse
import json
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from playwright.sync_api import sync_playwright

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = OUTPUT/'visualizations'/args.run_id
    root.mkdir(exist_ok=False)
    local_open = build_opener(ProxyHandler({})).open
    base = f'http://127.0.0.1:{args.port}'

    def read(path):
        with local_open(base+path, timeout=10) as response:
            return json.load(response)

    before = read('/api/review_status')
    collection_before = read('/api/natural_collection_status')
    requests, checks = [], []
    cards = json.loads((OUTPUT/'annotation/human_natural_collection_001/cards.json').read_text())
    with sync_playwright() as pl:
        browser = pl.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
        for name, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
            page = browser.new_page(viewport=dict(width=width, height=height))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.goto(base+'/annotation/human_natural_collection_001/index.html')
            page.wait_for_function("cards.length===30 && stats && loadedImage && loadedImage.naturalWidth===1600")
            assert page.locator('#command').input_value() == ''
            assert not page.locator('#attest').is_checked()
            assert page.locator('#scene option').count() == 30
            for scene in [cards[0], cards[-1]]:
                page.select_option('#scene', scene['scene'])
                for index in [0, len(scene['frames'])//2, len(scene['frames'])-1]:
                    page.evaluate("index=>{document.getElementById('frame').value=index;document.getElementById('frame').dispatchEvent(new Event('input'))}", index)
                    page.wait_for_function('(index)=>currentFrame.frame_idx===index && loadedImage && loadedImage.complete', arg=index)
                    observed = page.evaluate("({scene:currentScene.scene,frame:currentFrame.frame_idx,camera:currentFrame.camera_token,sha:currentFrame.rgb_sha256,width:loadedImage.naturalWidth,height:loadedImage.naturalHeight,point})")
                    frame = scene['frames'][index]
                    assert observed['camera'] == frame['camera_token'] and observed['sha'] == frame['rgb_sha256']
                    assert observed['width'] == 1600 and observed['height'] == 900 and observed['point'] is None
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    checks.append(dict(viewport=name, **observed))
            page.evaluate("()=>{document.getElementById('frame').value=0;document.getElementById('frame').dispatchEvent(new Event('input'))}")
            page.wait_for_function('loadedImage && currentFrame.frame_idx===0')
            page.click('#play')
            page.wait_for_function('currentFrame.frame_idx>=1 && loadedImage')
            page.click('#play')
            assert page.locator('#play').inner_text() == '播放'
            page.locator('#rgb').click(position=dict(x=60, y=40))
            assert page.evaluate('point !== null && point.every(v=>v>=0&&v<=1)')
            page.click('#clearPoint')
            assert page.evaluate('point === null')
            page.screenshot(path=str(root/f'{name}.png'), full_page=True)
            assert not errors
            page.close()
        browser.close()
    assert not [r for r in requests if r['method'] != 'GET']
    invalid = Request(base+'/api/natural_collections', data=b'{}', headers={'Content-Type': 'application/json'})
    try:
        local_open(invalid, timeout=10)
        raise AssertionError('无效提交被接收')
    except HTTPError as exc:
        assert exc.code == 400
    after = read('/api/review_status')
    collection_after = read('/api/natural_collection_status')
    assert before == after and collection_before == collection_after
    for path in ['/annotation/natural_command_review_001/index.html', '/visualizations/review.html']:
        with local_open(base+path, timeout=10) as response:
            assert response.status == 200
    dump(root/'requests.json', requests)
    dump(root/'report.json', dict(passed=True, port=args.port, viewports=2, original_frame_checks=checks,
        frame_checks=len(checks), playback_checked=True, author_point_checked=True,
        horizontal_overflow=False, browser_errors=0, browser_POST_requests=0,
        invalid_HTTP_POST_expected_status=400, successful_production_submissions=0,
        before=before, after=after, collection_before=collection_before, collection_after=collection_after,
        cards_sha256=digest(OUTPUT/'annotation/human_natural_collection_001/cards.json'),
        model_calls=0, new_test_opened=False, goal_complete=False))
    print(json.dumps(dict(passed=True, frame_checks=len(checks), successful_production_submissions=0)))


if __name__ == '__main__':
    main()
