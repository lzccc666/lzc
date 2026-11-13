"""主450查看器的独立浏览器、原始时间轴及16位PNG核对；全部GET。"""
import bisect
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


BASE = 'http://127.0.0.1:8768'


def read(path):
    return json.loads(Path(path).read_text())


def chunk(kind, content):
    return struct.pack('>I', len(content))+kind+content+struct.pack('>I', zlib.crc32(kind+content))


def paeth(a, b, c):
    p = a+b-c
    values = [abs(p-a), abs(p-b), abs(p-c)]
    return [a, b, c][values.index(min(values))]


def png_fixture(array, filtering):
    rows = array.astype('>u2').view(np.uint8).reshape(len(array), -1)
    encoded = bytearray()
    previous = np.zeros(rows.shape[1], dtype=np.uint8)
    for row in rows:
        encoded.append(filtering)
        for index, value in enumerate(row):
            a = int(row[index-2]) if index >= 2 else 0
            b = int(previous[index])
            c = int(previous[index-2]) if index >= 2 else 0
            prediction = [0, a, b, (a+b)//2, paeth(a, b, c)][filtering]
            encoded.append((int(value)-prediction) % 256)
        previous = row
    header = struct.pack('>IIBBBBB', array.shape[1], array.shape[0], 16, 0, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', header)+chunk(b'IDAT', zlib.compress(encoded))+chunk(b'IEND', b'')


def pixel_sha(array):
    return hashlib.sha256(np.asarray(array, dtype='<u2').tobytes()).hexdigest()


def decoded(page, path):
    return page.evaluate('''async path=>{
      const response=await fetch(path);if(!response.ok)throw Error('depth fixture unavailable');
      const result=await MATRIX_VIEW.decodeDepthPng(await response.arrayBuffer());
      const canonical=new ArrayBuffer(result.pixels.length*2),view=new DataView(canonical);
      let nonzero=0;for(let i=0;i<result.pixels.length;i++){view.setUint16(i*2,result.pixels[i],true);nonzero+=result.pixels[i]!==0}
      const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',canonical)),x=>x.toString(16).padStart(2,'0')).join('');
      return {width:result.width,height:result.height,nonzero,sha};}''', path)


def wait_case(page, run):
    page.wait_for_function('run=>document.body.dataset.caseReady===run && MATRIX_VIEW.snapshot.ready', arg=run, timeout=30000)


def seek(page, identifier, index):
    page.evaluate('arg=>{const element=document.getElementById(arg.id);element.value=arg.index;element.dispatchEvent(new Event("input"))}',
                  dict(id=identifier, index=index))
    page.wait_for_function('arg=>MATRIX_VIEW.snapshot.ready && MATRIX_VIEW.snapshot[arg.field]===arg.index',
                           arg=dict(field='tickIndex' if identifier == 'tick' else 'frameIndex', index=index), timeout=30000)


def main():
    started = time.monotonic()
    root = OUTPUT/'visualizations/realtime_main450_preview_003'
    manifest = read(root/'manifest.json')
    assert len(manifest['cases']) == 450 and not manifest['is_smoke']
    main_root = OUTPUT/'interactive_simulation/realtime_three_model_full_001'
    records = [json.loads(line) for line in (main_root/'trial_records.jsonl').open()]
    by_run = {record['run_id']: record for record in records}
    assert set(by_run) == {row['run'] for row in manifest['cases']}
    assert Counter(row['method'] for row in manifest['cases']) == dict(rule=150, base_full=150, new_full=150)
    destination = OUTPUT/'visualizations/realtime_main450_browser_qa_001'
    destination.mkdir(exist_ok=False)
    (destination/'png_filters').mkdir()
    first = np.array([0, 1, 255, 256, 1024, 32767, 32768, 65535, 5000], dtype=np.uint16)
    array = np.stack([np.roll(first, index) for index in range(7)])
    fixtures = []
    for filtering in range(5):
        path = destination/'png_filters'/f'filter_{filtering}.png'
        path.write_bytes(png_fixture(array, filtering))
        assert np.array_equal(np.asarray(Image.open(path)), array)
        fixtures.append(path)
    invalid = destination/'png_filters/unsupported_8bit.png'
    Image.fromarray(array.astype(np.uint8)).save(invalid)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE+'/api/review_status') as response:
        human_before = json.load(response)
    metadata = {}
    for row in manifest['cases']:
        assert time.monotonic()-started < 1200
        with opener.open(BASE+'/visualizations/realtime_main450_preview_003/'+row['file']) as response:
            content = response.read()
        assert hashlib.sha256(content).hexdigest() == manifest['files'][row['file']]
        value = json.loads(content)
        source = by_run[row['run']]
        assert (value['run'], value['method'], value['task']['task_id'], value['seed'], value['status']) == (
            source['run_id'], source['variant'], source['task'], source['seed'], source['status'])
        assert value['result'] == source['result']
        for path in value['independent_evidence']['proof_urls'].values():
            with opener.open(BASE+path) as response:
                proof = json.load(response)
            assert (proof['task'], proof['seed'], proof['method']) == (source['task'], source['seed'], source['variant'])
        metadata[row['run']] = value
    assert sum(value['independent_evidence']['initial_is_reconstructed'] for value in metadata.values()) == 54
    chosen = {}
    for method in manifest['methods']:
        for kind in ['reachable_positive', 'no_match', 'inaccessible']:
            row = next(row for row in manifest['cases'] if row['method'] == method and row['kind'] == kind)
            chosen[row['run']] = row
        row = next(row for row in manifest['cases'] if row['method'] == method and row['arrived'])
        chosen[row['run']] = row
    for status in sorted({row['status'] for row in manifest['cases']}):
        row = next(row for row in manifest['cases'] if row['status'] == status)
        chosen[row['run']] = row
    row = next(row for row in manifest['cases'] if row['arrived'] and metadata[row['run']]['independent_evidence']['initial_is_reconstructed'])
    chosen[row['run']] = row
    requests, failed_responses, checks, png_checks = [], [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
        for viewport, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
            page = browser.new_page(viewport=dict(width=width, height=height), device_scale_factor=2 if viewport == 'mobile' else 1)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
            page.on('response', lambda response: failed_responses.append(dict(url=response.url, status=response.status))
                    if response.status >= 400 and not response.url.endswith('/favicon.ico') else None)
            page.goto(BASE+'/visualizations/realtime_main450_preview_003/index.html')
            wait_case(page, manifest['cases'][0]['run'])
            assert page.locator('#case option').count() == 450
            assert not page.locator('#gt').is_checked()
            for path in fixtures:
                actual = decoded(page, '/'+str(path.relative_to(OUTPUT)))
                assert actual == dict(width=9, height=7, nonzero=int(np.count_nonzero(array)), sha=pixel_sha(array))
                png_checks.append(dict(viewport=viewport, filter=path.stem, **actual))
            rejected = page.evaluate('''async path=>{try{await MATRIX_VIEW.decodeDepthPng(await (await fetch(path)).arrayBuffer());return false}catch(error){return error.message.includes('16位')}}''', '/'+str(invalid.relative_to(OUTPUT)))
            assert rejected
            for index, row in enumerate(chosen.values()):
                page.select_option('#mode', 'observation')
                page.select_option('#case', row['file'])
                wait_case(page, row['run'])
                value = metadata[row['run']]
                raw = Path(by_run[row['run']]['raw_case'])
                frames = [json.loads(line) for line in (raw/'perception.jsonl').open()]
                controls = [json.loads(line) for line in (raw/'execution.jsonl').open()]
                assert page.locator('#outcomes .outcome').count() == 4
                if row['result_missing']:
                    assert page.locator('#outcomes').inner_text().count('未知（缺终止结果）') == 4
                if value['independent_evidence']['initial_is_reconstructed']:
                    assert '冻结种子' in page.locator('#caseEvidence').inner_text()
                else:
                    assert '冻结种子' not in page.locator('#caseEvidence').inner_text()
                state_checks = 0
                for frame_index in sorted({0, len(frames)//2, len(frames)-1}):
                    seek(page, 'frame', frame_index)
                    actual = page.evaluate('MATRIX_VIEW.snapshot')
                    expected = frames[frame_index]['sensor']['state']
                    assert actual['displayedState'] == expected and actual['perceptionFrame'] == frames[frame_index]['frame_idx']
                    assert actual['prefix'] == bisect.bisect_right([control['state']['time_s'] for control in controls], expected['time_s'])
                    state_checks += 1
                page.select_option('#mode', 'control')
                page.wait_for_function('MATRIX_VIEW.snapshot.ready && MATRIX_VIEW.snapshot.mode==="control"')
                for tick in sorted({0, len(controls)//2, len(controls)-1}):
                    seek(page, 'tick', tick)
                    actual = page.evaluate('MATRIX_VIEW.snapshot')
                    assert actual['displayedState'] == controls[tick]['state'] and actual['prefix'] == tick+1
                    frame_index = bisect.bisect_right([frame['completed_wall_elapsed_s'] for frame in frames], controls[tick]['wall_elapsed_s'])-1
                    assert actual['perceptionFrame'] == (frames[frame_index]['frame_idx'] if frame_index >= 0 else None)
                    assert actual['perceptionCompletedWall'] is None or actual['perceptionCompletedWall'] <= actual['controlWall']
                    state_checks += 1
                assert not page.locator('#gt').is_checked()
                page.locator('#gt').check()
                page.wait_for_function('MATRIX_VIEW.snapshot.ready && MATRIX_VIEW.snapshot.GT')
                assert '隔离评估层隐藏' not in page.locator('#execution').inner_text()
                page.locator('#gt').uncheck()
                page.wait_for_function('MATRIX_VIEW.snapshot.ready && !MATRIX_VIEW.snapshot.GT')
                assert '隔离评估层隐藏' in page.locator('#execution').inner_text()
                page.select_option('#spaceView', 'vehicle')
                canvas = page.evaluate('(()=>{const c=document.getElementById("space");return {w:c.width,h:c.height,cw:c.clientWidth,ch:c.clientHeight}})()')
                assert abs(canvas['w']/canvas['cw']-canvas['h']/canvas['ch']) < .01
                assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
                assert not errors and not page.locator('#error').inner_text()
                if index < 3:
                    frame = frames[len(frames)//2]
                    depthpath = Path(frame['sensor']['depth'])
                    depth = np.asarray(Image.open(depthpath))
                    actual = decoded(page, '/'+str(depthpath.relative_to(OUTPUT)))
                    assert actual == dict(width=640, height=360, nonzero=int(np.count_nonzero(depth)), sha=pixel_sha(depth))
                    cloud = page.evaluate('async index=>sparseDepth(frames[index])', len(frames)//2)
                    valid = np.flatnonzero(depth)
                    count = min(384, len(valid))
                    selected = valid[[i*(len(valid)-1)//(count-1) for i in range(count)]] if count > 1 else valid[:count]
                    v, u = np.divmod(selected, depth.shape[1])
                    z = depth[v, u]/1000.
                    reference = np.column_stack([u, v, z])
                    assert np.array_equal(np.asarray(cloud['sample']), reference)
                    camera = np.column_stack([(u-320)*z/320, (v-180)*z/320, z])
                    pose = np.asarray(frame['sensor']['camera_pose_cv'])
                    assert np.allclose(cloud['world'], camera@pose[:3, :3].T+pose[:3, 3], rtol=0, atol=1e-10)
                    png_checks.append(dict(viewport=viewport, real_frame=frame['frame_idx'], run=row['run'], pixels_sha=actual['sha'], world_points_checked=count))
                page.screenshot(path=str(destination/f'{viewport}_{row["index"]:04d}.png'), full_page=False)
                checks.append(dict(viewport=viewport, run=row['run'], method=row['method'], kind=row['kind'],
                    status=row['status'], actual_state_and_prefix_checks=state_checks, GT_hidden=True,
                    reconstructed_initial=value['independent_evidence']['initial_is_reconstructed'], canvas=canvas,
                    causal_completed_frame_alignment=True, page_errors=list(errors)))
                print(json.dumps(dict(viewport=viewport, case=len(checks), run=row['run'])), flush=True)
            page.select_option('#mode', 'observation')
            page.wait_for_function('MATRIX_VIEW.snapshot.ready && MATRIX_VIEW.snapshot.mode==="observation"')
            seek(page, 'frame', 0)
            page.locator('#play').click()
            page.wait_for_function('MATRIX_VIEW.snapshot.frameIndex>0', timeout=5000)
            page.locator('#play').click()
            paused = page.evaluate('MATRIX_VIEW.snapshot.frameIndex')
            page.wait_for_timeout(900)
            assert page.evaluate('MATRIX_VIEW.snapshot.frameIndex') == paused
            page.locator('#space').scroll_into_view_if_needed()
            position = page.locator('#space').bounding_box()
            state_before_drag = page.evaluate('MATRIX_VIEW.snapshot.displayedState')
            image_before_drag = page.evaluate('document.getElementById("space").toDataURL()')
            page.mouse.move(position['x']+60, position['y']+60)
            page.mouse.down()
            page.mouse.move(position['x']+95, position['y']+75, steps=3)
            page.mouse.up()
            page.wait_for_function('MATRIX_VIEW.snapshot.ready')
            assert page.evaluate('MATRIX_VIEW.snapshot.displayedState') == state_before_drag
            assert page.evaluate('document.getElementById("space").toDataURL()') != image_before_drag
            empty_config = next(config for config in manifest['configs'] if not any(row['method'] == 'new_full' and row['config_id'] == config and row['arrived'] for row in manifest['cases']))
            page.select_option('#method', 'new_full')
            page.select_option('#config', empty_config)
            page.select_option('#filter', 'arrived')
            page.wait_for_function('document.body.dataset.caseReady==="empty"')
            assert page.locator('#case option').count() == 0 and page.locator('#play').is_disabled()
            assert not page.locator('#outcomes').inner_text() and '当前筛选没有记录' in page.locator('#error').inner_text()
            page.select_option('#filter', 'all')
            page.select_option('#config', 'all')
            page.select_option('#method', 'all')
            wait_case(page, manifest['cases'][0]['run'])
            for method in manifest['methods']:
                page.select_option('#method', method)
                page.wait_for_function('document.querySelectorAll("#case option").length===150')
                for kind in ['no_match', 'inaccessible']:
                    page.select_option('#filter', kind)
                    page.wait_for_function('document.querySelectorAll("#case option").length===30')
                page.select_option('#filter', 'all')
            page.wait_for_function('MATRIX_VIEW.snapshot.ready')
            assert not errors and not page.locator('#error').inner_text()
            page.close()
        browser.close()
    with opener.open(BASE+'/api/review_status') as response:
        human_after = json.load(response)
    assert all(request['method'] == 'GET' for request in requests)
    assert not failed_responses
    assert time.monotonic()-started < 1200
    dump(destination/'report.json', dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        viewer='realtime_main450_preview_003', viewer_manifest_sha256=digest(root/'manifest.json'),
        QA_source_sha256=digest(__file__), roster_HTTP_cases=450, proof_HTTP_links=1350,
        unique_browser_cases=len(chosen), viewport_case_checks=len(checks), checks=checks, PNG_checks=png_checks,
        state_and_prefix_checks=sum(row['actual_state_and_prefix_checks'] for row in checks),
        only_GET=True, browser_requests=len(requests), failed_business_responses=failed_responses,
        human_status_before=human_before, human_status_after=human_after,
        actual_model_calls=0, new_executions=0, new_arrivals=0, human_records_written=0,
        CPU_shared_wall_s=time.monotonic()-started,
        limits=['浏览器代表条件检查不等于450次逐帧人工审核；全部450清单与源键/证据链接单独全查',
                'PNG/空间点核对限检查帧；原始57,460组输入另有生成器全量SHA核验',
                '按已完成感知时间保守对齐，不还原未记录的中间发布；查看不是新执行'], goal_complete=False))
    print(json.dumps(dict(passed=True, browser_cases=len(chosen), checks=len(checks),
                         only_GET=True, goal_complete=False)))


if __name__ == '__main__':
    main()
