"""新连续KITTI只读页90帧/三条件与桌面手机实际浏览器核验。"""

from datetime import datetime, timezone
import json
import urllib.request
from playwright.sync_api import sync_playwright
from conceptgraph.completion.stage9 import OUTPUT, digest, dump

BASE = "http://127.0.0.1:8768"
VIEW = "kitti_tracking_preview_001"
DEST = OUTPUT / "visualizations/kitti_tracking_browser_qa_001"
DEST.mkdir(exist_ok=False)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
errors = []
checks = []
links = set()
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--no-proxy-server"]
        )
        for viewport in [dict(width=1440, height=1000), dict(width=390, height=844)]:
            page = browser.new_page(
                viewport=viewport, is_mobile=viewport["width"] == 390
            )
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(
                f"{BASE}/visualizations/{VIEW}/index.html", wait_until="networkidle"
            )
            page.wait_for_function("data?.frames?.length===90")
            assert not page.is_checked("#gt")
            assert not page.evaluate(
                "document.documentElement.scrollWidth>innerWidth+1"
            )
            assert page.locator("#sequence option").count() == 3
            for seq in ["0000", "0005", "0013"]:
                page.select_option("#sequence", seq)
                for frame in [0, 14, 29]:
                    page.locator("#frame").evaluate(
                        '(e,v)=>{e.value=String(v);e.dispatchEvent(new Event("input"))}',
                        frame,
                    )
                    for method in [
                        "single_frame",
                        "original_fusion",
                        "one_to_one_depth_robust_motion",
                    ]:
                        page.select_option("#method", method)
                        page.wait_for_function(
                            'Array.from(document.querySelectorAll("img")).every(i=>i.complete&&i.naturalWidth>100)'
                        )
                        assert not page.locator("#error").inner_text()
                        assert page.locator("#objects tr").count() == page.evaluate(
                            'frames[index].predictions[el("method").value].length'
                        )
                        assert not page.evaluate(
                            "document.documentElement.scrollWidth>innerWidth+1"
                        )
                        checks.append(
                            dict(
                                viewport=viewport,
                                sequence=seq,
                                frame=frame,
                                method=method,
                            )
                        )
                    page.check("#gt")
                    assert "GT" in page.locator("#score").inner_text()
                    page.uncheck("#gt")
                    page.locator("#angle").evaluate(
                        '(e)=>{e.value="210";e.dispatchEvent(new Event("input"))}'
                    )
                    links.update(page.locator("a").evaluate_all("a=>a.map(x=>x.href)"))
            page.click("#prev")
            before = page.evaluate("index")
            page.click("#next")
            assert page.evaluate("index") == before + 1
            page.locator("#frame").evaluate(
                '(e)=>{e.value="0";e.dispatchEvent(new Event("input"))}'
            )
            page.click("#play")
            page.wait_for_function("index>=1")
            page.click("#play")
            paused = page.evaluate("index")
            page.wait_for_timeout(600)
            assert page.evaluate("index") == paused
            page.screenshot(
                path=str(DEST / f"viewport_{viewport['width']}.png"), full_page=True
            )
            page.close()
        browser.close()
    evidence = json.loads((OUTPUT / f"visualizations/{VIEW}/evidence.json").read_text())
    for row in evidence["frames"]:
        for field in ["rgb", "detection", "depth_visual", "raw"]:
            url = urllib.request.urljoin(
                f"{BASE}/visualizations/{VIEW}/index.html", row[field]
            )
            links.add(url)
    for url in links:
        with OPENER.open(url, timeout=15) as response:
            assert response.status == 200
    assert not errors, errors
    dump(
        DEST / "report.json",
        dict(
            passed=True,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            frames=90,
            sequence_count=3,
            conditions=3,
            viewport_checks=checks,
            HTTP_links=len(links),
            only_GET=True,
            GT_hidden_by_default=True,
            play_pause_verified=True,
            errors=errors,
            evidence_sha256=digest(OUTPUT / f"visualizations/{VIEW}/evidence.json"),
            source_sha256=digest(__file__),
            human_labels_added=0,
            goal_complete=False,
        ),
    )
    print(json.dumps(dict(passed=True, checks=len(checks), HTTP_links=len(links))))
except BaseException as exc:
    dump(
        DEST / "interrupted.json",
        dict(error=repr(exc), errors=errors, goal_complete=False),
    )
    raise
