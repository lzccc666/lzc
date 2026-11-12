from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实际渲染并检查阶段3交互关系图。")
    parser.add_argument("--visualization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--no-sandbox --disable-dev-shm-usage --use-gl=swiftshader",
    )
    from PyQt5.QtCore import QEventLoop, QTimer, QUrl
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWebEngineWidgets import QWebEngineView

    args = parse_args()
    application = QApplication.instance() or QApplication(sys.argv[:1])
    results = []
    for html_path in sorted(args.visualization_root.glob("scene-*/scenegraph_3d_interactive.html")):
        view = QWebEngineView()
        view.resize(1600, 1000)
        loaded = wait_for_load(view, QUrl.fromLocalFile(str(html_path.resolve())), QEventLoop, QTimer)
        wait_milliseconds(1800, QEventLoop, QTimer)
        initial = run_javascript(
            view,
            """(() => {
              const graph = window.scenegraphGraph;
              const traces = graph && graph.data ? graph.data : [];
              const edges = traces.filter(trace => trace.meta && trace.meta.kind === 'edge');
              return {
                plotlyLoaded: Boolean(window.Plotly),
                traceCount: traces.length,
                edgeTraceCount: edges.length,
                visibleEdgeTraceCount: edges.filter(trace => trace.visible !== false).length,
                relationFilterCount: document.querySelectorAll('.relation-filter').length,
                confidenceFilterPresent: Boolean(document.getElementById('confidence-filter')),
                canvasCount: document.querySelectorAll('canvas').length,
                svgCount: document.querySelectorAll('svg').length,
                usingFallback: Boolean(window.scenegraphUsingFallback),
                webglErrorVisible: document.body.innerText.includes('WebGL is not supported')
              };
            })()""",
            QEventLoop,
            QTimer,
        )
        screenshot_path = html_path.with_name("interactive_render.png")
        screenshot = view.grab()
        screenshot.save(str(screenshot_path), "PNG")

        toggled_relation = run_javascript(
            view,
            """(() => {
              const checkbox = document.querySelector('.relation-filter:checked');
              if (!checkbox) return null;
              checkbox.checked = false;
              checkbox.dispatchEvent(new Event('change', {bubbles: true}));
              return checkbox.value;
            })()""",
            QEventLoop,
            QTimer,
        )
        wait_milliseconds(450, QEventLoop, QTimer)
        after_relation = visible_edge_count(view, QEventLoop, QTimer)
        run_javascript(
            view,
            """(() => {
              document.querySelectorAll('.relation-filter').forEach(box => { box.checked = true; });
              const slider = document.getElementById('confidence-filter');
              slider.value = '0.99';
              slider.dispatchEvent(new Event('input', {bubbles: true}));
              return true;
            })()""",
            QEventLoop,
            QTimer,
        )
        wait_milliseconds(450, QEventLoop, QTimer)
        after_confidence = visible_edge_count(view, QEventLoop, QTimer)
        clicked_node = run_javascript(
            view,
            """(() => {
              const graph = window.scenegraphGraph;
              const nodeTrace = graph.data.find(trace => trace.meta && trace.meta.kind === 'node');
              if (!nodeTrace || !nodeTrace.customdata || !nodeTrace.customdata.length) return null;
              const nodeId = Number(nodeTrace.customdata[0]);
              graph.emit('plotly_click', {points: [{data: nodeTrace, customdata: nodeId}]});
              return nodeId;
            })()""",
            QEventLoop,
            QTimer,
        )
        wait_milliseconds(250, QEventLoop, QTimer)
        focus_status = run_javascript(
            view,
            "document.getElementById('focus-status').textContent",
            QEventLoop,
            QTimer,
        )
        pixel_statistics = image_statistics(screenshot_path)
        result = {
            "scene_name": html_path.parent.name,
            "html_path": str(html_path.resolve()),
            "html_bytes": html_path.stat().st_size,
            "load_success": bool(loaded),
            "initial_dom": initial,
            "toggled_relation": toggled_relation,
            "visible_edges_after_relation_filter": after_relation,
            "visible_edges_after_confidence_filter": after_confidence,
            "relation_filter_effective": (
                toggled_relation is not None
                and after_relation < int(initial.get("visibleEdgeTraceCount", 0))
            ),
            "confidence_filter_effective": (
                after_confidence < int(initial.get("visibleEdgeTraceCount", 0))
            ),
            "clicked_node": clicked_node,
            "focus_status": focus_status,
            "node_focus_effective": clicked_node is not None and str(clicked_node) in str(focus_status),
            "screenshot_path": str(screenshot_path.resolve()),
            "pixel_statistics": pixel_statistics,
        }
        result["success"] = bool(
            result["load_success"]
            and initial.get("plotlyLoaded")
            and int(initial.get("edgeTraceCount", 0)) > 0
            and (
                int(initial.get("canvasCount", 0)) > 0
                or int(initial.get("svgCount", 0)) > 0
            )
            and not initial.get("webglErrorVisible", False)
            and result["relation_filter_effective"]
            and result["confidence_filter_effective"]
            and result["node_focus_effective"]
            and pixel_statistics["nonwhite_ratio"] > 0.01
        )
        results.append(result)
        view.close()

    payload = {
        "schema_version": 1,
        "success": bool(results) and all(item["success"] for item in results),
        "scene_count": len(results),
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output.resolve())
    if not payload["success"]:
        raise RuntimeError("交互可视化实际渲染或交互检查失败。")
    application.quit()


def wait_for_load(view: Any, url: Any, event_loop_class: Any, timer_class: Any) -> bool:
    state = {"loaded": False}
    loop = event_loop_class()

    def finished(success: bool) -> None:
        state["loaded"] = bool(success)
        loop.quit()

    view.loadFinished.connect(finished)
    timer_class.singleShot(30000, loop.quit)
    view.load(url)
    view.show()
    loop.exec_()
    try:
        view.loadFinished.disconnect(finished)
    except TypeError:
        pass
    return state["loaded"]


def wait_milliseconds(milliseconds: int, event_loop_class: Any, timer_class: Any) -> None:
    loop = event_loop_class()
    timer_class.singleShot(milliseconds, loop.quit)
    loop.exec_()


def run_javascript(
    view: Any,
    script: str,
    event_loop_class: Any,
    timer_class: Any,
) -> Any:
    state = {"completed": False, "value": None}
    loop = event_loop_class()

    def completed(value: Any) -> None:
        state["completed"] = True
        state["value"] = value
        loop.quit()

    view.page().runJavaScript(script, completed)
    timer_class.singleShot(10000, loop.quit)
    loop.exec_()
    if not state["completed"]:
        raise RuntimeError("JavaScript QA callback timed out")
    return state["value"]


def visible_edge_count(view: Any, event_loop_class: Any, timer_class: Any) -> int:
    value = run_javascript(
        view,
        """(() => {
          const graph = window.scenegraphGraph;
          return graph.data.filter(trace => trace.meta && trace.meta.kind === 'edge' && trace.visible !== false).length;
        })()""",
        event_loop_class,
        timer_class,
    )
    return int(value or 0)


def image_statistics(path: Path) -> dict[str, Any]:
    pixels = np.asarray(Image.open(path).convert("RGB"))
    return {
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "nonwhite_ratio": float(np.mean(np.any(pixels < 245, axis=2))),
        "pixel_standard_deviation": float(pixels.std()),
        "file_bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    main()
