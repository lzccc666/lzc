import json

import pytest

from conceptgraph.nuscenes.expansion_benchmark import (
    iter_records,
    paired_scene_bootstrap,
    select_scenes,
)


def test_streaming_json_handles_boundaries_unicode_and_escaped_braces(tmp_path):
    expected = [{"text": '中文"},[\\end'}, {"nested": {"values": [1, 2]}}]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(expected, ensure_ascii=False))
    assert list(iter_records(path, chunk_size=3)) == expected


@pytest.mark.parametrize("text", ['[{"x":1},]', '[{"x":1}]x', '[{"x":', '{}', '[1]'])
def test_streaming_json_rejects_truncation_and_invalid_arrays(tmp_path, text):
    path = tmp_path / "table.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_records(path, chunk_size=2))


def test_selection_cannot_leak_legacy_or_test_logs():
    rows = []
    for i in range(60):
        rows.append({"scene": f"scene-{i:04}", "log_token": f"log-{i // 2}", "raw_complete": True, "strata": {"location": "test", "road_context": "parking", "light": "day", "weather": "dry"}})
    legacy = {"scene-0000": "train"}
    result = select_scenes(rows, legacy, target=30, test_count=10)
    test = [r for r in result["scenes"] if r["role"] == "sealed_test"]
    development = [r for r in result["scenes"] if r["role"] != "sealed_test"]
    assert len(result["scenes"]) == 30
    assert len(test) == 10
    assert not {r["log_token"] for r in test} & {r["log_token"] for r in development}
    assert "scene-0001" not in {r["scene"] for r in test}
    assert result == select_scenes(rows, legacy, target=30, test_count=10)


def test_selection_reports_data_shortfall_without_inventing_scenes():
    row = {"scene": "old", "log_token": "one", "raw_complete": True, "strata": {}}
    result = select_scenes([row], {"old": "train"}, target=50)
    assert len(result["scenes"]) == 1
    assert not result["audit"]["minimum_scene_count_met"]
    assert not result["audit"]["minimum_test_count_met"]


def test_bootstrap_weights_scenes_equally_and_pairs_queries():
    rows = []
    for scene, count, difference in [("many_frames", 100, 1), ("few_frames", 1, -1)]:
        for i in range(count):
            rows.extend([{"scene": scene, "query_id": str(i), "method": "a", "score": 0}, {"scene": scene, "query_id": str(i), "method": "b", "score": difference}])
    result = paired_scene_bootstrap(rows, "a", "b", "score")
    assert result["mean_difference"] == 0
    assert result["scene_count"] == 2
    assert result["paired_queries"] == 101
    assert result["ci95"] == [-1, 1]
    assert result == paired_scene_bootstrap(rows, "a", "b", "score")
    with pytest.raises(ValueError, match="duplicate"):
        paired_scene_bootstrap(rows + [rows[0]], "a", "b", "score")


def test_bootstrap_does_not_invent_certainty_or_drop_missing_pairs():
    rows = [{"scene": "s", "query_id": "q", "method": "a", "score": 1}]
    result = paired_scene_bootstrap(rows, "a", "b", "score")
    assert result["ci95"] is None
    assert result["paired_queries"] == 0
    assert result["missing_pairs"] == [["s", "q"]]
