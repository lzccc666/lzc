import os
from types import SimpleNamespace

import pytest

from conceptgraph.completion import runtime_profiling
from conceptgraph.completion.runtime_profiling import Recorder, summarize


def test_cold_warm_cache_and_failure_denominators_remain_separate():
    rows = [
        dict(stage="parse", phase="cold", duration_s=10.0, succeeded=True),
        dict(stage="parse", phase="warm", duration_s=1.0, succeeded=True),
        dict(stage="parse", phase="warm", duration_s=3.0, succeeded=False),
        dict(stage="parse", phase="cache", duration_s=0.01, succeeded=True),
    ]
    result = summarize(rows)
    assert result["parse/warm"]["calls"] == 2
    assert result["parse/warm"]["failures"] == 1
    assert result["parse/warm"]["median_s"] == 2.0
    assert result["parse/warm"]["p95_s"] == pytest.approx(2.9)
    assert result["parse/warm"]["successful_calls_per_measured_s"] == 0.25
    assert result["parse/cold"]["summed_s"] == 10.0
    assert result["parse/cache"]["summed_s"] == 0.01


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_seconds_cannot_form_plausible_performance_table(duration):
    with pytest.raises(ValueError):
        summarize([dict(stage="x", phase="warm", duration_s=duration, succeeded=True)])


def test_exception_restores_instrumented_function_and_records_failed_call(tmp_path):
    def broken(value):
        raise LookupError(value)

    owner = SimpleNamespace(update=broken)
    recorder = Recorder(tmp_path)
    recorder.context = dict(frame_idx=7)
    with pytest.raises(LookupError), recorder.patch(owner, "update", "mapping"):
        owner.update("keep original failure")
    assert owner.update is broken
    assert recorder.rows[0]["frame_idx"] == 7
    assert recorder.rows[0]["succeeded"] is False
    assert summarize(recorder.rows)["mapping/warm"]["failures"] == 1


def test_nested_measurements_preserve_actual_return_values(tmp_path):
    def real_operation(value):
        return {"value": value * 2}

    owner = SimpleNamespace(update=real_operation)
    recorder = Recorder(tmp_path)
    with recorder.patch(owner, "update", "inner"), recorder.measure("outer"):
        result = owner.update(3)
    assert result == {"value": 6}
    assert [r["stage"] for r in recorder.rows] == ["inner", "outer"]
    assert recorder.rows[1]["duration_s"] >= recorder.rows[0]["duration_s"]


def test_other_cuda_process_prevents_exclusive_measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_profiling, "cuda_processes", lambda: [os.getpid() + 1])
    monitor = runtime_profiling.ExclusiveCudaMonitor(tmp_path)
    with pytest.raises(RuntimeError, match="occupied"):
        with monitor:
            pytest.fail("must not start the measured workload")
    assert not monitor.thread.is_alive()


def test_transient_foreign_cuda_process_cannot_receive_exclusive_credit(
    monkeypatch, tmp_path
):
    samples = iter([[os.getpid()], [os.getpid(), os.getpid() + 1], [os.getpid()]])
    monkeypatch.setattr(runtime_profiling, "cuda_processes", lambda: next(samples))
    monitor = runtime_profiling.ExclusiveCudaMonitor(tmp_path, interval_s=10)
    with pytest.raises(RuntimeError, match="exclusivity"):
        with monitor:
            monitor.sample()
    assert monitor.events[1]["other_pids"]
    assert not monitor.thread.is_alive()
