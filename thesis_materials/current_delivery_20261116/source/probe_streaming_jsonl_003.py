"""对已结束真实日志测量逐条flush的无损gzip，不改写或删除原文件。"""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'storage_audit/streaming_jsonl_probe_003'
    out.mkdir(exist_ok=False)
    cases = [
        R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule',
        R/'carla_official/complex_semantic_navigation_007/complex_second_nearest20/base',
        R/'carla_official/clean_cpu_navigation_002/car20/full_rule_geometry']
    sources = [root/name for root in cases for name in ['execution.jsonl', 'perception.jsonl']]
    assert all((root/'summary.json').exists() for root in cases)
    original = {str(p): dict(sha256=digest(p), bytes=p.stat().st_size) for p in sources}
    dump(out/'registration.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=digest(__file__), inputs=original, compression='gzip level6 mtime0',
        flush_policy='每条原始JSONL记录写入后同步flush，保留记录边界',
        old_files_modified_or_removed=False, currently_running_episode_format_changed=False,
        actual_new_GPU_or_physical_credit=0, goal_complete=False))
    rows = []
    for index, path in enumerate(sources):
        dest = out/f'{index:02d}_{path.name}.gz'
        started = time.monotonic()
        count = 0
        with path.open('rb') as source, dest.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=6) as target:
                for line in source:
                    assert line.endswith(b'\n')
                    json.loads(line)
                    target.write(line)
                    target.flush()
                    count += 1
        write_s = time.monotonic()-started
        rebuilt, restored_records = hashlib.sha256(), 0
        with gzip.open(dest, 'rb') as rebuilt_stream:
            for line in rebuilt_stream:
                rebuilt.update(line)
                json.loads(line)
                restored_records += 1
        assert restored_records == count
        assert rebuilt.hexdigest() == original[str(path)]['sha256'] == digest(path)
        before, after = path.stat().st_size, dest.stat().st_size
        rows.append(dict(source=str(path), compressed=str(dest), records=count,
            original_bytes=before, compressed_bytes=after, saved_fraction=1-after/before,
            write_and_per_record_flush_wall_s=write_s, byte_exact_roundtrip=True,
            original_sha256=digest(path), compressed_sha256=digest(dest),
            original_unchanged=True))
    raw_bytes = sum(row['original_bytes'] for row in rows)
    compressed = sum(row['compressed_bytes'] for row in rows)
    dump(out/'report.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=rows, total_original_bytes=raw_bytes, total_compressed_bytes=compressed,
        aggregate_saved_fraction=1-compressed/raw_bytes,
        all_uncompressed_bytes_and_record_counts_identical=True,
        all_original_files_rehashed_unchanged=True,
        changed_running_experiment_format=False, adopted_for_new_experiments=False,
        limitation='只验证已结束日志的无损压缩和flush成本；不代表原RGB/深度/LiDAR压缩率，也未证明SIGKILL后尾部恢复',
        actual_new_GPU_calls=0, actual_new_physical_episodes=0, goal_complete=False))
    print(json.dumps(dict(files=len(rows), records=sum(r['records'] for r in rows),
        original_MiB=raw_bytes/2**20, compressed_MiB=compressed/2**20,
        saved_fraction=1-compressed/raw_bytes), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
