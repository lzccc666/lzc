"""Read-only bounded source prefetch; does not alter the inference process."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import time

from conceptgraph.completion import kitti_object_all_20261125 as impl


def main():
    records = list(impl.rows(impl.DATA / 'method_frames.jsonl'))
    local = threading.local(); opened = []; mutex = threading.Lock()
    phase = impl.REPORT / 'phase_timing_v3.jsonl'
    def consumed():
        # The numeric phase stream is monotonic and flushed once per frame.
        if not phase.exists():
            return -1
        with phase.open('rb') as stream:
            stream.seek(max(0, phase.stat().st_size - 4096)); tail = stream.read().splitlines()
        for line in reversed(tail):
            try:
                return int(json.loads(line)['frame_id'])
            except (ValueError, KeyError):
                continue
        return -1
    def read_input(index):
        if not hasattr(local, 'archives'):
            local.archives = impl.archive_open()
            with mutex:
                opened.append(local.archives)
        tick = time.monotonic(); row = records[index]; total = 0
        # Validate each original member's CRC through ZipFile.read. Bytes are
        # discarded after reading; only the existing filesystem cache is warmed.
        for key in ['rgb', 'lidar', 'calib']:
            content = local.archives[key].read(row['members'][key]['name'])
            if len(content) != row['members'][key]['uncompressed_bytes']:
                raise AssertionError('source_prefetch_member_size_mismatch')
            total += len(content)
        return dict(frame_id=row['frame_id'], read_bytes=total,
            read_wall_s=time.monotonic() - tick, source_only=True)
    impl.write(impl.DATA / 'read_only_prefetch_amendment_v1.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(Path(__file__)): impl.sha(__file__),
            str(impl.ROOT / 'scripts/kitti_object_prefetch_20261125.py'): impl.sha(impl.ROOT / 'scripts/kitti_object_prefetch_20261125.py')},
        workers=8, maximum_ahead_frames=32, retained_input_byte_queue=0,
        maximum_concurrent_uncompressed_members='8 readers, one current frame each; below256MiB by frozen file sizes',
        method_or_prediction_or_input_changes=False, actual_new_model_calls=0,
        data_action='read frozen original zip members through independent handles, validate CRC, discard bytes; no raw files written',
        budget='stops on GPU process release or6000s, no further inference config'))
    begin = time.monotonic(); pending = {}; next_index = consumed() + 1; count = byte_count = 0
    try:
        with ThreadPoolExecutor(max_workers=8) as pool, (impl.REPORT / 'source_prefetch_events_v1.jsonl').open('x') as log:
            while time.monotonic() - begin < 6000:
                if (impl.REPORT / 'gpu_released_v3.json').exists():
                    break
                limit = min(7480, consumed() + 32)
                while next_index <= limit and len(pending) < 32:
                    future = pool.submit(read_input, next_index); pending[future] = next_index; next_index += 1
                if not pending:
                    if next_index >= len(records):
                        break
                    time.sleep(.2); continue
                completed, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future); value = future.result(); count += 1; byte_count += value['read_bytes']
                    log.write(json.dumps(value) + '\n'); log.flush()
                    if count % 500 == 0:
                        print(json.dumps(dict(prefetched=count, latest_consumer=consumed(), elapsed_s=time.monotonic() - begin)), flush=True)
    finally:
        for group in opened:
            for archive in group.values():
                archive.close()
        impl.write(impl.REPORT / 'source_prefetch_summary_v1.json', dict(
            frames_read=count, source_bytes_read=byte_count, elapsed_s=time.monotonic() - begin,
            no_written_sensor_cache=True, no_model_calls=True,
            effect='performance diagnostic only; full inference output unchanged'))


if __name__ == '__main__':
    main()
