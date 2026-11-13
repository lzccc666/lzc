"""保留严格SHA比较失败；全量核对gzip仅mtime不同及未压缩检测业务字节一致。"""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    driver = Path(__file__).with_name('portable_business_reproduction_compare_001.py')
    spec = importlib.util.spec_from_file_location('portable_strict_comparison', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_root = OUTPUT/'reproduction/portable_business_comparison_001'
    strict = module.read(old_root/'report.json')
    assert not strict['passed'] and strict['comparisons']['perception.jsonl']['equal'] is False
    a, root_a, case_a = module.locate(OUTPUT/'reproduction/standalone_cpu_delivery_002')
    b, root_b, case_b = module.locate(OUTPUT/'reproduction/portable_executed_001')
    frames_a, frames_b = module.records(case_a/'perception.jsonl'), module.records(case_b/'perception.jsonl')
    assert len(frames_a) == len(frames_b) == 38
    frames_a, frames_b = module.canonical(frames_a, root_a, case_a), module.canonical(frames_b, root_b, case_b)
    containers = []
    for index, (left, right) in enumerate(zip(frames_a, frames_b)):
        pa, pb = case_a/'detections'/f'{index:06d}.pkl.gz', case_b/'detections'/f'{index:06d}.pkl.gz'
        raw_a, raw_b = pa.read_bytes(), pb.read_bytes()
        assert digest(pa) == left['detector_sha256'] and digest(pb) == right['detector_sha256']
        assert raw_a[:4] == raw_b[:4] == bytes([0x1f, 0x8b, 8, 8])
        payload_a, payload_b = gzip.decompress(raw_a), gzip.decompress(raw_b)
        payload_sha_a, payload_sha_b = hashlib.sha256(payload_a).hexdigest(), hashlib.sha256(payload_b).hexdigest()
        payload_same = payload_a == payload_b
        only_mtime = raw_a[:4]+raw_a[8:] == raw_b[:4]+raw_b[8:]
        containers.append(dict(frame_idx=index, original_file=str(pa), portable_file=str(pb),
            original_file_sha256=digest(pa), portable_file_sha256=digest(pb),
            original_payload_sha256=payload_sha_a, portable_payload_sha256=payload_sha_b,
            whole_uncompressed_pickle_bytes_equal=payload_same,
            compressed_bytes_outside_header_mtime_equal=only_mtime,
            original_mtime_s=int.from_bytes(raw_a[4:8], 'little'),
            portable_mtime_s=int.from_bytes(raw_b[4:8], 'little')))
        # SHA仍全量校验并保存在containers，业务比较改为实际解压后的整个pickle
        # 内容SHA，未忽略任何框/掩膜/crop/特征/来源字段或采用数值容差。
        left['detector_sha256'] = dict(identity_kind='whole_uncompressed_pickle_content', sha256=payload_sha_a)
        right['detector_sha256'] = dict(identity_kind='whole_uncompressed_pickle_content', sha256=payload_sha_b)
    destination = OUTPUT/'reproduction/portable_business_comparison_002'
    destination.mkdir(exist_ok=False)
    dump(destination/'all_detector_containers.json', containers)
    conditions = dict(strict['comparisons'])
    conditions['perception.jsonl'] = dict(equal=frames_a == frames_b,
        original_sha256=module.signature(frames_a), portable_sha256=module.signature(frames_b),
        detector_identity_basis='all uncompressed pickle bytes; raw container SHA retained and verified',
        original_records=len(frames_a), portable_records=len(frames_b))
    dump(destination/'report.json', dict(passed=all(value['equal'] for value in conditions.values())
        and all(row['whole_uncompressed_pickle_bytes_equal'] and row['compressed_bytes_outside_header_mtime_equal'] for row in containers),
        updated_utc=datetime.now(timezone.utc).isoformat(), comparisons=conditions,
        strict_raw_container_comparison_passed=False, strict_failed_report_preserved=True,
        strict_report_sha256=digest(old_root/'report.json'),
        detector_payloads_checked=len(containers),
        detector_all_pickle_payloads_byte_identical=all(row['whole_uncompressed_pickle_bytes_equal'] for row in containers),
        detector_compressed_difference_confined_to_gzip_header_mtime=all(row['compressed_bytes_outside_header_mtime_equal'] for row in containers),
        ignored_business_fields=[], numerical_tolerance_used=False,
        ignored_timing_fields=strict['ignored_timing_fields'],
        path_prefix_normalization=strict['path_prefix_normalization'],
        original_actual_frontend_calls=38, portable_actual_frontend_calls=38,
        actual_sensor_and_frontend_executed_in_both=True,
        Qwen_backend_calls_both_runs=0, full_three_task_Qwen_cache_gate_closed=False,
        detector_container_ledger_sha256=digest(destination/'all_detector_containers.json'),
        source_sha256=digest(__file__), strict_driver_sha256=digest(driver),
        new_independent_scenes=0, frozen_main_table_credit=False, goal_complete=False))
    print(json.dumps(module.read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
