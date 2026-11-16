"""补查真实封存检测crop与历史/训练crop的相似性；不访问模型成绩、不改数据。"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import time

import numpy as np

from conceptgraph.completion.crop_split_audit import crop_hashes
from conceptgraph.completion.stage9 import digest, dump

PROJECT = Path('/root/autodl-tmp/lizichong')
OLD = PROJECT / 'outputs/project_v2/thesis_completion_v1'
ROOT = OLD / 'public_experiment_completion_20261115'
RUN = ROOT / 'main_experiments/sealed_public_058'


def main():
    front = RUN / 'frontend/report.json'
    if not front.exists():
        raise RuntimeError('前端仍在运行，不创建部分全量审核')
    report = json.loads(front.read_text())
    assert report['frames'] == 1196
    references = OLD / 'splits/reserved_crop_audit_001/reference_hashes.jsonl'
    reference_report = json.loads((references.parent / 'report.json').read_text())
    assert digest(references) == reference_report['files'][references.name]
    dest = ROOT / 'annotation/actual_sealed_crop_audit_062'
    dest.mkdir(exist_ok=False)
    protected = {str(p): digest(p) for p in [front, references, Path(__file__),
        PROJECT / 'conceptgraph/completion/crop_split_audit.py', RUN / 'release.json']}
    dump(dest / 'protocol.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        protected_inputs_sha256=protected, minimum_crop_side_px_for_approximate=28,
        approximate_hamming=dict(dhash=4, phash=4), wall_budget_s=600,
        references='保留的历史官方框投影crop及实际训练/验证crop哈希，同旧协议',
        scope='真实前端保存的全部检测crop；未知相似只报告，不改标签、不删样本、不改变冻结方法',
        prior_model_test_opened=True, model_outputs_accessed=False, new_model_calls=0, goal_complete=False))
    reference = [json.loads(x) for x in references.open()]
    exact_index = {}
    for i, row in enumerate(reference):
        exact_index.setdefault(row['pixel_sha256'], []).append(i)
    dhashes = np.asarray([int(x['dhash64'], 16) for x in reference], dtype=np.uint64)
    phashes = np.asarray([int(x['phash64'], 16) for x in reference], dtype=np.uint64)
    bitcounts = np.asarray([x.bit_count() for x in range(256)], dtype=np.uint8)
    def hamming(values, text):
        return bitcounts[(values ^ np.uint64(int(text, 16))).view(np.uint8).reshape(-1, 8)].sum(1)
    counts = Counter()
    exact, approximate = [], []
    started = time.monotonic()
    with (dest / 'crop_hashes.jsonl').open('x') as output:
        for receipt_path, expected in report['receipts_sha256'].items():
            if time.monotonic()-started > 600:
                raise TimeoutError('crop audit budget，已写前缀保留')
            assert digest(receipt_path) == expected
            receipt = json.loads(Path(receipt_path).read_text())
            path = Path(receipt['detections'])
            assert digest(path) == receipt['files_sha256'][str(path)]
            with gzip.open(path, 'rb') as handle:
                observations = pickle.load(handle)
            assert len(observations['image_crops']) == len(observations['xyxy'])
            counts['frames'] += 1
            for index, crop in enumerate(observations['image_crops']):
                hashes = crop_hashes(crop)
                row = dict(scene=receipt['scene'], frame_idx=receipt['frame_idx'], detection_index=index,
                    predicted_category=observations['classes'][int(observations['class_id'][index])],
                    detection_file=str(path), **hashes)
                output.write(json.dumps(row, ensure_ascii=False)+'\n')
                counts['actual_detection_crops'] += 1
                identity = {k: row[k] for k in ['scene', 'frame_idx', 'detection_index', 'predicted_category', 'detection_file']}
                matches = exact_index.get(hashes['pixel_sha256'], [])
                if matches:
                    exact.append(dict(current=identity, reference_indices=matches, pixel_sha256=hashes['pixel_sha256']))
                if min(crop.size) < 28:
                    counts['small_crops_no_approximate_check'] += 1
                    continue
                counts['approximate_checked_crops'] += 1
                dh, ph = hamming(dhashes, hashes['dhash64']), hamming(phashes, hashes['phash64'])
                for i in np.flatnonzero((dh <= 4) & (ph <= 4)):
                    approximate.append(dict(current=identity, reference_index=int(i),
                        reference_kind=reference[int(i)]['source_kind'],
                        reference_scene=reference[int(i)]['scene'],
                        dhash_hamming=int(dh[i]), phash_hamming=int(ph[i]),
                        low_texture=hashes['grayscale_std'] < 8 or reference[int(i)]['grayscale_std'] < 8,
                        status='自动相似候选，未人审，不直接判泄漏'))
            if counts['frames'] % 100 == 0:
                print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    assert counts['frames'] == 1196
    assert all(digest(p) == h for p, h in protected.items())
    dump(dest / 'exact_candidates.json', exact)
    dump(dest / 'approximate_candidates.json', approximate)
    dump(dest / 'report.json', dict(UTC=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
        reference_crops=len(reference), reference_kinds=dict(Counter(x['source_kind'] for x in reference)),
        exact_candidates=len(exact), approximate_candidate_pairs=len(approximate),
        all_similarity_candidates_resolved=not exact and not approximate, independent_human_review_count=0,
        protected_input_hashes_unchanged=True, elapsed_s=time.monotonic()-started,
        generated_files_sha256={p.name: digest(p) for p in dest.iterdir() if p.is_file()},
        limitations=['无哈希相似不能证明全部视觉独立，比例/视角/裁剪变化可能漏检',
            '相似候选不直接等于泄漏；低纹理及同类外观需结合采集时空进一步核对',
            '旧参考集未覆盖未知预训练数据，不能证明基础模型完全没见过数据',
            '小于28像素只查精确像素，不进行近似哈希检查',
            '不根据相似性或方法成绩删除样本，不改变冻结评测集合'],
        model_outputs_accessed=False, new_GPU_calls=0, new_CARLA_calls=0, goal_complete=False))
    print(json.dumps(dict(counts=dict(counts), exact=len(exact), approximate=len(approximate)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
