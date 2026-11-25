"""复核冻结P0/S事实、保护指纹和小试复现；不重新运行建图。"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import shutil

import numpy as np

from conceptgraph.completion.stage9 import PROJECT, digest, dump


P = 'paper_graph_20261123_'
BASE = PROJECT/'outputs/results/nuscenes/validation/iterations'
REPORT = PROJECT/'outputs/reports/paper_campaign_20261123/graph'
METHOD = 'R1_S_image_continuity'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def main():
    verified = {}
    for name in [P+'p0_v1', P+'S_pilot_v1', P+'S_full_v1', P+'S_evaluation_v1']:
        run = BASE/name
        release, report = [read(run/(P+suffix+'.json')) for suffix in ['release', 'report']]
        output = report.get('outputs_sha256', report.get('output_sha256', {}))
        for path, sha in output.items():
            assert digest(run/path) == sha, path
        for path, sha in release['input_sha256'].items():
            assert digest(path) == sha, path
        sources = release.get('source_sha256', release.get('sources_sha256'))
        for record in sources.values():
            assert digest(record['snapshot']) == record['sha256'], record
        verified[name] = dict(outputs=len(output), inputs=len(release['input_sha256']),
                              source_snapshots=len(sources))
        if 'relations' not in report:
            continue
        total, frames = defaultdict(Counter), 0
        for path in run.glob('scene-*/*frame_relations.jsonl.gz'):
            for row in rows(path):
                gt = {tuple(e) for e in row['gt_relations']}
                mapping = {int(k): v for k, v in row['node_mapping'].items()}
                cond = {e for e in gt if e[0] in mapping.values() and e[2] in mapping.values()}
                for method, edges in row['predicted_relations'].items():
                    edges = {tuple(e) for e in edges}
                    resolved = {(mapping[s], rel, mapping[t]) for s, rel, t in edges
                                if s in mapping and t in mapping}
                    tp = len(resolved & gt)
                    counts = dict(TP=tp, all_pred=len(edges), resolved_pred=len(resolved),
                                  unresolved_pred=len(edges)-len(resolved), all_truth=len(gt),
                                  conditional_truth=len(cond), full_FP=len(edges)-tp,
                                  full_FN=len(gt)-tp, missing_endpoint_FN=len(gt)-len(cond),
                                  covered_endpoint_FN=len(cond-resolved))
                    assert counts == row['metrics'][method]['ALL']
                    assert counts['full_FN'] == counts['missing_endpoint_FN']+counts['covered_endpoint_FN']
                    total[method].update(counts)
                frames += 1
        assert frames == 1508
        for row in report['relations']:
            if row['relation'] == 'ALL':
                assert dict(total[row['method']]) == {k: row[k] for k in total[row['method']]}
        verified[name]['relation_frames_independently_recounted'] = frames

    protected = read(REPORT/(P+'start_p0_v1.json'))['protected_sha256']
    for path, sha in protected.items():
        assert digest(path) == sha, path
    baseline = BASE/'recall_refinement_R1_full_20261120_v2'
    full, pilot = BASE/(P+'S_full_v1'), BASE/(P+'S_pilot_v1')
    dense, lost, pilot_frames = Counter(), [], 0
    for scene in read(full/(P+'release.json'))['scenes']:
        name = scene['scene']
        old = rows(baseline/name/'frames.jsonl.gz')
        new = rows(full/name/(P+'frames.jsonl.gz'))
        if (pilot/name).is_dir():
            trial = rows(pilot/name/(P+'frames.jsonl.gz'))
            assert len(trial) == len(new)
            for a, b in zip(trial, new):
                for key in ['predictions', 'metrics', 'evaluation_only_official', 'input', 'audit',
                            'image_continuity_audit', 'frame_idx', 'camera_timestamp_s']:
                    assert a[key] == b[key], (name, key, a['frame_idx'])
                pilot_frames += 1
        for a, b in zip(old, new):
            assert a['evaluation_only_official'] == b['evaluation_only_official']
            assert a['input'] == b['input']
            aa = [q for q in a['predictions']['R1'] if q.get('shape_available') is not False]
            bb = [q for q in b['predictions'][METHOD] if q.get('shape_available') is not False]
            dense['frames'] += 1
            dense['output_lists_exact_frames'] += aa == bb
            dense['count_equal_frames'] += len(aa) == len(bb)
            dense['unordered_category_center_equal_frames'] += (
                sorted((q['category'], q['center']) for q in aa) ==
                sorted((q['category'], q['center']) for q in bb))
            for x, y in zip(aa, bb):
                for key in set(x) | set(y):
                    if x.get(key) != y.get(key):
                        dense['changed_node_field:'+key] += 1
            am = {m['truth_index'] for m in a['metrics']['R1']['matches']}
            bm = {m['truth_index'] for m in b['metrics'][METHOD]['matches']}
            for i in am-bm:
                truth = a['evaluation_only_official'][i]
                near = [q for q in b['predictions'][METHOD] if q['category'] == truth['category']
                        and np.linalg.norm(np.asarray(q['center'][:2])-np.asarray(truth['center'][:2])) <= 3]
                lost.append(dict(scene=name, frame_idx=a['frame_idx'], truth_index=i,
                                 category=truth['category'], same_class_predictions_within_3m=len(near)))
    assert pilot_frames == 160 and dense['frames'] == 1508
    assert dense['count_equal_frames'] == dense['unordered_category_center_equal_frames'] == 1508
    comparison = read(REPORT/(P+'S_comparison.json'))
    for section in ['input_sha256', 'output_sha256']:
        for path, sha in comparison[section].items():
            assert digest(path) == sha, path
    gain_loss = read(REPORT/(P+'S_gain_loss_cases.json'))
    assert (len(lost), gain_loss['gained'], gain_loss['lost']) == (4, 13, 4)
    assert comparison['method_counts'][METHOD]['TP']-comparison['method_counts']['R1']['TP'] == 13-4
    directories = [BASE/name for name in verified]+[REPORT]
    byte_counts = {str(d): sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) for d in directories}
    code = [p for d in ['conceptgraph', 'scripts', 'tests'] for p in (PROJECT/d).rglob(P+'*')
            if p.is_file() and p.suffix == '.py']
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), runs=verified,
                  protected_sha256_checked=len(protected), pilot_vs_full_exact_frames=pilot_frames,
                  pilot_exact_fields=['predictions', 'metrics', 'GT', 'inputs', 'mapper_audit', 'image_audit', 'timestamps'],
                  full_S_second_1508_frame_replay=False,
                  replay_limit='negative primary result; independent full fact/count audit plus deterministic 160-frame pilot/full overlap',
                  dense_output_audit=dict(dense), lost_GT_competition_crosscheck=lost,
                  bytes_by_directory=byte_counts, source_bytes=sum(p.stat().st_size for p in code),
                  free_disk_bytes=shutil.disk_usage(PROJECT).free,
                  source_sha256={str(p): digest(p) for p in code},
                  old_default_and_paper_unchanged=True, frontend_model_calls=0, test_accessed=False)
    dump(REPORT/(P+'campaign_final_verification.json'), result)
    print(json.dumps({k: result[k] for k in ['passed', 'completed_utc', 'protected_sha256_checked',
                     'pilot_vs_full_exact_frames', 'dense_output_audit', 'bytes_by_directory', 'free_disk_bytes']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
