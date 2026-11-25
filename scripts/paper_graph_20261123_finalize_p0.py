"""独立复核P0已保存关系事实及全量计数，并导出论文主表。"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json

from conceptgraph.completion.stage9 import PROJECT, digest, dump


P = 'paper_graph_20261123_'
ROOT = PROJECT/'outputs/reports/paper_campaign_20261123/graph'
RUN = PROJECT/'outputs/results/nuscenes/validation/iterations/paper_graph_20261123_p0_v1'


def main():
    r=json.loads((RUN/(P+'report.json')).read_text())
    total=defaultdict(Counter)
    frames=0
    for path in RUN.glob('scene-*/*.gz'):
        if 'frame_relations' not in path.name:
            continue
        for row in map(json.loads,gzip.open(path,'rt')):
            gt={tuple(e) for e in row['gt_relations']}
            mapping={int(k):v for k,v in row['node_mapping'].items()}
            cond={e for e in gt if e[0] in mapping.values() and e[2] in mapping.values()}
            for m,edges in row['predicted_relations'].items():
                edges={tuple(e) for e in edges}
                resolved={(mapping[s],rel,mapping[t]) for s,rel,t in edges if s in mapping and t in mapping}
                tp = len(resolved & gt)
                counts=dict(TP=tp,all_pred=len(edges),resolved_pred=len(resolved),unresolved_pred=len(edges)-len(resolved),
                    all_truth=len(gt),conditional_truth=len(cond),full_FP=len(edges)-tp,full_FN=len(gt)-tp,
                    missing_endpoint_FN=len(gt)-len(cond),covered_endpoint_FN=len(cond-resolved))
                if counts != row['metrics'][m]['ALL']:
                    raise AssertionError('独立事实重算不一致')
                total[m].update(counts)
            frames+=1
    for row in r['relations']:
        if row['relation']=='ALL' and dict(total[row['method']])!={k:row[k] for k in total[row['method']]}:
            raise AssertionError('全范围关系主表未复现')
    bad=[p for p,sha in r['outputs_sha256'].items() if digest(RUN/p)!=sha]
    release=json.loads((RUN/(P+'release.json')).read_text())
    bad_inputs=[p for p,sha in release['input_sha256'].items() if digest(p)!=sha]
    start=json.loads((ROOT/(P+'start_p0_v1.json')).read_text())
    bad_protected=[p for p,sha in start['protected_sha256'].items() if digest(p)!=sha]
    if bad or bad_inputs or bad_protected or frames!=1508:
        raise AssertionError(dict(bad=bad,bad_inputs=bad_inputs,bad_protected=bad_protected,frames=frames))
    node=dict(method='R1',**r['node_counts'],**r['original_node_metrics'])
    with (ROOT/(P+'P0_node_main.csv')).open('x',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(node))
        writer.writeheader()
        writer.writerow(node)
    dump(ROOT/(P+'P0_verification.json'),dict(passed=True,frames=frames,
        all_relation_facts_independently_recounted=True,checked_output_sha256=len(r['outputs_sha256']),
        checked_input_sha256=len(release['input_sha256']),checked_protected_sha256=len(start['protected_sha256']),
        bad_outputs=bad,bad_inputs=bad_inputs,bad_protected=bad_protected,verifier_sha256=digest(__file__)))
    print(dict(frames=frames,protected=len(start['protected_sha256']),outputs=len(r['outputs_sha256'])))


if __name__=='__main__':
    main()
