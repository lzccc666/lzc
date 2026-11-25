"""基于隔离官方属性的静动态分组召回和运动判别评价。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter

from conceptgraph.completion.stage9 import OUTPUT, dump, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mapping-run',required=True)
    parser.add_argument('--labels-run',default='official_motion_dev_001')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    labelpath=OUTPUT/'annotation'/args.labels_run/'evaluation_only_motion.jsonl'
    labels={(r['scene'],r['frame_idx']):r['objects'] for r in map(json.loads,labelpath.open())}
    run=OUTPUT/'mapping_evaluation'/args.mapping_run
    summary=json.loads((run/'summary.json').read_text())
    totals=defaultdict(lambda:defaultdict(lambda:dict(truth=0,matched=0,classified_correct=0,unknown_prediction=0)))
    confusion=defaultdict(Counter)
    by_scene=[]
    for report in summary['reports']:
        scene=json.loads(open(report).read())
        counts=defaultdict(lambda:defaultdict(lambda:dict(truth=0,matched=0,classified_correct=0,unknown_prediction=0)))
        path=run/scene['scene']/'frames.jsonl'
        for frame in map(json.loads,path.open()):
            gt=labels[(frame['scene'],frame['frame_idx'])]
            assert [o['annotation_token'] for o in gt]==[o['annotation_token'] for o in frame['evaluation_only_official']]
            for method,result in frame['metrics'].items():
                matched={m['truth_index']:frame['predictions'][method][m['prediction_index']] for m in result['matches']}
                for i,obj in enumerate(gt):
                    group=obj['official_motion']
                    counts[method][group]['truth']+=1
                    if i not in matched:
                        continue
                    predicted=matched[i]['motion_state']
                    counts[method][group]['matched']+=1
                    counts[method][group]['classified_correct']+=predicted==group and group!='unknown'
                    counts[method][group]['unknown_prediction']+=predicted=='unknown'
                    confusion[method][group+'→'+predicted]+=1
        for method,groups in counts.items():
            for group,values in groups.items():
                for k,v in values.items():
                    totals[method][group][k]+=v
        by_scene.append(dict(scene=scene['scene'],log_token=scene['log_token'],groups=counts,
            source_sha256=digest(path)))
    final={}
    for method,groups in totals.items():
        final[method]={}
        for group,v in groups.items():
            final[method][group]=dict(v,recall=v['matched']/v['truth'] if v['truth'] else None,
                matched_motion_accuracy=v['classified_correct']/v['matched'] if v['matched'] and group!='unknown' else None)
    dest=OUTPUT/'mapping_evaluation'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',dict(groups=final,confusion={m:dict(c) for m,c in confusion.items()},
        scenes=by_scene,ground_truth_sha256=digest(labelpath),source_summary_sha256=digest(run/'summary.json'),
        motion_truth='nuScenes官方属性，不由类别或本方法速度推定；未知保留',
        denominator='召回：全部该组GT观测；运动准确率：已匹配且官方属性已知的对象观测',
        limitations='运动准确率条件于成功关联；不是全部对象端到端运动准确率；误检/未知关联另列',
        independent_test=False))
    print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':
    main()
