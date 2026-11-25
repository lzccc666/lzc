"""Aggregate closed scene evaluations without confusing progress with full scores."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from conceptgraph.completion import nu_task_replay_20261125 as base
from conceptgraph.completion.nu_task_replay_fair_20261125_v2 import register, AMENDMENT


def main():
    register();manifest=base.verify()
    source=list(base.rows(base.DATA/'method_tasks.jsonl'))
    expected_scenes=sorted({r['scene'] for r in source})
    source_ids={r['task_id'] for r in source}
    out={};complete=True
    for channel,suffix in [('object_retrieval_diagnostic','retrieval'),('confirmed_static_selection','confirmation')]:
        root=base.experiment_directory('nuscenes','validation','nu_task_replay_20261125_v2_'+suffix)
        metrics=defaultdict(Counter);by_log=defaultdict(Counter);done=[];missing=[];observed=defaultdict(set)
        for scene in expected_scenes:
            path=root/scene/'evaluation_summary.json'
            if not path.exists():missing.append(scene);continue
            summary=json.loads(path.read_text())
            if base.sha(root/scene/'evaluation.jsonl')!=summary['evaluation_sha256']:
                raise AssertionError('evaluation_receipt_mismatch')
            for p,methods in summary['methods'].items():
                for m,c in methods.items():metrics[p,m].update(c)
            for group in summary['by_log']:by_log[group['protocol'],group['method'],group['log']].update(group['counts'])
            for r in base.rows(root/scene/'evaluation.jsonl'):
                if r['task_id'] not in source_ids:raise AssertionError('unregistered_task')
                key=r['protocol'],r['method']
                if r['task_id'] in observed[key]:raise AssertionError('duplicate_task_contribution')
                observed[key].add(r['task_id'])
            done.append(scene)
        all_done=not missing;complete=complete and all_done
        table={}
        for protocol,expected in manifest['task_counts'].items():
            table[protocol]={}
            for method in base.METHODS:
                values=dict(metrics[protocol,method]);count=values.get('tasks',0)
                if count!=len(observed[protocol,method]):raise AssertionError('counter_unique_id_mismatch')
                if all_done and count!=expected['all']:raise AssertionError('full_denominator_incomplete')
                entry=dict(observed_counts=values,frozen_full_denominator=expected,
                    unprocessed_tasks=expected['all']-count)
                if all_done:
                    entry['object_ID_selection_rate']=values.get('positive_selected_correct',0)/expected['positive']
                    entry['negative_false_selection_rate']=values.get('negative_false_selection',0)/expected['negative']
                if channel=='confirmed_static_selection' and method in ('single_frame','original_fusion'):
                    entry['confirmation_support']='unsupported_audited_multiframe_history; not a zero retrieval baseline'
                table[protocol][method]=entry
        out[channel]=dict(complete=all_done,scenes_done=done,scenes_missing=missing,methods=table,
            by_log=[dict(protocol=p,method=m,log=l,counts=dict(c)) for (p,m,l),c in by_log.items()],
            interpretation='object identity/spatial retrieval only; static condition not verified' if channel=='object_retrieval_diagnostic' else 'separate strict raw-evidence static confirmation contract')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=base.REPORT/f'corpus_summary_{stamp}.json'
    base.write(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=complete,
        full_denominator_rates_available=complete,channels=out,
        source_manifest_sha256=base.sha(base.DATA/'manifest.json'),fair_amendment_sha256=base.sha(AMENDMENT),
        postprocessing_source_sha256={str(Path(__file__)):base.sha(__file__),
            str(base.ROOT/'scripts/nu_task_summary_20261125.py'):base.sha(base.ROOT/'scripts/nu_task_summary_20261125.py')},
        excluded_claims=['navigation task completion from geometric retrieval','vehicle route correctness','arrival','parking legality','unseen language generalization']))
    print(json.dumps(dict(path=str(path),complete=complete)))


if __name__=='__main__':main()
