"""Pool the frozen validation and once-only regression outputs without method calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from conceptgraph.completion.mv136_20261123_protocol import evaluate
from conceptgraph.completion.stage9 import PROJECT, digest, dump
from conceptgraph.project_paths import experiment_directory

P='mapnext_20261124_'


def main():
    root=PROJECT/'outputs/reports/mapping_iteration_20261124'
    freeze_path=root/(P+'selection_freeze.json')
    freeze=json.loads(freeze_path.read_text())
    for p,sha in freeze['source_sha256'].items():
        assert digest(p)==sha
    runs=[experiment_directory('nuscenes','validation','mapnext_20261124_S2_'+part+'_v1')
          for part in ['validation','regression']]
    reports=[json.loads((run/(P+'report.json')).read_text()) for run in runs]
    releases=[json.loads((run/(P+'release.json')).read_text()) for run in runs]
    assert all(r['complete'] and r['tag']=='S2' and not r['GT_in_inference'] for r in reports)
    assert reports[0]['groups']==80 and reports[1]['groups']==40
    assert releases[0]['config']==releases[1]['config']==freeze['selected_config']
    assert datetime.fromisoformat(releases[1]['started_utc'])>datetime.fromisoformat(freeze['frozen_utc'])
    assert [{p:v['sha256'] for p,v in r['source_sha256'].items()} for r in reports]==[freeze['source_sha256']]*2
    manifest_path=PROJECT/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json'
    gt_path=PROJECT/'dataset/processed/mapnext_20261124_inputs_v1/all_GT.jsonl'
    manifest=json.loads(manifest_path.read_text())
    run=experiment_directory('nuscenes','validation','mapnext_20261124_S2_full_v1',create=True)
    dest=root/'S2_full_v1';dest.mkdir(exist_ok=False)
    shutil.copytree(runs[0]/'source_snapshot',run/'source_snapshot')
    provenance={}
    for scene in manifest['scenes']:
        name=scene['scene']
        source_run=runs[0] if (runs[0]/name).exists() else runs[1]
        source=source_run/name/(P+'predictions.jsonl.gz')
        target=run/name/(P+'predictions.jsonl.gz')
        target.parent.mkdir()
        shutil.copy2(source,target)
        assert digest(source)==digest(target)
        provenance[name]=dict(source=str(source),sha256=digest(source),unchanged_byte_copy=True)
    result=evaluate(run,manifest,gt_path,dest,P)
    release=dict(aggregation_only=True,no_method_calls=True,selection_freeze=str(freeze_path),
        selection_freeze_sha256=digest(freeze_path),source_reports={str(r/(P+'report.json')):digest(r/(P+'report.json')) for r in runs},
        source_sha256=freeze['source_sha256'],config=freeze['selected_config'],seed=17,
        predictions_provenance=provenance,input_sha256={str(p):digest(p) for p in [manifest_path,gt_path]},
        created_utc=datetime.now(timezone.utc).isoformat(),GT_in_inference=False,default_changed=False)
    dump(run/(P+'release.json'),release)
    report=dict(complete=True,run=str(run),tag='S2',groups=120,results=result,aggregation=release,
        GT_in_inference=False,default_changed=False,source_sha256=freeze['source_sha256'])
    dump(run/(P+'report.json'),report)
    dump(dest/(P+'result.json'),report)
    print(json.dumps(result['node_main'],ensure_ascii=False))


if __name__=='__main__':
    main()
