"""Replay frozen caches with a versioned mapper, full denominator, source snapshots and evidence."""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import importlib
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.completion.mv136_20261123_protocol import FRONT3, evaluate
from conceptgraph.completion.stage9 import PROJECT, digest, dump
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from scripts.multiview_20261123_run_graph import load_frame, write_csv

P = 'mapnext_20261124_'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--mapper', default='conceptgraph.slam.mapnext_20261124_mapper:AuditedNativeMapper')
    parser.add_argument('--tag', default='T3')
    parser.add_argument('--reference-exact', action='store_true')
    parser.add_argument('--include-front', action='store_true')
    args = parser.parse_args()
    begun = time.monotonic()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    registration = json.loads((PROJECT/'outputs/reports/mapping_iteration_20261124/mapnext_20261124_registration.json').read_text())
    deadline = datetime.fromisoformat(registration['deadline_utc'])
    assert all(digest(Path(p)) == sha for p,sha in registration['protected_sha256'].items())
    frozen_path = PROJECT/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json'
    frozen = json.loads(frozen_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    module_name, class_name = args.mapper.split(':')
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    run = experiment_directory('nuscenes','validation',args.run_id,create=True)
    paths = [Path(__file__).resolve(),Path(module.__file__),PROJECT/'conceptgraph/completion/mv136_20261123_protocol.py',
             PROJECT/'scripts/mapnext_20261124_prepare.py',*map(Path,frozen['source_sha256'])]
    paths += [PROJECT/p for p in getattr(module,'EXTRA_SOURCES',[])]
    sources = {}
    for path in dict.fromkeys(paths):
        rel = path.relative_to(PROJECT)
        target = run/'source_snapshot'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        sources[str(path)] = dict(sha256=digest(path),snapshot=str(target))
    inputs = {str(args.manifest):digest(args.manifest),str(args.ground_truth):digest(args.ground_truth),str(frozen_path):digest(frozen_path)}
    for scene in manifest['scenes']:
        for group in scene['groups']:
            for cam in FRONT3:
                view = group['views'][cam]
                for key in ['rgb','depth','detections','projection_evidence']:
                    inputs[view[key]] = digest(Path(view[key]))
    dump(run/(P+'release.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),
        registration=registration,config=frozen['config'],settings=klass(OmegaConf.create(frozen['config']),CLASSES).settings_record(),
        mapper=args.mapper,tag=args.tag,seed=17,GT_in_inference=False,source_sha256=sources,input_sha256=inputs,
        image_scope=FRONT3,reference_clock='same frozen max6 actual exposure; no rear image consumed',default_changed=False))
    costs, exact, total = {},0,0
    for scene in manifest['scenes']:
        name=scene['scene']
        reference_path=PROJECT/'outputs/results/nuscenes/validation/iterations/mv136_20261123_main_v1'/name/'mv136_20261123_predictions.jsonl.gz'
        reference={r['frame_idx']:r for r in map(json.loads,gzip.open(reference_path,'rt'))}
        methods={args.tag+'_three':FRONT3}
        if args.include_front:
            methods[args.tag+'_front']=['CAM_FRONT']
        mappers={tag:klass(OmegaConf.create(frozen['config']),CLASSES) for tag in methods}
        folder=run/name
        folder.mkdir()
        with gzip.open(folder/(P+'predictions.jsonl.gz'),'wt') as stream:
            for group in scene['groups']:
                if datetime.now(timezone.utc)>=deadline or shutil.disk_usage(PROJECT).free<20*2**30:
                    raise RuntimeError('time/disk budget; prefix retained')
                loaded={cam:load_frame(group,cam) for cam in FRONT3}
                projections={}
                for cam in FRONT3:
                    with np.load(group['views'][cam]['projection_evidence']) as file:
                        projections[cam]={k:file[k].copy() for k in file.files}
                    projections[cam]['lidar_token']=group['views'][cam].get('lidar_token')
                predictions,audits,timing={},{},{}
                for method,cameras in methods.items():
                    np.random.seed(17+group['frame_idx'])
                    frames=[(cam,copy.deepcopy(loaded[cam])) for cam in cameras]
                    captures={cam:group['views'][cam]['camera_timestamp_s'] for cam in cameras}
                    start=time.perf_counter()
                    predictions[method]=mappers[method].update_group(frames,captures,{cam:projections[cam] for cam in cameras})
                    elapsed=time.perf_counter()-start
                    audits[method]=mappers[method].audit
                    timing[method]=elapsed
                    costs[method]=costs.get(method,0)+elapsed
                    if args.reference_exact:
                        ref='T3_1' if method.endswith('_front') else 'T3_3'
                        assert predictions[method]==reference[group['frame_idx']]['predictions'][ref], (name,group['frame_idx'],method)
                        exact+=1
                stream.write(json.dumps(dict(scene=name,frame_idx=group['frame_idx'],sample_token=group['sample_token'],
                    reference_timestamp_s=group['reference_timestamp_s'],predictions=predictions,audit=audits,timing_s=timing,
                    GT_in_inference=False),allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
                total+=1
                print(name,group['frame_idx'],args.tag,'complete',flush=True)
    results=evaluate(run,manifest,args.ground_truth,args.report_dir,P)
    assert all(digest(Path(p))==sha for p,sha in inputs.items())
    assert all(digest(Path(p))==v['sha256'] for p,v in sources.items())
    assert all(digest(Path(p))==sha for p,sha in registration['protected_sha256'].items())
    report=dict(complete=True,run=str(run),tag=args.tag,groups=total,exact_reference_conditions=exact,
        results=results,wall_s=time.monotonic()-begun,cost_total_s=costs,seed=17,GT_in_inference=False,
        source_sha256=sources,input_sha256=inputs,default_changed=False,no_new_frontend=True,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        RSS_scope='all selected mapper conditions plus imports and images; not isolated method memory')
    dump(run/(P+'report.json'),report)
    dump(args.report_dir/(P+'result.json'),report)
    print(json.dumps(results['node_main'],ensure_ascii=False))


if __name__=='__main__':
    main()
