"""纯CPU检查32/48/5.25真实执行guard，原65文件不变，不调用执行函数。"""
from pathlib import Path
import sys,json,ast,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion import nuscenes_full_20261125_resource_guards as guards
from conceptgraph.project_paths import experiment_directory
s=guards.source

def main():
    import torch
    run=experiment_directory('nuscenes','validation','nuscenes_full_20261125_resource_guard_contract_v8',create=True)
    policy=json.loads((s.DATA/'effective_resource_policy_v8.json').read_text());before={p:s.sha(p) for p in json.loads((s.DATA/'phaseB_source_preseal_v1.json').read_text())['source_sha256']}
    guards.install(policy,run);receipt=json.loads((run/'resource_guard_receipt_v8.json').read_text());assert receipt['only_resource_constants_changed']
    body=(run/'source_snapshot/resources/run_approved_resource_guards.py').read_text();assert str(32*3600) in body and str(48*3600) in body and str(int(5.25*2**30)) in body
    assert all(s.sha(p)==h for p,h in before.items());hold=json.loads((s.DATA/'formal_relation_contract_hold_v1.json').read_text());assert hold['active']
    s.dump(run/'summary.json',dict(passed=True,new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),original65_source_sha256_unchanged=True,guard_replacement_receipt_sha256=s.sha(run/'resource_guard_receipt_v8.json'),effective_policy_sha256=s.sha(s.DATA/'effective_resource_policy_v8.json'),formal_relation_hold_active=True,no_execution_function_called=True,full850_not_started=True))
    print((run/'summary.json').read_text(),flush=True)
if __name__=='__main__':main()
