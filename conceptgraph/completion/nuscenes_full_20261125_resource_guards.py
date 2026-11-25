"""仅替换冻结执行函数的三个资源常量；建图/前端/输入数学保持。"""
import ast,hashlib,inspect,json
from pathlib import Path
from conceptgraph.completion import nuscenes_full_20261125_run as base
from conceptgraph.completion import nuscenes_full_20261125 as source
ORIGINAL=base.run

def install(policy,run):
    original=inspect.getsource(ORIGINAL)
    replacements={'int(4.5*2**30)':str(policy['nu_persistent_bytes']),'12*3600':str(policy['GPU_occupied_s']),'36*3600':str(policy['overall_wall_s'])}
    changed=original
    for old,new in replacements.items():assert changed.count(old)==1,old;changed=changed.replace(old,new)
    # Reversibility proves the rest of the frozen run function is byte exact.
    restored=changed
    for old,new in replacements.items():assert restored.count(new)==1,new;restored=restored.replace(new,old)
    assert restored==original
    folder=Path(run)/'source_snapshot/resources';folder.mkdir(parents=True,exist_ok=True);snapshot=folder/'run_approved_resource_guards.py'
    if snapshot.exists():assert snapshot.read_text()==changed
    else:snapshot.write_text(changed)
    record=dict(created_utc=source.now(),original_file_sha256=source.sha(base.__file__),original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),executed_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),replacements=replacements,only_resource_constants_changed=True,actual_GPU_occupancy_accounting_unchanged=True,method_math_unchanged=True)
    receipt=Path(run)/'resource_guard_receipt_v8.json'
    if not receipt.exists():source.dump(receipt,record)
    exec(compile(changed,str(snapshot),'exec'),base.__dict__)
