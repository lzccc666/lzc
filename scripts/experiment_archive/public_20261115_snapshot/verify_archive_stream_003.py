"""在内存流中逐项校验新源码归档，不重复展开占用预算。"""
import hashlib,json,tarfile
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];S=R/'source_bundle/reproduction_003'
if __name__=='__main__':
 receipt=json.loads((S/'receipt.json').read_text());manifest=json.loads((S/'source_manifest.json').read_text());expected={r['archive_path']:r for r in manifest};seen=set()
 archive=S/'actual_source_snapshot.tar.gz';assert digest(archive)==receipt['archive_sha256']
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   path=Path(member.name);assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name not in seen
   row=expected[member.name];h=hashlib.sha256();handle=tar.extractfile(member);size=0
   for block in iter(lambda:handle.read(1024**2),b''):h.update(block);size+=len(block)
   assert h.hexdigest()==row['sha256'] and size==row['bytes'];seen.add(member.name)
 assert seen==set(expected)
 old=json.loads((R/'source_bundle/reproduction_001/source_manifest.json').read_text());changed=[r['path'] for r in old if digest(r['path'])!=r['sha256']]
 assert not changed
 dump(S/'archive_integrity_001.json',dict(archive_sha256=digest(archive),manifest_sha256=digest(S/'source_manifest.json'),files_verified=len(seen),all_member_bytes_and_hashes_match=True,old_source_files_checked=len(old),old_source_files_changed=changed,archive_extracted=False,clean_environment_install_passed=False,previous_isolated_smoke_path=str(R/'source_bundle/reproduction_smoke_001/report.json'),verification_source_sha256=digest(__file__),goal_complete=False));print(json.dumps(dict(files=len(seen),old_source_changes=0)))
