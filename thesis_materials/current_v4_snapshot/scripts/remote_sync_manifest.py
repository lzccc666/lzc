from pathlib import Path
import hashlib,json
O=Path(__file__).resolve().parents[1]
skip={'.venv','__pycache__','lo_profile','lo_rebuild_profile','rendered','历史版本'}
files=[]
for p in sorted(O.rglob('*')):
 rel=p.relative_to(O)
 if not p.is_file() or p.is_symlink() or any(x in skip for x in rel.parts):continue
 if p.suffix in {'.zip','.part'} or p.name=='本机同步校验清单.json':continue
 if '公开数据版' in p.name and p.suffix in {'.docx','.pdf'}:continue
 data=p.read_bytes();files.append({'path':rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'source':str(O),'files':files},ensure_ascii=False))
