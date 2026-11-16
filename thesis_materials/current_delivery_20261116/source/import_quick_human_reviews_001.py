"""只导入用户真实导出的属性审核；源证据校验不等于人工结论正确或身份标注完成。"""
import argparse,datetime,hashlib,json,re
from pathlib import Path
from conceptgraph.completion.stage9 import OUTPUT as OLD,digest,dump
from conceptgraph.completion.review_server import validate_submission
R=Path(__file__).resolve().parents[1];MANIFEST=R/'annotation/quick_human_review_001/manifest.json'
def key(x):return (x['scene'],x['frame_idx'],x['detection_index'])
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def source_audit():
 m=json.loads(MANIFEST.read_text());assert len(m['cards'])==20;seen=set()
 for card in m['cards']:
  assert key(card) not in seen;seen.add(key(card));p=Path(card['source_evidence']);assert digest(p)==card['evidence_file_sha256'];evidence=json.loads(p.read_text());assert digest(evidence['source_frame_file'])==card['source_sha256'];detection=evidence['detections'][card['detection_index']];crop=p.parent/detection['actual_clip_crop'];assert digest(crop)==card['actual_crop_sha256']
 return dict(UTC=now(),campaign_cards=20,unique_sources=len(seen),manifest_sha256=digest(MANIFEST),source_hash_checks_passed=True,artificial_human_records_created=0,source_sha256=digest(__file__),goal_complete=False)
def validate_export(path):
 raw=path.read_bytes();payload=json.loads(raw);assert payload['campaign']=='quick_attributes_review_001' and payload['export_origin']=='human_offline_or_online_browser';records=payload['records'];assert isinstance(records,list) and 0<len(records)<=10000
 cards={key(c):c for c in json.loads(MANIFEST.read_text())['cards']};results=[];ids=set()
 for record in records:
  assert isinstance(record,dict) and record.get('human_attribute_only') is True
  assert record.get('reviewed_node_ids')==[] and record.get('instance_token') in [None,''] and record.get('label')=='证据不足'
  assert key(record) in cards;card=cards[key(record)]
  for field in ['evidence_run','evidence_index_sha256','evidence_file_sha256','source_sha256','mapping_method']:assert record[field]==card[field]
  identity=record['local_record_id'];assert isinstance(identity,str) and identity and identity not in ids;ids.add(identity)
  assert record['note'].startswith('本次仅核验图像属性，三维对象身份未核验。可见性：')
  date=datetime.datetime.fromisoformat(record['submitted_at'].replace('Z','+00:00'));assert date.tzinfo is not None
  # Legacy validator stringifies None to 'None'; normalize only this absent identity field.
  normalized=dict(record,instance_token='');accepted=validate_submission(normalized,output=OLD)
  accepted.update(human_attribute_only=True,annotation_origin='human_browser_export',human_3D_identity_credit=False,human_natural_instruction_credit=False,local_record_id=identity,raw_record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),imported_at=now(),independent_review_passed=False)
  results.append(accepted)
 return raw,results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['audit-source','import']);ap.add_argument('--file',type=Path);ap.add_argument('--run-id');a=ap.parse_args()
 if a.action=='audit-source':
  result=source_audit();p=R/'annotation/quick_human_review_001/source_binding_audit_001.json';dump(p,result);print(json.dumps(result,ensure_ascii=False));return
 if a.file is None or not a.run_id or not re.fullmatch('[a-zA-Z0-9_-]{1,80}',a.run_id):raise ValueError('导入需真实用户JSON和新的run-id')
 raw,records=validate_export(a.file);directory=R/'annotation/offline_imports'/a.run_id;directory.mkdir(parents=True,exist_ok=False);(directory/'original_user_export.json').write_bytes(raw)
 with (directory/'attribute_records.jsonl').open('x') as f:
  for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 latest={key(x):x for x in records}
 dump(directory/'receipt.json',dict(UTC=now(),raw_export_sha256=hashlib.sha256(raw).hexdigest(),manifest_sha256=digest(MANIFEST),records=len(records),unique_samples=len(latest),revisions=len(records)-len(latest),reviewers=sorted({r['reviewer'] for r in records}),known_color_samples=sum(x['observed_color']!='unknown' for x in latest.values()),human_attribute_only=True,human_3D_identity_records=0,human_natural_instructions=0,independent_reviewer_verified_records=0,source_and_schema_checked=True,agent_authored_records=0,records_sha256=digest(directory/'attribute_records.jsonl'),source_sha256=digest(__file__),goal_complete=False));print(str(directory))
if __name__=='__main__':main()
