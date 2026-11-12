"""保存真实人工提交；只验证来源与结构，不把自动检查当人工结论。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import threading
import uuid

from conceptgraph.completion.stage9 import OUTPUT, digest

LABELS = {'证据不足','正确关联','漏检','稀疏深度','投影偏差','错合并','重复节点','身份变化','语义错误','评测关联不足'}
COLORS = {'unknown','red','blue','white','black','gray','silver','green','yellow','orange','brown','multiple'}
CATEGORIES = {'unknown','car','truck','bus','bicycle','motorcycle','pedestrian','traffic sign','traffic light','other'}
LOCK = threading.Lock()


def validate_submission(record):
    if not isinstance(record,dict):
        raise ValueError('提交必须是JSON对象')
    reviewer = str(record.get('reviewer','')).strip()
    if not reviewer or len(reviewer)>120:
        raise ValueError('请填写真实审核者')
    cases = json.loads((OUTPUT/'annotation/corrected_dev_evidence_001/index.json').read_text())['cases']
    entry = next((r for r in cases if r['scene']==record.get('scene') and r['frame_idx']==record.get('frame_idx')),None)
    if entry is None:
        raise ValueError('场景或帧不在核验清单')
    evidencepath = OUTPUT/entry['url']
    evidence = json.loads(evidencepath.read_text())
    if record.get('source_sha256')!=evidence['source_frame_file_sha256'] or digest(evidence['source_frame_file'])!=record.get('source_sha256'):
        raise ValueError('源证据SHA-256不一致')
    index = record.get('detection_index')
    if type(index) is not int or not 0<=index<len(evidence['detections']):
        raise ValueError('非法检测序号')
    if record.get('label') not in LABELS or record.get('observed_color','unknown') not in COLORS or record.get('observed_category','unknown') not in CATEGORIES:
        raise ValueError('非法审核结论或属性')
    detection = evidence['detections'][index]
    crop = evidencepath.parent/detection['actual_clip_crop'] if detection.get('actual_clip_crop') else None
    note = str(record.get('note','')).strip()
    if not note:
        raise ValueError('请填写核验依据；看不清时写明原因并保留unknown')
    return dict(scene=evidence['scene'],frame_idx=evidence['frame_idx'],detection_index=index,
        source_sha256=evidence['source_frame_file_sha256'],evidence_file_sha256=digest(evidencepath),
        actual_crop_sha256=digest(crop) if crop else None,reviewer=reviewer,label=record['label'],
        observed_color=record.get('observed_color','unknown'),observed_category=record.get('observed_category','unknown'),
        instance_token=str(record.get('instance_token','')).strip() or None,
        body_type=str(record.get('body_type','unknown')).strip(),note=note[:5000],
        client_submitted_at=record.get('submitted_at'),server_received_utc=datetime.now(timezone.utc).isoformat(),
        annotation_origin='human_browser_submission',automatic_label=False,
        review_status='human_submitted_source_and_schema_checked',
        label_correctness_independently_verified=False)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(OUTPUT),**kwargs)

    def _json(self,status,value):
        content = json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path=='/api/review_status':
            rows = list((OUTPUT/'annotation/human_submissions').glob('*.json'))
            natural = list((OUTPUT/'annotation/human_natural_submissions').glob('*.json'))
            self._json(200,dict(human_submissions=len(rows)+len(natural),development_submissions=len(rows),
                natural_submissions=len(natural),storage='annotation/human_submissions',
                natural_storage='annotation/human_natural_submissions'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path not in {'/api/reviews','/api/natural_reviews'}:
            self._json(404,dict(error='unknown endpoint'))
            return
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0<size<20000:
                raise ValueError('非法提交大小')
            submitted = json.loads(self.rfile.read(size))
            if self.path=='/api/natural_reviews':
                from conceptgraph.completion.natural_review import validate_natural_submission
                record = validate_natural_submission(submitted)
                record['server_received_utc'] = datetime.now(timezone.utc).isoformat()
                record['review_status'] = 'human_submitted_source_and_schema_checked'
            else:
                record = validate_submission(submitted)
        except (ValueError,TypeError,OSError) as exc:
            self._json(400,dict(error=str(exc)))
            return
        directory = OUTPUT/'annotation'/('human_natural_submissions' if self.path=='/api/natural_reviews' else 'human_submissions')
        directory.mkdir(parents=True,exist_ok=True)
        identity = uuid.uuid4().hex
        record['submission_id'] = identity
        payload = json.dumps(record,ensure_ascii=False,indent=2)+'\n'
        with LOCK:
            path = directory/f'{identity}.json'
            with path.open('x') as handle:
                handle.write(payload)
        self._json(201,dict(saved=True,submission_id=identity,path=str(path),
                            sha256=hashlib.sha256(payload.encode()).hexdigest()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8767)
    args = parser.parse_args()
    ThreadingHTTPServer(('0.0.0.0',args.port),Handler).serve_forever()


if __name__=='__main__':
    main()
