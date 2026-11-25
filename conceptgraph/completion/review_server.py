"""保存真实人工提交；只验证来源与结构，不把自动检查当人工结论。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import threading
import uuid

from conceptgraph.completion.stage9 import OUTPUT, digest

LABELS = {'证据不足','正确关联','漏检','稀疏深度','投影偏差','错合并','重复节点','身份变化','语义错误','评测关联不足'}
COLORS = {'unknown','red','blue','white','black','gray','silver','green','yellow','orange','brown','multiple'}
CATEGORIES = {'unknown','car','truck','bus','bicycle','motorcycle','pedestrian','traffic sign','traffic light','other'}
LOCK = threading.Lock()
EVIDENCE_RUNS = {'corrected_dev_evidence_001', 'causal_dev_evidence_001'}


def validate_submission(record, *, output=None):
    output = OUTPUT if output is None else Path(output)
    if not isinstance(record,dict):
        raise ValueError('提交必须是JSON对象')
    reviewer = str(record.get('reviewer','')).strip()
    if not reviewer or len(reviewer)>120:
        raise ValueError('请填写真实审核者')
    evidence_run = record.get('evidence_run', 'corrected_dev_evidence_001')
    if evidence_run not in EVIDENCE_RUNS:
        raise ValueError('未知证据版本，不能指定任意目录')
    bundle = output/'annotation'/evidence_run
    index_path = bundle/'index.json'
    index_sha = digest(index_path)
    if evidence_run == 'causal_dev_evidence_001' and record.get('evidence_index_sha256') != index_sha:
        raise ValueError('过去LiDAR审核必须绑定当前证据索引指纹')
    cases = json.loads(index_path.read_text())['cases']
    entry = next((r for r in cases if r['scene']==record.get('scene') and r['frame_idx']==record.get('frame_idx')),None)
    if entry is None:
        raise ValueError('场景或帧不在核验清单')
    evidencepath = (output/entry['url']).resolve()
    if not evidencepath.is_relative_to(bundle.resolve()):
        raise ValueError('证据文件越过版本目录')
    audited_files = None
    if evidence_run == 'causal_dev_evidence_001':
        audit = json.loads((bundle/'audit_001/report.json').read_text())
        files_path = bundle/'audit_001/files.json'
        if (not audit['passed'] or audit['source_index_sha256'] != index_sha
                or audit['files_manifest_sha256'] != digest(files_path)):
            raise ValueError('过去LiDAR证据独立审计指纹不一致')
        audited_files = json.loads(files_path.read_text())
        evidence_sha = digest(evidencepath)
        if (record.get('evidence_file_sha256') != evidence_sha
                or audited_files.get(str(evidencepath.relative_to(output.resolve()))) != evidence_sha):
            raise ValueError('过去LiDAR证据内容已改变，不能复用旧审核')
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
    if crop is not None and not crop.resolve().is_relative_to(evidencepath.parent):
        raise ValueError('裁剪越过当前证据目录')
    if audited_files is not None and crop is not None:
        if audited_files.get(str(crop.relative_to(output.resolve()))) != digest(crop):
            raise ValueError('真实CLIP裁剪已改变，不能接收旧像素审核')
    instance_token = str(record.get('instance_token','')).strip() or None
    mapping_method, reviewed_node_ids = None, []
    if evidence_run == 'causal_dev_evidence_001':
        mapping_method = record.get('mapping_method')
        if not isinstance(mapping_method, str) or mapping_method not in evidence['predictions']:
            raise ValueError('过去LiDAR审核必须指定当前证据中的建图方法')
        reviewed_node_ids = record.get('reviewed_node_ids', [])
        if (not isinstance(reviewed_node_ids, list)
                or any(not isinstance(node, str) for node in reviewed_node_ids)
                or len(set(reviewed_node_ids)) != len(reviewed_node_ids)):
            raise ValueError('审核节点必须是不重复的节点标识列表')
        current_nodes = {node['node_id'] for node in evidence['predictions'][mapping_method]}
        if not set(reviewed_node_ids) <= current_nodes:
            raise ValueError('审核节点不在当前帧的所选方法中')
        if record['label'] in {'正确关联', '错合并', '重复节点', '身份变化'} and not reviewed_node_ids:
            raise ValueError('对象图身份结论需要明确选择待审核节点')
        official = {row['instance_token'] for row in evidence.get('official_evaluation_only', [])}
        if instance_token is not None and instance_token not in official:
            raise ValueError('实例不在当前帧官方评价清单，保持未知或重新核验')
        if record['label'] == '正确关联' and instance_token is None:
            raise ValueError('确认正确关联需要填写当前帧实例，无法确认时保留证据不足')
    note = str(record.get('note','')).strip()
    if not note:
        raise ValueError('请填写核验依据；看不清时写明原因并保留unknown')
    return dict(scene=evidence['scene'],frame_idx=evidence['frame_idx'],detection_index=index,
        evidence_run=evidence_run,evidence_index_sha256=index_sha,
        source_sha256=evidence['source_frame_file_sha256'],evidence_file_sha256=digest(evidencepath),
        actual_crop_sha256=digest(crop) if crop else None,reviewer=reviewer,label=record['label'],
        observed_color=record.get('observed_color','unknown'),observed_category=record.get('observed_category','unknown'),
        instance_token=instance_token,
        mapping_method=mapping_method, reviewed_node_ids=reviewed_node_ids,
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
