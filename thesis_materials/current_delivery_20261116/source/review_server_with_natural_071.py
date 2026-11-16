"""保留已部署属性审核，接通既有自然指令页面；新记录仅写本轮目录。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import uuid

from conceptgraph.completion.natural_collection import CAMPAIGN, validate_collection
from conceptgraph.completion.stage9 import OUTPUT
from review_server_isolated_002 import Handler as PreviousHandler

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / 'annotation/human_generated_natural_submissions_071'
LOCK = threading.Lock()


def reject_constant(value):
    raise ValueError('不接受非有限JSON数值：'+value)


def save_submission(record, *, storage=None, validator=None):
    storage = STORAGE if storage is None else Path(storage)
    validator = validate_collection if validator is None else validator
    # 沿用只读的原画面/哈希/人工自述校验，保存位置单独限定到新批次。
    checked = validator(record, output=OUTPUT)
    with LOCK:
        for path in storage.glob('*.json'):
            previous = json.loads(path.read_text())
            if previous['scene'] == checked['scene'] and previous['normalized_command'] == checked['normalized_command']:
                raise ValueError('同场景已有相同表达；重复不会增加样本数')
        storage.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        checked.update(submission_id=identity, server_received_utc=datetime.now(timezone.utc).isoformat(),
            review_status='human_authored_source_and_schema_checked_pending_label_review')
        path = storage/f'{identity}.json'
        pending = storage/f'{identity}.pending'
        with pending.open('x') as stream:
            json.dump(checked, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        if path.exists():
            raise FileExistsError(path)
        pending.rename(path)
    return dict(saved=True, submission_id=identity, path=str(path),
                qualifies_for_sealed_test=False, label_status='待独立核验')


def collection_status(*, storage=None):
    storage = STORAGE if storage is None else Path(storage)
    records = [json.loads(p.read_text()) for p in storage.glob('*.json')]
    by_scene = Counter(row['scene'] for row in records)
    return dict(campaign=CAMPAIGN, human_authored_submissions=len(records), by_scene=dict(by_scene),
        qualifying_labels=0, storage=str(storage), automatic_labels_created=0,
        status='仅统计原始人工表达；不等于有效标签、独立实例或封存测试成绩')


class Handler(PreviousHandler):
    def do_GET(self):
        if self.path == '/api/natural_collection_status':
            try:
                self._json(200, collection_status())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._json(400, dict(error=str(exc)))
            return
        super().do_GET()

    def do_POST(self):
        if self.path != '/api/natural_collections':
            super().do_POST()
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 20000:
                raise ValueError('非法提交大小')
            record = json.loads(self.rfile.read(size), parse_constant=reject_constant)
            result = save_submission(record)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._json(400, dict(error=str(exc)))
            return
        self._json(201, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
