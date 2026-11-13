"""保持既有审核接口，新增人工自由表达采集接口。"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json

from conceptgraph.completion.natural_collection import collection_status, save_collection
from conceptgraph.completion.review_server import Handler as ReviewHandler


def _reject_constant(value):
    raise ValueError('非法非有限JSON常量：'+value)


class Handler(ReviewHandler):
    def do_GET(self):
        if self.path == '/api/natural_collection_status':
            try:
                self._json(200, collection_status())
            except (OSError, ValueError, TypeError) as exc:
                self._json(400, dict(error=str(exc)))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/natural_collections':
            super().do_POST()
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 20000:
                raise ValueError('非法提交大小')
            record = json.loads(self.rfile.read(size), parse_constant=_reject_constant)
            result = save_collection(record)
        except (OSError, ValueError, TypeError) as exc:
            self._json(400, dict(error=str(exc)))
            return
        self._json(201, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
