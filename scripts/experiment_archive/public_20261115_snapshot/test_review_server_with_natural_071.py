"""只在临时测试目录写夹具，不向真实人工队列提交自动记录。"""
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler, Request

import review_server_with_natural_071 as service


class CollectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.storage = Path(self.directory.name)/'test_fixtures_only'

    def tearDown(self):
        self.directory.cleanup()

    def fixture(self, scene='test-scene', command='测试夹具，不计入真实人工标注'):
        return dict(scene=scene, normalized_command=command, command=command,
                    automatic_label=True, annotation_origin='unit_test_fixture')

    def validator(self, record, *, output):
        return dict(record)

    def test_save_is_only_to_requested_storage(self):
        record = self.fixture()
        result = service.save_submission(record, storage=self.storage, validator=self.validator)
        self.assertTrue(result['saved'])
        self.assertEqual(Path(result['path']).parent, self.storage)
        saved = json.loads(Path(result['path']).read_text())
        self.assertEqual(saved['annotation_origin'], 'unit_test_fixture')
        self.assertFalse(result['qualifies_for_sealed_test'])

    def test_duplicate_rejected_without_overwriting_first(self):
        first = service.save_submission(self.fixture(), storage=self.storage, validator=self.validator)
        content = Path(first['path']).read_bytes()
        with self.assertRaises(ValueError):
            service.save_submission(self.fixture(), storage=self.storage, validator=self.validator)
        self.assertEqual(Path(first['path']).read_bytes(), content)
        self.assertEqual(len(list(self.storage.glob('*.json'))), 1)

    def test_same_text_in_different_scene_preserves_source(self):
        for scene in ['test-a', 'test-b']:
            service.save_submission(self.fixture(scene=scene), storage=self.storage, validator=self.validator)
        self.assertEqual(service.collection_status(storage=self.storage)['human_authored_submissions'], 2)

    def test_validation_failure_cannot_create_record(self):
        def rejecting(record, **kwargs):
            raise ValueError('test rejection')
        with self.assertRaises(ValueError):
            service.save_submission({}, storage=self.storage, validator=rejecting)
        self.assertFalse(self.storage.exists())

    def test_non_finite_JSON_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"x":NaN}', parse_constant=service.reject_constant)

    def test_HTTP_status_and_invalid_post(self):
        with patch.object(service, 'STORAGE', self.storage):
            server = ThreadingHTTPServer(('127.0.0.1', 0), service.Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            client = build_opener(ProxyHandler({}))
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with client.open(url+'/api/natural_collection_status') as reply:
                    self.assertEqual(json.load(reply)['human_authored_submissions'], 0)
                request = Request(url+'/api/natural_collections', data=b'{}',
                    headers={'Content-Type': 'application/json'})
                with self.assertRaises(HTTPError) as error:
                    client.open(request)
                self.assertEqual(error.exception.code, 400)
                self.assertFalse(self.storage.exists())
                # 有效HTTP路径使用显式测试夹具，只写临时目录，不能增加真实人工记录。
                with patch.object(service, 'validate_collection', return_value=self.fixture()):
                    with client.open(request) as reply:
                        self.assertEqual(reply.status, 201)
                        self.assertTrue(json.load(reply)['saved'])
                with client.open(url+'/api/natural_collection_status') as reply:
                    self.assertEqual(json.load(reply)['human_authored_submissions'], 1)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == '__main__':
    unittest.main()
