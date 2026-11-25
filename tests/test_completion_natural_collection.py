"""临时目录合成夹具；不会制造真实人工记录。"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from conceptgraph.completion.natural_collection import (
    CAMPAIGN, STORAGE, collection_status, save_collection, validate_collection,
)
from conceptgraph.completion.stage9 import digest


@pytest.fixture
def collection_fixture(tmp_path):
    root = tmp_path/'annotation'/CAMPAIGN
    root.mkdir(parents=True)
    image = tmp_path/'synthetic_evidence.txt'
    image.write_text('合成测试字节，不是真实画面或人工标注')
    frame = dict(frame_idx=0, rgb=image.name, rgb_sha256=digest(image), camera_token='camera',
                 sample_token='sample', camera_timestamp=123)
    cards = [dict(scene='synthetic-scene', scene_token='scene-token', log_token='log-token', frames=[frame])]
    (root/'cards.json').write_text(json.dumps(cards))
    (root/'manifest.json').write_text(json.dumps(dict(cards_sha256=digest(root/'cards.json'))))
    record = dict(campaign=CAMPAIGN, campaign_sha256=digest(root/'manifest.json'), scene='synthetic-scene',
                  frame_idx=0, camera_token='camera', rgb_sha256=digest(image), author='synthetic_unit_test',
                  command='仅供单元测试的合成指令', note='临时目录中的合成夹具，不计真实人工记录',
                  language='zh', author_intent='unknown', human_authorship_attested=True,
                  author_target_point_normalized=[0.3, 0.4])
    return tmp_path, record, image


def test_collection_preserves_unknowns_without_creating_qualified_labels(collection_fixture):
    root, record, _ = collection_fixture
    validated = validate_collection(record, output=root)
    assert validated['command'] == record['command'] and validated['target_identity'] is None
    assert validated['reachability'] == 'unknown' and not validated['qualifies_for_sealed_test']
    assert not validated['label_correctness_independently_verified']
    assert not (root/'annotation'/STORAGE).exists()
    saved = save_collection(record, output=root)
    assert saved['saved'] and not saved['qualifies_for_sealed_test']
    status = collection_status(output=root)
    assert status['human_authored_submissions'] == 1 and status['qualified_test_queries'] == 0
    assert status['by_scene'] == {'synthetic-scene': 1}


def test_pixel_and_manifest_changes_reject_old_submissions(collection_fixture):
    root, record, image = collection_fixture
    image.write_text('changed synthetic pixels')
    with pytest.raises(ValueError, match='指纹'):
        save_collection(record, output=root)
    assert not (root/'annotation'/STORAGE).exists()


def test_duplicate_commands_cannot_inflate_count_even_concurrently(collection_fixture):
    root, record, _ = collection_fixture

    def submit(_):
        try:
            return save_collection(record, output=root)['saved']
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, range(4)))
    assert sum(results) == 1
    assert len(list((root/'annotation'/STORAGE).glob('*.json'))) == 1
    record['command'] = '  仅供单元测试的 合成指令 '
    with pytest.raises(ValueError, match='重复'):
        save_collection(record, output=root)


@pytest.mark.parametrize('field,value', [('frame_idx', True), ('human_authorship_attested', False),
    ('author_target_point_normalized', [float('nan'), 0]), ('author_target_point_normalized', [True, 0]),
    ('language', 'invalid'), ('author', {}), ('command', '   '), ('note', ''), ('camera_token', 'wrong')])
def test_malformed_records_never_persist(collection_fixture, field, value):
    root, record, _ = collection_fixture
    record[field] = value
    with pytest.raises(ValueError):
        save_collection(record, output=root)
    assert not (root/'annotation'/STORAGE).exists()


def test_http_collection_and_legacy_review_routes_on_temporary_fixture(collection_fixture, monkeypatch):
    from conceptgraph.completion import natural_collection_server, review_server

    root, record, _ = collection_fixture
    monkeypatch.setattr(review_server, 'OUTPUT', root)
    monkeypatch.setattr(natural_collection_server, 'save_collection', partial(save_collection, output=root))
    monkeypatch.setattr(natural_collection_server, 'collection_status', partial(collection_status, output=root))
    server = ThreadingHTTPServer(('127.0.0.1', 0), natural_collection_server.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    local_open = build_opener(ProxyHandler({})).open
    try:
        with local_open(base+'/api/review_status', timeout=5) as response:
            assert json.load(response)['human_submissions'] == 0
        request = Request(base+'/api/natural_collections', data=json.dumps(record).encode(),
                          headers={'Content-Type': 'application/json'})
        with local_open(request, timeout=5) as response:
            assert response.status == 201 and not json.load(response)['qualifies_for_sealed_test']
        with pytest.raises(HTTPError) as error:
            local_open(request, timeout=5)
        assert error.value.code == 400
        with local_open(base+'/api/natural_collection_status', timeout=5) as response:
            status = json.load(response)
            assert status['human_authored_submissions'] == 1 and status['qualified_test_queries'] == 0
        invalid = Request(base+'/api/natural_collections', data=b'{"submitted_at":NaN}',
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            local_open(invalid, timeout=5)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
