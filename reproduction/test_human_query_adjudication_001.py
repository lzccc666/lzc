"""仅隔离合成夹具；不向正式采集或审核目录写入任何记录。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image
import pytest


SPEC = importlib.util.spec_from_file_location(
    'adjudication_fixture_module', Path(__file__).with_name('human_query_adjudication_001.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.fixture
def case(tmp_path):
    assert str(tmp_path).startswith(str(MODULE.OUTPUT / 'reproduction'))
    collection = tmp_path / 'annotation/human_natural_collection_001'
    frames, truths = [], []
    for index in range(3):
        image = tmp_path / f'images/{index}.png'
        image.parent.mkdir(exist_ok=True)
        Image.new('RGB', (100, 100), (250-index, 250, 250)).save(image)
        frames.append(dict(frame_idx=index, sample_token=f'sample{index}',
            camera_token=f'camera{index}', camera_timestamp=index*500000,
            rgb=str(image.relative_to(tmp_path)), rgb_sha256=MODULE.sha(image)))
        truths.append(dict(scene='fixture-scene', frame_idx=index, sample_token=f'sample{index}',
            camera_timestamp_s=index*.5, camera_pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            intrinsic=[[100,0,50],[0,100,50],[0,0,1]], width=100, height=100,
            objects=[dict(instance_token='b'*32, annotation_token='c'*32, category='car',
                          center=[0,0,5], size_wlh=[2,2,2], rotation_wxyz=[1,0,0,0],
                          official_motion='static')]))
    write(collection / 'cards.json', [dict(scene='fixture-scene', scene_token='fixture-scene-token',
                                          log_token='fixture-log', frames=frames)])
    write(collection / 'manifest.json', dict(cards_sha256=MODULE.sha(collection / 'cards.json')))
    truth = tmp_path / 'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl'
    truth.parent.mkdir(parents=True)
    truth.write_text(''.join(json.dumps(row)+'\n' for row in truths))
    review_manifest = tmp_path / 'annotation' / MODULE.CAMPAIGN / 'manifest.json'
    write(review_manifest, dict(synthetic_fixture=True))
    raw = dict(campaign='human_natural_collection_001',
               campaign_sha256=MODULE.sha(collection / 'manifest.json'), scene='fixture-scene',
               frame_idx=0, camera_token='camera0', rgb_sha256=frames[0]['rgb_sha256'],
               author='SYNTHETIC_FIXTURE', command='靠近画面里的白色汽车', note='仅合成测试夹具，不是真实人工表达',
               human_authorship_attested=True, language='zh', author_intent='approach_static_target',
               author_target_point_normalized=None, submission_id='a'*32,
               annotation_origin='synthetic_test_fixture', synthetic_fixture=True)
    raw_path = tmp_path / 'annotation/human_generated_natural_submissions' / ('a'*32+'.json')
    write(raw_path, raw)
    store = MODULE.Store(tmp_path)
    payload = dict(campaign=MODULE.CAMPAIGN, campaign_sha256=MODULE.sha(review_manifest),
        source_submission_id='a'*32, source_sha256=MODULE.sha(raw_path),
        human_review_attested=True, reviewer='SYNTHETIC_FIXTURE', evidence_note='仅隔离合成夹具测试',
        decision='accept', task_state='approach_static_target', viewed_frames=[0,1,2],
        accepted_instance_tokens=['b'*32], ambiguity='unique', observed_motion='static',
        observed_color='white', required_fields=['color'], target_category='car',
        fine_attribute='none', relation='none', anchor_instances='none', negation='none',
        expression_family='synthetic-fixture-family', supersedes_review_id=None)
    return store, payload, raw_path


def test_wrong_instance_never_creates_review(case):
    store, payload, _ = case
    payload['accepted_instance_tokens'] = ['d'*32]
    with pytest.raises(ValueError, match='官方实例'):
        store.save(payload)
    assert not list(store.reviews.glob('*.json'))


def test_original_hash_drift_rejected(case):
    store, payload, source = case
    source.write_text(source.read_text()+' ')
    with pytest.raises(ValueError, match='指纹'):
        store.save(payload)


def test_rgb_drift_rejected(case):
    store, payload, _ = case
    (store.output / 'images/1.png').write_bytes(b'corrupt fixture')
    with pytest.raises(AssertionError):
        store.save(payload)


def test_human_confirmation_required(case):
    store, payload, _ = case
    payload['human_review_attested'] = False
    with pytest.raises(ValueError, match='真实审核'):
        store.save(payload)


def test_single_frame_is_not_static_evidence_and_revision_preserves_it(case):
    store, payload, source = case
    source_sha = MODULE.sha(source)
    payload['viewed_frames'] = [0]
    first = store.save(payload)
    assert not first['label_eligible_for_retrieval']
    assert 'multi_frame_static_evidence' in first['unresolved_fields']
    first_path = store.reviews / (first['review_id']+'.json')
    first_sha = MODULE.sha(first_path)
    payload['viewed_frames'] = [0,1,2]
    with pytest.raises(ValueError, match='审核版本'):
        store.save(payload)
    payload['supersedes_review_id'] = first['review_id']
    second = store.save(payload)
    assert second['label_eligible_for_retrieval']
    assert not second['qualifies_for_sealed_test']
    record = MODULE.load(store.reviews / (second['review_id']+'.json'))
    assert record['navigation_reachability'] == 'unknown'
    assert record['synthetic_fixture'] and not record['expression_family_independence_audited']
    assert MODULE.sha(first_path) == first_sha and MODULE.sha(source) == source_sha


def test_category_and_anchor_conflicts_stay_unresolved(case):
    store, payload, _ = case
    payload.update(target_category='tree', required_fields=['anchor'], anchor_instances='fake-anchor')
    result = store.save(payload)
    assert not result['label_eligible_for_retrieval']
    assert {'target_category_mismatch','anchor_identity'}.issubset(result['unresolved_fields'])


def test_unknown_color_is_not_a_color_label(case):
    store, payload, _ = case
    payload['observed_color'] = 'unknown'
    result = store.save(payload)
    assert not result['label_eligible_for_retrieval'] and 'color' in result['unresolved_fields']


def test_other_scene_or_unavailable_frame_rejected(case):
    store, payload, _ = case
    payload['viewed_frames'] = [0,1,999]
    with pytest.raises(ValueError, match='同一场景'):
        store.save(payload)


def test_projection_front_and_behind_camera(case):
    store, _, _ = case
    frame = store.frames[('fixture-scene',0)]
    obj = deepcopy(frame['objects'][0])
    assert MODULE.projected_box(obj, frame) == pytest.approx([25,25,50,50])
    obj['center'] = [0,0,-5]
    assert MODULE.projected_box(obj, frame) is None


def test_http_round_trip_in_fixture_only(case):
    store, payload, _ = case
    class Handler(MODULE.Handler):
        def log_message(self, *_args):
            pass
    Handler.store = store
    server = MODULE.ThreadingHTTPServer(('127.0.0.1',0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = build_opener(ProxyHandler({}))
    url = 'http://127.0.0.1:'+str(server.server_port)
    try:
        with opener.open(url+'/api/adjudication_frame?scene=fixture-scene&frame=0') as response:
            frame = json.load(response)
        assert frame['official_truth_is_annotation_only'] and frame['objects'][0]['box_xywh']
        with pytest.raises(HTTPError) as caught:
            opener.open(Request(url+'/api/adjudications',data=b'{}',headers={'Content-Type':'application/json'}))
        assert caught.value.code == 400
        with opener.open(Request(url+'/api/adjudications', data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})) as response:
            result = json.load(response)
        assert result['saved'] and not result['qualifies_for_sealed_test']
        with opener.open(url+'/api/adjudication_status') as response:
            status = json.load(response)
        assert status['reviewed_submissions'] == 1 and status['qualified_test_queries'] == 0
        assert status['synthetic_fixture'] and not status['new_test_opened']
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
