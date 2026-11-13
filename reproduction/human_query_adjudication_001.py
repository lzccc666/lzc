"""独立人工表达标签审核；真值仅在标注服务使用，不开放封存方法评测。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np

from conceptgraph.completion.natural_collection import validate_collection


OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
CAMPAIGN = 'human_natural_adjudication_001'
STATES = {'approach_static_target', 'no_match_in_observed_prefix', 'target_not_yet_seen',
          'temporarily_occluded', 'target_present_inaccessible', 'insufficient_evidence', 'out_of_scope'}
COLORS = {'unknown', 'white', 'black', 'gray', 'silver', 'red', 'blue', 'green',
          'yellow', 'orange', 'brown', 'purple', 'other'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def text(record, name, minimum=1, maximum=2000):
    value = record.get(name)
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ValueError('字段缺失或长度无效：' + name)
    return value.strip()


def projected_box(obj, frame):
    """官方world box投到camera；裁剪近面交点，投影框不等于像素掩膜。"""
    w, x, y, z = obj['rotation_wxyz']
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    width, length, height = obj['size_wlh']
    local = np.array([[a*length/2, b*width/2, c*height/2]
                      for a in [-1, 1] for b in [-1, 1] for c in [-1, 1]])
    world = local @ rotation.T + np.asarray(obj['center'])
    pose = np.asarray(frame['camera_pose'])
    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    near = .1
    points = [p for p in camera if p[2] >= near]
    for a in range(8):
        for bit in [1, 2, 4]:
            b = a ^ bit
            if a >= b:
                continue
            p, q = camera[a], camera[b]
            if (p[2] < near) != (q[2] < near):
                points.append(p + (near-p[2])/(q[2]-p[2])*(q-p))
    if not points:
        return None
    points = np.asarray(points) @ np.asarray(frame['intrinsic']).T
    pixels = points[:, :2] / points[:, 2:]
    x0, y0 = np.maximum(pixels.min(axis=0), [0, 0])
    x1, y1 = np.minimum(pixels.max(axis=0), [frame['width'], frame['height']])
    return [float(x0), float(y0), float(x1-x0), float(y1-y0)] if x1 > x0 and y1 > y0 else None


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.root = self.output / 'annotation' / CAMPAIGN
        self.raw = self.output / 'annotation/human_generated_natural_submissions'
        self.reviews = self.output / 'annotation/human_natural_adjudication_records'
        self.cards_path = self.output / 'annotation/human_natural_collection_001/cards.json'
        cards = load(self.cards_path)
        self.cards = {(scene['scene'], frame['frame_idx']): dict(frame, scene=scene['scene'])
                      for scene in cards for frame in scene['frames']}
        self.truth_path = self.output / 'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl'
        self.truth_sha = sha(self.truth_path)
        self.frames = {}
        for line in self.truth_path.read_text().splitlines():
            row = json.loads(line)
            key = row['scene'], row['frame_idx']
            assert key in self.cards and row['sample_token'] == self.cards[key]['sample_token']
            self.frames[key] = row
        self.lock = threading.Lock()

    def sources(self):
        assert sha(self.truth_path) == self.truth_sha, '标注真值来源变化'
        return dict(truth_sha256=self.truth_sha, cards_sha256=sha(self.cards_path))

    def originals(self):
        rows = []
        for path in sorted(self.raw.glob('*.json')):
            record = load(path)
            validate_collection(record, output=self.output)
            if record['submission_id'] != path.stem:
                raise ValueError('原始提交ID与文件不符')
            rows.append(dict(record, source_sha256=sha(path)))
        return rows

    def records(self):
        return [load(path) for path in sorted(self.reviews.glob('*.json'))]

    def status(self):
        records = self.records()
        latest = {}
        for row in sorted(records, key=lambda r: r['server_received_utc']):
            latest[row['source_submission_id']] = row
        original = self.originals()
        return dict(campaign=CAMPAIGN, original_submissions=len(original),
                    synthetic_fixture=self.output != OUTPUT,
                    review_records=len(records), reviewed_submissions=len(latest),
                    label_eligible_for_retrieval=sum(r['label_eligible_for_retrieval'] for r in latest.values()),
                    by_scene=dict(Counter(r['scene'] for r in latest.values() if r['label_eligible_for_retrieval'])),
                    qualified_test_queries=0, new_test_opened=False,
                    qualified_status='审核记录先保留；表达家族/划分、覆盖、导航独立几何和release仍须验收',
                    human_identity_independently_verified=False, goal_complete=False)

    def frame(self, scene, index):
        card = self.cards[(scene, index)]
        truth = self.frames[(scene, index)]
        image = (self.output / card['rgb']).resolve()
        assert image.is_relative_to(self.output) and sha(image) == card['rgb_sha256']
        return dict(card, width=truth['width'], height=truth['height'],
                    objects=[dict(obj, box_xywh=projected_box(obj, truth)) for obj in truth['objects']],
                    camera_timestamp_s=truth['camera_timestamp_s'],
                    official_truth_is_annotation_only=True, temporal_alignment_is_approximate=True)

    def save(self, payload):
        if not isinstance(payload, dict) or payload.get('campaign') != CAMPAIGN:
            raise ValueError('非法审核批次')
        if payload.get('campaign_sha256') != sha(self.root / 'manifest.json'):
            raise ValueError('审核页面或标签协议已变化，请刷新')
        identity = payload.get('source_submission_id', '')
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{32}', identity):
            raise ValueError('非法原始提交ID')
        source = self.raw / (identity + '.json')
        original = load(source)
        validate_collection(original, output=self.output)
        if payload.get('source_sha256') != sha(source):
            raise ValueError('原始表达指纹不符')
        if payload.get('human_review_attested') is not True:
            raise ValueError('需要真实审核者确认亲自观察证据')
        reviewer = text(payload, 'reviewer', maximum=120)
        note = text(payload, 'evidence_note', minimum=5)
        decision, state = payload.get('decision'), payload.get('task_state')
        if decision not in {'accept', 'reject', 'needs_evidence'} or state not in STATES:
            raise ValueError('非法审核决定或任务状态')
        scene, frame_idx = original['scene'], original['frame_idx']
        viewed = payload.get('viewed_frames')
        if (not isinstance(viewed, list) or not viewed or len(viewed) > 100
                or any(type(i) is not int or (scene, i) not in self.frames for i in viewed)
                or frame_idx not in viewed):
            raise ValueError('必须查看原始指令绑定帧，且证据帧属于同一场景')
        viewed = sorted(set(viewed))
        objects = {obj['instance_token']: obj for i in viewed for obj in self.frames[(scene, i)]['objects']}
        selected = payload.get('accepted_instance_tokens')
        if not isinstance(selected, list) or len(selected) > 100 or len(set(selected)) != len(selected):
            raise ValueError('非法可接受对象集合')
        if any(not isinstance(token, str) or token not in objects for token in selected):
            raise ValueError('对象ID不属于已查看的官方实例证据')
        if state in {'approach_static_target', 'temporarily_occluded', 'target_present_inaccessible', 'target_not_yet_seen'} and decision == 'accept' and not selected:
            raise ValueError('正目标审核必须给出可接受官方实例集合')
        if state == 'no_match_in_observed_prefix' and selected:
            raise ValueError('当前观察范围无匹配不能同时指定匹配目标')
        ambiguity = payload.get('ambiguity')
        motion = payload.get('observed_motion')
        color = payload.get('observed_color')
        if ambiguity not in {'unique', 'multiple_acceptable', 'underspecified', 'unknown'}:
            raise ValueError('非法歧义标签')
        if motion not in {'static', 'dynamic', 'unknown'} or color not in COLORS:
            raise ValueError('非法运动或颜色标签')
        requirements = payload.get('required_fields')
        if not isinstance(requirements, list) or any(k not in {'color', 'fine_attribute', 'relation', 'anchor', 'negation'} for k in requirements):
            raise ValueError('请逐项标记指令实际要求的语义字段')
        labels = {key: text(payload, key, maximum=500) for key in
                  ['target_category', 'fine_attribute', 'relation', 'anchor_instances', 'negation', 'expression_family']}
        missing = []
        categories = {objects[token]['category'] for token in selected}
        if selected and categories != {labels['target_category']}:
            missing.append('target_category_mismatch')
        if ambiguity == 'unique' and len(selected) > 1:
            missing.append('unique_identity_set_mismatch')
        if ambiguity == 'multiple_acceptable' and len(selected) < 2:
            missing.append('multiple_identity_set_mismatch')
        if 'anchor' in requirements:
            anchors = [token.strip() for token in labels['anchor_instances'].split(',')]
            if any(token not in objects for token in anchors):
                missing.append('anchor_identity')
        if ambiguity in {'unknown', 'underspecified'}:
            missing.append('ambiguity')
        if labels['target_category'] == 'unknown':
            missing.append('target_category')
        if 'color' in requirements and color == 'unknown':
            missing.append('color')
        for key in ['fine_attribute', 'relation', 'negation']:
            if key in requirements and labels[key] in {'unknown', 'none'}:
                missing.append(key)
        if 'anchor' in requirements and labels['anchor_instances'] in {'unknown', 'none'}:
            missing.append('anchor')
        if state == 'approach_static_target':
            timestamps = [self.frames[(scene, i)]['camera_timestamp_s'] for i in viewed]
            if motion != 'static' or len(viewed) < 3 or max(timestamps)-min(timestamps) < 1:
                missing.append('multi_frame_static_evidence')
            if any(token not in {o['instance_token'] for o in self.frames[(scene, frame_idx)]['objects']} for token in selected):
                missing.append('current_target_identity')
        if state == 'no_match_in_observed_prefix' and not set(range(frame_idx+1)).issubset(viewed):
            missing.append('whole_observed_prefix_review')
        if state not in {'approach_static_target', 'no_match_in_observed_prefix'}:
            missing.append('special_state_protocol_review')
        frame_evidence = {str(i): dict(rgb_sha256=self.cards[(scene, i)]['rgb_sha256'],
                                     sample_token=self.frames[(scene, i)]['sample_token']) for i in viewed}
        for i in viewed:
            self.frame(scene, i)
        sources = self.sources()
        with self.lock:
            previous = [r for r in self.records() if r['source_submission_id'] == identity]
            previous.sort(key=lambda r: r['server_received_utc'])
            expected_parent = previous[-1]['review_id'] if previous else None
            if payload.get('supersedes_review_id') != expected_parent:
                raise ValueError('审核版本已变化，请刷新后追加修订；旧记录不会覆盖')
            review_id = uuid.uuid4().hex
            record = dict(campaign=CAMPAIGN, review_id=review_id, source_submission_id=identity,
                campaign_sha256=sha(self.root / 'manifest.json'),
                source_sha256=sha(source), scene=scene, frame_idx=frame_idx,
                reviewer=reviewer, evidence_note=note, decision=decision, task_state=state,
                accepted_instance_tokens=selected, ambiguity=ambiguity, observed_motion=motion,
                observed_color=color, required_fields=requirements, semantic_labels=labels,
                viewed_frames=viewed, frame_evidence=frame_evidence, **sources,
                human_review_attested=True, human_identity_independently_verified=False,
                author_identity_verified=False, label_eligible_for_retrieval=decision == 'accept' and not missing,
                unresolved_fields=missing, qualifies_for_sealed_test=False,
                navigation_reachability='unknown', independent_navigation_certificate_attached=False,
                expression_family_independence_audited=False,
                synthetic_fixture=self.output != OUTPUT,
                supersedes_review_id=expected_parent, annotation_only_GT=True,
                model_output_used=False, new_test_opened=False, automatic_label=False,
                server_received_utc=datetime.now(timezone.utc).isoformat(), goal_complete=False)
            self.reviews.mkdir(parents=True, exist_ok=True)
            with (self.reviews / (review_id + '.json')).open('x') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        return dict(saved=True, review_id=review_id,
                    label_eligible_for_retrieval=record['label_eligible_for_retrieval'],
                    unresolved_fields=missing, qualifies_for_sealed_test=False)


class Handler(SimpleHTTPRequestHandler):
    store = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.store.output), **kwargs)

    def reply(self, status, value):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/adjudication_status':
                self.reply(200, self.store.status())
            elif parsed.path == '/api/adjudication_submissions':
                self.reply(200, dict(submissions=self.store.originals(), reviews=self.store.records(),
                    scenes={scene: sorted(index for s, index in self.store.cards if s == scene)
                            for scene in sorted({s for s, _ in self.store.cards})}))
            elif parsed.path == '/api/adjudication_frame':
                args = parse_qs(parsed.query, strict_parsing=True)
                self.reply(200, self.store.frame(args['scene'][0], int(args['frame'][0])))
            elif parsed.path == '/':
                self.send_response(302)
                self.send_header('Location', '/annotation/' + CAMPAIGN + '/index.html')
                self.end_headers()
            else:
                super().do_GET()
        except (KeyError, ValueError, TypeError, OSError, AssertionError) as error:
            self.reply(400, dict(error=str(error)))

    def do_POST(self):
        if self.path != '/api/adjudications':
            self.reply(404, dict(error='未知提交入口'))
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 30000:
                raise ValueError('非法提交大小')

            def finite(value):
                raise ValueError('JSON不允许非有限常量：' + value)

            payload = json.loads(self.rfile.read(size), parse_constant=finite)
            self.reply(201, self.store.save(payload))
        except (KeyError, ValueError, TypeError, OSError, AssertionError) as error:
            self.reply(400, dict(error=str(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(OUTPUT)
    Handler.store = Store(args.output)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
