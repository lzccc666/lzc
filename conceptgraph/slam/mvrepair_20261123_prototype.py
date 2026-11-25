"""T2：见证退休后的有限关联原型；分配前阻断不连续接管。

原型不输出，不制造点/hits；同组多个有物理连续证据的部件选一条代表。
"""
from __future__ import annotations

import copy
from itertools import combinations

import numpy as np

from conceptgraph.slam.mvrepair_20261123_partial_fragment import (
    EXTRA_SOURCES as TEMPORAL_SOURCES,
    PartialFragmentMapper,
    TemporalLifecycle,
    UnifiedSettings,
    projected_support,
)
from conceptgraph.slam.slam_classes import DetectionList
from conceptgraph.slam.mapping import compute_spatial_similarities, compute_visual_similarities, aggregate_similarities
from conceptgraph.slam.sparse_observations import cosine


EXTRA_SOURCES = [*TEMPORAL_SOURCES,
    'conceptgraph/slam/mvrepair_20261123_partial_fragment.py',
    'tests/mvrepair_20261123_test_prototype.py']


def independent_competitors(observation, root, objects):
    return [o['canonical'] for o in objects if o['canonical'] != root
        and o['category'] == observation['category']
        and np.linalg.norm((np.asarray(o['center'])-observation['center'])[:2]) <= 2.5
        and cosine(o['feature'], observation['feature']) >= .8]


def current_parts_compatible(a, b):
    if a['camera'] != b['camera']:
        return False
    ma, mb = a['mask'], b['mask']
    smaller = min(int(ma.sum()), int(mb.sum()))
    if not smaller or (ma & mb).sum()/smaller < .9:
        return False
    aa, bb = np.asarray(a['box']), np.asarray(b['box'])
    small, large = (aa, bb) if np.prod(aa[2:]-aa[:2]) <= np.prod(bb[2:]-bb[:2]) else (bb, aa)
    center = (small[:2]+small[2:])/2
    return bool(np.all(center >= large[:2]) and np.all(center <= large[2:]))


def merge_prototypes(prototypes, resolve, now):
    merged = {}
    for root, prototype in prototypes.items():
        if now-prototype['last_seen'] > 1.5:
            continue
        root = resolve(root)
        if root not in merged:
            merged[root] = dict(contexts=[], last_seen=prototype['last_seen'])
        target = merged[root]
        target['last_seen'] = max(target['last_seen'], prototype['last_seen'])
        known = {tuple(c['source_key']) for c in target['contexts']}
        for context in prototype['contexts']:
            if tuple(context['source_key']) not in known:
                target['contexts'].append(context)
                known.add(tuple(context['source_key']))
    return merged


def prototype_witness(observation, view, contexts, now):
    pairs = []
    for context in contexts:
        age = now-context['timestamp']
        if not 0 < age <= 1.5 or observation['camera'] != context['source_key'][0] or observation['category'] != context['category']:
            continue
        delta = np.asarray(observation['center'])-context['center']
        appearance = cosine(observation['feature'], context['feature'])
        if np.linalg.norm(delta[:2]) > 2.5 or abs(delta[2]) > 1.5 or appearance < .65:
            continue
        prior = projected_support(context['points'], view, observation['mask'])
        current = projected_support(observation['world_points'], context['view'], context['mask'])
        # 完整观测→当前部件，或原部分观测→当前更完整观测；没有纯距离分支。
        physical = max(min(prior['fraction']/.9, current['fraction']/.2),
            min(current['fraction']/.9, prior['fraction']/.2)) >= 1.
        pairs.append(dict(context_source=context['source_key'], age_s=age,
            xy_distance_m=float(np.linalg.norm(delta[:2])), z_distance_m=float(abs(delta[2])),
            feature_cosine=appearance, prior_into_current=prior, current_into_prior=current,
            physical_continuity=physical, primary=physical and appearance >= .8))
    return dict(primary=any(p['primary'] for p in pairs),
        physical_continuity=any(p['physical_continuity'] for p in pairs), context_pairs=pairs)


class PrototypeLifecycle(TemporalLifecycle):
    def observe_group(self, group_id, timestamp_s, observations):
        result = super().observe_group(group_id, timestamp_s, observations)
        layer = self.mapper.position_layer
        states = {s['node_id']:s for s in [*layer.tracks, *layer.dense_states.values()]}
        for witness in self.mapper.temporal_audit:
            if not witness['accepted']:
                continue
            root = self.resolve(witness['strong_canonical'])
            prototype = self.mapper.prototypes.setdefault(root, dict(contexts=[], last_seen=timestamp_s))
            weak, strong = states[witness['weak_raw']], states[witness['strong_raw']]
            self.mapper.add_context(prototype, weak, self.mapper.previous_views)
            self.mapper.add_context(prototype, self.mapper.previous_states[witness['strong_raw']], self.mapper.previous_views)
            self.mapper.add_context(prototype, strong, self.mapper.current_views)
            prototype['last_seen'] = timestamp_s
        return result


class PrototypeContinuityMapper(PartialFragmentMapper):
    def __init__(self, cfg, classes, settings=None):
        super().__init__(cfg, classes, settings or UnifiedSettings())
        self.lifecycle = PrototypeLifecycle(self)
        self.prototypes = {}
        self.prototype_audit = {}

    def settings_record(self):
        return dict(**super().settings_record(), association_prototype=dict(
            lifetime_s=1.5, lifetime_origin='original sparse/internal association TTL',
            position_gate_m=2.5, z_gate_m=1.5, primary_cosine=.8,
            native_secondary_cosine=.65, secondary_requires_unique_primary=True,
            forward_fraction=.9, reverse_fraction=.2,
            representative='one actual reliable/support/confidence-ranked observation; no point or hits sum',
            motion='ego-pose world projection only; no object-motion compensation; static/slow partial-surface scope, dynamic costs retained',
            output='none; prototype clocks update only with accepted real observations'))

    def add_context(self, prototype, state, views):
        key = state.get('_observation_keys', [None])[-1]
        if key is None or key[0] not in views or views[key[0]]['capture_timestamp_s'] != key[1]:
            return
        if any(c['source_key'] == key for c in prototype['contexts']):
            return
        view = views[key[0]]
        prototype['contexts'].append(dict(source_key=copy.deepcopy(key), timestamp=state['timestamp'],
            category=state['category'], feature=np.asarray(state['feature']).copy(),
            center=np.asarray(state['last_measured_center']).copy(), points=np.asarray(state['_last_world_points']).copy(),
            mask=view['masks'][key[2]].copy(), view=dict(pose=view['pose'].copy(), intrinsic=view['intrinsic'].copy())))

    def extract_detections(self, frame):
        fg, bg = super().extract_detections(frame)
        self.prototypes = merge_prototypes(self.prototypes, self.lifecycle.resolve, frame.timestamp_s)
        self.prototype_bindings, self.prototype_checks = {}, []
        candidates, current_records = {}, {}
        native = None
        if fg and self.objects and self.prototypes:
            native = aggregate_similarities(self.cfg, compute_spatial_similarities(self.cfg, fg, self.objects),
                compute_visual_similarities(self.cfg, fg, self.objects)).detach().cpu().numpy()
        layer = self.position_layer
        live_roots = {self.lifecycle.resolve(layer.aliases.get(o['stable_object_id'], o['stable_object_id'])) for o in self.objects}
        object_records = [dict(canonical=self.lifecycle.resolve(layer.aliases.get(o['stable_object_id'], o['stable_object_id'])),
            category=o['class_name'][-1], center=np.asarray(o['bbox'].get_center()), feature=o['clip_ft'].detach().cpu().numpy()) for o in self.objects]
        for i, obj in enumerate(fg):
            key = obj['mask_idx'][-1]
            source = self.measurement_sources[key]
            obs = dict(source, mask=self.current_views[source['camera']]['masks'][source['_observation_key'][2]])
            obs['box'] = np.asarray(obj['xyxy'][-1])
            current_records[i] = obs
            matches = []
            for root, prototype in self.prototypes.items():
                if root not in live_roots:
                    continue
                witness = prototype_witness(obs, self.current_views[source['camera']], prototype['contexts'], frame.timestamp_s)
                witness.update(observation_index=i, source_key=source['_observation_key'], canonical=root)
                witness['native_candidate'] = bool(native is not None and any(
                    self.lifecycle.resolve(layer.aliases.get(o['stable_object_id'], o['stable_object_id']))==root
                    and source['category']==o['class_name'][-1] and native[i,j]>=self.cfg.sim_threshold
                    and np.linalg.norm(np.asarray(o['bbox'].get_center())-source['center'])<=self.cfg.max_tracking_distance_m
                    for j,o in enumerate(self.objects)))
                # 已有别的canonical在原类/位置/.8外观内同样竞争时，不强制认领。
                competitors = []
                if witness['physical_continuity']:
                    competitors = independent_competitors(source, root, object_records)
                    if competitors:
                        witness['primary'] = False
                witness['independent_competing_canonicals'] = competitors
                self.prototype_checks.append(witness)
                if witness['primary']:
                    matches.append(root)
            if len(matches) == 1:
                candidates.setdefault(matches[0], []).append(i)
        removed = set()
        for root, primaries in candidates.items():
            # 连续但外观较弱的原生部件只能在唯一强主观测存在时作为同组辅助。
            members = set(primaries)
            for check in self.prototype_checks:
                if check['canonical'] == root and check['physical_continuity'] and check['native_candidate'] and not check['independent_competing_canonicals']:
                    members.add(check['observation_index'])
            # 每条输入只能归一个canonical；跨原型有多解就不收束。
            members = {i for i in members if sum(c['observation_index']==i and c['physical_continuity'] for c in self.prototype_checks) == 1}
            if not members or not (members & set(primaries)):
                continue
            if any(not current_parts_compatible(current_records[a], current_records[b]) for a,b in combinations(members,2)):
                for check in self.prototype_checks:
                    if check['canonical']==root and check['observation_index'] in members:
                        check['current_component_conflict']=True
                continue
            representative = max(members, key=lambda i:(int(self.measurement_sources[fg[i]['mask_idx'][-1]]['reliable']),
                self.measurement_sources[fg[i]['mask_idx'][-1]]['points'], self.measurement_sources[fg[i]['mask_idx'][-1]]['confidence'], -i))
            self.prototype_bindings[fg[representative]['mask_idx'][-1]] = root
            removed.update(members-{representative})
        self.prototype_audit = dict(checks=self.prototype_checks, removed_secondary_source_keys=[
            self.measurement_sources[fg[i]['mask_idx'][-1]]['_observation_key'] for i in sorted(removed)],
            bound_representatives=self.prototype_bindings,
            counted_support='representative actual points only; secondary points/hits never summed')
        return DetectionList([obj for i,obj in enumerate(fg) if i not in removed]), bg

    def refine_association_scores(self, frame, detections, spatial_sim, visual_sim, scores):
        scores = super().refine_association_scores(frame, detections, spatial_sim, visual_sim, scores).clone()
        changes = []
        layer = self.position_layer
        for i, obj in enumerate(detections):
            index = obj['mask_idx'][-1]
            binding = self.prototype_bindings.get(index)
            for j, other in enumerate(self.objects):
                root = self.lifecycle.resolve(layer.aliases.get(other['stable_object_id'], other['stable_object_id']))
                physical = next((c['physical_continuity'] for c in self.prototype_checks if c['canonical']==root and c['source_key']==self.measurement_sources[index]['_observation_key']), False)
                if root in self.prototypes and not physical:
                    scores[i,j] = -float('inf')
                    changes.append(dict(source_key=self.measurement_sources[index]['_observation_key'], raw_target=other['stable_object_id'], canonical=root, action='reject_discontinuous_takeover_before_assignment'))
                if binding is not None:
                    scores[i,j] = 100. if root == binding else -float('inf')
                    changes.append(dict(source_key=self.measurement_sources[index]['_observation_key'], raw_target=other['stable_object_id'], canonical=root, action='physical_unique_prototype_binding' if root==binding else 'exclude_other_canonical'))
        self.prototype_audit['association_edge_changes'] = changes
        return scores

    def update_group(self, frames, capture_times, projections=None):
        output = super().update_group(frames, capture_times, projections)
        layer = self.position_layer
        now = frames[0][1].timestamp_s
        for state in [*layer.tracks, *layer.dense_states.values()]:
            root = self.lifecycle.resolve(layer.aliases.get(state['node_id'], state['node_id']))
            key = state.get('_observation_keys', [None])[-1]
            supported = any(c['canonical']==root and c['source_key']==key and c['physical_continuity'] for c in self.prototype_checks)
            if root in self.prototypes and state['timestamp'] == now and supported:
                prototype = self.prototypes[root]
                self.add_context(prototype, state, self.current_views)
                prototype['last_seen'] = now
        for p in self.prototypes.values():
            p['contexts'] = [c for c in p['contexts'] if now-c['timestamp'] <= 1.5]
        self.audit['association_prototypes'] = dict(**self.prototype_audit,
            active=[dict(canonical=root,last_actual_measurement_s=p['last_seen'],
                contexts=[dict(source_key=c['source_key'],timestamp=c['timestamp'],actual_points=len(c['points'])) for c in p['contexts']]) for root,p in self.prototypes.items()])
        return output


__all__ = ['PrototypeContinuityMapper', 'UnifiedSettings']
