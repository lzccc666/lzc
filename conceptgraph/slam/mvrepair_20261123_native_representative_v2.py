"""T3：已通过物理连续/完整链接的当前部件中，保留原生关联代表。

只替换 T2 的代表排序；所有继承/竞争/投影/寿命/确认门不变。
"""
from itertools import combinations
import numpy as np
from conceptgraph.slam.mapping import compute_spatial_similarities, compute_visual_similarities, aggregate_similarities
from conceptgraph.slam.slam_classes import DetectionList
from conceptgraph.slam.mvrepair_20261123_partial_fragment import PartialFragmentMapper
from conceptgraph.slam.mvrepair_20261123_prototype import (
    EXTRA_SOURCES as PROTOTYPE_SOURCES, PrototypeContinuityMapper,
    merge_prototypes, prototype_witness, independent_competitors, current_parts_compatible,
)
EXTRA_SOURCES = [*PROTOTYPE_SOURCES,
    'conceptgraph/slam/mvrepair_20261123_prototype.py',
    'tests/mvrepair_20261123_test_native_representative.py']


def representative_rank(source, native_score, index):
    return (int(source['reliable']), native_score, source['points'], source['confidence'], -index)


class NativeRepresentativeMapper(PrototypeContinuityMapper):
    def settings_record(self):
        record = super().settings_record()
        record['association_prototype']['representative'] = ('current reliable first, then original canonical aggregate association score '
            'within original class/distance/similarity gates; point count/confidence/index only break ties; no support/hits sum')
        return record

    def extract_detections(self, frame):
        fg, bg = PartialFragmentMapper.extract_detections(self, frame)
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
        representative_decisions = []
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
            native_by_member = {i: max((float(native[i,j]) for j,o in enumerate(self.objects)
                if self.lifecycle.resolve(layer.aliases.get(o['stable_object_id'], o['stable_object_id'])) == root
                and current_records[i]['category'] == o['class_name'][-1]
                and np.linalg.norm(np.asarray(o['bbox'].get_center())-current_records[i]['center']) <= self.cfg.max_tracking_distance_m
                and native[i,j] >= self.cfg.sim_threshold), default=-float('inf')) for i in members}
            representative = max(members, key=lambda i: representative_rank(
                self.measurement_sources[fg[i]['mask_idx'][-1]], native_by_member[i], i))
            representative_decisions.append(dict(canonical=root, selected_source_key=current_records[representative]['_observation_key'],
                candidates=[dict(source_key=current_records[i]['_observation_key'], native_canonical_score=native_by_member[i],
                    reliable=bool(current_records[i]['reliable']), actual_points=current_records[i]['points']) for i in sorted(members)]))
            self.prototype_bindings[fg[representative]['mask_idx'][-1]] = root
            removed.update(members-{representative})
        self.prototype_audit = dict(checks=self.prototype_checks, removed_secondary_source_keys=[
            self.measurement_sources[fg[i]['mask_idx'][-1]]['_observation_key'] for i in sorted(removed)],
            bound_representatives=self.prototype_bindings,
            counted_support='representative actual points only; secondary points/hits never summed',
            native_representative_decisions=representative_decisions)
        return DetectionList([obj for i,obj in enumerate(fg) if i not in removed]), bg
