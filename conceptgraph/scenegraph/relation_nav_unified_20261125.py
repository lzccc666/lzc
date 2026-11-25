"""Current-frame relation facts, exact pair-cache refresh and explicit frame semantics.

The legacy builder and incremental cache remain untouched.  A cached geometry
candidate is never a remembered current fact: source/time/confidence are rebuilt
from the current snapshot before global conflict resolution and pruning.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import itertools
import json
import time

import numpy as np

from conceptgraph.scenegraph.relations import (
    RelationConfig, RelationNode, build_relations, coerce_relation_nodes,
    finalize_relation_candidates, generate_relation_candidates, inverse_relation,
    resolve_coordinate_frame, _make_edge, _relation_statistics,
)


POLICY = dict(coordinate_frame='ego_heading', coordinate_frame_fallback=None,
    relation_distance_thresholds_m=dict(near=6., directional=12., left_of=12., right_of=12., in_front_of=12., behind=12.),
    directional_mode='all_supported_axes', minimum_direction_component_m=1.,
    position_update_threshold_m=.1)
FRAME_SOURCE = 'camera_forward_xy_proxy_of_ego_heading; CAM_FRONT optical +Z projected into world XY; not object body heading'


def frame_metadata(heading, *, timestamp_s, frame_source=FRAME_SOURCE):
    forward=np.asarray(heading,dtype=float)
    if forward.shape!=(2,) or not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-8:
        raise ValueError('missing_or_degenerate_camera_forward_xy')
    forward=forward/np.linalg.norm(forward)
    return dict(reference_frame='ego_heading', reference_frame_source=frame_source,
        position_frame='world_xy_m', forward_axis_world=forward.tolist(),
        left_axis_world=[-float(forward[1]),float(forward[0])],timestamp_s=float(timestamp_s),
        relation_semantics='source minus target dot forward/left; positive means source front/left of target',
        object_relative_supported=False,object_heading_source='unknown_not_inferred_from_velocity')


def enrich(edges, context):
    return [replace(e,metadata={**e.metadata,**context,
        'current_fact_only':True,'historical_memory_used_as_current_fact':False,
        'inverse_query_relation':inverse_relation(e.relation),
        'canonical_storage_counting':'one canonical edge; inverse expansion only for queries/declared ordered-fact evaluation'}) for e in edges]


def full_current(nodes, *, frame_index, heading, timestamp_s, frame_source=FRAME_SOURCE):
    result=build_relations(nodes,POLICY,frame_index=frame_index,ego_heading=heading)
    context=frame_metadata(heading,timestamp_s=timestamp_s,frame_source=frame_source)
    result.edges=enrich(result.edges,context)
    return result


class CurrentRelationCache:
    """Cache pair geometry; rederive current metadata and discard absent facts.

    A full current node snapshot is required.  An empty snapshot expires every
    current relation.  There is deliberately no extra dynamic-edge memory in
    this public fact set: the upstream node tracker owns node lifetime.
    """
    def __init__(self):
        self.config=RelationConfig.from_mapping(POLICY)
        self.geometry_config=replace(self.config,minimum_confidence=0.)
        self.pairs={};self.frame_signature=None;self.previous_ids=set()
        self.last_frame=-1;self.last_timestamp=None;self.historical_observations={}

    def update_snapshot(self,nodes,*,frame_index,heading,timestamp_s,frame_source=FRAME_SOURCE):
        begun=time.perf_counter()
        if frame_index<=self.last_frame or (self.last_timestamp is not None and timestamp_s<=self.last_timestamp):
            raise ValueError('current_relation_snapshots_require_increasing_frame_and_time')
        current,rejected=coerce_relation_nodes(nodes,frame_index=frame_index)
        context=frame_metadata(heading,timestamp_s=timestamp_s,frame_source=frame_source)
        frame=resolve_coordinate_frame('ego_heading',None,ego_heading=heading)
        signature=(tuple(frame.forward_axis_world),tuple(frame.left_axis_world))
        axis_changed=self.frame_signature is not None and signature!=self.frame_signature
        active=set(current);removed=self.previous_ids-active;added=active-self.previous_ids
        current_keys=set(itertools.combinations(sorted(current),2));expired=set(self.pairs)-current_keys
        for key in expired:del self.pairs[key]
        candidates=[];generated=hits=0
        for a,b in sorted(current_keys):
            na,nb=current[a],current[b]
            # Exact geometry, not a displacement tolerance: a sub-0.1m move can
            # cross a 6m/12m/1m predicate boundary.
            geom=(tuple(na.center),tuple(nb.center),signature)
            cached=self.pairs.get((a,b))
            if cached is None or cached['geometry']!=geom:
                raw,_,_=generate_relation_candidates({a:na,b:nb},self.geometry_config,frame,frame_index=frame_index)
                cached=dict(geometry=geom,proposals=[copy.deepcopy(e) for e in raw]);self.pairs[a,b]=cached;generated+=1
            else:hits+=1
            for old in cached['proposals']:
                # Refresh even reused geometry: never merge a stale source
                # timestamp, geometry, count or confidence into a current fact.
                edge=_make_edge(current[old.source],current[old.target],relation=old.relation,
                    distance_m=old.distance_m,angle_margin=old.metadata['confidence_components']['angle_margin'],
                    threshold_m=self.config.relation_distance_thresholds_m[old.relation],
                    config=self.config,frame=frame,frame_index=frame_index)
                if edge.confidence>=self.config.minimum_confidence:candidates.append(edge)
        edges,_,conflicts,pruning=finalize_relation_candidates(candidates,self.config)
        result=full_current({},frame_index=frame_index,heading=heading,timestamp_s=timestamp_s,frame_source=frame_source)
        result.edges=enrich(edges,context);result.conflict_statistics=conflicts;result.pruning_statistics=pruning
        result.mode='incremental_current_snapshot';result.node_count=len(current);result.edge_count=len(edges)
        result.relation_statistics=_relation_statistics(result.edges)
        result.rejected_edges=rejected
        result.update_statistics=dict(candidate_pair_count=len(current_keys),geometry_recomputed_pairs=generated,
            geometry_cache_hits=hits,expired_pair_cache=len(expired),added_nodes=sorted(added),removed_nodes=sorted(removed),
            coordinate_axes_changed=axis_changed,current_source_refreshed_edges=len(candidates),
            authoritative_current_snapshot=True,dynamic_memory_edges_retained=0)
        result.timing={'total_s':time.perf_counter()-begun}
        for edge in result.edges:
            key=(edge.source,edge.target,edge.relation)
            history=self.historical_observations.setdefault(key,dict(first_computed_frame=frame_index,snapshot_support=0))
            history.update(last_computed_frame=frame_index,snapshot_support=history['snapshot_support']+1)
        self.previous_ids=active;self.frame_signature=signature;self.last_frame=frame_index;self.last_timestamp=timestamp_s
        return result


def canonical_payload(edges):
    """All current fields, including current source provenance and confidence."""
    return sorted((json.dumps(e.to_dict(),sort_keys=True,separators=(',',':')) for e in edges))


def expanded_facts(edges):
    facts=set()
    for edge in edges:
        facts.add((edge.source,edge.relation,edge.target))
        facts.add((edge.target,inverse_relation(edge.relation),edge.source))
    return facts


def reference_query_status(text,reference_frame='ego_heading',object_heading=None):
    if reference_frame=='object_relative':
        return dict(supported=False,status='OBJECT_RELATIVE_UNSUPPORTED',
            reason='object body orientation is not evidenced; motion direction is not a substitute')
    if reference_frame!='ego_heading':
        return dict(supported=False,status='REFERENCE_FRAME_UNSUPPORTED')
    return dict(supported=True,status='SUPPORTED_EGO_HEADING_CONVENTION',reference_frame='ego_heading',
        source=FRAME_SOURCE)
