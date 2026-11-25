"""单pair一个八方向+XY距离。标签描述source相对target；不存near/反边。"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import itertools,math
from types import SimpleNamespace
import numpy as np
from conceptgraph.scenegraph.relations import coerce_relation_nodes
LABELS=('front','front_left','left','back_left','back','back_right','right','front_right')
FRAME_SOURCE='CAM_FRONT optical +Z projected into world XY; ego-heading proxy, not object body heading'
MAX_DISTANCE_M=12.
COINCIDENT_M=1e-6

def axes(forward_xy):
    f=np.asarray(forward_xy,dtype=float)
    if f.shape!=(2,) or not np.isfinite(f).all() or np.linalg.norm(f)<1e-8:raise ValueError('valid_camera_forward_xy_required')
    f=f/np.linalg.norm(f);return f,np.asarray([-f[1],f[0]])

def direction_for_delta(subject_minus_reference_xy,forward_xy):
    d=np.asarray(subject_minus_reference_xy,dtype=float)
    if d.shape!=(2,) or not np.isfinite(d).all():raise ValueError('finite_delta_xy_required')
    f,left=axes(forward_xy)
    if np.linalg.norm(d)<=COINCIDENT_M:return None
    angle=math.degrees(math.atan2(float(d@left),float(d@f)))
    sector=(angle+22.5)/45.
    nearest=round(sector)
    # Only representational boundary error (8 ULP), never a metric tolerance.
    if abs(sector-nearest)<=8*math.ulp(max(abs(sector),1.)):sector=float(nearest)
    return LABELS[math.floor(sector)%8]

def invert_label(label):
    return LABELS[(LABELS.index(label)+4)%8]

inverse_relation=invert_label
label_for_delta=direction_for_delta

def build_edge_snapshot(objects,forward_xy,max_distance_m=MAX_DISTANCE_M):
    axes(forward_xy);eligible={};rejected=[]
    for obj in objects:
        identity=obj['id'];key=str(identity)
        if key in eligible:raise ValueError('duplicate_normalized_stable_node_id')
        center=np.asarray(obj['center'],dtype=float).reshape(-1)
        if center.size<2 or not np.isfinite(center[:2]).all():rejected.append(identity);continue
        eligible[key]=(identity,center[:2])
    edges=[];coincident=far=0
    for a,b in itertools.combinations(sorted(eligible),2):
        ai,ac=eligible[a];bi,bc=eligible[b];d=ac-bc;distance=float(np.linalg.norm(d))
        if distance<=COINCIDENT_M:coincident+=1;continue
        if distance>max_distance_m:far+=1;continue
        edges.append(dict(source=ai,target=bi,relation=direction_for_delta(d,forward_xy),distance_m=distance))
    return dict(edges=edges,statistics=dict(valid_nodes=len(eligible),invalid_nodes=len(rejected),candidate_pairs=len(eligible)*(len(eligible)-1)//2,
        coincident_pairs=coincident,outside_distance_pairs=far,stored_edges=len(edges),one_edge_per_unordered_pair=True),rejected_node_ids=rejected)

def build_edges_from_centers(objects,forward_xy,max_distance_m=MAX_DISTANCE_M):
    return build_edge_snapshot(objects,forward_xy,max_distance_m)['edges']

def query_pair(edges,subject,reference):
    if str(subject)==str(reference):return None
    for edge in edges:
        if str(edge['source'])==str(subject) and str(edge['target'])==str(reference):
            return dict(source=subject,target=reference,relation=edge['relation'],distance_m=edge['distance_m'])
        if str(edge['source'])==str(reference) and str(edge['target'])==str(subject):
            return dict(source=subject,target=reference,relation=invert_label(edge['relation']),distance_m=edge['distance_m'])
    return None

def canonical_fact(source_id,relation,target_id):
    # Required after predicted-ID -> GT-ID matching, since sort order may flip.
    if str(source_id)>str(target_id):return (str(target_id),invert_label(relation),str(source_id))
    return (str(source_id),relation,str(target_id))

@dataclass
class Edge:
    source:int
    target:int
    relation:str
    distance_m:float
    def to_dict(self):return dict(source=self.source,target=self.target,relation=self.relation,distance_m=self.distance_m)

class CurrentSingle8RelationCache:
    """Current positions only. Upstream owns node lifetime; motion never gates geometry."""
    def __init__(self):self.last_frame=-1;self.last_timestamp=None;self.last_statistics={}
    def update_snapshot(self,nodes,*,frame_index,heading,timestamp_s,frame_source=FRAME_SOURCE):
        if frame_index<=self.last_frame or (self.last_timestamp is not None and timestamp_s<=self.last_timestamp):raise ValueError('increasing_frame_and_timestamp_required')
        current,rejected=coerce_relation_nodes(nodes,frame_index=frame_index)
        objects=[];local={}
        for index,node in current.items():
            stable=node.metadata.get('source_node_id',node.node_id);objects.append(dict(id=stable,center=node.center));local[str(stable)]=index
        result=build_edge_snapshot(objects,heading);self.last_statistics=result['statistics'];self.last_frame=frame_index;self.last_timestamp=timestamp_s
        edges=[Edge(local[str(e['source'])],local[str(e['target'])],e['relation'],e['distance_m']) for e in result['edges']]
        return SimpleNamespace(edges=edges,statistics=result['statistics'],rejected_nodes=rejected,reference_frame_source=frame_source)
