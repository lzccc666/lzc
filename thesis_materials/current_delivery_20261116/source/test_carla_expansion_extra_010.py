"""追加只读测试，不改已经冻结并在运行的009源码。"""
from types import SimpleNamespace
import unittest

import numpy as np
from shapely.geometry import box

from audit_carla_expansion_009 import match_identities
from carla_real_graph_bridge_003 import observed_row
from conceptgraph.navigation.vehicle_geometry import VehicleState


class ExtraTests(unittest.TestCase):
    def test_lane_scope_filters_targets_and_keeps_off_lane_anchor(self):
        def node(category,x,y):
            return SimpleNamespace(object_tag=category,bbox_center=np.array([x,y,0.]),
                motion_state='static',metadata=dict(last_seen_frame=2,timestamp_s=.4,num_detections=3,
                    object_identity=f'{category}-{x}-{y}',depth_evidence=[dict(source='current_projected_lidar',retained_pixels=6,support_fraction=.8,sufficient_support=True)]))
        nodes={0:node('car',10.,0.),1:node('car',7.,4.),2:node('truck',10.,-4.3)}
        row,_,_,_=observed_row('以卡车为参照，选择本车道内左侧的小汽车',nodes,[],VehicleState(0.,0.,0.,time_s=.4),2,road=box(-5.,-1.75,50.,1.75))
        self.assertEqual(set(row['candidate_ids']),{100,102})

    def test_duplicate_predictions_cannot_both_match_the_same_target(self):
        nodes=[dict(node_id=0,category='car',center=[10.,0.,0.]),dict(node_id=1,category='car',center=[10.1,0.,0.])]
        mapping=match_identities(nodes,[dict(instance_id='official-target',category='car',center_xy=[10.,0.])])
        self.assertEqual(mapping,{0:'official-target'})


if __name__=='__main__':unittest.main(verbosity=2)
