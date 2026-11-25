"""H2: change the depth prior only when history crosses camera views."""
from conceptgraph.slam.mapnext_20261124_depth_history_v1 import UniqueHistoryMapper
from conceptgraph.slam.quality_mapper import QualityEvidenceMapper
from conceptgraph.slam.sparse_observations import cosine

EXTRA_SOURCES = ['conceptgraph/slam/mapnext_20261124_mapper.py',
                 'conceptgraph/slam/mapnext_20261124_depth_history_v1.py']


class CrossCameraHistoryMapper(UniqueHistoryMapper):
    def settings_record(self):
        record=super().settings_record()
        record['mapnext_depth_history'].update(version='H2',
            intervention='only if original eligible projected-center history includes a different camera',
            same_camera_history='original T3 unchanged')
        return record

    def observation_policy(self, frame):
        original=QualityEvidenceMapper.observation_policy(self,frame)
        robust=super().observation_policy(frame)
        camera=next(c for c,f in self.group_frames if f is frame)
        def select(depth,mask,gobs,index):
            box=gobs['xyxy'][index]
            category=gobs['classes'][gobs['class_id'][index]]
            feature=gobs['image_feats'][index]
            cross=[]
            for track in [*self.position_layer.tracks,*self.position_layer.dense_states.values()]:
                if (track['category']!=category or frame.timestamp_s-track['timestamp']>1.5
                        or cosine(feature,track['feature'])<.65):
                    continue
                point=frame.camera_pose[:3,:3].T@(track['center']-frame.camera_pose[:3,3])
                uv=frame.cam_K@point
                if (point[2]>1 and box[0]<=uv[0]/uv[2]<=box[2] and box[1]<=uv[1]/uv[2]<=box[3]
                        and track.get('_last_camera') not in [None,camera]):
                    cross.append(track['node_id'])
            selected,evidence=(robust if cross else original)(depth,mask,gobs,index)
            evidence['mapnext_cross_camera_history']=dict(intervened=bool(cross),cross_view_candidates=cross)
            return selected,evidence
        return select
