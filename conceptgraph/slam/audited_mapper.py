"""对真实观测入口附加诊断，默认算法与EvidenceAwareMapper完全一致。"""
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper, confirmed_observation
from conceptgraph.slam.streamlined_mapping import BG_CLASSES
from conceptgraph.slam.utils import gobs_to_detection_list


class AuditedEvidenceMapper(EvidenceAwareMapper):
    def extract_detections(self, frame):
        self.frame_audit = []
        return gobs_to_detection_list(
            cfg=self.cfg, image=frame.image_rgb, depth_array=frame.depth_array,
            cam_K=frame.cam_K, idx=int(frame.frame_idx), gobs=frame.detections,
            trans_pose=frame.camera_pose, class_names=self.classes,
            BG_CLASSES=BG_CLASSES, color_path=frame.color_path,
            audit=self.frame_audit,
            observation_policy=self.observation_policy(frame),
            sparse_callback=self.sparse_callback(frame))

    def observation_policy(self, frame):
        return None

    def sparse_callback(self, frame):
        return None

    def update(self, frame):
        result = super().update(frame)
        self.frame_audit.append(dict(
            stage='confirmation', reason='map_snapshot', internal_objects=len(self.objects),
            confirmed_objects=sum(confirmed_observation(
                obj, strong_confidence=float(getattr(self.cfg, 'strong_confidence', .2)),
                minimum_observations=int(getattr(self.cfg, 'weak_confirmation_observations', 2)))
                for obj in self.objects),
            current_observations=len(self.last_observations)))
        return result
