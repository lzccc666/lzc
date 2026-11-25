"""当前观测更新对象几何，保留历史语义与身份；不使用评价标签。"""
import copy

from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper


class CurrentGeometryMapper(EvidenceAwareMapper):
    """检验对交通参与者避免旧点云累积是否能减少位置偏移。"""

    def update(self, frame):
        update = super().update(frame)
        latest = {int(obj['mask_idx'][-1]): obj for obj in self.last_observations}
        for obj in self.objects:
            if obj['image_idx'][-1] != frame.frame_idx:
                continue
            current = latest.get(int(obj['mask_idx'][-1]))
            if current is not None:
                obj['pcd'] = copy.deepcopy(current['pcd'])
                obj['bbox'] = copy.deepcopy(current['bbox'])
        return update
