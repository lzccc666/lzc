"""L4：冻结L2前视位置辅助 + 冻结L3同曝光车辆竞争假设。

类别竞争先在真实观测入口收束，L1持久身份只输出一个canonical；
前视证据仅校正该canonical位置，不创建第二张对象图。
"""
from dataclasses import asdict

from conceptgraph.slam.mvrepair_20261123_front_position import (
    EXTRA_SOURCES as FRONT_SOURCES,
    CanonicalFrontMapper,
    UnifiedSettings,
)
from conceptgraph.slam.mvrepair_lifecycle_20261123_crossclass import CrossClassPhysicalMapper


EXTRA_SOURCES = [*FRONT_SOURCES,
    'conceptgraph/slam/mvrepair_20261123_front_position.py',
    'conceptgraph/slam/mvrepair_lifecycle_20261123_crossclass.py',
    'scripts/mvrepair_lifecycle_20261123_crossclass_run.py']


class CanonicalPhysicalFrontMapper(CanonicalFrontMapper, CrossClassPhysicalMapper):
    """MRO只组合已有两个冻结组件；没有新门限、GT或额外输出状态。"""

    def settings_record(self):
        return dict(**super().settings_record(), physical_hypotheses=asdict(self.physical_settings),
            combination='L4=L2 front position evidence + L3 same-exposure vehicle hypothesis competition')

    def update_group(self, frames, capture_times, projections=None):
        output = super().update_group(frames, capture_times, projections)
        self.audit['physical_hypotheses'] = self.physical_audit
        return output


__all__ = ['CanonicalPhysicalFrontMapper', 'UnifiedSettings']
