"""关闭时间生命周期子系统，保留累积地图、关联、深度、关系及原证据门限。"""
import copy

from carla_ablation_agent_001 import Agent as OriginalAgent, CONDITIONS as ORIGINAL_CONDITIONS
from carla_agent_isolated_002 import mapping_config
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper

CONDITION = 'without_history_maintenance'
CONDITIONS = [*ORIGINAL_CONDITIONS, CONDITION]
MODULE_PARAMETERS = dict(enable_dynamic_object_handling=False, dynamic_ttl_s=1.e9)
DEFINITION = (
    '去掉对象时间生命周期维护：不初始化/更新运动历史，不采用动态目标当前点云替换，'
    '不作短期过期清理；保留跨帧一对一匹配和普通点云累积。观测元数据按原代码保存，'
    '缺失运动证据保持unknown，不补造static；原有身份、时间、深度及车辆安全检查不变。'
    '这是整个时间生命周期子系统移除，与仅关闭短期TTL、丢弃所有历史地图分别列出。')


def lifecycle_config(base, enabled=True):
    cfg = copy.deepcopy(base)
    if not enabled:
        for name, value in MODULE_PARAMETERS.items():
            setattr(cfg, name, value)
    return cfg


def lifecycle_observation(mapper, now_s):
    return dict(
        enable_dynamic_object_handling=bool(mapper.cfg.enable_dynamic_object_handling),
        dynamic_ttl_s=float(mapper.cfg.dynamic_ttl_s),
        objects=[dict(stable_object_id=obj.get('stable_object_id'),
                      num_detections=int(obj.get('num_detections', 1)),
                      motion_state=obj.get('motion_state', 'unknown'),
                      motion_state_field_present='motion_state' in obj,
                      center_history_length=len(obj.get('center_history', [])),
                      frame_history_length=len(obj.get('frame_history', [])),
                      timestamp_history_length=len(obj.get('timestamp_history_s', [])),
                      timestamp_s=obj.get('timestamp_s'),
                      timestamp_age_s=(float(now_s)-float(obj['timestamp_s'])
                                       if obj.get('timestamp_s') is not None else None),
                      image_idx=list(map(int, obj.get('image_idx', []))),
                      pcd_points=len(obj['pcd'].points))
                 for obj in mapper.objects])


class Agent(OriginalAgent):
    def __init__(self, query, prior, classes, dest, condition):
        super().__init__(query, prior, classes, dest,
                         'full_rule_geometry' if condition == CONDITION else condition)
        self.condition = condition
        if condition == CONDITION:
            self.mapper = StreamlinedMapper(lifecycle_config(mapping_config(), False), classes)

    def perceive(self, index, state, *args, **kwargs):
        result = super().perceive(index, state, *args, **kwargs)
        result['temporal_lifecycle_observation'] = lifecycle_observation(self.mapper, state.time_s)
        return result
