"""S2: independence override also needs multiple actual current LiDAR points."""
from conceptgraph.slam.mapnext_20261124_sparse_separation_v1 import IndependentSparseMixin
from conceptgraph.slam.mapnext_20261124_mapper import AuditedNativeMapper
from conceptgraph.slam.mapnext_20261124_depth_history_v2 import CrossCameraHistoryMapper

EXTRA_SOURCES = ['conceptgraph/slam/mapnext_20261124_mapper.py',
                 'conceptgraph/slam/mapnext_20261124_depth_history_v1.py',
                 'conceptgraph/slam/mapnext_20261124_depth_history_v2.py',
                 'conceptgraph/slam/mapnext_20261124_sparse_separation_v1.py']


class MultiplePointMixin(IndependentSparseMixin):
    def settings_record(self):
        record=super().settings_record()
        record['mapnext_sparse_independence'].update(version='S2',minimum_current_real_points=2)
        return record

    def position_predictions(self,predictions,frame):
        output=super().position_predictions(predictions,frame)
        states={s['node_id']:s for s in self.position_layer.tracks}
        rejected=set()
        for decision in self.mapnext_sparse_decisions:
            if not decision['accepted']:
                continue
            state=next((s for s in states.values() if self.lifecycle.resolve(
                self.position_layer.aliases.get(s['node_id'],s['node_id']))==decision['canonical']),None)
            points=len(state.get('_last_world_points',[])) if state else 0
            decision['current_actual_points']=points
            if points<2:
                decision.update(accepted=False,reason='single_current_point_not_enough_for_independence_override')
                rejected.add(decision['canonical'])
        return [p for p in output if not (p['node_id'] in rejected
            and p.get('position_source')=='confirmed_sparse_independent_mask')]


class IndependentSparseMapperV2(MultiplePointMixin,AuditedNativeMapper):
    pass


class CrossCameraIndependentMapper(MultiplePointMixin,CrossCameraHistoryMapper):
    pass
