"""New iteration wrappers; frozen T3 source and all original numeric gates stay intact."""
import numpy as np

from conceptgraph.slam.mvrepair_20261123_native_representative_v2 import NativeRepresentativeMapper


class AuditedNativeMapper(NativeRepresentativeMapper):
    def update_group(self, frames, capture_times, projections=None):
        result = super().update_group(frames, capture_times, projections)
        observations = []
        for source in self.measurement_sources.values():
            observations.append(dict(kind='dense', category=source['category'], camera=source['camera'],
                center=np.asarray(source['center']).tolist(), points=source['points'], reliable=source['reliable'],
                confidence=source['confidence'], source_key=source['_observation_key'],
                variance=np.asarray(source['variance']).tolist()))
        for source in self.sparse_measurements:
            observations.append(dict(kind='sparse', category=source['category'], camera=source.get('camera'),
                center=np.asarray(source['center']).tolist(), points=source['points'], reliable=False,
                confidence=source['confidence'], source_key=source.get('_observation_key'),
                variance=np.asarray(source['variance']).tolist()))
        self.audit['mapnext_observations'] = observations
        self.audit['mapnext_internal'] = self.internal_snapshot()
        return result
