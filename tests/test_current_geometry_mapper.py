from types import SimpleNamespace
from unittest.mock import patch

from conceptgraph.slam.current_geometry_mapper import CurrentGeometryMapper


def test_updates_current_geometry_without_replacing_identity_or_stale_objects():
    mapper = CurrentGeometryMapper.__new__(CurrentGeometryMapper)
    mapper.last_observations = [dict(mask_idx=[2], pcd={'x': 8}, bbox={'x': 9})]
    mapper.objects = [dict(image_idx=[0, 1], mask_idx=[0, 2], pcd={'x': 1}, bbox={'x': 2}, stable_object_id='old'),
                      dict(image_idx=[0], mask_idx=[2], pcd={'x': 3}, bbox={'x': 4}, stable_object_id='stale')]
    with patch('conceptgraph.slam.evidence_mapper.EvidenceAwareMapper.update', return_value='updated'):
        assert mapper.update(SimpleNamespace(frame_idx=1)) == 'updated'
    assert mapper.objects[0]['bbox']['x'] == 9
    assert mapper.objects[0]['stable_object_id'] == 'old'
    assert mapper.objects[1]['bbox']['x'] == 4
    mapper.objects[0]['bbox']['x'] = 99
    assert mapper.last_observations[0]['bbox']['x'] == 9
