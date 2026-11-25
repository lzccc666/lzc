import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box, mapping

from conceptgraph.completion.reference_region_audit import audit_region
from conceptgraph.completion.reference_regions import (
    RegionPolicy, check_region, conservative_envelope, object_polygon,
    reference_prior_from_json,
)
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon


def test_nominal_body_free_but_orientation_uncertainty_hits_obstacle():
    road = box(-20, -20, 20, 20)
    obstacle = box(2., 1.05, 2.3, 1.2)
    assert not footprint_polygon(VehicleState(0., 0., 0.)).intersects(obstacle)
    region, reason = check_region(0., 0., 0., box(-6., -1., -5., 1.), road, road, obstacle)
    assert region is None and reason == 'envelope_intersects_current_official_box'


def test_official_road_hole_cannot_be_filled_by_region_certificate():
    road = Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)],
                   holes=[[(2, -.2), (3, -.2), (3, .2), (2, .2)]])
    region, reason = check_region(0., 0., 0., box(-6., -1., -5., 1.), road, box(-20, -20, 20, 20), Polygon())
    assert region is None and reason == 'envelope_outside_official_drivable_area'


def test_full_3d_box_roll_is_projected_and_invalid_box_is_not_omitted():
    record = dict(center=[0., 0., 0.], size_wlh=[2., 4., 6.],
                  rotation_wxyz=[math.cos(math.pi/4), math.sin(math.pi/4), 0., 0.])
    assert np.allclose(object_polygon(record).bounds, [-2, -3, 2, 3])
    with pytest.raises(ValueError, match='cannot omit'):
        object_polygon(dict(record, size_wlh=[0., 4., 6.]))


def test_independent_audit_accepts_region_but_never_promotes_reachability():
    road = box(-20, -20, 20, 20)
    target = box(-6., -1., -5., 1.)
    region, reason = check_region(0., 0., 0., target, road, road, target)
    assert not reason and region['reachability'] == 'unknown'
    from dataclasses import asdict
    result = audit_region(region, target, dict(drivable_geojson=mapping(road),
        lane_union_geojson=mapping(road), objects=[dict(polygon_geojson=mapping(target))]),
        asdict(RegionPolicy()), asdict(VehicleSpec()))
    assert result['passed'] and result['checked_pose_samples'] == 99
    region['reachability'] = 'reachable'
    assert not audit_region(region, target, dict(drivable_geojson=mapping(road),
        lane_union_geojson=mapping(road), objects=[dict(polygon_geojson=mapping(target))]),
        asdict(RegionPolicy()), asdict(VehicleSpec()))['passed']


def test_conservative_envelope_covers_extreme_rotations_and_translations():
    envelope = conservative_envelope(0., 0., 0.)
    for x in [-.2, .2]:
        for y in [-.1, .1]:
            for yaw in [-math.radians(5), math.radians(5)]:
                assert envelope.covers(footprint_polygon(VehicleState(x, y, yaw)))
    assert envelope.covers(Point(0., 0.))


def test_null_official_reference_is_missing_geometry_and_does_not_fill_holes():
    xy = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-1, -1), (1, -1), (1, 1), (-1, 1)]
    data = dict(node=[dict(token=str(i), x=x, y=y) for i, (x, y) in enumerate(xy)],
        polygon=[dict(token='p', exterior_node_tokens=list('0123'), holes=[dict(node_tokens=list('4567'))])],
        drivable_area=[dict(token='road', polygon_tokens=['p', None, 'missing'])])
    prior = reference_prior_from_json(data)
    assert not prior['drivable'].covers(Point(0., 0.))
    assert prior['drivable'].covers(Point(5., 5.))
    assert len(prior['diagnostics']['missing_drivable_polygon_references']) == 2
    assert prior['diagnostics']['missing_geometry_not_filled']
