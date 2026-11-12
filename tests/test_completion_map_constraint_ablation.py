import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from conceptgraph.completion.map_constraint_ablation import Context, audit, plan, polygon_with_holes
from conceptgraph.navigation.lane_constraints import LaneGeometry, LaneTopology


def context_for(lanes, road=None):
    geometries = {}
    for token, polygon, line in lanes:
        line = np.asarray(line, dtype=float)
        geometries[token] = LaneGeometry(token, 'lane', np.asarray(polygon), line,
            float(line[0, 2]), tuple(line[0]), tuple(line[-1]))
    shape = (12, 12)
    topology = LaneTopology(np.full(shape, -1), np.full(shape, np.nan), np.full(shape, np.inf),
        list(geometries), ['lane']*len(geometries), {token: () for token in geometries},
        geometries=geometries, origin_xy=np.zeros(2), resolution=1.)
    road = road if road is not None else box(0, 0, 11, 11)
    mask = np.array([[road.covers(Point(c, r)) for c in range(12)] for r in range(12)])
    return Context(topology, np.zeros(shape, dtype=bool), mask, road, [],
                   unary_union([Polygon(g.polygon) for g in geometries.values()]))


def test_direction_without_connection_can_switch_to_disconnected_crossing_lane():
    context = context_for([
        ('east', [(0, 1), (10, 1), (10, 3), (0, 3)], [[0, 2, 0], [10, 2, 0]]),
        ('north', [(4, 2), (6, 2), (6, 10), (4, 10)], [[5, 2, np.pi/2], [5, 10, np.pi/2]])])
    direction = plan(context, (2, 1), (9, 5), [1, 0], 'lane_direction')
    assert direction['path']
    checked = audit(context, direction['path'], [1, 0])
    assert not checked['full_lane_legal']
    assert checked['independent_lane_audit']['illegal_transitions']
    assert plan(context, (2, 1), (9, 5), [1, 0], 'lane_connection')['path'] is None
    assert plan(context, (2, 1), (9, 5), [1, 0], 'lane_direction_connection')['path'] is None


def test_connection_without_direction_allows_reverse_but_full_constraint_rejects():
    context = context_for([
        ('east', [(0, 1), (10, 1), (10, 3), (0, 3)], [[0, 2, 0], [10, 2, 0]])])
    connection = plan(context, (2, 8), (2, 1), [1, 0], 'lane_connection')
    assert connection['path']
    checked = audit(context, connection['path'], [1, 0])
    assert checked['independent_lane_audit']['reverse_count'] > 0
    assert not checked['full_lane_legal']
    assert plan(context, (2, 8), (2, 1), [1, 0], 'lane_direction_connection')['path'] is None


def test_official_polygon_hole_is_preserved_and_drivable_path_goes_around_it():
    xy = [(0, 0), (11, 0), (11, 11), (0, 11), (4, 3), (6, 3), (6, 7), (4, 7)]
    nodes = {str(i): dict(x=x, y=y) for i, (x, y) in enumerate(xy)}
    road = polygon_with_holes(dict(exterior_node_tokens=['0', '1', '2', '3'],
        holes=[dict(node_tokens=['4', '5', '6', '7'])]), nodes)
    context = context_for([
        ('east', xy[:4], [[0, 5, 0], [11, 5, 0]])], road)
    free = plan(context, (5, 1), (5, 9), [1, 0], 'free_space')
    restricted = plan(context, (5, 1), (5, 9), [1, 0], 'drivable_area')
    assert free['path'] and restricted['path']
    assert not audit(context, free['path'], [1, 0])['continuous_point_drivable']
    assert audit(context, restricted['path'], [1, 0])['continuous_point_drivable']
    assert restricted['cost_grid'] > free['cost_grid']


def test_no_route_does_not_receive_safety_selection_or_arrival_credit():
    result = audit(None, None, None)
    assert not result['path_generated']
    assert result['full_lane_legal'] is None and result['body_pose_legal'] is None
    assert result['actual_arrival'] is None and result['selected_correct_object'] is None
    assert result['stable_correct_lock'] is None
