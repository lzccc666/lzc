"""官方对象/场景标签及道路拓扑仅在评价侧；规划器不参与真值生成。"""
from dataclasses import asdict, replace
import math

import carla
from shapely.geometry import Point, Polygon


def barrier_certificate(road, region, start_xy, barrier):
    remainder = road.difference(barrier)
    components = list(remainder.geoms) if hasattr(remainder, 'geoms') else [remainder]
    start_components = [p for p in components if p.covers(Point(start_xy))]
    separated = bool(start_components and all(p.disjoint(region) for p in start_components))
    return dict(independently_inaccessible=separated,
        proof='声明单车道减去真实阻挡车实体多边形；起点所在所有连通分量均与整个终区不相交。点已不连通则有限车身也不可能合法穿越。',
        component_count=len(components), start_components=len(start_components),
        free_components_wkt=[p.wkt for p in components],
        no_planner_failure_used_as_truth=True,
        scope='仅冻结的有向单车道；不允许越界、穿越车辆或另行道路连接')


def finalize_truth(world, gt, truth, params, road, region, origin, barrier):
    gt['controlled_object_scope']=f"{1+len(gt['other_controlled_actors'])}个预注册目标/干扰车辆，另记录官方静态标签；身份和区域均只供评价。"
    gt['task_kind']=params['task_kind']
    truth=replace(truth, max_speed_mps=.08, max_heading_error_rad=math.radians(15), hold_s=2.)
    if params['task_kind']=='positive':
        static=[]
        for obj in world.get_environment_objects(carla.CityObjectLabel.Car):
            center=[obj.bounding_box.location.x,-obj.bounding_box.location.y]
            if road.covers(Point(center)):
                static.append(dict(id=int(obj.id),name=obj.name,center_xy=center))
        # 任一本车道静态车可能改变最近/关系答案，整回合保留但不授予独立可达任务信用。
        verified=not static
        gt['positive_scope_certificate']=dict(official_static_car_objects_in_declared_lane=static,
            no_additional_static_car_in_declared_lane=verified,
            controlled_target_and_other_actor_categories=[params['target_category'],params['distractor_category']],
            query_scope='目标在声明的本车道内，参照物可在车道外；当前预测候选按同一公开道路先验过滤目标类别')
        if not verified:truth=replace(truth,reachability='unknown')
    if params['task_kind']=='blocked_target':
        proof=barrier_certificate(road, region, origin, barrier)
        gt['independent_barrier_certificate']=proof
        truth=replace(truth, reachability='inaccessible' if proof['independently_inaccessible'] else 'unknown',
            provenance=truth.provenance+'；独立道路连通分量证明，未使用规划失败')
    elif params['task_kind']=='no_match':
        scope=road.buffer(6.)
        objects=[]
        for obj in world.get_environment_objects(carla.CityObjectLabel.Bus):
            vertices=obj.bounding_box.get_world_vertices(carla.Transform())
            polygon=Polygon([(v.x,-v.y) for v in vertices]).convex_hull
            objects.append(dict(id=int(obj.id),name=obj.name,polygon_xy=list(polygon.exterior.coords),
                in_scope=polygon.intersects(scope)))
        actors=[dict(id=str(a.id),type_id=a.type_id) for a in world.get_actors().filter('vehicle.*')]
        known={params['target_blueprint'], params['distractor_blueprint'],'vehicle.toyota.prius'}
        controlled_only=all(a['type_id'] in known for a in actors)
        valid=controlled_only and not any(o['in_scope'] for o in objects)
        gt['no_match_scope_certificate']=dict(
            official_environment_label='CityObjectLabel.Bus',
            static_bus_environment_objects=objects, actors=actors,
            only_registered_non_bus_dynamic_actors=controlled_only,
            scope_polygon_xy=list(scope.exterior.coords),
            verified_no_bus_in_declared_scope=valid,
            limitation='官方静态语义标签及受控actor范围，不扩展为整张地图或所有时刻均无公交车')
        truth=replace(truth,spatial_scope=truth.spatial_scope+'；声明道路走廊向外6m范围',
            provenance=truth.provenance+'；官方静态Bus标签及受控actor清单')
    gt['arrival_truth']=asdict(truth)
    return truth,gt
