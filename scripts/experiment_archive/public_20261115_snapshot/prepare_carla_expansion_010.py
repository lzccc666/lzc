"""009在生成阶段失败、尚无图像/模型成绩；修正无关车生成并增加全量几何预检。"""
import copy
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    bank=copy.deepcopy(json.loads((R/'carla_official/expansion_registration_009.json').read_text()))
    for task in bank['episodes']:
        relation=task['episode'].endswith('_relation22')
        task['spawn_distractor']=relation or task['task_kind']=='blocked_target'
        if relation:
            task.update(distractor_waypoint_index=31,distractor_lateral_right_m=0.,
                query='以卡车为参照，选择本车道内后方的小汽车')
    bank.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='010开发扩展；009在无图像/无Qwen的初始化阶段失败。先检查75个预注册静态布局，再运行相同50任务/3种子。',
        bulk_sensor_root=bank['bulk_sensor_root']+'/runtime010',
        prior_failed_setup='carla_official/expansion_navigation_009/town10_spawn001_ClearNoon_car20/seed17/adapter/full_rule_geometry/failure.json',
        prior_failure_had_perception_frames=0,prior_failure_had_Qwen_calls=0,
        preflight_required=True,preflight_evaluates_performance=False,
        conditions=list(bank['condition_definitions']),
        common_scene_setup_change='简单任务/无匹配不生成无关路外干扰车；复杂任务参照卡车改为车道内31m，目标车22m，指令改为相对后方。所有方法/消融/种子相同，修改依据是官方生成碰撞而非模型成绩。')
    for name in ['prepare_carla_expansion_010.py','carla_expansion_episode_010.py','carla_expansion_truth_010.py','run_carla_expansion_batch_010.py','run_official_expansion_018.py','preflight_carla_expansion_010.py']:
        p=R/'source'/name;bank['source_sha256'][str(p.resolve())]=digest(p)
    dump(R/'carla_official/expansion_registration_010.json',bank)
    print('010注册完成：50任务/3种子；先实际生成75布局并核验独立真值，再允许性能运行。009失败保留。')


if __name__=='__main__':main()
