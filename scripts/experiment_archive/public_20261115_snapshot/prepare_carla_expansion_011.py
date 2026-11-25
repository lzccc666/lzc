"""修正官方异步缓存位姿预检；导航参数继续使用原冻结010，未读取任何导航成绩。"""
import copy
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    runtime=R/'carla_official/expansion_registration_010.json'
    bank=copy.deepcopy(json.loads(runtime.read_text()))
    bank.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='011修正预检：新spawn后必须world.tick并以同步snapshot验证每个actor位置，再作独立真值证明。010预检的75项生成API通过，但42项真值通过全部撤销；未产生导航图像/模型成绩。',
        immutable_navigation_registration=str(runtime),immutable_navigation_registration_sha256=digest(runtime),
        actual_navigation_runtime='source/carla_expansion_episode_010.py',
        actual_navigation_configuration_unchanged_from_010=True,
        superseded_preflight='carla_official/expansion_spawn_preflight_010/report.json',
        superseded_preflight_truth_credit=0,
        positive_preflight_requires_constructive_vehicle_reference=True,
        constructive_reference_scope='只供评价证明可达，不向方法提供路线/障碍GT；不把参考失败定义为不可达。')
    for name in ['prepare_carla_expansion_011.py','preflight_carla_expansion_011.py','run_carla_expansion_batch_011.py','run_official_expansion_019.py']:
        p=R/'source'/name;bank['source_sha256'][str(p.resolve())]=digest(p)
    dump(R/'carla_official/expansion_registration_011.json',bank)
    print('011预检注册；导航配置、任务、种子、输入与运行源码保持010冻结版本。')


if __name__=='__main__':main()
