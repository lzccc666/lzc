"""复制正在运行回合中已完整写出的前20时刻，提前核验审计器；不改变回合。"""
import datetime
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import digest, dump
from audit_carla_guarded_execution_001 import audit

R = Path(__file__).resolve().parents[1]


def complete_rows(path):
    data = path.read_bytes()
    # 最后未flush完的行不作为输入；终止结果仍须另行核验全日志。
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b'\n')]


def main():
    original = R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule'
    dest = R/'carla_official/guarded_motion_prefix_001/rule'
    dest.mkdir(parents=True, exist_ok=False)
    execution = complete_rows(original/'execution.jsonl')[:20]
    assert len(execution) == 20 and execution[-1]['tick'] == 19
    frames = complete_rows(original/'perception.jsonl')[:10]
    assert len(frames) == 10 and frames[-1]['frame_idx'] == 9
    for name, rows in [('execution.jsonl', execution), ('perception.jsonl', frames)]:
        with (dest/name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    shutil.copyfile(original/'release.json', dest/'release.json')
    sources = []
    for row in execution:
        relative = row['tracking']['current_motion_observation']['raw_file']
        source, copy = original/relative, dest/relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        assert digest(source) == digest(copy)
        sources.append(dict(relative_path=relative, sha256=digest(source)))
    dump(dest/'prefix_provenance.json', dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_episode=str(original), copied_complete_actual_physics_ticks=20,
        copied_actual_perception_frames=10, raw_files=sources,
        original_source_changed=False, source_sha256=digest(__file__),
        new_physical_episodes=0, new_GPU_calls=0, full_episode_acceptance=False,
        scope='实际运行早期已写出20时刻的固定快照，仅提前验证运动审计器，完整回合须终止后全量重查',
        goal_complete=False))
    audit(dest)


if __name__ == '__main__':
    main()
