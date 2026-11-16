"""只改变019诊断的日志表示和资源保护；真实输入、mapper及门限保持相同。"""
import json
from pathlib import Path

from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]


def main():
    source = R/'source/replay_carla_history_module_019.py'
    text = source.read_text()
    changes = [
        ('history_module_recorded_diagnostic_019', 'history_module_recorded_diagnostic_021'),
        ("                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\\n')",
         "                    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)+'\\n'\n"
         "                    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())\n"
         "                    if used+len(encoded.encode('utf-8'))+128*1024 > 32*1024**2:\n"
         "                        raise RuntimeError('32MiB诊断日志保护停止；保留已写前缀')\n"
         "                    stream.write(encoded)"),
        ('motion=node.motion_state, metadata=node.metadata)',
         "motion=node.motion_state, metadata=dict(\n"
         "                                               num_detections=node.metadata['num_detections'],\n"
         "                                               object_identity=node.metadata['object_identity'],\n"
         "                                               timestamp_s=node.metadata['timestamp_s'],\n"
         "                                               first_seen_frame=node.metadata['first_seen_frame'],\n"
         "                                               last_seen_frame=node.metadata['last_seen_frame'],\n"
         "                                               depth_evidence_count=len(node.metadata['depth_evidence']),\n"
         "                                               latest_depth_evidence=(node.metadata['depth_evidence'][-1]\n"
         "                                                   if node.metadata['depth_evidence'] else None)))")]
    for before, after in changes:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    dest = R/'source/replay_carla_history_module_021.py'
    with dest.open('x') as stream:
        stream.write(text)
    receipt = dict(original_source=str(source), original_sha256=digest(source),
                   derived_source=str(dest), derived_sha256=digest(dest),
                   replacements=[dict(before=a, after=b) for a, b in changes],
                   mapper_inference_changes=False,
                   evidence_decisions_still_consume_original_full_metadata=True,
                   original_interrupted_results_preserved=True, goal_complete=False)
    with (R/'json/history_compact_replay_derivation_020.json').open('x') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
