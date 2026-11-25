"""已有真实感知候选上的CPU开发验证；不访问封存查询或标签。"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from public_query_intake_055 import CATEGORIES, choose, parse, strong_rule
from real_graph_cooperation_002 import categories as old_categories
from real_graph_cooperation_002 import choose as old_choose
from real_graph_cooperation_002 import complete_geometry_rule as old_rule

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'qwen_adapter_fair_comparison/real_graph_dataset_003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    dest = ROOT / 'method_comparison/public_intake_development_055'
    dest.mkdir(exist_ok=False)
    paths = [DATA / (split + '.jsonl') for split in ['train', 'validation', 'development_holdout']]
    paths += [Path(__file__), ROOT / 'source/public_query_intake_055.py',
              ROOT / 'source/test_public_query_intake_055.py',
              ROOT / 'source/real_graph_contract_003.py',
              ROOT / 'source/real_graph_cooperation_002.py']
    hashes = {str(p): sha(p) for p in paths}
    write(dest / 'release.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        inputs_and_sources_sha256=hashes, original_query_comparison_relation_radius_m=12,
        public_grammar_probe_relation_radius_m=15, seed=17,
        public_probe_selection='每个simple_category开发样本构造4种声明句式；排除/参照类取固定类别表的下一类，不读取标签或方法成败',
        public_probe_truth='仅有预先声明的句法字段；未把原任务答案转为改写任务答案，不计算改写任务正确率',
        method='原强规则与新强规则，使用相同原始候选和原任务标签',
        old_results_unchanged=True, model_inference=False, new_test_opened=False, goal_complete=False))
    counts = defaultdict(Counter)
    probes = Counter()
    original_records = []
    public_records = []
    for split in ['train', 'validation', 'development_holdout']:
        for row in map(json.loads, (DATA / (split + '.jsonl')).open()):
            text, nodes = row['query_text'], row['candidate_nodes']
            parsed = parse(text)
            old = old_rule(text, nodes, row['relation_metadata'])
            new = strong_rule(text, nodes, relation_radius_m=12)
            explicit = parsed.explicit_authorization
            answer_key = 'authorized_node_id' if explicit else 'proposed_node_id'
            old_answer, new_answer = old[answer_key], new[answer_key]
            counters = dict(queries=1, supported=parsed.supported,
                old_correct=old_answer in row['acceptable_node_ids'],
                new_correct=new_answer in row['acceptable_node_ids'],
                changed_answer=old_answer != new_answer)
            counts[split].update(counters)
            counts[split + '/' + row['query_kind']].update(counters)
            original_records.append(dict(sample_id=row['sample_id'], split=split,
                query_kind=row['query_kind'], parsed=asdict(parsed),
                old_answer=old_answer, new_answer=new_answer,
                acceptable_node_ids=row['acceptable_node_ids'], **counters))
            if row['query_kind'] != 'simple_category':
                continue
            target = parsed.targets[0]
            categories = list(CATEGORIES)
            other = categories[(categories.index(target) + 1) % len(categories)]
            texts = [
                '将车辆驶向观察范围内最近的静止' + CATEGORIES[target] + '旁的合法区域',
                '观察范围内若有静止' + CATEGORIES[target] + '，请安全接近',
                '本次接近观察范围内静止' + CATEGORIES[target] + '，而不是' + CATEGORIES[other],
                '目标是在' + CATEGORIES[other] + '右侧的静止' + CATEGORIES[target] + '，请接近它',
            ]
            for family, public_text in enumerate(texts):
                parsed_probe = parse(public_text)
                correct = (parsed_probe.supported and parsed_probe.targets == (target,)
                    and parsed_probe.requires_static
                    and parsed_probe.excluded == ((other,) if family == 2 else ())
                    and parsed_probe.anchors == ((other,) if family == 3 else ())
                    and parsed_probe.relation == ('right_of' if family == 3 else None)
                    and parsed_probe.prefer_nearest == (family == 0))
                prior_route = old_choose(public_text, nodes, row['relation_metadata'], None)
                current_route = choose(public_text, nodes)
                wrong_classes = sorted(set(old_categories(public_text)) - {target})
                probes.update(probes=1, exact_declared_semantics=correct,
                              legacy_target_category_contamination=bool(wrong_classes),
                              relation_probes=family == 3,
                              legacy_relation_as_clear_rule=family == 3 and prior_route['source'] == 'clear_rule')
                public_records.append(dict(source_sample_id=row['sample_id'], split=split,
                    grammar_family=family, query_text=public_text,
                    parsed=asdict(parsed_probe), exact_declared_semantics=correct,
                    legacy_extra_target_categories=wrong_classes,
                    legacy_route=prior_route, new_route=current_route,
                    model_accuracy_evaluated=False, actual_arrival_credit=False))
    for name, rows in [('original_query_checks.jsonl', original_records),
                       ('public_grammar_probes.jsonl', public_records)]:
        with (dest / name).open('x') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    preserved = all(sha(Path(p)) == h for p, h in hashes.items())
    passed = (preserved and all(c['queries'] == c['supported'] for c in counts.values())
              and probes['probes'] == probes['exact_declared_semantics'])
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=passed,
        original_queries=len(original_records), unique_frames=len({r['source_sample_id'].rsplit('_', 1)[0]
            for r in public_records}), original_query_counts={k: dict(v) for k, v in counts.items()},
        public_grammar_probes=dict(probes), original_inputs_rehashed_unchanged=preserved,
        new_GPU_calls=0, new_CARLA_calls=0, new_test_opened=False,
        files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        limitations=['只覆盖声明的受限中文句式，不是任意自然语言解析',
                    '旧新强规则比较为已有开发数据，包含训练与验证；独立开发留出另列，仅2日志',
                    '公开句式探针改变了静止要求，不沿用原任务答案，不报告其目标正确率',
                    '没有真实模型推理，也没有新的锁定或到达证据',
                    '候选版本用于新实验准备，未改写历史实验或部署到CARLA'], goal_complete=False)
    write(dest / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
