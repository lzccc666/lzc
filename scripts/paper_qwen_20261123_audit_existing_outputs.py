"""复核既有Qwen原答；不调用模型、不训练、不读取测试集。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conceptgraph.project_paths import experiment_directory
from conceptgraph.completion.semantic_contract import validate as legacy_validate

OLD = ROOT / 'outputs/project_v2/thesis_completion_v1'
FAIR = OLD / 'main_experiments/fair_three_task_full_validation_001'
REAL_ROOT = OLD / 'public_experiment_completion_20261115'
REAL = REAL_ROOT / 'qwen_adapter_fair_comparison/real_graph_fair_gpu_001'
DATA = ROOT / 'dataset/qwen/real_graph_v003'
STRUCT = OLD / 'semantic_dataset/structured_supervision_002'
REPORT = ROOT / 'outputs/reports/paper_campaign_20261123/qwen'
PILOT = ROOT / 'dataset/qwen/paper_qwen_20261123_pilot_v1'
FIELDS = ['target_categories', 'target_colors', 'anchor_categories', 'anchor_colors',
          'relation', 'prefer_nearest', 'prefer_front', 'excluded_categories',
          'excluded_colors', 'requires_static']
SEED = 17


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, records):
    assert records
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def rate(n, d):
    return n / d if d else None


def footprint(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def files_to_freeze():
    files = [Path(__file__), ROOT / 'AGENTS.md', ROOT / 'MEMORY.md', ROOT / 'PAPER_MEMORY.md',
             ROOT / 'paper/当前版本.json', ROOT / 'paper/当前版_v4/sources/论文正文.md',
             ROOT / 'paper/当前版_v4/硕士论文初稿.docx', ROOT / 'paper/当前版_v4/硕士论文初稿.pdf',
             ROOT / 'conceptgraph/project_paths.py', ROOT / 'conceptgraph/completion/semantic_contract.py',
             ROOT / 'conceptgraph/vlm/schemas.py', FAIR / 'report.json', FAIR / 'release.json',
             FAIR / 'source_snapshot.tar.gz', STRUCT / 'manifest.json', STRUCT / 'validation.jsonl',
             OLD / 'statistics/fair_three_task_full_validation_001/report.json', REAL / 'release.json',
             DATA / 'manifest.json']
    files += list(FAIR.glob('*/raw_records.jsonl')) + list(REAL.glob('*/raw_records.jsonl'))
    files += [DATA / f'{s}.jsonl' for s in ['train', 'validation', 'development_holdout']]
    files += [REAL_ROOT / 'source' / n for n in ['real_graph_contract_003.py',
               'evaluate_real_graph_models_002.py', 'train_real_graph_adapter_004.py']]
    if (ROOT / '.git/index').exists():
        files.append(ROOT / '.git/index')
    return sorted(set(p.absolute() for p in files))


def prepare(run_id):
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('空闲低于20GiB，停止新增产物')
    dest = experiment_directory('qwen', 'validation', run_id, create=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    release = dict(run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                   stage='existing_raw_output_cpu_audit', seed=SEED,
                   budget=dict(cpu_wall_minutes=45, new_storage_mib=256,
                               optional_gpu_inference_minutes=30, gpu_used=False),
                   model_calls=0, training=False, test_data_opened=False,
                   scope=dict(three_task_requests=603, real_validation_inputs=171,
                              real_development_requests=108),
                   fixed_metrics=['contract_correct/all_requests', 'ordinary_target_correct/60',
                                  'covered_target_correct/28', 'valid_negative_selection/12',
                                  'negative_invalid/12', 'controlled_fault_selection/36'],
                   invalid_outputs_retained=True, fallback_credit_separate=True,
                   source_hashes={str(p): sha(p) for p in files_to_freeze()},
                   report_directory=str(REPORT), pilot_directory=str(PILOT),
                   pilot_selection='seed17 sha256; train and validation separate logs; '
                                   '20 covered positives+4 missing-target+4 negatives+12 controlled-fault per split',
                   dataset_roles='all existing development/validation; no fresh unseen-system test claim')
    dump(dest / 'release.json', release)
    snapshot = dest / 'source_snapshot'
    snapshot.mkdir()
    for p in [Path(__file__), ROOT / 'conceptgraph/completion/semantic_contract.py',
              ROOT / 'conceptgraph/vlm/schemas.py', REAL_ROOT / 'source/real_graph_contract_003.py']:
        shutil.copyfile(p, snapshot / p.name)
    print(json.dumps(dict(prepared=str(dest), frozen_files=len(release['source_hashes'])), ensure_ascii=False))


def success(task, pred, row):
    if not pred:
        return False
    if task == 'query_parse':
        return all(pred.get(k) == row['response'][k] for k in FIELDS)
    if task == 'object_attributes':
        return pred.get('object_tag') == row['response']['object_tag']
    return pred.get('selected_node_id') in row['acceptable_node_ids']


def raw_intent(parsed):
    return isinstance(parsed, dict) and parsed.get('selected_node_id') is not None


def quantile(values, q):
    if not values:
        return None
    z = sorted(values)
    x = (len(z) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(z) - 1)
    return z[lo] + (z[hi] - z[lo]) * (x - lo)


def summarize(records):
    c = Counter()
    for r in records:
        c['requests'] += 1
        for k in ['model_correct', 'schema_valid', 'schema_invalid', 'fallback_used',
                  'backend_failed', 'valid_selection', 'valid_refusal', 'raw_nonnull_attempt',
                  'stored_envelope_correct', 'fallback_rescued', 'fallback_spoiled']:
            c[k] += bool(r.get(k))
    d = dict(c)
    d['model_accuracy'] = rate(d.get('model_correct', 0), len(records))
    d['schema_valid_rate'] = rate(d.get('schema_valid', 0), len(records))
    return d


def inventory(rr):
    frames = {(r['scene_id'], r['frame_idx']) for r in rr if 'frame_idx' in r}
    return dict(requests=len(rr), tasks=dict(Counter(r['task'] for r in rr)), unique_frames=len(frames),
                scenes=len({r['scene_id'] for r in rr}), logs=len({r['log_token'] for r in rr}),
                human_reviewed=sum(r.get('manual_reviewed') is True for r in rr),
                ordinary_positive=sum(r.get('ordinary_proposal_only') is True and r.get('official_positive') is True for r in rr),
                controlled_fault=sum(bool(r.get('controlled_fault')) for r in rr),
                negative=sum(r.get('official_positive') is False for r in rr),
                expected_selection=sum(any(v is not None for v in r.get('acceptable_node_ids', [])) for r in rr),
                expected_null=sum(r.get('acceptable_node_ids') == [None] for r in rr),
                candidate_counts=dict(Counter(len(r.get('candidate_ids', [])) for r in rr)),
                query_kinds=dict(Counter(r.get('query_kind', 'not_applicable') for r in rr)))


def run(run_id):
    start = time.monotonic()
    dest = experiment_directory('qwen', 'validation', run_id)
    release = load(dest / 'release.json')
    assert all(sha(p) == h for p, h in release['source_hashes'].items()), 'frozen inputs changed'
    with tarfile.open(FAIR / 'source_snapshot.tar.gz') as t:
        m = next(x for x in t.getmembers() if x.name.endswith('semantic_contract.py'))
        assert t.extractfile(m).read() == (ROOT / 'conceptgraph/completion/semantic_contract.py').read_bytes()
    spec = importlib.util.spec_from_file_location('paper_qwen_real_contract', REAL_ROOT / 'source/real_graph_contract_003.py')
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    three_release = load(FAIR / 'release.json')
    three_report = load(FAIR / 'report.json')
    structure_lookup = {r['sample_id']: r for r in rows(STRUCT / 'validation.jsonl')}
    structure = [structure_lookup[i] for i in three_release['sample_ids']]
    real_data = {s: rows(DATA / f'{s}.jsonl') for s in ['train', 'validation', 'development_holdout']}
    real_lookup = {r['sample_id']: r for r in real_data['development_holdout']}
    validation_ids = {r['sample_id'] for r in real_data['validation']}
    assert len(structure) == 603 and len(real_lookup) == 108 and len(validation_ids) == 171
    assert not set(real_lookup) & validation_ids
    all_scores, three_summary, rank_strata, parse_fields, categories = [], [], [], [], []
    disagreements = []
    for model in three_report['variants']:
        path = FAIR / model / 'raw_records.jsonl'
        rr = rows(path)
        assert [r['sample_id'] for r in rr] == three_release['sample_ids']
        scores = []
        for line, (record, row) in enumerate(zip(rr, structure), 1):
            env = record['envelope']
            v = legacy_validate(row['task'], env['raw_output'], allowed_candidate_ids=row.get('candidate_ids'))
            pred = v.normalized_output or {}
            valid = v.valid and env['used_qwen'] and not env['fallback_used']
            correct = valid and success(row['task'], pred, row)
            stored_correct = success(row['task'], env.get('normalized_output'), row)
            score = dict(cohort='three_task_historical', model=model, sample_id=row['sample_id'],
                         task=row['task'], query_kind=row.get('query_kind', ''), log_token=row['log_token'],
                         model_correct=bool(correct), schema_valid=bool(v.valid), schema_invalid=not v.valid,
                         fallback_used=bool(env['fallback_used']), backend_failed=env['raw_output'] is None,
                         valid_selection=valid and pred.get('selected_node_id') is not None if row['task'] == 'candidate_rerank' else False,
                         valid_refusal=valid and pred.get('selected_node_id') is None if row['task'] == 'candidate_rerank' else False,
                         raw_nonnull_attempt=raw_intent(v.parsed_output),
                         stored_envelope_correct=bool(stored_correct),
                         fallback_rescued=bool(env['fallback_used'] and stored_correct and not correct),
                         fallback_spoiled=bool(env['fallback_used'] and not stored_correct and success(row['task'], pred, row)),
                         normalized_selected=pred.get('selected_node_id'),
                         acceptable_node_ids=json.dumps(row.get('acceptable_node_ids')),
                         source_path=str(path), source_line=line)
            if bool(v.valid) != bool(env['output_schema_valid']):
                disagreements.append(dict(model=model, sample_id=row['sample_id'], kind='schema'))
            scores.append(score)
            if row['task'] == 'query_parse':
                for field in FIELDS:
                    parse_fields.append(dict(model=model, sample_id=row['sample_id'], field=field,
                                             correct=bool(valid and pred.get(field) == row['response'][field]),
                                             truth=json.dumps(row['response'][field]),
                                             prediction=json.dumps(pred.get(field)), schema_valid=v.valid))
        for task in ['query_parse', 'object_attributes', 'candidate_rerank']:
            subset = [r for r in scores if r['task'] == task]
            result = dict(model=model, task=task, **summarize(subset))
            three_summary.append(result)
            expected = three_report['variants'][model]['metrics'][task]
            assert result['model_correct'] == expected['semantic_correct']
            assert result['schema_valid'] == expected['schema_valid']
            assert result['fallback_used'] == expected['fallback']
        for kind in sorted({r.get('query_kind') for r in structure if r['task'] == 'candidate_rerank'}):
            subset = [r for r in scores if r['task'] == 'candidate_rerank' and r['query_kind'] == kind]
            rank_strata.append(dict(model=model, query_kind=kind, **summarize(subset)))
        for category in sorted({r['response']['object_tag'] for r in structure if r['task'] == 'object_attributes'}):
            ids = {r['sample_id'] for r in structure if r['task'] == 'object_attributes' and r['response']['object_tag'] == category}
            categories.append(dict(model=model, category=category, **summarize([r for r in scores if r['sample_id'] in ids])))
        all_scores += scores
    real_models, by_kind, by_log, real_scores, real_failures, costs = [], [], [], [], [], []
    real_release = load(REAL / 'release.json')
    for model in real_release['variants']:
        path = REAL / model / 'raw_records.jsonl'
        rr = rows(path)
        assert [r['sample_id'] for r in rr] == real_release['sample_ids']
        scores = []
        for line, record in enumerate(rr, 1):
            row = real_lookup[record['sample_id']]
            v = contract.validate('candidate_rerank', record['raw_output'], allowed_candidate_ids=row['candidate_ids'])
            pred = v.normalized_output or {}
            correct = v.valid and not record['fallback_used'] and success('candidate_rerank', pred, row)
            assert bool(v.valid) == bool(record['schema_valid'])
            assert bool(correct) == bool(record['semantic_correct'])
            if row['controlled_fault']:
                stratum = 'controlled_fault_authorization'
            elif row['official_positive'] is False:
                stratum = 'true_no_match'
            elif row['candidate_target_coverage']:
                stratum = 'ordinary_positive_covered'
            else:
                stratum = 'ordinary_positive_missing_candidate'
            score = dict(cohort='real_graph_development_108', model=model, task='candidate_rerank',
                         sample_id=row['sample_id'], log_token=row['log_token'], query_kind=row['query_kind'],
                         stratum=stratum, model_correct=bool(correct), schema_valid=v.valid, schema_invalid=not v.valid,
                         fallback_used=record['fallback_used'], backend_failed=not record['backend_success'],
                         valid_selection=bool(v.valid and pred.get('selected_node_id') is not None),
                         valid_refusal=bool(v.valid and pred.get('selected_node_id') is None),
                         raw_nonnull_attempt=raw_intent(v.parsed_output), stored_envelope_correct=bool(correct),
                         fallback_rescued=False, fallback_spoiled=False,
                         normalized_selected=pred.get('selected_node_id'),
                         acceptable_node_ids=json.dumps(row['acceptable_node_ids']),
                         source_path=str(path), source_line=line)
            scores.append(score)
            if not v.valid:
                real_failures.append(dict(model=model, sample_id=row['sample_id'], query_kind=row['query_kind'],
                                          parsed_keys=sorted(v.parsed_output) if isinstance(v.parsed_output, dict) else [],
                                          validation_errors=v.validation_errors, raw_output=record['raw_output'],
                                          source_path=str(path), source_line=line))
        c = {s: summarize([r for r in scores if r['stratum'] == s]) for s in [
             'ordinary_positive_covered', 'ordinary_positive_missing_candidate', 'true_no_match', 'controlled_fault_authorization']}
        total = summarize(scores)
        positive_n = c['ordinary_positive_covered']['requests'] + c['ordinary_positive_missing_candidate']['requests']
        positive_correct = c['ordinary_positive_covered']['model_correct']
        assert positive_n == 60 and c['ordinary_positive_covered']['requests'] == 28
        assert c['true_no_match']['requests'] == 12 and c['controlled_fault_authorization']['requests'] == 36
        negative = c['true_no_match']
        result = dict(model=model, full_contract_correct=total['model_correct'], requests=108,
                      full_contract_accuracy=total['model_accuracy'], ordinary_positive_correct=positive_correct,
                      ordinary_positive_denominator=positive_n, ordinary_positive_accuracy=positive_correct / positive_n,
                      covered_positive_correct=positive_correct, covered_positive_denominator=28,
                      covered_positive_accuracy=positive_correct / 28, candidate_coverage=28 / 60,
                      negative_valid_selection=negative['valid_selection'], negative_denominator=12,
                      negative_valid_selection_rate=negative['valid_selection'] / 12,
                      negative_invalid=negative['schema_invalid'], negative_valid_refusal=negative['valid_refusal'],
                      negative_raw_nonnull_attempt=negative['raw_nonnull_attempt'],
                      missing_candidate_valid_selection=c['ordinary_positive_missing_candidate']['valid_selection'],
                      missing_candidate_denominator=32,
                      fault_valid_selection=c['controlled_fault_authorization']['valid_selection'], fault_denominator=36,
                      fault_valid_refusal=c['controlled_fault_authorization']['valid_refusal'],
                      fault_invalid=c['controlled_fault_authorization']['schema_invalid'],
                      schema_valid=total['schema_valid'], schema_invalid=total['schema_invalid'],
                      fallback_used=total['fallback_used'], backend_failed=total['backend_failed'],
                      comparison_scope='文字化真实候选开发诊断;非自然任务;非导航到达')
        assert result['negative_valid_selection'] + result['negative_invalid'] + result['negative_valid_refusal'] == 12
        real_models.append(result)
        for kind in sorted({r['query_kind'] for r in scores}):
            by_kind.append(dict(model=model, query_kind=kind, **summarize([r for r in scores if r['query_kind'] == kind])))
        for log in sorted({r['log_token'] for r in scores}):
            for stratum in c:
                subset = [r for r in scores if r['log_token'] == log and r['stratum'] == stratum]
                by_log.append(dict(model=model, log_token=log, stratum=stratum, **summarize(subset)))
        warm = [r['latency_s'] for r in rr if not r['cold_first_request']]
        costs.append(dict(model=model, requests=108, existing_inference_record_count=108,
                          new_gpu_calls_this_run=0, input_tokens_min=min(r['input_tokens'] for r in rr),
                          input_tokens_max=max(r['input_tokens'] for r in rr),
                          warm_latency_p50_s=quantile(warm, .5), warm_latency_p95_s=quantile(warm, .95),
                          first_request_latency_s=rr[0]['latency_s'],
                          output_tokens='not_saved;not_estimated', same_gpu_realtime_claim=False))
        real_scores += scores
    assert not disagreements, disagreements
    write_csv(dest / 'paper_qwen_20261123_three_task.csv', three_summary)
    write_csv(dest / 'paper_qwen_20261123_symbolic_ranking_strata.csv', rank_strata)
    write_csv(dest / 'paper_qwen_20261123_crop_categories.csv', categories)
    field_summary = []
    for model in three_report['variants']:
        for field in FIELDS:
            ss = [r for r in parse_fields if r['model'] == model and r['field'] == field]
            field_summary.append(dict(model=model, field=field, correct=sum(r['correct'] for r in ss), requests=len(ss),
                                      accuracy=sum(r['correct'] for r in ss) / len(ss)))
    write_csv(dest / 'paper_qwen_20261123_parse_fields.csv', field_summary)
    write_csv(dest / 'paper_qwen_20261123_real_graph_models.csv', real_models)
    write_csv(dest / 'paper_qwen_20261123_real_graph_by_query_kind.csv', by_kind)
    write_csv(dest / 'paper_qwen_20261123_real_graph_by_log.csv', by_log)
    write_csv(dest / 'paper_qwen_20261123_real_graph_costs.csv', costs)
    write_csv(dest / 'paper_qwen_20261123_request_scores.csv', all_scores + real_scores)
    dump(dest / 'paper_qwen_20261123_format_failures.json', real_failures)
    structure_rank = [r for r in structure if r['task'] == 'candidate_rerank']
    difference = dict(three_task=inventory(structure), real_graph={s: inventory(rr) for s, rr in real_data.items()},
                      validation_development_sample_id_intersection=0,
                      candidate_input_fields_historical=sorted({k for r in structure_rank for n in r['candidate_nodes'] for k in n}),
                      candidate_input_fields_real=sorted({k for r in real_data['validation'] for n in r['candidate_nodes'] for k in n}),
                      historical_ranking='受控符号候选;含构造颜色/预给安全字段;不是真实道路属性GT',
                      real_ranking='保存的真实预测节点;文字输入;普通提议和受控授权分开;无RGB',
                      real_model_payload_keys=['query_text', 'candidate_nodes', 'relation_metadata'],
                      label_and_official_binding_in_model_payload=False, real_rgb_input=False,
                      color_accuracy_available=False, navigation_arrival_evaluated=False,
                      missing_annotations=['自然自由表达与指代身份的人审', '可辨识颜色/细属性真值',
                                           '歧义/未知标签独立审核', '连续任务可评价机会和锁定真值',
                                           '合法接近区域/车身终端姿态'],
                      source_validator_matches_historical_snapshot=True)
    dump(dest / 'paper_qwen_20261123_input_differences.json', difference)
    pilot = make_pilot(real_data, contract)
    changed = [p for p, h in release['source_hashes'].items() if sha(p) != h]
    assert not changed
    result = dict(run_id=run_id, run_directory=str(dest), three_task=three_summary,
                  real_graph_models=real_models, input_differences=difference, training_pilot=pilot,
                  verification=dict(three_task_records_reproduced=3015, real_graph_records_reproduced=756,
                                    schema_or_score_disagreements=0, protection_files=len(release['source_hashes']),
                                    protected_files_unchanged=not changed),
                  training_started=False, new_gpu_inference_calls=0, test_data_opened=False,
                  gpu_not_needed_reason='现有原答已足够定位候选覆盖/格式迁移/过度拒选;不为重复已知证据启动推理',
                  training_recommendation='只建议先做40条受控弱监督平衡小试;自然任务长训练先补标注和候选覆盖',
                  next_pilot_budget=dict(gpu_hours_upper_bound=1, new_storage_mib=256,
                                         training_not_authorized_by_this_report=True),
                  elapsed_s=time.monotonic() - start)
    dump(dest / 'report.json', result)
    dump(REPORT / 'paper_qwen_20261123_result_index.json', dict(run_directory=str(dest),
         report=str(dest / 'report.json'), pilot_directory=str(PILOT), new_gpu_calls=0,
         training_started=False, test_data_opened=False))
    total_storage = footprint(dest) + footprint(REPORT) + footprint(PILOT)
    assert total_storage < 256 * 1024**2
    assert time.monotonic() - start < 45 * 60
    print(json.dumps(dict(run_directory=str(dest), real_models=real_models, pilot=pilot,
                         elapsed_s=time.monotonic() - start, total_new_bytes=total_storage), ensure_ascii=False))


def make_pilot(real_data, contract):
    PILOT.mkdir(parents=True, exist_ok=False)
    manifests = {}
    groups = ['complex_relation', 'multiple_targets', 'negation', 'similar_instances', 'simple_category']
    for split in ['train', 'validation']:
        source = real_data[split]
        order = lambda r: hashlib.sha256(f'paper-qwen-pilot|{SEED}|{r["sample_id"]}'.encode()).hexdigest()
        selected = []
        for kind in groups:
            selected += sorted([r for r in source if r['query_kind'] == kind and any(x is not None for x in r['acceptable_node_ids'])], key=order)[:4]
        selected += sorted([r for r in source if r['ordinary_proposal_only'] and r['official_positive'] is True and not r['candidate_target_coverage']], key=order)[:4]
        selected += sorted([r for r in source if r['official_positive'] is False], key=order)[:4]
        for kind in ['insufficient_evidence', 'stale_information', 'conflicting_information']:
            selected += sorted([r for r in source if r['query_kind'] == kind], key=order)[:4]
        assert len(selected) == 40 and len({r['sample_id'] for r in selected}) == 40
        assert all(contract.validate('candidate_rerank', r['response'], allowed_candidate_ids=r['candidate_ids']).valid for r in selected)
        path = PILOT / f'{split}.jsonl'
        with path.open('x') as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
        manifests[split] = dict(inventory(selected), sample_ids=[r['sample_id'] for r in selected],
                               source=str(DATA / f'{split}.jsonl'), source_sha256=sha(DATA / f'{split}.jsonl'),
                               file_sha256=sha(path), logs=sorted({r['log_token'] for r in selected}),
                               labels_unchanged=True, schema_valid=40)
    assert not set(manifests['train']['logs']) & set(manifests['validation']['logs'])
    manifest = dict(dataset_id='paper_qwen_20261123_pilot_v1', seed=SEED, roles=manifests,
                    original_data_unchanged=True, natural_language_human_annotation=False,
                    purpose='受控弱监督接口/正目标与拒选平衡小试;不是新增自然指令数据集',
                    sampling='每split20个已有目标候选正例+4个候选漏目标+4真无匹配+12受控故障;仅训练/开发小试',
                    evaluation_main_denominator_not_replaced=True, full_validation_requests=171,
                    train_and_validation_logs_disjoint=True, training_started=False,
                    pilot_ready_for_controlled_text_training=True, rgb_inputs=False,
                    token_budget_check='待按冻结原processor审计后开训;不假称已核全部2048token限制',
                    missing_for_natural_claim=['人审自然指令及目标身份', '图像属性真值及真实RGB训练入口'])
    dump(PILOT / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    prepare(args.run_id) if args.prepare else run(args.run_id)


if __name__ == '__main__':
    main()
