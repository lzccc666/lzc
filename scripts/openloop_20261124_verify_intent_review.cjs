/* Real DOM save/export/import checks. All synthetic review events stay in memory. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');
const {JSDOM} = require('/tmp/lzc-pilot-ui/node_modules/jsdom');
const root = path.resolve(__dirname, '..');
const folder = path.join(root, 'dataset/qwen/multiview_task_20261124_openloop_candidates_v2');
const report = path.join(root, 'outputs/reports/openloop_review_fix_20261124');
const run = path.join(root, 'outputs/results/qwen/preparation/iterations/openloop_review_fix_20261124_v2');
const html = fs.readFileSync(path.join(folder, 'index.html'), 'utf8');
const clone = x => JSON.parse(JSON.stringify(x));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const packageHashes = () => Object.fromEntries(fs.readdirSync(folder).filter(x => fs.statSync(path.join(folder, x)).isFile()).map(x => [x, sha(path.join(folder, x))]));
const before = packageHashes();
let downloads = [], blobs = [];
function create(stored) {
  return new JSDOM(html, {runScripts: 'dangerously', url: 'https://openloop-review.invalid/', beforeParse(w) {
    w.Blob = class { constructor(parts, options) { this.content = parts.join(''); this.options = options; } };
    w.URL.createObjectURL = blob => {blobs.push(JSON.parse(blob.content)); return 'blob:openloop-fixture';};
    w.URL.revokeObjectURL = () => {};
    w.HTMLAnchorElement.prototype.click = function () { downloads.push(this.download); };
    if (stored) w.localStorage.setItem(stored.key, stored.value);
  }});
}
const dom = create(), w = dom.window, d = w.document, $ = id => d.getElementById(id);
const bundle = JSON.parse($('bundle').textContent);
const protocol = JSON.parse(fs.readFileSync(path.join(folder, 'review_protocol.json'), 'utf8'));
assert.deepStrictEqual(bundle.review_protocol, protocol);
assert.equal(bundle.rows.length, 72);
assert.equal(d.querySelectorAll('#list button').length, 24);
assert.equal(w.IntentReview.exportPayload().review_events.length, 0);
assert($('summary').textContent.includes('预测轨迹末端'));
assert(!$('summary').textContent.includes('在允许区域停稳后完成'));
for (const key of ['requires_actual_execution', 'requires_actual_stop_position', 'requires_actual_speed', 'requires_actual_follow_duration', 'text_approval_is_trajectory_success', 'nonempty_path_is_sufficient', 'recorded_ego_future_is_automatic_task_ground_truth']) assert.strictEqual(protocol[key], false);
assert.equal(protocol.evaluation_mode, 'open_loop');
assert.equal(protocol.evaluation_object, 'predicted_trajectory');
assert.equal(protocol.trajectory_evaluation_status, 'not_evaluated');
assert.equal(protocol.actual_arrival_result, null);

// Approval needs a reviewer and meaning confirmation, with zero actual telemetry inputs.
$('decision').value = 'approve'; $('meaningConfirmed').checked = true; $('save').click();
assert.equal(w.IntentReview.exportPayload().review_events.length, 0);
$('reviewer').value = 'AUTOMATED_DOM_FIXTURE_NOT_HUMAN'; $('meaningConfirmed').checked = false; $('save').click();
assert.equal(w.IntentReview.exportPayload().review_events.length, 0);
$('meaningConfirmed').checked = true; $('save').click();
let payload = w.IntentReview.exportPayload();
assert.equal(payload.review_events.length, 1);
assert.equal(payload.review_events[0].sample_id, 'PO01__p1');
assert.strictEqual(payload.identity_verified, false);
assert.strictEqual(payload.training_export_enabled, false);
assert.equal(payload.review_protocol.trajectory_evaluation_status, 'not_evaluated');
assert.strictEqual(payload.review_protocol.text_approval_is_trajectory_success, false);
assert.equal($('decision').value, 'pending');
$('decision').value = 'needs_revision'; $('notes').value = 'DOM fixture only'; $('save').click();
payload = w.IntentReview.exportPayload();
assert.equal(payload.review_events.length, 2);
$('export').click();
assert.equal(downloads.length, 1);
assert(downloads[0].startsWith('开环轨迹任务意图_审核意见_'));
assert.deepStrictEqual(blobs[0].review_protocol, protocol);
assert.equal(blobs[0].review_events.length, 2);

function clickRow(id) {
  const button = [...d.querySelectorAll('#list button')].find(b => b.textContent.startsWith(id + ' · '));
  assert(button, id); button.click();
}
clickRow('FV02__p1');
assert($('summary').textContent.includes('固定评估时域'));
assert($('summary').textContent.includes('不等待实际跟满'));
assert($('summary').textContent.includes('10秒（不是实际执行等待条件）'));
clickRow('FV01__p1'); assert($('summary').textContent.includes('不等待实际用户结束'));
clickRow('PO06__p1'); assert($('summary').textContent.includes('预测轨迹请求最高速度：5'));
assert(d.body.textContent.includes('只有空间路径而没有时间/速度输出时，速度和停稳状态保持未核验'));

// Import cannot silently upgrade the scope, correctness, identity, or training permissions.
for (const key of ['package_id', 'manifest_sha256', 'intent_schema_sha256', 'review_protocol_sha256']) {
  const bad = clone(payload); bad[key] = 'WRONG'; assert.throws(() => w.IntentReview.importPayload(bad));
}
for (const [key, value] of Object.entries(protocol)) {
  const bad = clone(payload);
  bad.review_protocol[key] = typeof value === 'boolean' ? !value : value === null ? true : 'WRONG';
  assert.throws(() => w.IntentReview.importPayload(bad), key);
}
let bad = clone(payload); bad.review_protocol.actual_speed = 0;
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.training_export_enabled = true;
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.identity_verified = true;
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.review_events[0].reviewed_intent.actual_speed_mps = 0;
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.review_events[0].reviewed_intent.allowed = true;
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.review_events[0].reviewed_intent.termination_request.mode = 'actual_arrived';
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.review_events[0].source_instruction_sha256 = 'WRONG';
assert.throws(() => w.IntentReview.importPayload(bad));
bad = clone(payload); bad.review_events.push(clone(bad.review_events[0]));
assert.throws(() => w.IntentReview.importPayload(bad));
const old = clone(payload);
for (const key of ['review_protocol_sha256', 'review_protocol']) delete old[key];
assert.throws(() => w.IntentReview.importPayload(old));
assert.equal(w.IntentReview.exportPayload().review_events.length, 2);

const restored = create();
assert.equal(restored.window.IntentReview.importPayload(clone(payload)), 2);
assert.equal(restored.window.IntentReview.exportPayload().review_events.length, 2);
const newKey = 'openloopIntentReview:v2:' + bundle.manifest_sha256;
const local = create({key: newKey, value: JSON.stringify(payload)});
assert.equal(local.window.IntentReview.exportPayload().review_events.length, 2);
const isolated = create({key: 'intentReview:' + bundle.manifest_sha256, value: JSON.stringify(payload)});
assert.equal(isolated.window.IntentReview.exportPayload().review_events.length, 0);
assert.deepStrictEqual(packageHashes(), before);
const rows = fs.readFileSync(path.join(folder, 'candidates.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
assert(rows.every(r => r.human_review_status === 'pending' && r.training_eligible === false));
const result = {
  passed: true, input_rows: 72, default_family_rows: 24,
  parking_predicted_terminal_semantics: true, follow_finite_horizon_semantics: true,
  user_end_no_actual_wait: true, spatial_path_does_not_verify_speed: true,
  save_without_actual_telemetry: true, downloaded_protocol_and_events_verified: true,
  import_and_local_restore_verified: true, legacy_storage_isolated: true,
  rejected_wrong_package_hash_schema_protocol: true, protocol_fields_tamper_rejected: Object.keys(protocol).length,
  rejected_actual_and_training_privilege_upgrade: true, old_payload_rejected_not_migrated: true,
  human_annotations_created: 0, synthetic_review_events_in_memory_only: 2,
  dataset_SHA_unchanged: true, trajectory_predictions_or_scores_created: 0,
  real_browser_visual_QA: false, dependency: 'existing /tmp/lzc-pilot-ui/node_modules/jsdom; no installation',
  source_sha256: sha(__filename), package_sha256: before,
};
fs.writeFileSync(path.join(report, 'review_ui_dom_verification.json'), JSON.stringify(result, null, 2) + '\n');
fs.writeFileSync(path.join(run, 'review_ui_dom_verification.json'), JSON.stringify(result, null, 2) + '\n');
fs.copyFileSync(__filename, path.join(run, 'source_snapshot', path.basename(__filename)));
console.log(JSON.stringify(result));
for (const item of [dom, restored, local, isolated]) item.window.close();
