// Checks the existing offline form in memory; no synthetic export enters dataset/.
const fs = require('fs');
const assert = require('node:assert/strict');
const {JSDOM} = require('/tmp/lzc-pilot-ui/node_modules/jsdom');
const page = process.argv[2];
const output = process.argv[3];
const html = fs.readFileSync(page, 'utf8');
let exported = null;

function open(stored = null) {
  return new JSDOM(html, {
    runScripts: 'dangerously', url: 'https://offline.local/',
    beforeParse(w) {
      w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
        get: (o, k) => o[k] || (() => {}), set: (o, k, v) => (o[k] = v, true),
      });
      w.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({left: 0, top: 0, width: 1280, height: 720});
      w.Image = class {
        constructor() { this.naturalWidth = 1280; this.naturalHeight = 720; }
        set src(v) { this.onload?.(); }
      };
      w.Blob = class { constructor(parts) { exported = parts.join(''); } };
      w.URL.createObjectURL = () => 'blob:mock';
      w.URL.revokeObjectURL = () => {};
      w.HTMLAnchorElement.prototype.click = () => {};
      if (stored) w.localStorage.setItem(stored.campaign, JSON.stringify(stored));
    },
  });
}

async function main() {
  const dom = open(), w = dom.window;
  const $ = id => w.document.getElementById(id);
  const input = (id, value) => { $(id).value = value; $(id).dispatchEvent(new w.Event('input')); };
  const embedded = JSON.parse($('data').textContent);
  input('author', '自动测试，不是人工标签');
  input('command', '界面功能检查，禁止用于训练或人工标注');
  input('description', '测试数据只存在内存');
  input('visible', 'visible'); input('mapstate', 'correct'); input('nodeid', 'N0001'); input('color', '测试色');
  $('rgb').dispatchEvent(new w.MouseEvent('click', {clientX: 640, clientY: 360}));
  $('clickmode').value = 'anchor';
  $('rgb').dispatchEvent(new w.MouseEvent('click', {clientX: 320, clientY: 180}));
  $('save').click();
  assert.match($('status').textContent, /已保存/);
  $('from').value = '0'; $('to').value = '4'; $('range').click();
  $('export').click();
  const original = JSON.parse(exported);
  assert.deepEqual(original.frames[0].target, [[0.5, 0.5]]);
  assert.deepEqual(original.frames[0].anchor, [[0.25, 0.25]]);
  assert.equal(original.frames[4].visible, 'visible');
  assert.deepEqual(original.frames[4].target, []); assert.deepEqual(original.frames[4].anchor, []);
  assert.equal(original.frames[4].nodeid, ''); assert.equal(original.frames[4].mapstate, 'unreviewed');
  assert.equal(original.frames[4].color, '未知');
  assert.equal(original.campaign, embedded.campaign); assert.equal(original.scene, embedded.scene);
  assert.equal(original.manifest_sha256, embedded.manifest_sha256); assert.equal(original.frame_manifest.length, 40);
  for (const frame of original.frame_manifest) {
    assert.equal(frame.rgb_sha256, embedded.frames[frame.frame_idx].rgb_sha256);
    assert.match(frame.rgb_sha256, /^[0-9a-f]{64}$/);
    assert.equal('image' in frame, false); assert.equal('nodes' in frame, false); assert.equal('edges' in frame, false);
  }
  const stored = JSON.parse(w.localStorage.getItem(original.campaign));
  dom.window.close();

  const restored = open(stored), rw = restored.window, $r = id => rw.document.getElementById(id);
  assert.equal($r('command').value, original.command);
  $r('export').click();
  assert.deepEqual(JSON.parse(exported).frames[0].target, [[0.5, 0.5]]);
  restored.window.close();

  const imported = open(), iw = imported.window, $i = id => iw.document.getElementById(id);
  async function importValue(value) {
    Object.defineProperty($i('import'), 'files', {configurable: true, value: [{text: async () => JSON.stringify(value)}]});
    await $i('import').onchange();
  }
  await importValue(original); assert.equal($i('command').value, original.command);
  $i('export').click(); assert.deepEqual(JSON.parse(exported).frames[0].anchor, [[0.25, 0.25]]);
  const checks = [];
  for (const [name, mutate] of [
    ['different_campaign', a => a.campaign = 'other_batch'],
    ['different_manifest', a => a.manifest_sha256 = '0'.repeat(64)],
    ['out_of_range_frame', a => a.frames['40'] = {...a.frames['0'], frame_idx: 40}],
    ['invalid_click_coordinates', a => a.frames['0'].target = [[1.2, 0.5]]],
  ]) {
    const a = JSON.parse(JSON.stringify(original)); mutate(a); await importValue(a);
    assert.match($i('status').textContent, /导入失败/);
    $i('export').click(); assert.equal(JSON.parse(exported).campaign, original.campaign);
    checks.push(name);
  }
  imported.window.close();
  const result = {
    status: 'passed', target_and_anchor_clicks: true, explicit_save: true,
    range_did_not_copy_points_node_ids_graph_judgment_or_color: true,
    export_includes_campaign_manifest_scene_and_40_source_rgb_hashes: true,
    local_storage_restore: true, json_import_restore: true, rejected_imports: checks,
    source_package_modified: false, synthetic_export_written_to_dataset: false,
    human_annotations_created: 0, rendering_verified: false,
    limitation: 'jsdom DOM logic; canvas/image rendering stubbed. Actual 40 JPEGs decoded separately with Pillow; no real browser visual QA.',
  };
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify(result));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
