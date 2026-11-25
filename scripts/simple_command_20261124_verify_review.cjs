/* Review events below are DOM fixtures only and never written into candidates. */
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {JSDOM}=require('/tmp/lzc-pilot-ui/node_modules/jsdom');
const root=path.resolve(__dirname,'..'),folder=path.join(root,'dataset/qwen/multiview_task_20261124_simple_candidates_v3');
const report=path.join(root,'outputs/reports/command_review_simplify_20261124'),run=path.join(root,'outputs/results/qwen/preparation/iterations/simple_command_review_20261124_v3');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const hashes=()=>Object.fromEntries(fs.readdirSync(folder).map(x=>[x,sha(path.join(folder,x))]));
const before=hashes(),html=fs.readFileSync(path.join(folder,'index.html'),'utf8'),clone=x=>JSON.parse(JSON.stringify(x));
let blobs=[],downloads=[];
function create(stored){return new JSDOM(html,{runScripts:'dangerously',url:'https://simple-review.invalid/',beforeParse(w){w.Blob=class{constructor(parts){this.data=parts.join('');}};w.URL.createObjectURL=b=>{blobs.push(JSON.parse(b.data));return'blob:fixture';};w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=function(){downloads.push(this.download);};if(stored)for(const [k,v]of Object.entries(stored))w.localStorage.setItem(k,v);}});}
const dom=create(),w=dom.window,d=w.document,$=id=>d.getElementById(id),bundle=JSON.parse($('bundle').textContent);
assert.equal(bundle.rows.length,72);assert.equal(d.querySelectorAll('#list button').length,24);
assert.equal(w.SimpleCommandReview.exportPayload().review_events.length,0);
for(const id of ['intent','meaningConfirmed','reviewer'])assert.equal(d.getElementById(id),null);
assert.equal(d.querySelectorAll('details').length,0);
assert(d.body.textContent.includes('只看这句话和系统理解是否一致，完成条件由程序处理'));
assert(!/termination_request|termination_mode|review_protocol|source_history\.jsonl|original_instruction/.test(JSON.stringify(bundle)));
assert(!/预测轨迹|实际遥测|停稳后完成|直到我说结束/.test(d.body.textContent));
for(const r of bundle.rows){assert(!/结束|便完成|便算任务完成|叫停任务/.test(r.instruction));assert(r.sample_id.endsWith('__v3'));assert(!Object.hasOwn(r.intent_draft,'termination_request'));}
assert.equal(bundle.rows.filter(r=>r.intent_draft.requested_constraints.duration_s!=null).length,15);
$('decision').value='approve';$('save').click();
let p=w.SimpleCommandReview.exportPayload();assert.equal(p.review_events.length,1);
assert.equal(p.review_events[0].sample_id,'PO01__p1__v3');assert.equal(p.review_events[0].decision,'approve');
assert.strictEqual(p.identity_verified,false);assert.strictEqual(p.training_export_enabled,false);assert.equal(p.trajectory_evaluation_status,'not_evaluated');
assert.equal($('decision').value,'pending');
$('instruction').value='请靠右边停车。';$('decision').value='needs_revision';$('notes').value='DOM FIXTURE ONLY';$('save').click();
p=w.SimpleCommandReview.exportPayload();assert.equal(p.review_events.length,2);
assert.strictEqual(p.review_events[1].instruction_changed,true);assert.strictEqual(p.review_events[1].requires_intent_recheck,true);
$('export').click();assert.equal(downloads.length,1);assert(downloads[0].startsWith('普通导航指令_审核意见_'));
assert.equal(blobs[0].review_events.length,2);assert(!JSON.stringify(blobs[0]).includes('termination_request'));
const click=id=>{const b=[...d.querySelectorAll('#list button')].find(b=>b.textContent.startsWith(id+' · '));assert(b);b.click();};
click('FV01__p1');assert($('instruction').value==='请跟随本车道前车。');assert(!$('summary').textContent.includes('时长'));
click('FV02__p1');assert($('summary').textContent.includes('跟随时长：10秒'));
click('FV08__p1');assert($('summary').textContent.includes('先问清楚'));
const r=w.SimpleCommandReview.validateIntent;
let bad=clone(bundle.rows[0].intent_draft);bad.termination_request={mode:'user_end',duration_s:null};assert.throws(()=>r(bad));
bad=clone(bundle.rows[0].intent_draft);bad.actual_speed_mps=0;assert.throws(()=>r(bad));
for(const k of ['package_id','manifest_sha256','intent_schema_sha256']){bad=clone(p);bad[k]='WRONG';assert.throws(()=>w.SimpleCommandReview.importPayload(bad));}
for(const [k,v]of [['training_export_enabled',true],['identity_verified',true],['trajectory_evaluation_status','passed']]){bad=clone(p);bad[k]=v;assert.throws(()=>w.SimpleCommandReview.importPayload(bad));}
bad=clone(p);bad.review_events[0].intent_draft.termination_request={mode:'user_end',duration_s:null};assert.throws(()=>w.SimpleCommandReview.importPayload(bad));
bad=clone(p);bad.review_events[0].source_instruction_sha256='WRONG';assert.throws(()=>w.SimpleCommandReview.importPayload(bad));
bad=clone(p);bad.review_events[1].requires_intent_recheck=false;assert.throws(()=>w.SimpleCommandReview.importPayload(bad));
bad=clone(p);bad.review_events.push(clone(bad.review_events[0]));assert.throws(()=>w.SimpleCommandReview.importPayload(bad));
const old=clone(p);old.review_protocol={evaluation_mode:'open_loop'};assert.throws(()=>w.SimpleCommandReview.importPayload(old));
const restored=create();assert.equal(restored.window.SimpleCommandReview.importPayload(clone(p)),2);
const key='simpleCommandReview:v3:'+bundle.manifest_sha256,local=create({[key]:JSON.stringify(p)});
assert.equal(local.window.SimpleCommandReview.exportPayload().review_events.length,2);
const oldKeys={'intentReview:legacy':'OLD_V1_VALUE','openloopIntentReview:v2:legacy':'OLD_V2_VALUE'};
const isolated=create(oldKeys);assert.equal(isolated.window.SimpleCommandReview.exportPayload().review_events.length,0);
for(const [k,v]of Object.entries(oldKeys))assert.equal(isolated.window.localStorage.getItem(k),v);
assert.deepStrictEqual(hashes(),before);
const result={passed:true,rows:72,default_rows:24,simplified_instruction_rows:15,
  no_completion_fields_in_review_bundle_or_export:true,no_advanced_JSON_or_completion_confirmation_UI:true,
  plain_follow_no_forced_duration:true,explicit_requested_duration_rows:15,
  explicit_save_without_extra_confirmation:true,edited_text_requires_later_intent_recheck:true,
  downloaded_payload_verified:true,import_and_local_restore_verified:true,
  rejected_old_package_and_extra_completion_fields:true,old_storage_preserved:true,
  rejected_training_identity_or_trajectory_success_upgrade:true,
  synthetic_events_in_memory_only:2,new_human_annotations:0,new_trajectory_scores:0,
  candidate_package_SHA_unchanged:true,real_browser_visual_QA:false,
  source_sha256:sha(__filename),package_SHA:before};
for(const dest of [path.join(report,'review_ui_dom_verification.json'),path.join(run,'review_ui_dom_verification.json')])fs.writeFileSync(dest,JSON.stringify(result,null,2)+'\n');
fs.copyFileSync(__filename,path.join(run,'source_snapshot',path.basename(__filename)));console.log(JSON.stringify(result));
for(const x of [dom,restored,local,isolated])x.window.close();
