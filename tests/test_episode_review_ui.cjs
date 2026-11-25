// Real form logic in a DOM; canvas/image rendering is stubbed, not a visual browser test.
const fs=require('fs'),assert=require('node:assert/strict');
const {JSDOM}=require(process.env.LZC_JSDOM_MODULE||'jsdom');
const page=process.argv[2];let exported,stored;
const dom=new JSDOM(fs.readFileSync(page,'utf8'),{runScripts:'dangerously',url:'https://offline.local/',beforeParse(w){
 w.HTMLCanvasElement.prototype.getContext=()=>new Proxy({},{get:(o,k)=>o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
 w.HTMLCanvasElement.prototype.getBoundingClientRect=()=>({left:0,top:0,width:1280,height:720});
 w.Image=class{constructor(){this.naturalWidth=1280;this.naturalHeight=720}set src(v){this.onload?.()}};
 w.Blob=class{constructor(parts){exported=parts.join('')}};
 w.URL.createObjectURL=()=> 'blob:mock';w.URL.revokeObjectURL=()=>{};
 w.HTMLAnchorElement.prototype.click=()=>{};
}});
const w=dom.window,$=id=>w.document.getElementById(id),input=(id,v)=>{$(id).value=v;$(id).dispatchEvent(new w.Event('input'))};
assert.equal($('timeline').max,'39');assert.match($('frameinfo').textContent,/19|0\.00/);
input('author','界面自动测试（不作为人工标签）');input('command','这是自动测试输入，禁止作为人工样本');input('visible','visible');input('note','自动测试，不作为人工数据');
$('rgb').dispatchEvent(new w.MouseEvent('click',{clientX:640,clientY:360}));$('save').click();
assert.match($('status').textContent,/已保存/);
$('from').value='0';$('to').value='4';$('range').click();$('export').click();
let a=JSON.parse(exported);assert.equal(a.frames[0].target.length,1);assert.equal(a.frames[4].visible,'visible');
assert.equal(a.frames[4].target.length,0);assert.equal(a.frames[4].mapstate,'unreviewed');assert.equal(a.frames[4].nodeid,'');
assert.equal(a.frame_manifest.length,40);assert.equal(a.command,'这是自动测试输入，禁止作为人工样本');
$('timeline').value='39';$('timeline').dispatchEvent(new w.Event('input'));assert.match($('frameinfo').textContent,/帧39\/39/);
$('showgraph').checked=true;$('showgraph').dispatchEvent(new w.Event('change'));
stored=w.localStorage.getItem(a.campaign);assert.ok(stored);
console.log(JSON.stringify({status:'passed',frames:40,task_saved:true,keyframe_click_saved:true,range_visibility_saved:true,range_did_not_copy_points_or_map_labels:true,graph_toggle_worked:true,export_parsed:true,local_storage_saved:true,scope:'DOM functional check; rendering stubbed; no human annotation generated'}));
w.close();
