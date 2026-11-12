"""真实浏览器检查版本化证据；只发送必定拒绝的请求，不生成伪人工记录。"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--review-page', default='causal_development_review_002')
args = parser.parse_args()
out = Path(__file__).resolve().parents[1]
destination = out/'visualizations'/args.run_id
destination.mkdir(exist_ok=False)
base = f'http://127.0.0.1:{args.port}'
def submissions():
    return sorted(str(p.relative_to(out)) for name in ['human_submissions', 'human_natural_submissions']
        for p in (out/'annotation'/name).glob('*.json'))
before = submissions()
checks, requests = [], []
with sync_playwright() as pl:
    browser = pl.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
    for viewport, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
        page = browser.new_page(viewport=dict(width=width, height=height))
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(dict(method=request.method, url=request.url)))
        page.goto(base+'/visualizations/'+args.review_page+'/index.html')
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.includes('one_to_one_depth_quality')")
        assert page.locator('#cases option').count() == 114
        assert page.locator('#method option').count() == 7
        assert page.locator('#color').input_value() == page.locator('#category').input_value() == 'unknown'
        assert page.locator('#label').input_value() == '证据不足'
        page.locator('#submit').click()
        assert '请填写真实审核者' in page.locator('#review-status').inner_text()
        # 避免把测试者写成人工审核。两次请求都故意破坏来源指纹，必须被服务拒绝。
        page.locator('#reviewer').fill('自动无效请求检查_不入库')
        page.locator('#note').fill('仅验证服务器拒绝错版本，不能保存为真实人工记录')
        original = page.evaluate('manifest.evidence_index_sha256')
        page.evaluate("manifest.evidence_index_sha256='wrong-index-unit-check'")
        with page.expect_response(lambda response: response.url.endswith('/api/reviews')) as result:
            page.locator('#submit').click()
        assert result.value.status == 400
        assert '索引指纹' in page.locator('#review-status').inner_text()
        page.evaluate('(value)=>{manifest.evidence_index_sha256=value;current.entry.evidence_file_sha256="wrong-file-unit-check"}', original)
        with page.expect_response(lambda response: response.url.endswith('/api/reviews')) as result:
            page.locator('#submit').click()
        assert result.value.status == 400
        assert '证据内容已改变' in page.locator('#review-status').inner_text()
        page.reload()
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.includes('one_to_one_depth_quality')")
        page.locator('#cases').select_option('50')
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.startsWith(manifest.cases[50].scene+'_'+manifest.cases[50].frame_idx+'_')")
        page.locator('#review-nodes').select_option(index=0)
        page.locator('#label').select_option('错合并')
        page.locator('#method').select_option('one_to_one_depth_robust_motion')
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.endsWith('one_to_one_depth_robust_motion')")
        assert page.locator('#review-nodes').evaluate('(node)=>node.selectedOptions.length') == 0
        assert page.locator('#label').input_value() == '证据不足'
        page.locator('#gt').uncheck()
        page.locator('#mask').uncheck()
        page.locator('#depth').uncheck()
        page.locator('#cases').select_option('113')
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.startsWith(manifest.cases[113].scene+'_'+manifest.cases[113].frame_idx+'_')")
        for node in ['#gt', '#mask', '#depth']:
            page.locator(node).check()
        page.wait_for_function("document.querySelector('#rgb').dataset.rendered?.endsWith('one_to_one_depth_robust_motion')")
        page.wait_for_function("[...document.querySelectorAll('#crops img')].every(image=>image.complete&&image.naturalWidth>0)")
        value = page.evaluate("""() => ({overflow:document.documentElement.scrollWidth>innerWidth,
            error:document.querySelector('#error').textContent,rendered:document.querySelector('#rgb').dataset.rendered,
            color:document.querySelector('#color').value,category:document.querySelector('#category').value,
            source:document.querySelector('#source').textContent,human:document.querySelector('#human').textContent,
            cropImages:document.querySelectorAll('#crops img').length})""")
        assert not value['overflow'] and not value['error'] and not errors
        assert value['color'] == value['category'] == 'unknown' and value['cropImages'] == 2
        assert 'causal_dev_evidence_001' in value['source']
        page.screenshot(path=str(destination/(viewport+'.png')), full_page=True)
        checks.append(dict(viewport=viewport, result=value, browser_errors=errors,
            empty_reviewer_refused_before_request=True, method_switch_clears_previous_identity_claim=True, wrong_index_HTTP=400, wrong_evidence_HTTP=400))
        page.close()
    browser.close()
after = submissions()
assert before == after
assert sum(row['method'] == 'POST' for row in requests) == 4
report = dict(passed=True, port=args.port, checks=checks, real_human_records_before=len(before),
    real_human_records_after=len(after), human_records_written_by_QA=0,
    rejected_test_requests=4, test_requests_explicitly_invalid=True, new_model_calls=0,
    limitations=['没有模拟成功人工提交，标签正确性与真实用户审核仍待完成',
        '浏览器测试只核对读取和拒绝路径；有效提交逻辑以临时合成结构单测核验，不作为人工标注'],
    goal_complete=False)
(destination/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False))
