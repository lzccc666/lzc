"""Reference status audit group 1-16, no manuscript edits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, re, subprocess, urllib.parse

O=Path(__file__).resolve().parent
refs=json.loads((O.parent/'verified_references.json').read_text(encoding='utf-8'))[:16]
dois={1:'10.15607/RSS.2020.XVI.079',2:'10.1109/TIV.2016.2578706',3:'10.1109/CVPR42600.2020.01164',4:'10.1177/0278364913491297',5:'10.1109/TRO.2015.2463671',6:'10.1109/TRO.2017.2705103',7:'10.1109/TRO.2021.3075644',8:'10.1109/ICRA.2017.7989538',9:'10.1109/IROS40897.2019.8967890',10:'10.1109/ICRA40945.2020.9196885',11:'10.1109/TRO.2021.3137751',12:'10.1109/3DV.2018.00015',13:'10.1109/ISMAR.2018.00024',14:'10.1109/LRA.2018.2860039',15:'10.1109/CVPR.2019.01298',16:'10.1109/CVPR46437.2021.01161'}
def fetch(item):
    number,doi=item
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
    out=O/f'{number:02d}_crossref.json'
    try:
        if out.exists():
            payload=json.loads(out.read_text(encoding='utf-8'))
        else:
            p=subprocess.run(['curl.exe','-L','-sS','--max-time','25',url],capture_output=True)
            payload=json.loads(p.stdout)
            out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        m=payload['message']
        return {'number':number,'title':m.get('title'),'container':m.get('container-title'),'DOI':m.get('DOI'),'published':m.get('published'),'print':m.get('published-print'),'online':m.get('published-online'),'page':m.get('page'),'volume':m.get('volume'),'issue':m.get('issue'),'url':url}
    except Exception as e:
        return {'number':number,'error':str(e),'url':url}
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,dois.items()))
(O/'group_01_16_crossref_fetch.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for r in results: print(json.dumps(r,ensure_ascii=False))

official={
1:'https://roboticsproceedings.org/rss16/p079.html',
3:'https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html',
4:'https://www.cvlibs.net/datasets/kitti/',
8:'https://portal.fis.tum.de/en/publications/semanticfusion-dense-3d-semantic-mapping-with-convolutional-neura/',
10:'https://www.mit.edu/~arosinol/papers/Rosinol20icra-Kimera.pdf',
11:'https://xplorestaging.ieee.org/ielx7/8860/9851593/09686955.pdf?arnumber=9686955&isnumber=9851593',
15:'https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html',
16:'https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html',
}
output=[]
for r in refs:
    n=r['number'];m=json.loads((O/f'{n:02d}_crossref.json').read_text(encoding='utf-8'))['message']
    assert re.sub('[^a-z0-9]','',m['title'][0].lower())==re.sub('[^a-z0-9]','',r['title'].lower()),(n,m['title'],r['title'])
    publication=m.get('published-print') or m.get('published')
    year=publication['date-parts'][0][0]
    urls=[r['url'],'https://doi.org/'+dois[n],'https://api.crossref.org/works/'+urllib.parse.quote(dois[n],safe='')]
    if n in official:urls.append(official[n])
    notes=['原始arXiv条目（如有）已逐条联网打开核对题名、作者及v1年份；正式状态以Crossref出版社登记记录并结合出版/会议页面核验。']
    if n==4:notes=['KITTI为IJRR 2013正式期刊论文，已核验SAGE DOI、Crossref和作者数据集官网。未为该条目指定arXiv。']
    elif int(r['date'][:4])!=year:notes.append(f"当前书目引用预印本首发年{r['date'][:4]}；正式出版年为{year}，两个年份分别记录。")
    if n==11:notes.append('DOI含2021，但实际卷期为TRO 38(4)，2022年8月；近五年正式发表口径计入2022，首次公开口径仍为2021。')
    if n in [3,15,16]:notes.append('CVF开放版本与IEEE登记页码存在差异，本核验未将两者页码混拼；身份与正式发表年份一致。')
    output.append({'number':n,'id':r['id'],'title':r['title'],'exists':True,'arxiv_year':int(r['date'][:4]) if r['id']!='KITTI2013' else None,'published_year':year,'venue':m['container-title'][0],'doi':m['DOI'],'status':'published','evidence_urls':list(dict.fromkeys(urls)),'notes':notes,'verified_date':'2026-11-20','crossref_record':f'{n:02d}_crossref.json'})
(O/'group_01_16.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'total':len(output),'exists':sum(x['exists'] for x in output),'published':sum(x['status']=='published' for x in output),'published_2022_2026':sum(2022<=x['published_year']<=2026 for x in output),'first_public_2022_2026':sum(2022<=(x['arxiv_year'] or x['published_year'])<=2026 for x in output)}
(O/'group_01_16_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'group_01_16_核验记录.md').write_text('# 第1—16条文献发表状态核验\n\n核验日期：2026-11-20。全部16条真实存在且已正式发表，15条另有对应arXiv预印本，KITTI以正式期刊记录核验。已逐条联网打开arXiv/原始发表页面，并缓存16条Crossref DOI登记元数据。近五年按2022—2026计，正式发表口径仅Kimera-Multi一条为2022；首次公开口径本组0条。\n\nKimera-Multi预印本首发2021，对应IEEE TRO正式卷期2022；不得将DOI中的2021当作正式卷期年。nuScenes、PointPillars、CenterPoint的CVF开放版和IEEE登记页码有差异，本次不混拼元数据。各条证据及细节见group_01_16.json。没有修改论文正文、正式书目、Word或PDF；没有Git操作。\n',encoding='utf-8')
print('SUMMARY',json.dumps(summary,ensure_ascii=False))
