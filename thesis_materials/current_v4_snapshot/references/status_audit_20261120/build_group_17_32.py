from pathlib import Path
import json,re,html,urllib.parse

O=Path(__file__).resolve().parent
refs=json.loads((O.parent/'verified_references.json').read_text(encoding='utf-8'))
live={r['number']:r for r in json.loads((O/'group_17_32_live_arxiv.json').read_text(encoding='utf-8'))}
# 各正式来源已通过网页检索/原始论文头部、出版方会议记录或Crossref登记核验。
publication={
17:(2023,'2023 IEEE International Conference on Robotics and Automation (ICRA)','10.1109/ICRA48891.2023.10160968','https://par.nsf.gov/servlets/purl/10467175'),
18:(2022,'Computer Vision – ECCV 2022','10.1007/978-3-031-20077-9_1','https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/694_ECCV_2022_paper.php'),
19:(2021,'Proceedings of the 38th International Conference on Machine Learning, PMLR 139',None,'https://proceedings.mlr.press/v139/radford21a.html'),
20:(2022,'International Conference on Learning Representations (ICLR 2022)',None,'https://openreview.net/pdf/9214428c3de4dee047f2ade9ea22c54498592843.pdf'),
21:(2024,'Transactions on Machine Learning Research (TMLR)',None,'https://openreview.net/pdf?id=a68SUt6zFt'),
22:(2022,'2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)','10.1109/CVPR52688.2022.01069','https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html'),
23:(2022,'Computer Vision – ECCV 2022','10.1007/978-3-031-20077-9_21','https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690344.pdf'),
24:(2024,'Computer Vision – ECCV 2024','10.1007/978-3-031-72970-6_3','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf'),
25:(2024,'2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)','10.1109/CVPR52733.2024.01599','https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html'),
26:(2023,'2023 IEEE/CVF International Conference on Computer Vision (ICCV)','10.1109/ICCV51070.2023.00371','https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html'),
27:(2025,'International Conference on Learning Representations (ICLR 2025)',None,'https://openreview.net/pdf/7c41968163abe4e3700e3e3a15174a9d679fcd52.pdf'),
28:(2023,'2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)','10.1109/CVPR52729.2023.00085','https://openaccess.thecvf.com/content/CVPR2023/html/Peng_OpenScene_3D_Scene_Understanding_With_Open_Vocabularies_CVPR_2023_paper.html'),
29:(2023,'Robotics: Science and Systems XIX','10.15607/RSS.2023.XIX.066','https://www.roboticsproceedings.org/rss19/p066.html'),
30:(2023,'2023 IEEE/CVF International Conference on Computer Vision (ICCV)','10.1109/ICCV51070.2023.01807','https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html'),
31:(2023,'Advances in Neural Information Processing Systems 36 (NeurIPS 2023)','10.52202/075280-2989','https://proceedings.neurips.cc/paper_files/paper/2023/hash/d77b5482e38339a8068791d939126be2-Abstract-Conference.html'),
32:(2024,'2024 IEEE International Conference on Robotics and Automation (ICRA)','10.1109/ICRA57147.2024.10610193','https://ieeexplore.ieee.org/abstract/document/10610193/figures'),
}
notes={
19:'CLIP为ICML 2021正式论文，2022—2026口径不计入近五年；PMLR正式记录可核验。',
20:'OpenReview论坛页有浏览器验证；OpenReview论文PDF及作者机构托管PDF明确标注ICLR 2022，arXiv注释亦为ICLR 2022。',
21:'arXiv首发2023，正式TMLR论文PDF页眉为01/2024；OpenReview论坛页有浏览器验证，PDF可核验。',
22:'预印本首发2021，正式CVPR 2022。CVF页面页码10965—10975，Crossref登记10955—10965；如采用正式版须以出版社登记核对整条元数据，勿与预印本信息混搭。本轮不修改主书目。',
24:'预印本首发2023，正式ECCV 2024。正式版作者表含Qing Jiang；引用正式版时须按正式版作者顺序整条核验。',
27:'预印本首发2024，OpenReview正式论文PDF明确标注ICLR 2025，非仅投稿记录。',
29:'ConceptFusion为RSS 2023独立论文，并非用户排除的ConceptGraphs。RSS正式作者顺序将Ganesh Iyer、Soroush Saryazdi列在Tao Chen之前，与当前arXiv顺序不同；应按所引版本保留整套元数据。',
}
norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
out=[]
for r in refs:
    n=r['number']
    if n not in publication:continue
    v=live[n];assert v['http_status']==200 and norm(v['title'])==norm(r['title'])
    year,venue,doi,url=publication[n]
    urls=[r['url'],url]
    if doi:urls.append('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''))
    row={'number':n,'id':r['id'],'title':r['title'],'exists':True,'arxiv_year':int(v['dates'][0][:4]),'published_year':year,'venue':venue,'doi':doi,'status':'published','evidence_urls':urls,'notes':notes.get(n,'已核对arXiv首发记录及正式论文记录；预印本与正式版为同一工作，不重复计数。'),'preprint_first_date':v['dates'][0],'checked_date':'2026-11-20','recent_2022_2026_by_preprint':2022<=int(v['dates'][0][:4])<=2026,'recent_2022_2026_by_publication':2022<=year<=2026}
    cp=O/'group_17_32_cache'/f'{n}_crossref.json'
    if cp.exists():
        m=json.loads(cp.read_text(encoding='utf-8'))['message']
        assert norm(m['title'][0])==norm(r['title'])
        row['published_metadata_cache']=cp.relative_to(O).as_posix()
        row['published_authors']=m.get('author',[])
        row['published_title']=m['title'][0]
        row['published_pages']=m.get('page')
        row['published_date_parts']=m.get('published',{}).get('date-parts')
    else:
        searches=json.loads((O/'group_17_32_crossref_search.json').read_text(encoding='utf-8'))
        candidates=[x for x in searches if x['number']==n and x.get('matches')]
        if candidates:row['crossref_search_match']=candidates[0]['matches'][0]
    out.append(row)
(O/'group_17_32.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
md='# 第17—32条论文身份与发表状态核验\n\n核验日期：2026-11-20。仅更新独立核验文件，未改主书目、正文、Word/PDF或实验；未执行Git操作。\n\n16项均有真实arXiv论文记录，且均已找到正式发表来源。按2022—2026计，预印本首发14/16，正式发表15/16；CLIP为2021，GLIP由2021预印本对应2022正式发表。\n\n| 序号 | 工作 | arXiv首发 | 正式发表 | 出版来源 |\n| --- | --- | --- | --- | --- |\n'
for r in out:
    md+=f"| {r['number']} | {r['title']} | {r['arxiv_year']} | {r['published_year']} | [{r['venue']}]({r['evidence_urls'][1]}) |\n"
md+='\nDOI字段为正式版DOI；null表示本组未登记正式版DOI，不表示论文不存在。12项正式版DOI以Crossref命中或出版社记录核对；ICLR/PMLR/TMLR使用正式开放论文记录。Crossref标题检索曾遇到一次429及一次TLS EOF，改用确切DOI请求后二者成功。OpenReview论坛触发浏览器验证时使用正式论文PDF页眉及作者/会议记录交叉验证。\n\n需要注意：GLIP正式页码在CVF与Crossref登记不同；ConceptFusion及Grounding DINO正式版作者表与预印本有差异，后续改正式引用时要整条替换，不混合版本。ConceptFusion与ConceptGraphs为不同论文。\n'
(O/'group_17_32核验记录.md').write_text(md,encoding='utf-8')
print(json.dumps({'items':len(out),'exists':sum(r['exists'] for r in out),'formally_published':sum(r['status']=='published' for r in out),'recent_preprint_2022_2026':sum(r['recent_2022_2026_by_preprint'] for r in out),'recent_formal_2022_2026':sum(r['recent_2022_2026_by_publication'] for r in out),'formal_dois':sum(bool(r['doi']) for r in out)},ensure_ascii=False))
