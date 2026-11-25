"""保存第33—48项联网核验结果；不改写正式书目或正文。"""
from pathlib import Path
import json

O=Path(__file__).resolve().parent
refs=json.loads((O.parent/'verified_references.json').read_text(encoding='utf-8'))
facts={
33: (2016, '2016 IEEE International Conference on Image Processing (ICIP)', '10.1109/ICIP.2016.7533003', ['https://api.crossref.org/works/10.1109/ICIP.2016.7533003'], 'Crossref出版社注册记录与arXiv题名一致；会议页3464—3468。'),
34: (2017, '2017 IEEE International Conference on Image Processing (ICIP)', '10.1109/ICIP.2017.8296962', ['https://api.crossref.org/works/10.1109/ICIP.2017.8296962','https://ieeexplore.ieee.org/abstract/document/8296962/authors'], 'IEEE页面给出会议2017、Xplore收录日2018；按会议出版年2017记录，页3645—3649。'),
35: (2022, 'Computer Vision – ECCV 2022', '10.1007/978-3-031-20047-2_1', ['https://link.springer.com/chapter/10.1007/978-3-031-20047-2_1'], 'Springer原始页面First Online为2022-10-23，页1—21；预印本首次提交2021，两个年份分别保留。Crossref本次请求429，使用出版社原始页面核实。'),
36: (None, 'ECCV 2020 Workshops（作者说明为extended abstract）', None, [], 'arXiv原始页面明确为ECCV 2020 Workshops的extended abstract。当前准确题名的可确认身份为2020年公开预印本；本次没有查得该准确题名的独立出版社出版记录，published_year留空，不将工作坊呈现说明冒充正式出版证明。相关完整版可能采用不同题名，不混填。'),
37: (2016, 'Computer Vision – ECCV 2016 Workshops', '10.1007/978-3-319-48881-3_2', ['https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2','https://api.crossref.org/works/10.1007/978-3-319-48881-3_2'], 'Springer原始页面First Online为2016-11-03，页17—35；属于正式工作坊论文。'),
38: (2021, 'International Journal of Computer Vision', '10.1007/s11263-020-01375-2', ['https://link.springer.com/article/10.1007/s11263-020-01375-2','https://api.crossref.org/works/10.1007/s11263-020-01375-2'], 'Springer页面与Crossref一致：online published 2020-10-08；正式卷129(2):548—578归属2021。published_year记录卷年2021，online_year另外记录为2020；两者均不在2022—2026。'),
39: (2020, 'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)', None, ['https://openaccess.thecvf.com/content_CVPR_2020/html/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.html'], 'CVF原始书目2020，页3961—3970；作者及题名与arXiv一致。本次CVF直接打开403，检索工具返回同一CVF原始页面完整书目。未核实DOI，不填猜测值。'),
40: (2022, 'Proceedings of Robotics: Science and Systems XVIII', '10.15607/RSS.2022.XVIII.050', ['https://roboticsproceedings.org/rss18/p050.html'], 'RSS正式论文集页面BibTeX明确YEAR=2022及DOI；Crossref本次429，原始论文集核实成功。'),
41: (2024, 'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)', None, ['https://openaccess.thecvf.com/content/CVPR2024/html/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.html','https://openaccess.thecvf.com/content/CVPR2024/papers/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.pdf'], 'CVF正式书目2024，页14183—14193；作者及题名与arXiv一致。本次CVF直接打开403，检索工具返回同一CVF原始页面书目及原始PDF。未核实DOI，不填猜测值。'),
42: (2024, 'Proceedings of Robotics: Science and Systems XX', '10.15607/RSS.2024.XX.077', ['https://roboticsproceedings.org/rss20/p077.html','https://api.crossref.org/works/10.15607/RSS.2024.XX.077'], 'RSS正式页面YEAR=2024，Crossref已核实相同题名、DOI和2024-07-15出版日期。'),
43: (2025, 'IEEE Robotics and Automation Letters', '10.1109/LRA.2025.3547643', ['https://api.crossref.org/works/10.1109/LRA.2025.3547643','https://ieeexplore.ieee.org/document/10909193','https://github.com/BJHYZJ/DovSG'], 'IEEE向Crossref注册的题名、卷期页均已取得：10(5):4252—4259，2025-05。作者arXiv明确RA-L2025 accepted；作者官方项目仓库BibTeX与注册记录一致。Xplore直接页面触发JavaScript机器人检查，未把受限访问误记为论文不存在。'),
44: (2025, '2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)', '10.1109/IROS60139.2025.11245861', ['https://api.crossref.org/works/10.1109/IROS60139.2025.11245861','https://ieeexplore.ieee.org/document/11245861','https://ov-curb.cs.uni-freiburg.de/'], 'IEEE向Crossref注册的原始元数据已取得：题名一致，IROS2025，2025-10-19，页6000—6007。Xplore直接页面触发JavaScript机器人检查，出版状态以取得的注册记录确认。'),
45: (2018, 'Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)', None, ['https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html','https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf'], 'CVF正式页面列2018及页3674—3683；预印本首次提交2017。未核实DOI，不填猜测值。'),
46: (2023, 'Proceedings of The 6th Conference on Robot Learning, PMLR 205', None, ['https://proceedings.mlr.press/v205/shah23b.html'], '会议为CoRL2022；原始PMLR页面和BibTeX均采用正式论文集年份2023，页492—504。预印本2022、会议2022、论文集2023分别保留；不混用。PMLR页面无DOI字段。'),
47: (2024, 'Advances in Neural Information Processing Systems 37 (NeurIPS 2024)', '10.52202/079017-0171', ['https://proceedings.neurips.cc/paper_files/paper/2024/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html','https://api.crossref.org/works/10.52202/079017-0171'], 'NeurIPS正式论文集页面明确Main Conference Track、2024及DOI；Crossref已核实相同题名和页5285—5307。'),
48: (2023, '2023 IEEE International Conference on Robotics and Automation (ICRA)', '10.1109/ICRA48891.2023.10160969', ['https://ieeexplore.ieee.org/abstract/document/10160969/media','https://api.crossref.org/works/10.1109/ICRA48891.2023.10160969'], 'IEEE原始页面及Crossref注册记录一致：2023年ICRA，页10608—10615；预印本首次提交2022。'),
}
rows=[]
for r in refs:
    n=r['number']
    if n not in facts: continue
    year,venue,doi,urls,notes=facts[n]
    row={
        'number':n,'id':r['id'],'title':r['title'],'exists':True,
        'arxiv_year':int(r['date'][:4]),'published_year':year,
        'venue':venue,'doi':doi,
        'status':'published' if year is not None else 'preprint_confirmed; workshop_extended_abstract_claimed_by_authors',
        'evidence_urls':[r['url']]+urls,
        'notes':notes,'verified_on':'2026-11-20',
        'primary_arxiv_page_opened':True,
        'current_citation_year':int(r['date'][:4]),
        'recent_by_arxiv_2022_2026':2022<=int(r['date'][:4])<=2026,
        'recent_by_publication_2022_2026':year is not None and 2022<=year<=2026,
    }
    if n==38: row['published_online_year']=2020
    if n==46: row['conference_year']=2022
    rows.append(row)
assert len(rows)==16 and len({r['number'] for r in rows})==16
(O/'group_33_48.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'all_exist':all(x['exists'] for x in rows),'formal_published':sum(x['published_year'] is not None for x in rows),'recent_2022_2026_published':sum(x['recent_by_publication_2022_2026'] for x in rows),'recent_2022_2026_arxiv':sum(x['recent_by_arxiv_2022_2026'] for x in rows)},ensure_ascii=False))
