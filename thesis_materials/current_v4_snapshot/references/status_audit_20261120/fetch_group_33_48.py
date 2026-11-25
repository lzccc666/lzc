from pathlib import Path
import concurrent.futures
import json
import urllib.request

out = Path(__file__).resolve().parent / 'primary_cache_33_48'
out.mkdir(parents=True, exist_ok=True)
dois = {
33: '10.1109/ICIP.2016.7533003',
34: '10.1109/ICIP.2017.8296962',
35: '10.1007/978-3-031-20047-2_1',
37: '10.1007/978-3-319-48881-3_2',
38: '10.1007/s11263-020-01375-2',
40: '10.15607/RSS.2022.XVIII.050',
42: '10.15607/RSS.2024.XX.077',
43: '10.1109/LRA.2025.3547643',
44: '10.1109/IROS60139.2025.11245861',
47: '10.52202/079017-0171',
48: '10.1109/ICRA48891.2023.10160969',
}
def fetch(item):
    number, doi = item
    url = 'https://api.crossref.org/works/' + doi
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            data = json.load(response)
        (out / f'{number:02d}_crossref.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        m=data['message']
        return {'number': number, 'doi': doi, 'url': url, 'title': m.get('title'), 'venue':m.get('container-title'), 'published':m.get('published'), 'print':m.get('published-print'), 'online':m.get('published-online'), 'page':m.get('page'), 'volume':m.get('volume'), 'issue':m.get('issue'), 'type':m.get('type')}
    except Exception as e:
        return {'number':number, 'url':url, 'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    result = list(pool.map(fetch, dois.items()))
(out/'fetch_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
