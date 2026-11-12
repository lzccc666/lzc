"""原稿书目元数据核对：CrossRef/arXiv原始元数据，冲突和失败逐项保留。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import unicodedata

import requests
from bs4 import BeautifulSoup

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, digest, dump


def normalized(text):
    text = unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode().casefold()
    return re.sub(r'[^a-z0-9]','',text)


def parse_reference(line):
    number,text = re.match(r'^\[(\d+)\]\s+(.+)$',line).groups()
    match = re.match(r'(.+?)\.\s+(.+?)\[([^\]]+)\](.*)',text)
    if not match:
        return dict(number=int(number),original=line,status='manual_needed',reason='unparsed reference')
    authors,title,kind,venue = match.groups()
    year = re.search(r'\b(?:19|20)\d{2}\b',venue)
    arxiv = re.search(r'arXiv:(\d{4}\.\d{4,5})',text,re.I)
    return dict(number=int(number),original=line,authors=authors,title=title,kind=kind,
                year=int(year.group()) if year else None,venue_text=venue,
                arxiv_id=arxiv.group(1) if arxiv else None)


def resolve(row, dest):
    if row.get('status'):
        return row
    result = dict(row)
    try:
        if row.get('arxiv_id'):
            url = 'https://arxiv.org/abs/'+row['arxiv_id']
            response = requests.get(url,timeout=25)
            response.raise_for_status()
            soup = BeautifulSoup(response.text,'html.parser')
            def meta(name):
                return [m.get('content','') for m in soup.find_all('meta',attrs={'name':name})]
            paper = dict(title=(meta('citation_title') or [''])[0],
                authors=meta('citation_author'),published=(meta('citation_date') or [''])[0],url=url,
                source='arxiv',arxiv_id=row['arxiv_id'])
            if not paper['title']:
                raise ValueError('arXiv citation metadata absent')
            score = difflib.SequenceMatcher(None,normalized(row['title']),normalized(paper['title'])).ratio()
            first = paper['authors'][0] if paper['authors'] else ''
            years = re.findall(r'\b(?:19|20)\d{2}\b',paper['published'])
            paper['year'] = int(years[0]) if years else None
            candidates = [paper]
        else:
            url = 'https://api.crossref.org/works'
            params = {'query.title':row['title'],'rows':3,
                'select':'DOI,title,author,container-title,published,published-print,published-online,page,volume,issue,URL'}
            response = requests.get(url,params=params,timeout=25,
                                    headers={'User-Agent':'ThesisCitationAudit/1.0 (metadata verification)'})
            response.raise_for_status()
            payload = response.json()
            candidates = []
            for item in payload['message']['items']:
                title = (item.get('title') or [''])[0]
                dates = [v['date-parts'][0][0] for k,v in item.items() if k.startswith('published') and isinstance(v,dict) and v.get('date-parts')]
                candidates.append(dict(item,title=title,year=dates[0] if dates else None,
                    possible_publication_years=sorted(set(dates)),
                    title_similarity=difflib.SequenceMatcher(None,normalized(row['title']),normalized(title)).ratio()))
            paper = max(candidates,key=lambda r:r['title_similarity'])
            score = paper['title_similarity']
            author = (paper.get('author') or [{}])[0]
            first = author.get('family','')
        first_original = row['authors'].split(',')[0].split()[0]
        author_matches = normalized(first_original) in normalized(first)
        year_matches = row['year'] in (paper.get('possible_publication_years') or [paper.get('year')])
        result.update(metadata=paper,candidates=candidates,title_similarity=score,
            first_author_matches=author_matches,year_matches=year_matches,
            status='verified' if score>=.94 and author_matches and year_matches else
                'mismatch' if score>=.85 else 'manual_needed',
            retrieval_url=response.url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
            verification_scope='书目标题、首作者、年份的元数据核验；不是全部正文主张已核实')
    except Exception as exc:
        result.update(status='manual_needed',error=repr(exc),verification_scope='检索失败，不能视为不存在')
    dump(dest/f"ref_{row['number']:03d}.json",result)
    print(json.dumps(dict(number=row['number'],status=result['status']),ensure_ascii=False),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    source = PROJECT/'论文1-3章初稿.md'
    rows = [parse_reference(s) for s in source.read_text().splitlines() if re.match(r'^\[\d+\]',s)]
    dest = OUTPUT/'thesis'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda r:resolve(r,dest/'records'),rows))
    dump(dest/'report.json',dict(original_sha256=digest(source),source_sha256=digest(Path(__file__)),
        counts=dict(Counter(r['status'] for r in results)),references=results,
        original_manuscript_modified=False,all_content_claims_verified=False,
        missing_tools='academic-search MCP unavailable; direct CrossRef/arXiv public metadata used',
        goal_complete=False))


if __name__=='__main__':
    main()
