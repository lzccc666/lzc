"""新回合流式gzip；旧回合文件保持原样。每行flush，读取同时支持原JSONL。"""
import gzip
import io
import json


class LosslessJsonl:
    def __init__(self, path):
        self.raw = path.open('xb')
        self.zipped = gzip.GzipFile(fileobj=self.raw, mode='wb', filename='',
                                   compresslevel=6, mtime=0)
        self.text = io.TextIOWrapper(self.zipped, encoding='utf-8', newline='\n')

    def __enter__(self):
        return self

    def write(self, text):
        return self.text.write(text)

    def flush(self):
        self.text.flush()
        self.raw.flush()

    def __exit__(self, *args):
        self.text.close()
        self.raw.close()


def read_rows(episode, name):
    plain = episode/name
    compressed = episode/(name+'.gz')
    path = plain if plain.exists() else compressed
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def actual_log(episode, name):
    return episode/name if (episode/name).exists() else episode/(name+'.gz')
