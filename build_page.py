"""Embed saved framework data for a server-free HTML teaching interface."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent


def main():
    data = json.loads((BASE / 'data.json').read_text(encoding='utf-8'))
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    template = (BASE / 'page_template.html').read_text(encoding='utf-8')
    (BASE / 'index.html').write_text(template.replace('__DATA__', encoded), encoding='utf-8')
    print('Framework page written; open index.html directly without a server')


if __name__ == '__main__':
    main()
