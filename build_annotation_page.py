"""Build an independent annotation page; never embeds candidate predictions."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
def main():
    document=json.loads((BASE/'annotation_template.json').read_text(encoding='utf-8'))
    encoded=json.dumps(document,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    page=(BASE/'annotation_page_template.html').read_text(encoding='utf-8').replace('__ANNOTATION_DATA__',encoded)
    (BASE/'annotation.html').write_text(page,encoding='utf-8')
    print('annotation.html built; all labels remain blank')
if __name__=='__main__':main()
