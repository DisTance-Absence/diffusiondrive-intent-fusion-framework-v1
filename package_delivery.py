"""Package the local replay, code and verification without caches or screenshots."""
import hashlib
import json
from pathlib import Path
import zipfile

BASE=Path(__file__).resolve().parent
DEST=BASE/'外部方向意图融合框架_v1.zip'


def main():
    names=['index.html','app.js','page_template.html','data.json','fusion.py','safety.py',
           'runtime_adapter.py','build_framework.py','build_page.py','test_safety.py',
           'verify_page.mjs','protocol.json','source_provenance.json','框架说明.txt',
           'package_delivery.py','work/audit_framework.py','work/audit_framework.json',
           'work/safety_validation.json','work/page_validation.json',
           'scene_classification.json','classify_scenes.py','场景分类说明.txt',
           'work/classification_validation.json','work/classification_evidence_audit.json',
           'work/visual_scene_classification.json','work/dynamic_guard_audit.json',
           'work/dynamic_guard_scoring_parity.json',
           'S13_S14冻结障碍检查说明.txt','work/s13_s14_guard_diagnosis.json',
           'work/s13_s14_original_vs_guard.json','work/s13_s14_actor_motion_sources.json',
           'test_fusion_reject.py','calibration.py','test_calibration.py','标定说明.txt',
           'annotation_template.json','annotation.html','annotation_page_template.html','build_annotation_page.py',
           'work/audit_unified_rejection.py','work/unified_rejection_validation.json',
           'work/verify_annotation.mjs','work/annotation_page_validation.json',
           'work/audit_reject_option_incremental.py','work/reject_option_validation.json']
    names.extend(str(p.relative_to(BASE)).replace('\\','/') for p in sorted((BASE/'results').glob('*.json')))
    names.extend(str(p.relative_to(BASE)).replace('\\','/') for p in sorted((BASE/'figures/classification').glob('*.png')))
    names.extend(str(p.relative_to(BASE)).replace('\\','/') for p in sorted((BASE/'figures/diagnostics').glob('*.png')))
    entries=[]
    for name in names:
        path=BASE/name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    manifest=dict(format='frozen_local_replay_and_source',entries=entries,
                  note='Open index.html locally, or run fusion.py with included data.json. Rebuilding from original sources and independent source audits require the original workspace; no dependencies or model weights bundled.')
    with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(BASE/entry['path'],entry['path'])
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for entry in entries:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    print(json.dumps(dict(path=str(DEST),entries=len(entries)+1,bytes=DEST.stat().st_size,
                          sha256=hashlib.sha256(DEST.read_bytes()).hexdigest()),ensure_ascii=False))


if __name__=='__main__':
    main()
