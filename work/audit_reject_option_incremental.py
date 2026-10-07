"""Incremental pure-number and saved-file audit; no model/controller execution.

Run after the existing main audit and calibration-status export are refreshed.
Does not replace the main audit, geometry checks, or browser sensitivity QA.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import calibration
import fusion


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    errors=[]
    def check(ok,description):
        if not ok: errors.append(description)

    # Direct high-precision logistic expression, independent of the core's
    # numerically stable branch. Tiny nonzero tails are checked relatively/
    # against their floating-point ULP, never hidden by a large absolute tol.
    logits=[-1000.,-746.,-745.,-740.,-100.,-40.,-20.,-3.,0.,3.,20.,40.,100.,740.,1000.]
    bank=[dict(id=i,model_logit=z,poses=[[float(k),0.,0.] for k in range(1,9)])
          for i,z in enumerate(logits)]
    guards=[dict(candidate_id=c['id'],eligible=True,reasons=[]) for c in bank]
    output=fusion.fuse_candidates(bank,guards,dict(active=True,angle_deg=0.),
        dict(normalization='sigmoid',decision_rule='score_reject'))
    numeric=[]
    with localcontext() as ctx:
        ctx.prec=100
        for row,z in zip(output['rows'],logits):
            expected_decimal=Decimal(1)/(Decimal(1)+(-Decimal(str(z))).exp())
            expected=float(expected_decimal)
            actual=row['model_score']
            tolerance=max(2*math.ulp(expected),abs(expected)*4e-15)
            passed=math.isfinite(actual) and 0<=actual<=1 and abs(actual-expected)<=tolerance
            check(passed,f'Native sigmoid differs from Decimal reference at z={z}')
            if z in (-100.,-40.):
                check(actual>0.,f'Nonzero sigmoid tail erased at z={z}')
            numeric.append(dict(raw_logit=z,core_native_sigmoid=actual,
                decimal_reference=str(expected_decimal),reference_rounded_float=expected,
                absolute_error=abs(actual-expected),allowed_error=tolerance,passed=passed))
    check(all(a['model_score']<=b['model_score'] for a,b in zip(output['rows'],output['rows'][1:])),
          'Native sigmoid output is not monotone')
    extreme_values={str(z):fusion._stable_sigmoid(z) for z in (-1e308,1e308)}
    check(extreme_values=={'-1e+308':0.,'1e+308':1.},'Finite +/-1e308 sigmoid stability failed')

    data_path=BASE/'data.json'
    baseline_path=BASE/'work/pre_reject_option_data.json'
    data_hash=sha(data_path);baseline_hash=sha(baseline_path)
    data=read(data_path);baseline=read(baseline_path)
    scenes={s['id']:s for s in data['scenes']};before={s['id']:s for s in baseline['scenes']}
    check(set(scenes)==set(before) and len(scenes)==16,'Scene membership changed from pre-reject snapshot')
    candidates_checked=0;guards_checked=0
    for sid,scene in scenes.items():
        old=before.get(sid,{})
        check(scene['candidates']==old.get('candidates'),sid+': candidate bank or guard changed')
        check(scene['token']==old.get('token'),sid+': token changed')
        check([c['id'] for c in scene['candidates']]==list(range(20)),sid+': IDs are not the original 0..19')
        for c in scene['candidates']:
            candidates_checked+=1;guards_checked+=len(c['guards'])
            check(set(c['guards'])=={'current_environment_prediction','current_environment_geometry','offline_pdm'},
                  sid+': guard-mode membership changed')
    check(candidates_checked==320 and guards_checked==960,'Expected 320 candidates and 960 saved guard assessments')
    check(data['meta']['defaults']==fusion.DEFAULTS,'Data/core defaults differ')
    check(fusion.DEFAULTS['normalization']=='sigmoid' and fusion.DEFAULTS['decision_rule']=='score_reject'
          and fusion.DEFAULTS['rejection_threshold']==.75,'Current demonstration defaults differ')

    inventory_path=BASE.parent/'scene-choice-inventory-v3/inventory.json'
    annotation_path=BASE/'annotation_template.json'
    status_path=BASE/'results/calibration_status.json'
    annotations=read(annotation_path)
    expected=calibration.make_template(read(inventory_path),inventory_path)
    validation=calibration.validate(annotations,expected)
    check(validation['status']=='valid','Annotation template source/protocol validation failed')
    check(validation.get('records')==112 and validation.get('blank')==112 and validation.get('binary_labels')==0,
          'Expected 112 blank independent labels, zero binary labels')
    for r in annotations['records']:
        check(all(r[k]=='' for k in calibration.REVIEW_KEYS) and r['independent_of_model_outputs'] is None,
              r['record_id']+': human review field was prefilled')
    stored_status=read(status_path)
    check(stored_status.get('status')=='not_calibrated' and stored_status.get('best_config') is None,
          'A parameter configuration was selected without independent labels')
    check(stored_status.get('validation',{}).get('blank')==112 and
          stored_status.get('validation',{}).get('binary_labels')==0,'Stored calibration label counts are stale')
    for key,path in [('annotations_sha256',annotation_path),('data_sha256',data_path),('inventory_sha256',inventory_path)]:
        check(stored_status.get('inputs',{}).get(key)==sha(path),'Stored calibration source hash is stale: '+key)
    check(data['meta']['calibration']['status']=='not_calibrated' and
          data['meta']['calibration']['independent_label_count']==0 and
          data['meta']['calibration']['defaults_are_demonstration_only'] is True,
          'Data metadata implies completed calibration')

    main_path=BASE/'work/unified_rejection_validation.json'
    main_report=read(main_path)
    check(main_report.get('status')=='PASS','Existing main replay audit did not pass')
    check(main_report.get('data_sha256')==data_hash and main_report.get('fusion_sha256')==sha(BASE/'fusion.py'),
          'Existing main replay audit does not cover current data/core')
    check(main_report.get('fixture_count')==len(data['fixtures']) and
          main_report.get('candidates_logits_anchors_guards_maps_classification_unchanged') is True,
          'Existing main audit coverage declaration is incomplete')
    check(sha(data_path)==data_hash and sha(baseline_path)==baseline_hash,'Source changed while audit was running')
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,
        scope='Incremental numerical/snapshot/annotation audit only. No new model inference, geometry execution, vehicle control, or parameter selection.',
        numeric_sigmoid=dict(decimal_precision_digits=100,cases=numeric,finite_extremes=extreme_values,
            note='Uses relative/ULP-aware tolerances so tiny nonzero sigmoid tails are not accepted as zero.'),
        preserved_snapshot=dict(path=str(baseline_path),sha256=baseline_hash,scenes=len(scenes),
            candidates_checked=candidates_checked,guard_assessments_checked=guards_checked,
            equality_scope='All candidate fields exactly equal, including poses/logits/anchors/offline metrics and all three guard diagnostics.'),
        independent_annotations=dict(template_sha256=sha(annotation_path),validation=validation,
            current_status=stored_status['status'],best_config=stored_status['best_config'],
            status_report_sha256=sha(status_path)),
        reused_main_audit=dict(path=str(main_path),sha256=sha(main_path),status=main_report['status'],
            fixtures=main_report.get('fixture_count'),legacy_fixtures=main_report.get('legacy_fixture_count'),
            grid_rows=main_report.get('grid_count'),random_cases=main_report.get('random_cases')),
        sensitivity_scope='The alpha-rho grid is computed in the browser for the current scene/command; cell/count/click verification belongs to browser QA, not this report.',
        interpretation='Acceptance rate describes an algorithm decision frequency. With zero independent labels it is not intent success, safety, balanced accuracy, or calibrated performance.',
        data_sha256=data_hash,fusion_sha256=sha(BASE/'fusion.py'),audit_sha256=sha(__file__))


def main():
    try:
        report=audit()
    except Exception as exc:
        report=dict(status='FAIL',errors=[type(exc).__name__+': '+str(exc)],audit_sha256=sha(__file__))
    path=BASE/'work/reject_option_validation.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='numeric_sigmoid'},ensure_ascii=True,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
