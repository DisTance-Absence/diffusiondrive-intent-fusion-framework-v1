"""SYNTHETIC UNIT TESTS ONLY. These labels are not research annotations."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
import calibration as cal


def synthetic_document():
    scenes=[]
    for i in range(3):
        sid=f'UNIT_SYNTHETIC_{i}'
        scenes.append(dict(scene_id=sid,token=sid,summary=dict(trip=f'UNIT_TRIP_{i}'),
            metadata=dict(ego_speed_mps=2.,split='synthetic'),road_context={},actor_context={},map_paths=[]))
    doc=cal.make_template(dict(scenes=scenes))
    doc['synthetic']=True
    for r in doc['records']:
        r.update(label='accept' if r['command']['angle_deg']<=0 else 'reject',reason='UNIT TEST ONLY',
            goal='UNIT TEST ONLY',annotator_id='SYNTHETIC_TEST_NOT_PERSON',reviewed_at='2000-01-01T00:00:00+00:00',
            evidence_source='GENERATED UNIT FIXTURE NOT REAL EVIDENCE',label_source='human_review',
            independent_of_model_outputs=True,split=('train','dev','test')[int(r['scene_id'][-1])])
    data=dict(scenes=[dict(id=s['scene_id'],token=s['token'],candidates=[dict(id=0,poses=[],model_logit=0.,
        guards={cal.GUARD:dict(eligible=True,reasons=[])})]) for s in scenes])
    return doc,data


SYNTHETIC_DEFAULTS=dict(alpha=.7,normalization='sigmoid',decision_rule='score_reject',
    rejection_threshold=.75,temperature=2.,sigma_deg=15.,tolerance_deg=10.,min_displacement_m=1.)


def synthetic_evaluator(candidates,guards,command,config):
    assert config['decision_rule']=='score_reject'
    assert config['normalization'] in cal.MODES
    # Deliberately simple mock of the core API, not a second production scorer.
    score=.9 if command['angle_deg']<=0 else .2
    accepted=score>config['rejection_threshold']
    return dict(status='accepted' if accepted else 'rejected_score',selected_id=0)


class CalibrationTests(unittest.TestCase):
    def test_canonical_numeric_equivalence_preserves_booleans_and_large_ints(self):
        self.assertEqual(cal.canonical_hash({'nested':[0.,-0.,1.,2.5]}),
                         cal.canonical_hash({'nested':[0,0,1,2.5]}))
        self.assertNotEqual(cal.canonical_hash(False),cal.canonical_hash(0))
        self.assertNotEqual(cal.canonical_hash(True),cal.canonical_hash(1))
        self.assertNotEqual(cal.canonical_hash(2**60+1),cal.canonical_hash(2**60))
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError): cal.canonical_hash({'value':value})

    @unittest.skipUnless(shutil.which('node'),'Node is needed for actual JSON.stringify roundtrip.')
    def test_actual_template_js_stringify_roundtrip_and_real_coordinate_change(self):
        expected=cal.read(cal.BASE/'annotation_template.json')
        script="process.stdout.write(JSON.stringify(JSON.parse(require('fs').readFileSync(0,'utf8'))))"
        run=subprocess.run([shutil.which('node'),'-e',script],input=json.dumps(expected,ensure_ascii=False),
            text=True,encoding='utf-8',capture_output=True,check=True)
        exported=json.loads(run.stdout)
        result=cal.validate(exported,expected)
        self.assertEqual(result['status'],'valid',result)
        self.assertEqual(result['blank'],112)
        self.assertEqual(exported['context_sha256'],cal.canonical_hash(exported['contexts']))
        polygon=exported['contexts']['S01']['road_context']['drivable_polygons_ego'][0]
        polygon['exterior'][0][0]+=.25
        changed=cal.validate(exported,expected)
        self.assertIn('context_hash_mismatch',changed['errors'])
        # Rehashing altered source data cannot bypass the trusted-template check.
        exported['context_sha256']=cal.canonical_hash(exported['contexts'])
        self.assertIn('source_template_mismatch:context_sha256',cal.validate(exported,expected)['errors'])

    def test_real_template_112_blank_and_no_method_outputs(self):
        doc=cal.read(cal.BASE/'annotation_template.json')
        self.assertEqual(len(doc['records']),112)
        self.assertEqual(len(doc['contexts']),16)
        self.assertNotIn('angular_interpretation_tolerance_deg',doc['instruction'])
        for r in doc['records']:
            for k in cal.REVIEW_KEYS: self.assertEqual(r[k],'')
            self.assertIsNone(r['independent_of_model_outputs'])
            self.assertEqual(r['split'],'')
        forbidden={'candidates','model_score','model_logit','fused_score','guards','offline_metrics','NC','DAC','TTC','selected_id','independent_reference_supported'}
        def check(value):
            if isinstance(value,dict):
                self.assertFalse(forbidden.intersection(value))
                for v in value.values(): check(v)
            elif isinstance(value,list):
                for v in value: check(v)
        check(doc)
        expected=cal.make_template(cal.read(cal.BASE.parent/'scene-choice-inventory-v3/inventory.json'),cal.BASE.parent/'scene-choice-inventory-v3/inventory.json')
        self.assertEqual(cal.validate(doc,expected)['status'],'valid')

    def test_blank_production_document_never_calls_core_or_picks_best(self):
        doc=cal.read(cal.BASE/'annotation_template.json')
        def forbidden(*a): self.fail('Core must not run with no labels.')
        result=cal.calibrate({},doc,evaluator=forbidden)
        self.assertEqual(result['status'],'not_calibrated')
        self.assertIsNone(result['best_config'])
        self.assertEqual(result['reason'],'no_completed_independent_binary_labels')

    def test_synthetic_fixture_is_rejected_in_production(self):
        doc,data=synthetic_document()
        self.assertEqual(cal.validate(doc)['status'],'invalid')
        self.assertEqual(cal.calibrate(data,doc)['reason'],'invalid_annotations')

    def test_rules_and_offline_labels_cannot_masquerade_as_human(self):
        doc,_=synthetic_document();doc['records'][0]['label_source']='offline_PDM_rule'
        self.assertIn('only_independent_human_review_allowed',' '.join(cal.validate(doc,allow_synthetic=True)['errors']))

    def test_completed_label_requires_metadata_and_attestation(self):
        for key,value in [('reason',''),('goal',''),('annotator_id',''),('evidence_source',''),('independent_of_model_outputs',None),('reviewed_at','2026-10-07')]:
            doc,_=synthetic_document();doc['records'][0][key]=value
            self.assertEqual(cal.validate(doc,allow_synthetic=True)['status'],'invalid',key)

    def test_identity_and_context_edits_detected(self):
        doc,_=synthetic_document();doc['records'][0]['command']['angle_deg']=15
        self.assertIn('identity_hash_mismatch',cal.validate(doc,allow_synthetic=True)['errors'])
        doc,_=synthetic_document();doc['contexts']['UNIT_SYNTHETIC_0']['current_ego_speed_mps']=100
        self.assertIn('context_hash_mismatch',cal.validate(doc,allow_synthetic=True)['errors'])

    def test_full_instruction_protocol_must_match_expected(self):
        doc,_=synthetic_document();expected=copy.deepcopy(doc)
        for key,value in [('goal','Changed goal definition'),('angular_interpretation_tolerance_deg',10.)]:
            changed=copy.deepcopy(doc);changed['instruction'][key]=value
            self.assertIn('source_template_mismatch:instruction',
                cal.validate(changed,expected,allow_synthetic=True)['errors'])
        changed=copy.deepcopy(doc);del changed['instruction']['reject']
        self.assertIn('source_template_mismatch:instruction',
            cal.validate(changed,expected,allow_synthetic=True)['errors'])

    def test_trip_group_cannot_cross_splits(self):
        doc,_=synthetic_document();doc['records'][0]['split']='test'
        self.assertIn('trip_group_split_leakage',cal.validate(doc,allow_synthetic=True)['errors'])

    def test_duplicate_and_unexpected_labels_detected(self):
        doc,_=synthetic_document();doc['records'].append(copy.deepcopy(doc['records'][0]))
        self.assertIn('duplicate_record_id',cal.validate(doc,allow_synthetic=True)['errors'])
        doc,_=synthetic_document();doc['records'][0]['label']='qualified'
        self.assertEqual(cal.validate(doc,allow_synthetic=True)['status'],'invalid')

    def test_splits_explicit_not_automatic_and_not_mutated(self):
        doc,_=synthetic_document();before=copy.deepcopy(doc)
        assignments={r['group_id']:'train' for r in doc['records']}
        new=cal.assign_split(doc,assignments)
        self.assertEqual(doc,before)
        self.assertTrue(all(r['split']=='train' for r in new['records']))
        with self.assertRaises(ValueError): cal.assign_split(doc,{})

    def test_partial_and_single_class_not_calibrated(self):
        doc,data=synthetic_document();doc['records'][0]['label']=''
        result=cal.calibrate(data,doc,allow_synthetic=True)
        self.assertEqual(result['reason'],'train_annotations_incomplete')
        doc,data=synthetic_document()
        for r in doc['records']:
            if r['split']=='train': r['label']='accept'
        self.assertEqual(cal.calibrate(data,doc,allow_synthetic=True)['reason'],'train_requires_both_binary_classes')

    def test_confusion_metrics_and_single_class_not_fake_balanced_accuracy(self):
        m=cal.metrics(['accept','accept','reject','reject'],[True,False,True,False])
        self.assertEqual((m['false_accept'],m['false_reject'],m['coverage'],m['balanced_accuracy']),(1,1,.5,.5))
        self.assertIsNone(cal.metrics(['accept'],[True])['balanced_accuracy'])

    def test_grid_uses_core_contract_and_deterministic_ties(self):
        doc,data=synthetic_document()
        with patch('fusion.DEFAULTS',SYNTHETIC_DEFAULTS):
            result=cal.calibrate(data,doc,evaluator=synthetic_evaluator,alphas=[0.,1.],thresholds=[0.,.5,1.],allow_synthetic=True)
        self.assertEqual(result['status'],'exploratory_calibrated')
        self.assertTrue(result['synthetic'])
        self.assertEqual(len(result['train_grid']),12)
        self.assertEqual(result['best_config']['normalization'],'sigmoid')
        self.assertEqual(result['best_config']['alpha'],0.)
        self.assertEqual(result['best_config']['rejection_threshold'],.5)
        self.assertFalse(result['test_evaluated'])

    def test_test_labels_do_not_change_selection_or_enter_evaluator(self):
        doc,data=synthetic_document()
        for s in data['scenes']:
            s['candidates'][0]['unit_scene']=s['id']
        def no_test(candidates,*args):
            self.assertNotEqual(candidates[0]['unit_scene'],'UNIT_SYNTHETIC_2')
            return synthetic_evaluator(candidates,*args)
        with patch('fusion.DEFAULTS',SYNTHETIC_DEFAULTS):
            first=cal.calibrate(data,doc,evaluator=no_test,alphas=[0.,1.],thresholds=[.5],allow_synthetic=True)
            for r in doc['records']:
                if r['split']=='test': r['label']='reject' if r['label']=='accept' else 'accept'
            second=cal.calibrate(data,doc,evaluator=no_test,alphas=[0.,1.],thresholds=[.5],allow_synthetic=True)
        self.assertEqual(first['best_config'],second['best_config'])
        self.assertEqual(first['normalization_winners'],second['normalization_winners'])

    def test_uncertain_labels_excluded_not_forced_negative(self):
        doc,data=synthetic_document();doc['records'][0]['label']='uncertain'
        with patch('fusion.DEFAULTS',SYNTHETIC_DEFAULTS):
            result=cal.calibrate(data,doc,evaluator=synthetic_evaluator,alphas=[0.],thresholds=[.5],allow_synthetic=True)
        self.assertEqual(result['excluded_uncertain']['train'],1)
        self.assertEqual(result['normalization_winners'][0]['train_metrics']['n'],6)

    def test_real_core_strict_threshold_and_fallback_not_counted_accept(self):
        from fusion import DEFAULTS
        doc,data=synthetic_document();row=doc['records'][3]
        candidate=data['scenes'][0]['candidates'][0]
        candidate['poses']=[[(i+1)*.5,0.,0.] for i in range(8)]
        config=dict(DEFAULTS,alpha=0.,normalization='sigmoid',decision_rule='score_reject',rejection_threshold=.5)
        rejected=cal.evaluate_records(data,[row],config)
        self.assertEqual(rejected['metrics']['coverage'],0.)
        self.assertEqual(rejected['decisions'][0]['selected_id'],0)
        config['rejection_threshold']=.499
        self.assertEqual(cal.evaluate_records(data,[row],config)['metrics']['coverage'],1.)

    def test_real_core_undefined_direction_never_counts_accepted(self):
        from fusion import DEFAULTS
        doc,data=synthetic_document();row=doc['records'][3]
        candidate=data['scenes'][0]['candidates'][0]
        candidate['poses']=[[(i+1)*.01,0.,0.] for i in range(8)]
        candidate['model_logit']=100.
        config=dict(DEFAULTS,alpha=0.,normalization='sigmoid',decision_rule='score_reject',rejection_threshold=0.)
        result=cal.evaluate_records(data,[row],config)
        self.assertFalse(result['decisions'][0]['predicted_accept'])


if __name__=='__main__':
    unittest.main(verbosity=2)
