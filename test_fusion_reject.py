"""Score rejection contracts plus a pre-change 128-case legacy snapshot.

The historical digest below was captured before changing fusion.py. It covers
the unchanged 16 scene banks at seven directions and an inactive command,
including all old output/row fields under the current CV environment protocol.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from fusion import DEFAULTS, config_values, fuse_candidates


def candidate(identifier=0, angle=0, logit=0, length=8):
    radians = math.radians(angle)
    return dict(id=identifier, model_logit=logit,
                poses=[[length*k/8*math.cos(radians),
                        length*k/8*math.sin(radians), radians] for k in range(1, 9)])


def guards(candidates, blocked=()):
    return [dict(candidate_id=c['id'], eligible=c['id'] not in blocked,
                 reasons=['test_environment_block'] if c['id'] in blocked else [])
            for c in candidates]


def run(candidates, config=None, angle=0, active=True, blocked=()):
    return fuse_candidates(candidates, guards(candidates, blocked),
                           dict(active=active, angle_deg=angle), config)


class ScoreRejectTests(unittest.TestCase):
    def test_native_sigmoid_scores_stability_and_ranking(self):
        bank = [candidate(i, logit=z) for i, z in enumerate([-1000, -3, 0, 3, 1000])]
        result = run(bank)
        scores = [r['model_score'] for r in result['rows']]
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[-1], 1)
        self.assertEqual(scores[2], .5)
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(math.isfinite(x) for x in scores))
        self.assertEqual(result['normalization'], 'sigmoid')
        self.assertEqual(result['effective_temperature'], 1)

    def test_native_sigmoid_ignores_temperature_and_other_candidates(self):
        bank = [candidate(0, logit=-2), candidate(1, logit=2)]
        a, b = run(bank, {'temperature': .1}), run(bank, {'temperature': 100})
        self.assertEqual(a['rows'], b['rows'])
        changed = run(bank + [candidate(2, logit=1000)])
        self.assertEqual(a['rows'][0]['model_score'], changed['rows'][0]['model_score'])
        self.assertEqual(a['rows'][1]['model_score'], changed['rows'][1]['model_score'])

    def test_softmax_keeps_full_bank_denominator_and_uses_temperature(self):
        bank = [candidate(0, logit=2), candidate(1, logit=0)]
        a = run(bank, {'normalization': 'softmax', 'temperature': 1}, blocked=[0])
        b = run(bank, {'normalization': 'softmax', 'temperature': 2}, blocked=[0])
        self.assertAlmostEqual(a['rows'][1]['model_score'], 1/(math.exp(2)+1))
        self.assertAlmostEqual(b['rows'][1]['model_score'], 1/(math.e+1))
        self.assertAlmostEqual(sum(r['model_score'] for r in b['rows']), 1)
        self.assertEqual(a['effective_temperature'], 1)
        self.assertEqual(b['effective_temperature'], 2)

    def test_threshold_equality_rejects_strictly(self):
        result = run([candidate()], {'alpha': 0, 'rejection_threshold': .5})
        self.assertEqual(result['proposed_score'], .5)
        self.assertEqual(result['status'], 'rejected_low_support')
        self.assertTrue(result['fallback_used'])
        self.assertFalse(result['rows'][0]['intent_eligible'])
        self.assertFalse(result['rows'][0]['selection_eligible'])

    def test_proposal_is_saved_before_rejection_and_fallback(self):
        bank = [candidate(0, angle=30, logit=8), candidate(1, angle=0, logit=-8)]
        result = run(bank, {'rejection_threshold': .75})
        self.assertEqual(result['proposed_id'], 1)
        self.assertEqual(result['selected_id'], 0)
        self.assertEqual(result['proposal_pool_count'], 2)
        self.assertEqual(result['intent_eligible_count'], 0)
        self.assertEqual(result['status'], 'rejected_low_support')
        self.assertEqual(result['selected_trajectory'], bank[0]['poses'])
        self.assertNotEqual(result['selected_score'], result['proposed_score'])

    def test_score_rule_has_no_hidden_direction_tolerance(self):
        bank = [candidate(angle=20)]
        result = run(bank, {'alpha': 1, 'rejection_threshold': .3, 'tolerance_deg': 0})
        self.assertEqual(result['status'], 'accepted')
        self.assertGreater(result['selected_error_deg'], 10)
        self.assertFalse(result['direction_tolerance_applied'])
        legacy = run(bank, {'normalization': 'softmax', 'decision_rule': 'direction_tolerance'})
        self.assertEqual(legacy['status'], 'rejected_unsupported')
        self.assertTrue(legacy['direction_tolerance_applied'])

    def test_environment_constraint_precedes_proposal(self):
        bank = [candidate(0, logit=10), candidate(1, logit=2)]
        result = run(bank, blocked=[0])
        self.assertEqual(result['original_model_top1'], 0)
        self.assertEqual(result['proposed_id'], 1)
        self.assertEqual(result['selected_id'], 1)
        self.assertFalse(result['rows'][0]['intent_eligible'])

    def test_no_environment_candidate_is_not_intent_rejection(self):
        result = run([candidate()], blocked=[0])
        self.assertEqual(result['status'], 'no_feasible_candidate')
        self.assertIsNone(result['proposed_id'])
        self.assertIsNone(result['proposed_score'])
        self.assertIsNone(result['selected_trajectory'])
        self.assertTrue(result['stop_requested'])
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['direction_only_status'], 'no_feasible_candidate')

    def test_no_command_bypasses_acceptance_and_uses_raw_logit_rank(self):
        # Both sigmoids saturate to 1: autonomous choice must still use logits.
        bank = [candidate(1, logit=100), candidate(9, logit=1000)]
        result = run(bank, {'rejection_threshold': 1}, active=False)
        self.assertEqual(result['status'], 'no_command')
        self.assertEqual(result['selected_id'], 9)
        self.assertEqual(result['proposed_id'], 9)
        self.assertIsNone(result['proposed_error_deg'])
        self.assertEqual(result['direction_only_output_id'], 9)
        self.assertIsNone(result['direction_only_score'])
        self.assertTrue(all(r['selection_eligible'] for r in result['rows']))

    def test_threshold_zero_and_one_and_empty_bank(self):
        bank = [candidate(logit=1000)]
        self.assertEqual(run(bank, {'rejection_threshold': 0})['status'], 'accepted')
        self.assertEqual(run(bank, {'rejection_threshold': 1})['status'], 'rejected_low_support')
        # Score exactly zero does not beat a zero threshold.
        self.assertEqual(run([candidate(logit=-1000)], {'alpha': 0, 'rejection_threshold': 0})['status'], 'rejected_low_support')
        with self.assertRaises(ValueError):
            run([])

    def test_undefined_bearing_cannot_win_active_command_but_can_fallback(self):
        bank = [candidate(logit=100, length=.5)]
        result = run(bank, {'alpha': 0, 'rejection_threshold': 0})
        self.assertEqual(result['status'], 'rejected_low_support')
        self.assertIsNone(result['proposed_id'])
        self.assertEqual(result['selected_id'], 0)
        self.assertEqual(result['proposal_pool_count'], 0)
        self.assertEqual(result['intent_eligible_count'], 0)
        self.assertFalse(result['rows'][0]['selection_eligible'])
        self.assertEqual(run(bank, active=False)['status'], 'no_command')

    def test_equal_fused_scores_use_smallest_id_independent_of_order(self):
        bank = [candidate(9), candidate(2)]
        a, b = run(bank), run(list(reversed(bank)))
        self.assertEqual(a['selected_id'], 2)
        self.assertEqual(a['proposed_id'], b['proposed_id'])

    def test_direction_only_has_own_threshold_and_matches_alpha_one(self):
        bank = [candidate(0, angle=30, logit=8), candidate(1, angle=0, logit=-8)]
        joint = run(bank)
        direction = run(bank, {'alpha': 1})
        self.assertEqual(joint['status'], 'rejected_low_support')
        self.assertEqual(joint['direction_only_status'], 'accepted')
        self.assertEqual(joint['direction_only_output_id'], 1)
        self.assertEqual(joint['direction_only_status'], direction['status'])
        self.assertEqual(joint['direction_only_output_id'], direction['selected_id'])
        far = run(bank, angle=180)
        self.assertEqual(far['direction_only_status'], 'rejected_low_support')
        self.assertEqual(far['direction_only_output_id'], 0)

    def test_zero_angle_is_active_and_wrap_is_circular(self):
        result = run([candidate(angle=-179, logit=2)], angle=179)
        self.assertAlmostEqual(result['proposed_error_deg'], 2)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(run([candidate()])['status'], 'accepted')

    def test_input_isolation_and_trajectory_copy(self):
        class Candidate(dict):
            def __getitem__(self, key):
                if key not in ('id', 'model_logit', 'poses'):
                    raise AssertionError('Unexpected input access: '+key)
                return super().__getitem__(key)
        bank = [Candidate(candidate())]
        result = run(bank)
        result['selected_trajectory'][0][0] = 1234
        self.assertEqual(bank[0]['poses'][0][0], 1)
        self.assertEqual(bank[0]['poses'][-1][0], 8)

    def test_configuration_enums_and_finite_numeric_contract(self):
        self.assertEqual(DEFAULTS['normalization'], 'sigmoid')
        self.assertEqual(DEFAULTS['decision_rule'], 'score_reject')
        for bad in [{'normalization': 'minmax'}, {'decision_rule': 1},
                    {'rejection_threshold': -1}, {'rejection_threshold': 1.01},
                    {'rejection_threshold': float('nan')}, {'alpha': True},
                    {'temperature': 0}, {'sigma_deg': float('inf')}, []]:
            with self.assertRaises(ValueError):
                config_values(bad)

    def test_legacy_full_output_snapshot_128_scene_commands(self):
        path = Path(__file__).resolve().with_name('data.json')
        data = json.loads(path.read_text(encoding='utf-8'))
        old_fields = ['config', 'command', 'status', 'reason', 'selected_id',
                      'selected_error_deg', 'selected_score', 'selected_trajectory',
                      'intent_satisfied', 'fallback_used', 'stop_requested',
                      'original_model_top1', 'safe_model_top1', 'model_only_output_id',
                      'direction_only_output_id', 'pure_intent_top1', 'safe_pure_intent_top1',
                      'ungated_fused_top1', 'environment_eligible_count',
                      'intent_eligible_count', 'rows']
        records = []
        for scene in sorted(data['scenes'], key=lambda s:s['id']):
            for angle in [-45, -30, -15, 0, 15, 30, 45, None]:
                result = fuse_candidates(scene['candidates'],
                    [dict(candidate_id=c['id'], **c['guards']['current_environment_prediction'])
                     for c in scene['candidates']],
                    dict(active=angle is not None, angle_deg=angle or 0),
                    dict(normalization='softmax', decision_rule='direction_tolerance'))
                result = {key: result[key] for key in old_fields}
                for field in ('normalization', 'decision_rule', 'rejection_threshold'):
                    result['config'].pop(field)
                records.append(dict(scene_id=scene['id'], angle=angle, result=result))
        digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':'),
                                            ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(len(records), 128)
        self.assertEqual(digest, '07f0856bc6fc7543d81990d23028eb80908bdbf4548eb3e1b2885e2f7c1e9913')


if __name__ == '__main__':
    unittest.main(verbosity=2)
