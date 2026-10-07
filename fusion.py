"""Frozen-candidate, direction-only late fusion; no perception or safety oracle inside.

Safety assessments are supplied by an explicit adapter. This module does not
read NC/DAC/TTC, maps, future actors, or mutate candidate trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

DEFAULTS = dict(alpha=0.7, temperature=2.0, sigma_deg=15.0,
                tolerance_deg=10.0, min_displacement_m=1.0,
                normalization='sigmoid', decision_rule='score_reject',
                rejection_threshold=0.75)


def _stable_sigmoid(logit):
    """Native sigmoid of the raw focal-classification logit, without temperature.

    This score is not asserted to be a calibrated acceptance/safety probability.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def wrap_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def direction_features(poses, min_displacement_m=1.0):
    """Use current rear-axle origin to t=4 s endpoint, not terminal yaw."""
    if len(poses) != 8 or any(len(p) != 3 for p in poses):
        raise ValueError('Expected eight future [x,y,yaw] poses at 0.5 s.')
    if any(not math.isfinite(float(v)) for p in poses for v in p):
        raise ValueError('Candidate poses must be finite.')
    x, y = map(float, poses[-1][:2])
    displacement = math.hypot(x, y)
    direction = math.degrees(math.atan2(y, x)) if displacement >= min_displacement_m else None
    points = [[0.0, 0.0]] + [p[:2] for p in poses]
    length = sum(math.hypot(float(b[0])-float(a[0]), float(b[1])-float(a[1]))
                 for a, b in zip(points, points[1:]))
    return dict(direction_deg=direction, net_displacement_m=displacement,
                length_m=length, mean_speed_mps=length/4.0)


def config_values(config=None):
    result = dict(DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError('Configuration must be a mapping.')
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown config fields: {sorted(unknown)}')
        result.update(config)
    choices = {'normalization': ('sigmoid', 'softmax'),
               'decision_rule': ('score_reject', 'direction_tolerance')}
    for key, values in choices.items():
        if not isinstance(result[key], str) or result[key] not in values:
            raise ValueError(f'Invalid {key}: {result[key]!r}')
    numeric = [key for key in result if key not in choices]
    try:
        if any(isinstance(result[key], bool) for key in numeric):
            raise ValueError('Numeric configuration values must not be booleans.')
        for key in numeric:
            result[key] = float(result[key])
    except (TypeError, ValueError) as exc:
        raise ValueError('Numeric configuration values must be real numbers.') from exc
    if not all(math.isfinite(result[key]) for key in numeric):
        raise ValueError('Configuration must be finite.')
    if not 0 <= result['alpha'] <= 1:
        raise ValueError('alpha must be in [0,1].')
    if result['temperature'] <= 0 or result['sigma_deg'] <= 0:
        raise ValueError('temperature and sigma_deg must be positive.')
    if not 0 <= result['tolerance_deg'] <= 180 or result['min_displacement_m'] <= 0:
        raise ValueError('Invalid tolerance or minimum displacement.')
    if not 0 <= result['rejection_threshold'] <= 1:
        raise ValueError('rejection_threshold must be in [0,1].')
    return result


def fuse_candidates(candidates, guards, command=None, config=None):
    """Return every score and a selected saved trajectory, or no trajectory.

    candidates: [{id:int, poses:8x3, model_logit:float, ...}]
    guards: [{candidate_id:int, eligible:bool, reasons:[str], ...}]
    command: {active:bool, angle_deg:float}; 0 degrees is an active forward
        direction, NOT an instruction to stop. Missing command means inactive.
    The deployed 20-candidate contract is checked by its adapter/builder; this
    function also accepts smaller banks for unit tests and other integrations.

    score_reject: propose the highest fused score in the environment-passing,
    direction-defined pool; accept only when F > rejection_threshold. Equality
    rejects. The default threshold is an uncalibrated demonstration setting.
    direction_tolerance + softmax exactly reproduces the historical rule.
    Native sigmoid ignores temperature; only the softmax comparison uses it.
    """
    cfg = config_values(config)
    if not candidates:
        raise ValueError('Candidate bank is empty.')
    ids = [c['id'] for c in candidates]
    if any(not isinstance(i, int) or isinstance(i, bool) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Candidate IDs must be unique integers.')
    logits = [float(c['model_logit']) for c in candidates]
    if not all(math.isfinite(z) for z in logits):
        raise ValueError('Model logits must be finite.')
    guard_map = {}
    for g in guards:
        aid = g['candidate_id']
        if not isinstance(aid, int) or isinstance(aid, bool):
            raise ValueError('Assessment candidate IDs must be integers, not boolean.')
        reasons = g.get('reasons', [])
        if not isinstance(reasons, list) or any(not isinstance(s, str) for s in reasons):
            raise ValueError('Assessment reasons must be a list of strings.')
        if aid in guard_map:
            raise ValueError('Duplicate safety assessment.')
        guard_map[aid] = g
    if set(guard_map) - set(ids):
        raise ValueError('Assessment refers to an unknown candidate.')
    command = dict(active=False) if command is None else dict(command)
    if not isinstance(command.get('active', False), bool):
        raise ValueError('command.active must be boolean.')
    active = command.get('active', False)
    angle = float(command.get('angle_deg', 0.0))
    if not math.isfinite(angle):
        raise ValueError('Command angle must be finite.')
    angle = wrap_deg(angle)
    # Convert all original logits before masking. Native sigmoid is pointwise;
    # softmax keeps its historical denominator over the unchanged full bank.
    if cfg['normalization'] == 'sigmoid':
        model_scores = [_stable_sigmoid(z) for z in logits]
    else:
        maximum = max(logits)
        exponents = [math.exp((z-maximum)/cfg['temperature']) for z in logits]
        total = sum(exponents)
        model_scores = [value/total for value in exponents]
    score_reject = cfg['decision_rule'] == 'score_reject'
    rows = []
    for c, model_score in zip(candidates, model_scores):
        geom = direction_features(c['poses'], cfg['min_displacement_m'])
        theta = geom['direction_deg']
        error = abs(wrap_deg(theta-angle)) if active and theta is not None else None
        intent_score = math.exp(-0.5*(error/cfg['sigma_deg'])**2) if error is not None else None
        fused_score = ((1-cfg['alpha'])*model_score + cfg['alpha']*(intent_score or 0.0)) if active else model_score
        g = guard_map.get(c['id'], {})
        eligible = g.get('eligible') is True  # unknown/missing is never a pass
        reasons = list(g.get('reasons', [])) if g else ['missing_environment_assessment']
        if not eligible and not reasons:
            reasons = ['environment_check_not_passed']
        intent_eligible = eligible and error is not None and (
            fused_score > cfg['rejection_threshold'] if score_reject
            else error <= cfg['tolerance_deg'])
        rows.append(dict(id=c['id'], model_logit=float(c['model_logit']),
                         model_score=model_score, intent_score=intent_score,
                         fused_score=fused_score, error_deg=error,
                         eligible=eligible, safety_reasons=reasons,
                         intent_eligible=intent_eligible,
                         selection_eligible=intent_eligible if active else eligible,
                         **geom))
    def highest(pool, field):
        return min(pool, key=lambda r: (-r[field], r['id']))['id'] if pool else None
    def closest(pool):
        pool = [r for r in pool if r['error_deg'] is not None]
        return min(pool, key=lambda r: (r['error_deg'], r['id']))['id'] if pool else None
    environmental = [r for r in rows if r['eligible']]
    directional = [r for r in environmental if r['error_deg'] is not None]
    matched = [r for r in rows if r['intent_eligible']]
    original = highest(rows, 'model_logit')
    fallback = highest(environmental, 'model_logit')
    proposal_pool = (directional if score_reject else matched) if active else environmental
    proposed = highest(proposal_pool, 'fused_score') if active else fallback
    proposed_row = next((r for r in rows if r['id'] == proposed), None)
    if not environmental:
        status, selected = 'no_feasible_candidate', None
        reason = '当前环境检查没有通过的候选；返回空轨迹并请求停车控制器处理，尚未执行停车。'
    elif not active:
        status, selected = 'no_command', fallback
        reason = '外部指令未启用，选择环境检查通过候选中模型分数最高者。'
    elif score_reject and (proposed_row is None or proposed_row['fused_score'] <= cfg['rejection_threshold']):
        status, selected = 'rejected_low_support', fallback
        reason = '最高融合分未严格高于拒绝阈值，或没有方向可定义的候选；拒绝本次意图，回退到检查通过的模型首选。'
    elif score_reject:
        status, selected = 'accepted', proposed
        reason = '环境检查通过候选中的最高融合分严格高于拒绝阈值；接受外部意图。'
    elif not matched:
        status, selected = 'rejected_unsupported', fallback
        reason = '当前候选中没有同时通过环境检查且处于方向容差内的轨迹；拒绝本次意图，回退到检查通过的模型首选。'
    else:
        status, selected = 'accepted', highest(matched, 'fused_score')
        reason = '在同时通过环境检查和方向容差的候选中，选择融合分数最高者。'
    # Direction-only is an alpha=1 rejection ablation in the new rule. It never
    # borrows the jointly-qualified pool; its own I must beat the same rho.
    if not environmental:
        direction_status, direction_proposed, direction_output = 'no_feasible_candidate', None, None
    elif not active:
        direction_status, direction_proposed, direction_output = 'no_command', fallback, fallback
    elif score_reject:
        direction_proposed = highest(directional, 'intent_score')
        direction_row = next((r for r in rows if r['id'] == direction_proposed), None)
        direction_accepts = direction_row is not None and direction_row['intent_score'] > cfg['rejection_threshold']
        direction_status = 'accepted' if direction_accepts else 'rejected_low_support'
        direction_output = direction_proposed if direction_accepts else fallback
    else:
        direction_proposed = closest(matched)
        direction_status = 'accepted' if direction_proposed is not None else 'rejected_unsupported'
        direction_output = direction_proposed if direction_proposed is not None else fallback
    direction_row = next((r for r in rows if r['id'] == direction_proposed), None)
    selected_row = next((r for r in rows if r['id'] == selected), None)
    for row in rows:
        row['selected'] = row['id'] == selected
    selected_candidate = next((c for c in candidates if c['id'] == selected), None)
    return dict(config=cfg, command=dict(active=active, angle_deg=angle),
                normalization=cfg['normalization'], decision_rule=cfg['decision_rule'],
                effective_temperature=1.0 if cfg['normalization'] == 'sigmoid' else cfg['temperature'],
                rejection_threshold=cfg['rejection_threshold'],
                direction_tolerance_applied=active and not score_reject,
                proposed_id=proposed,
                proposed_score=proposed_row['fused_score'] if proposed_row else None,
                proposed_error_deg=proposed_row['error_deg'] if proposed_row else None,
                proposal_pool_count=len(proposal_pool),
                status=status, reason=reason, selected_id=selected,
                selected_error_deg=selected_row['error_deg'] if selected_row else None,
                selected_score=selected_row['fused_score'] if selected_row else None,
                selected_trajectory=deepcopy(selected_candidate['poses']) if selected_candidate else None,
                intent_satisfied=status == 'accepted',
                fallback_used=status in ('rejected_unsupported', 'rejected_low_support'),
                stop_requested=status == 'no_feasible_candidate',
                original_model_top1=original, safe_model_top1=fallback,
                model_only_output_id=fallback,
                direction_only_output_id=direction_output,
                direction_only_proposed_id=direction_proposed,
                direction_only_score=direction_row['intent_score'] if active and direction_row else None,
                direction_only_proposed_error_deg=direction_row['error_deg'] if active and direction_row else None,
                direction_only_status=direction_status,
                pure_intent_top1=closest(rows) if active else None,
                safe_pure_intent_top1=closest(environmental) if active else None,
                ungated_fused_top1=highest(rows, 'fused_score'),
                environment_eligible_count=len(environmental), intent_eligible_count=len(matched),
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description='Replay late fusion on frozen DiffusionDrive candidates.')
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('data.json'))
    parser.add_argument('--scene', default='S08')
    parser.add_argument('--angle', type=float, default=0.0)
    parser.add_argument('--no-command', action='store_true')
    parser.add_argument('--safety-mode', choices=['current_environment_prediction','current_environment_geometry','offline_pdm'], default='current_environment_prediction')
    parser.add_argument('--alpha', type=float, default=DEFAULTS['alpha'])
    parser.add_argument('--temperature', type=float, default=DEFAULTS['temperature'])
    parser.add_argument('--sigma-deg', type=float, default=DEFAULTS['sigma_deg'])
    parser.add_argument('--tolerance-deg', type=float, default=DEFAULTS['tolerance_deg'])
    parser.add_argument('--normalization', choices=['sigmoid', 'softmax'], default=DEFAULTS['normalization'])
    parser.add_argument('--decision-rule', choices=['score_reject', 'direction_tolerance'], default=DEFAULTS['decision_rule'])
    parser.add_argument('--rejection-threshold', type=float, default=DEFAULTS['rejection_threshold'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    scene = next((s for s in data['scenes'] if s['id'] == args.scene), None)
    if scene is None:
        parser.error(f'Unknown scene {args.scene}')
    result = fuse_candidates(scene['candidates'],
        [dict(candidate_id=c['id'], **c['guards'][args.safety_mode]) for c in scene['candidates']],
        dict(active=not args.no_command, angle_deg=args.angle),
        dict(alpha=args.alpha, temperature=args.temperature, sigma_deg=args.sigma_deg,
             tolerance_deg=args.tolerance_deg, normalization=args.normalization,
             decision_rule=args.decision_rule, rejection_threshold=args.rejection_threshold))
    result.update(scene_id=args.scene, safety_mode=args.safety_mode,
                  execution='offline_replay_no_new_model_inference_no_vehicle_control')
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output)


if __name__ == '__main__':
    main()
