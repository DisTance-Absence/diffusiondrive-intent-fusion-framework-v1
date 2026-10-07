"""Export a traceable development replay from the unchanged v3 inventory."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from copy import deepcopy
import math

from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from fusion import DEFAULTS, direction_features, fuse_candidates
from safety import evaluate_scene

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SOURCE = ROOT/'scene-choice-inventory-v3/inventory.json'
ANCHORS = ROOT/'joystick-experiment-copy/rule-joystick-routing-v10/anchors.json'
MODES = ('current_environment_prediction', 'current_environment_geometry', 'offline_pdm')
DEFAULT_MODE = MODES[0]
NORMALIZATIONS = ('sigmoid', 'softmax')
DECISION_RULES = ('score_reject', 'direction_tolerance')


def save(path, obj, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=None if compact else 2,
                              allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_map(scene):
    # This clipped copy is for display only; the guard reads the original map.
    points = [p for c in scene['candidates'] for p in c['poses']]
    bounds = [min(-7.0,min(p[0] for p in points)-4),
              min(-12.0,min(p[1] for p in points)-6),
              max(30.0,max(p[0] for p in points)+7),
              max(12.0,max(p[1] for p in points)+6)]
    clip = box(*bounds)
    road = scene['road_context']
    polys = [Polygon(p['exterior'],p.get('holes',[])) for p in road['drivable_polygons_ego']]
    clipped = unary_union(polys).intersection(clip)
    geometries = [clipped] if clipped.geom_type == 'Polygon' else list(clipped.geoms)
    polygons = [dict(exterior=list(p.exterior.coords), holes=[list(h.coords) for h in p.interiors])
                for p in geometries if p.geom_type=='Polygon' and not p.is_empty]
    actors=[]
    for a in scene['actor_context']['actors']:
        x,y = a['box_center_xy_ego_m']
        if bounds[0]-10<=x<=bounds[2]+10 and bounds[1]-10<=y<=bounds[3]+10:
            actors.append(dict(x=x,y=y,yaw=a['heading_ego_rad'],length=a['length_m'],
                               width=a['width_m'],category=a['category'],
                               annotation_index=a.get('annotation_index'),
                               velocity_xy_ego_mps=a.get('velocity_xy_ego_mps'),
                               velocity_known=a.get('velocity_known'),
                               is_dynamic_category=a.get('is_dynamic_category')))
    return dict(bounds=bounds,drivable=polygons,
                paths=[dict(id=p['path_id'],xy=p['centerline_xy']) for p in scene['map_paths']],
                actors=actors,vehicle=road['vehicle_parameters'],
                coordinate_frame='current ego rear axle; x forward, y left; metres')


def evaluate_export(scene, mode, angle=0.0, active=True, config=None):
    return fuse_candidates(scene['candidates'],
        [dict(candidate_id=c['id'],**c['guards'][mode]) for c in scene['candidates']],
        dict(active=active,angle_deg=angle),config)


def build():
    frozen_hash=sha(SOURCE)
    inventory=json.loads(SOURCE.read_text(encoding='utf-8'))
    anchors=json.loads(ANCHORS.read_text(encoding='utf-8'))['anchors']
    assert len(anchors)==20
    scenes=[]; comparison=[]
    for source_scene in inventory['scenes']:
        print('ENVIRONMENT_CHECK',source_scene['scene_id'],flush=True)
        assessments=evaluate_scene(source_scene, {'actor_motion_mode':'constant_velocity'})
        frozen_assessments=evaluate_scene(source_scene, {'actor_motion_mode':'frozen'})
        assessment_by_id={g['candidate_id']:g for g in assessments}
        frozen_by_id={g['candidate_id']:g for g in frozen_assessments}
        candidates=[]
        for c in source_scene['candidates']:
            r=c['row']; aid=r['anchor_id']
            offline={k:r[k] for k in ('NC','DAC','TTC')}
            reasons=[f'{k}={v:g}' for k,v in offline.items() if v!=1.0]
            prediction=deepcopy(assessment_by_id[aid]);prediction.pop('candidate_id')
            current=deepcopy(frozen_by_id[aid]);current.pop('candidate_id')
            offline_guard=dict(eligible=not reasons,reasons=reasons,
                 diagnostics=dict(status='pass' if not reasons else 'fail',
                    scope='Saved offline PDM simulation metrics; retrospective replay only; NOT online information.',
                    metrics=offline))
            candidates.append(dict(id=aid,poses=c['poses'],anchor_poses=anchors[aid],
                 model_logit=r['model_score'],**direction_features(c['poses']),offline_metrics=offline,
                 guards=dict(current_environment_prediction=prediction,
                             current_environment_geometry=current,offline_pdm=offline_guard)))
            comparison.append(dict(scene_id=source_scene['scene_id'],candidate_id=aid,
                 current_environment_eligible=current['eligible'],current_reasons=current['reasons'],
                 prediction_environment_eligible=prediction['eligible'],prediction_reasons=prediction['reasons'],
                 offline_pdm_eligible=not reasons,offline_metrics=offline))
        assert sorted(c['id'] for c in candidates)==list(range(20))
        scenes.append(dict(id=source_scene['scene_id'],token=source_scene['token'],
             trip=source_scene['metadata']['trip'],
             seed=c['row']['seed'],ego_speed_mps=source_scene['metadata']['ego_speed_mps'],
             map=display_map(source_scene),candidates=candidates))
    meta=dict(title='DiffusionDrive 外部方向意图融合框架 v1',date='2026-10-07',
        default_scene='S08',default_command_deg=0.0,defaults=DEFAULTS,
        default_safety_mode=DEFAULT_MODE,
        scope='16 historical development scenes / 320 saved predictions; no new inference or vehicle execution',
        command_scope='direction-only; 0 degrees is forward; no speed target; explicit active switch',
        model_score_note='Native sigmoid(logit), with T=1, is the default; full20 softmax(logit/T) is a comparison. Both are uncalibrated model preference scores, not safety probabilities.',
        calibration=dict(status='not_calibrated',independent_label_count=0,
                         reason='No completed independent accept/reject annotations; the included 112-request template remains blank.',
                         template='annotation_template.json',annotation_page='annotation.html',
                         instructions='标定说明.txt',report='results/calibration_status.json',
                         defaults_are_demonstration_only=True,
                         sensitivity_scope='Current scene and current command only; alpha-rho response is not success rate or parameter optimization.'),
        guard_notes=dict(current_environment_prediction='Current HD map + static actors fixed + dynamic actors translated by current absolute velocity at each matching future time, with fixed heading. Sampled constant-velocity prediction, not official DAC/TTC or a validated dynamic safety guarantee. Unknown dynamic velocity fails closed.',
                         current_environment_geometry='Frozen-current-actor comparison only: current HD map + all current actor footprints fixed in space. This deliberately freezes moving actors too; not an appropriate general future collision model.',
                         offline_pdm='Existing NC/DAC/TTC used as a retrospective oracle-like gate for comparison; unavailable to an online controller.'),
        defaults_note='Weights, Softmax temperature, angular scale, rejection threshold, historical angular tolerance and 0.1m geometry margin are development settings; native Sigmoid always uses T=1. No calibrated or validated performance claims.',
        coordinate_frame='x forward, y left. 8 future poses at 0.5s intervals, current origin separately (0,0).',
        source_inventory=str(SOURCE),source_sha256=frozen_hash)
    fixtures=[]
    requests=[('S08',0,True,{}),('S08',9,True,{}),('S08',14.3,True,{}),
              ('S08',25,True,{}),('S08',-30,True,{}),('S08',0,False,{}),
              ('S04',-20,True,{}),('S08',20,True,{'alpha':0}),
              ('S08',20,True,{'alpha':1}),('S01',179,True,{}),
              ('S08',0,True,{'alpha':0.4,'temperature':1,'sigma_deg':8,'tolerance_deg':5}),
              ('S13',0,True,{}),('S13',0,False,{}),('S14',0,True,{}),('S14',0,False,{})]
    for mode in MODES:
        for normalization in NORMALIZATIONS:
            for decision_rule in DECISION_RULES:
                for sid,angle,active,config in requests:
                    scene=next(s for s in scenes if s['id']==sid)
                    cfg=dict(config,normalization=normalization,decision_rule=decision_rule)
                    result=evaluate_export(scene,mode,angle,active,cfg)
                    fixtures.append(dict(scene_id=sid,safety_mode=mode,command=result['command'],
                                         config=result['config'],result=result))
    grid=[]
    for scene in scenes:
        for mode in MODES:
            for normalization in NORMALIZATIONS:
                for angle in (-45,-30,-15,0,15,30,45):
                    result=evaluate_export(scene,mode,angle,config=dict(normalization=normalization,decision_rule='score_reject'))
                    grid.append(dict(scene_id=scene['id'],safety_mode=mode,normalization=normalization,
                    decision_rule='score_reject',command_angle_deg=angle,
                    status=result['status'],selected_id=result['selected_id'],
                    selected_error_deg=result['selected_error_deg'],
                    original_model_top1=result['original_model_top1'],
                    model_only_output_id=result['model_only_output_id'],
                    direction_only_output_id=result['direction_only_output_id'],
                    environment_eligible_count=result['environment_eligible_count'],
                    intent_eligible_count=result['intent_eligible_count']))
    data=dict(meta=meta,scenes=scenes,fixtures=fixtures)
    save(BASE/'data.json',data,compact=True)
    save(BASE/'results/guard_comparison.json',dict(
        interpretation='Different protocols: time-aligned constant-velocity prediction, frozen-current boxes, and offline simulated tracking/logged future actors. Disagreement is not an error rate. Legacy current_* fields refer to the frozen comparison.',
        rows=comparison,totals=dict(current_pass=sum(r['current_environment_eligible'] for r in comparison),
        offline_pass=sum(r['offline_pdm_eligible'] for r in comparison),
        disagreement=sum(r['current_environment_eligible']!=r['offline_pdm_eligible'] for r in comparison),
        prediction_pass=sum(r['prediction_environment_eligible'] for r in comparison),
        prediction_vs_offline_disagreement=sum(r['prediction_environment_eligible']!=r['offline_pdm_eligible'] for r in comparison),
        prediction_vs_frozen_disagreement=sum(r['prediction_environment_eligible']!=r['current_environment_eligible'] for r in comparison))))
    save(BASE/'results/development_grid.json',dict(
        scope='Development replay only; no independent test, no closed-loop safety or task success measurement.',
        angles_deg=[-45,-30,-15,0,15,30,45],defaults=DEFAULTS,
        status_counts={mode:dict(Counter(r['status'] for r in grid if r['safety_mode']==mode))
                       for mode in MODES},rows=grid))
    for sid,angle,active,name in [('S08',0,True,'S08_forward'),('S08',9,True,'S08_left9'),
            ('S08',14.3,True,'S08_left14_3'),('S08',-30,True,'S08_right30'),
            ('S04',-20,True,'S04_right20'),('S13',0,True,'S13_forward'),('S14',0,True,'S14_forward')]:
        scene=next(s for s in scenes if s['id']==sid)
        save(BASE/f'results/{name}.json',dict(scene_id=sid,safety_mode=DEFAULT_MODE,
                                          result=evaluate_export(scene,DEFAULT_MODE,angle,active)))
    save(BASE/'results/normalization_comparison.json',dict(
        scope='Same saved candidates/current prediction guard/alpha/rho. Demonstration only, not a calibrated performance comparison.',
        defaults=DEFAULTS,rows=[dict(scene_id=s['id'],command_angle_deg=u,
        methods={n:evaluate_export(s,DEFAULT_MODE,u,config=dict(normalization=n,decision_rule='score_reject'))
                 for n in NORMALIZATIONS}) for s in scenes for u in (-45,-30,-15,0,15,30,45)]))
    save(BASE/'protocol.json',dict(version='1.2-development-unified-rejection',date='2026-10-07',
        stage='late_fusion_after_all_20_refined_candidates',defaults=DEFAULTS,
        ranking='Native sigmoid(raw logit), or softmax over all20; fused=(1-alpha)*model_weight+alpha*exp(-error^2/(2*sigma^2)); ties smaller original ID',
        acceptance='Default score_reject: among environment-eligible candidates with defined direction, propose maximum fused score; accept only when score > rejection_threshold. Equality rejects. No hard direction tolerance.',
        historical_rule='direction_tolerance + softmax reproduces previous angular-gate selection; not the default.',
        calibration=meta['calibration'],
        fallback='unsupported intent: highest model logit among environment-eligible candidates; none: null trajectory and stop_requested',
        command_input=dict(active='explicit boolean',angle_deg='relative to current ego forward; left positive; 0 is forward, not stop'),
        safety_modes=meta['guard_notes'],default_safety_mode=DEFAULT_MODE,candidate_modification=False,
        motion_prediction=dict(model='current absolute velocity, constant translation, fixed actor yaw',
                               time_frame='all future times expressed in the current fixed ego rear-axle frame',
                               uncertainty='no calibrated prediction uncertainty; fixed 0.1 m vehicle margin only',
                               missing_dynamic_velocity='unknown, not eligible',
                               future_ground_truth_used=False),
        new_inference_performed=False,online_integration_performed=False,closed_loop_execution_performed=False,
        parameter_calibration_performed=False,independent_test_performed=False,
        prior_s04_comparison_unchanged=True,
        next_validation=['tune on development scenes, freeze settings, use independent test scenes',
          'integrate current perception/local costmap + predicted dynamic obstacles',
          'compare same environment checks and fallback, report rejection and actual task success',
          'real anchor-pre-fusion requires a separate pre-inference selector; final logits must not leak into it']))
    sources=[SOURCE,ANCHORS,ROOT/'joystick-experiment-copy/rule-joystick-routing-v10/full20_reference.jsonl']
    save(BASE/'source_provenance.json',dict(
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        code=[dict(path=str(BASE/name),sha256=sha(BASE/name)) for name in ('fusion.py','safety.py','build_framework.py','runtime_adapter.py','calibration.py','build_annotation_page.py')],
        sources_unchanged=sha(SOURCE)==frozen_hash,
        distinction='Map/actors are dataset-provided current-information proxies, not perception estimates. Existing metrics only used by separately labelled offline_pdm adapter.'))
    classification_path=BASE/'scene_classification.json'
    if classification_path.exists():
        from classify_scenes import attach
        attach(json.loads(classification_path.read_text(encoding='utf-8')))
    assert sha(SOURCE)==frozen_hash
    print(json.dumps(dict(scenes=len(scenes),candidates=sum(len(s['candidates']) for s in scenes),
                          fixtures=len(fixtures),grid_rows=len(grid)),ensure_ascii=False))


if __name__=='__main__':
    build()
