"""Independent human-request annotation and grouped exploratory calibration.

The annotator document never contains candidates, scores, guard outcomes or
method decisions. Empty labels never produce fitted parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
SCHEMA = 'scene_command_human_annotation_v1'
ANGLES = [-45, -30, -15, 0, 15, 30, 45]
LABELS = ('accept', 'reject', 'uncertain')
MODES = ('sigmoid', 'softmax')
GUARD = 'current_environment_prediction'
ALPHAS = [i / 10 for i in range(11)]
THRESHOLDS = [i / 100 for i in range(101)]
IDENTITY_KEYS = ('record_id','scene_id','token','trip','group_id','source_split','context_id','command')
REVIEW_KEYS = ('label','reason','goal','annotator_id','reviewed_at','evidence_source','label_source')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    """Hash JSON values independently of integral-float spelling (0.0 vs 0).

    Browser JSON.stringify normalizes integral numbers, including -0.0. Keep
    booleans and integers intact; in particular, never cast large integers to
    float, which would silently discard precision.
    """
    def normalize(item):
        if isinstance(item,float):
            if not math.isfinite(item):
                raise ValueError('Canonical JSON numbers must be finite.')
            return int(item) if item.is_integer() else item
        if isinstance(item,dict):
            return {k:normalize(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):
            return [normalize(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(normalize(value),sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def make_template(inventory, source_path=None):
    contexts={};records=[]
    for s in inventory['scenes']:
        sid=s['scene_id'];trip=s['summary']['trip']
        contexts[sid]=dict(scene_id=sid,token=s['token'],trip=trip,
            current_ego_speed_mps=s['metadata']['ego_speed_mps'],
            road_context=deepcopy(s['road_context']),actor_context=deepcopy(s['actor_context']),
            road_paths=[{k:deepcopy(p.get(k)) for k in ('path_id','path_ids','centerline_xy',
                'signal_supported','red_signal','unknown_signal','known_min_speed_limit_mps')} for p in s['map_paths']],
            context_limitations='当前标注对象是感知代理，地图查询有限；道路路径不是完整合法行为枚举；缺少信号、路权或时间可行性证据时选 uncertain。')
        for angle in ANGLES:
            r=dict(record_id=f'{sid}_direction_{angle:+d}',scene_id=sid,token=s['token'],trip=trip,
                group_id='trip:'+trip,source_split=s['metadata'].get('split','unknown'),context_id=sid,
                command=dict(active=True,angle_deg=angle),split='',
                independent_of_model_outputs=None,**{k:'' for k in REVIEW_KEYS})
            records.append(r)
    template=dict(schema_version=SCHEMA,synthetic=False,dataset_role='historical_development_not_independent_test',
        source=dict(inventory_path=str(source_path) if source_path else None,
                    inventory_sha256=sha(source_path) if source_path else None),
        instruction=dict(question='根据当前场景和独立任务目标，此方向请求是否应被接受？请勿查看模型候选、评分、筛查结果或方法决定。',
            horizon_s=4.,coordinate_frame='当前自车后轴原点，x向前、y向左；正角向左，负角向右；0°为向前，非停车。',
            direction_definition='未来4秒终点相对当前后轴原点的位移方位角；此处不指定目标速度。',
            accept='独立场景证据和任务目标支持该方向请求在4秒尺度应被接受；不能仅因模型有候选或评分高而接受。',
            reject='独立场景证据或任务目标明确表明不应接受；不能仅因模型无候选或评分低而拒绝。',
            uncertain='证据不足、目标不明确、通行许可或4秒可达性无法判定；不得强行二分。',
            goal='填写标注者独立确认的任务目标，明确目标区域或可接受的行为范围；不自动指定角度容差，不得由模型输出反推。目标范围不清楚时选 uncertain。',
            evidence_source='填写实际检查的当前资料及必要外部依据；不接受专家行车轨迹、PDM分数或旧规则输出冒充人工标签。',
            allowed_labels=list(LABELS),allowed_splits=['','train','dev','test'],
            label_source='完成时填写 human_review；只有实际人工填写才可声明 independent_of_model_outputs=true。',
            split_policy='保持同一trip全部请求处于同一分组；默认留空，不按单条请求或候选随机切分。',
            review_policy='所有人工字段留空；建议盲审并复核分歧。地图或当前信息不足时保留 uncertain。'),
        contexts=contexts,records=records)
    template['identity_sha256']=canonical_hash([ {k:r[k] for k in IDENTITY_KEYS} for r in records])
    template['context_sha256']=canonical_hash(contexts)
    return template


def validate(document, expected=None, allow_synthetic=False):
    errors=[];rows=document.get('records',[])
    if document.get('schema_version')!=SCHEMA: errors.append('unsupported_schema')
    if document.get('synthetic') is not False and not allow_synthetic: errors.append('synthetic_or_unverified_source_not_allowed')
    if not isinstance(rows,list) or not rows: return dict(status='invalid',errors=errors+['no_records'])
    ids=[r.get('record_id') for r in rows]
    if len(ids)!=len(set(ids)): errors.append('duplicate_record_id')
    groups={};scene_groups={};complete=Counter();blank=0
    for r in rows:
        rid=r.get('record_id','?')
        if any(k not in r for k in (*IDENTITY_KEYS,*REVIEW_KEYS,'split','independent_of_model_outputs')):
            errors.append(rid+':missing_fields');continue
        if not isinstance(r['trip'],str) or not r['trip'] or not isinstance(r['group_id'],str):
            errors.append(rid+':invalid_trip_group');continue
        if r['group_id']!='trip:'+r['trip']: errors.append(rid+':group_must_equal_trip')
        groups.setdefault(r['group_id'],set()).add(r['split'])
        scene_groups.setdefault(r['scene_id'],set()).add(r['group_id'])
        if r['split'] not in ('','train','dev','test'): errors.append(rid+':invalid_split')
        if not isinstance(r['command'],dict) or r['command'].get('active') is not True or isinstance(r['command'].get('angle_deg'),bool) or r['command'].get('angle_deg') not in ANGLES:
            errors.append(rid+':unexpected_command')
        label=r['label']
        if label=='': blank+=1;continue
        if label not in LABELS: errors.append(rid+':invalid_label');continue
        complete[label]+=1
        if any(not isinstance(r[k],str) or not r[k].strip() for k in REVIEW_KEYS): errors.append(rid+':incomplete_review_metadata')
        if r['label_source']!='human_review': errors.append(rid+':only_independent_human_review_allowed')
        if r['independent_of_model_outputs'] is not True: errors.append(rid+':independence_not_attested')
        try:
            stamp=datetime.fromisoformat(r['reviewed_at'].replace('Z','+00:00'))
            if stamp.tzinfo is None: raise ValueError()
        except (ValueError,TypeError,AttributeError): errors.append(rid+':reviewed_at_requires_ISO8601_timezone')
    if any(len(v)>1 for v in groups.values()): errors.append('trip_group_split_leakage')
    if any(len(v)>1 for v in scene_groups.values()): errors.append('scene_group_identity_conflict')
    identity=[{k:r.get(k) for k in IDENTITY_KEYS} for r in rows]
    if canonical_hash(identity)!=document.get('identity_sha256'): errors.append('identity_hash_mismatch')
    if canonical_hash(document.get('contexts',{}))!=document.get('context_sha256'): errors.append('context_hash_mismatch')
    if expected:
        if document.get('instruction')!=expected.get('instruction'):
            errors.append('source_template_mismatch:instruction')
        for key in ('identity_sha256','context_sha256','dataset_role'):
            if document.get(key)!=expected.get(key): errors.append('source_template_mismatch:'+key)
        if document.get('source',{}).get('inventory_sha256')!=expected.get('source',{}).get('inventory_sha256'):
            errors.append('source_template_mismatch:inventory_sha256')
    return dict(status='valid' if not errors else 'invalid',errors=errors,records=len(rows),
        groups=len(groups),blank=blank,completed=dict(complete),binary_labels=complete['accept']+complete['reject'],
        note='人工来源只能通过标注者声明与留存证据核验；程序不能证明是谁填写或其判断正确。')


def assign_split(document, assignments):
    result=deepcopy(document)
    groups={r['group_id'] for r in result['records']}
    if set(assignments)!=groups or any(v not in ('train','dev','test') for v in assignments.values()):
        raise ValueError('Assignments must map every full trip group exactly once to train/dev/test.')
    for r in result['records']: r['split']=assignments[r['group_id']]
    return result


def metrics(labels, predictions):
    if len(labels)!=len(predictions): raise ValueError('Label/prediction lengths differ.')
    tp=sum(y=='accept' and p for y,p in zip(labels,predictions))
    tn=sum(y=='reject' and not p for y,p in zip(labels,predictions))
    fp=sum(y=='reject' and p for y,p in zip(labels,predictions))
    fn=sum(y=='accept' and not p for y,p in zip(labels,predictions))
    far=fp/(fp+tn) if fp+tn else None
    frr=fn/(fn+tp) if fn+tp else None
    error=(far+frr)/2 if far is not None and frr is not None else None
    return dict(n=len(labels),true_accept=tp,true_reject=tn,false_accept=fp,false_reject=fn,
        false_accept_rate=far,false_reject_rate=frr,coverage=sum(predictions)/len(labels) if labels else None,
        balanced_accuracy=1-error if error is not None else None,equal_class_error=error)


def evaluate_records(data, records, config, evaluator=None):
    if evaluator is None:
        from fusion import fuse_candidates
        evaluator=fuse_candidates
    scenes={s['id']:s for s in data['scenes']};predictions=[];decisions=[]
    for r in records:
        scene=scenes[r['scene_id']]
        if scene['token']!=r['token']: raise ValueError('Scene/token mismatch.')
        assessment=[dict(candidate_id=c['id'],**c['guards'][GUARD]) for c in scene['candidates']]
        output=evaluator(scene['candidates'],assessment,r['command'],config)
        accepted=output['status']=='accepted'
        predictions.append(accepted)
        decisions.append(dict(record_id=r['record_id'],predicted_accept=accepted,status=output['status'],selected_id=output['selected_id']))
    return dict(metrics=metrics([r['label'] for r in records],predictions),decisions=decisions)


def calibrate(data, document, expected=None, evaluator=None, alphas=None, thresholds=None, allow_synthetic=False):
    check=validate(document,expected,allow_synthetic)
    base=dict(status='not_calibrated',best_config=None,validation=check,
        parameter_scope='Exploratory development calibration; not a deployment safety threshold.',
        selection_protocol=dict(train_objective='minimize (false_accept_rate + false_reject_rate) / 2',
            train_tie_break=['lower_false_accept_rate','higher_rejection_threshold','lower_alpha'],
            dev_objective='compare only the two train-selected normalization winners by equal_class_error',
            dev_tie_break=['lower_false_accept_rate','sigmoid_before_softmax'],
            test_usage='test label content and predictions are excluded from parameter/normalization selection; format/identity validation still applies'),
        warnings=['仅16个历史开发场景，候选320条不是320个独立样本。',
                  '本批已多次用于开发；重划test也不能成为未接触的独立测试集。'])
    if check['status']!='valid': return dict(base,reason='invalid_annotations')
    if not check['binary_labels']: return dict(base,reason='no_completed_independent_binary_labels')
    if any(not r['split'] for r in document['records']): return dict(base,reason='explicit_trip_group_split_required')
    parts={p:[r for r in document['records'] if r['split']==p] for p in ('train','dev')}
    for p,rows in parts.items():
        if not rows or any(r['label']=='' for r in rows): return dict(base,reason=p+'_annotations_incomplete')
        parts[p]=[r for r in rows if r['label'] in ('accept','reject')]
        if {r['label'] for r in parts[p]}!={'accept','reject'}: return dict(base,reason=p+'_requires_both_binary_classes')
    from fusion import DEFAULTS
    if not all(k in DEFAULTS for k in ('normalization','decision_rule','rejection_threshold')):
        return dict(base,reason='core_score_reject_contract_unavailable')
    aa=ALPHAS if alphas is None else alphas;rr=THRESHOLDS if thresholds is None else thresholds
    if not aa or not rr or any(not math.isfinite(x) or not 0<=x<=1 for x in [*aa,*rr]): raise ValueError('Invalid calibration grid.')
    winners=[];grid=[]
    for normalization in MODES:
        trials=[]
        for alpha in aa:
            for rho in rr:
                config=dict(DEFAULTS,alpha=alpha,normalization=normalization,decision_rule='score_reject',rejection_threshold=rho)
                outcome=evaluate_records(data,parts['train'],config,evaluator)
                trial=dict(config=config,train_metrics=outcome['metrics'])
                trials.append(trial);grid.append(trial)
        winner=min(trials,key=lambda t:(t['train_metrics']['equal_class_error'],t['train_metrics']['false_accept_rate'],
                                       -t['config']['rejection_threshold'],t['config']['alpha']))
        winner=deepcopy(winner)
        winner['dev']=evaluate_records(data,parts['dev'],winner['config'],evaluator)
        winners.append(winner)
    best=min(winners,key=lambda w:(w['dev']['metrics']['equal_class_error'],w['dev']['metrics']['false_accept_rate'],
                                   MODES.index(w['config']['normalization'])))
    return dict(base,status='exploratory_calibrated',reason=None,best_config=best['config'],
        core_sha256=sha(BASE/'fusion.py'),normalization_winners=winners,train_grid=grid,
        selection_record_ids={p:[r['record_id'] for r in rows] for p,rows in parts.items()},
        selection_groups={p:sorted({r['group_id'] for r in document['records'] if r['split']==p}) for p in ('train','dev')},
        excluded_uncertain={p:sum(r['split']==p and r['label']=='uncertain' for r in document['records']) for p in ('train','dev')},
        test_evaluated=False,has_independent_test=False,synthetic=document.get('synthetic',False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['template','validate','split','calibrate','evaluate'])
    parser.add_argument('--inventory',type=Path,default=BASE.parent/'scene-choice-inventory-v3/inventory.json')
    parser.add_argument('--annotations',type=Path,default=BASE/'annotation_template.json')
    parser.add_argument('--data',type=Path,default=BASE/'data.json')
    parser.add_argument('--assignments',type=Path)
    parser.add_argument('--frozen-calibration',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    expected=make_template(read(args.inventory),args.inventory)
    if args.action=='template': result=expected
    else:
        document=read(args.annotations)
        if args.action=='validate': result=validate(document,expected)
        elif args.action=='split':
            if not args.assignments: parser.error('--assignments is required for split.')
            if validate(document,expected)['status']!='valid': parser.error('Invalid source annotations.')
            result=assign_split(document,read(args.assignments))
        elif args.action=='calibrate':
            result=calibrate(read(args.data),document,expected)
            result['inputs']=dict(annotations_sha256=sha(args.annotations),data_sha256=sha(args.data),inventory_sha256=sha(args.inventory))
        else:
            if not args.frozen_calibration: parser.error('--frozen-calibration is required; never tune on test.')
            check=validate(document,expected)
            frozen=read(args.frozen_calibration)
            if check['status']!='valid' or frozen.get('status')!='exploratory_calibrated' or not frozen.get('best_config'):
                parser.error('Valid independent annotations and an already frozen calibration are required.')
            if frozen.get('core_sha256')!=sha(BASE/'fusion.py'): parser.error('Core changed since parameter selection.')
            if frozen.get('inputs',{}).get('data_sha256')!=sha(args.data): parser.error('Candidate/guard data changed since parameter selection.')
            rows=[r for r in document['records'] if r['split']=='test']
            if not rows or any(r['label']=='' for r in rows): parser.error('All explicit test requests must be reviewed first.')
            rows=[r for r in rows if r['label'] in ('accept','reject')]
            selected=set(sum(frozen['selection_record_ids'].values(),[]))
            if any(r['record_id'] in selected for r in rows): parser.error('Selection/test record overlap.')
            selected_groups=set(sum(frozen['selection_groups'].values(),[]))
            if any(r['group_id'] in selected_groups for r in rows): parser.error('Selection/test trip-group overlap.')
            result=dict(status='fixed_parameter_test_evaluation',config=frozen['best_config'],
                result=evaluate_records(read(args.data),rows,frozen['best_config']),
                frozen_calibration_sha256=sha(args.frozen_calibration),has_independent_test=False,
                note='Historical development cases remain historical even when explicitly held out for this split.')
    write(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','reason','best_config','validation') if k in result},ensure_ascii=True))


if __name__=='__main__':
    main()
