"""Independent replay plus old angular-gate regression for unified rejection."""
from pathlib import Path
import hashlib
import json
import math
import random
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import fusion

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def reference(cs,gs,cmd,cfg):
    active=cmd.get('active',False);target=cmd.get('angle_deg',0)
    top=max(c['model_logit'] for c in cs)
    ws=[math.exp((c['model_logit']-top)/cfg['temperature']) for c in cs];total=sum(ws)
    allowed={g['candidate_id'] for g in gs if g.get('eligible') is True};rows=[]
    for c,w in zip(cs,ws):
        m=(1+math.tanh(c['model_logit']/2))/2 if cfg['normalization']=='sigmoid' else w/total
        x,y=c['poses'][-1][:2];angle=math.degrees(math.atan2(y,x)) if math.hypot(x,y)>=cfg['min_displacement_m'] else None
        e=abs(math.remainder(angle-target,360)) if active and angle is not None else None
        intent=math.exp(-e*e/(2*cfg['sigma_deg']**2)) if e is not None else None
        f=(1-cfg['alpha'])*m+cfg['alpha']*(intent or 0) if active else m
        rows.append(dict(id=c['id'],logit=c['model_logit'],model_score=m,intent_score=intent,error_deg=e,fused_score=f,eligible=c['id'] in allowed))
    pool=[r for r in rows if r['eligible']]
    if not pool:return None,'no_feasible_candidate',rows
    fallback=min(pool,key=lambda r:(-r['logit'],r['id']))['id']
    if not active:return fallback,'no_command',rows
    defined=[r for r in pool if r['error_deg'] is not None]
    if cfg['decision_rule']=='direction_tolerance':
        pool=[r for r in defined if r['error_deg']<=cfg['tolerance_deg']]
        if not pool:return fallback,'rejected_unsupported',rows
        return min(pool,key=lambda r:(-r['fused_score'],r['id']))['id'],'accepted',rows
    best=min(defined,key=lambda r:(-r['fused_score'],r['id'])) if defined else None
    if best and best['fused_score']>cfg['rejection_threshold']:return best['id'],'accepted',rows
    return fallback,'rejected_low_support',rows

def verify(cs,gs,cmd,cfg,result):
    wanted,status,rows=reference(cs,gs,cmd,cfg)
    assert (result['selected_id'],result['status'])==(wanted,status),(result['selected_id'],result['status'],wanted,status)
    actual={r['id']:r for r in result['rows']}
    for r in rows:
        for key in ('model_score','intent_score','error_deg','fused_score'):
            a,b=r[key],actual[r['id']][key]
            assert (a is b) if a is None or b is None else math.isclose(a,b,abs_tol=2e-12,rel_tol=2e-12),(key,a,b)
    if wanted is not None:assert result['selected_trajectory']==next(c['poses'] for c in cs if c['id']==wanted)

def main():
    data=json.loads((BASE/'data.json').read_text(encoding='utf-8'))
    old=json.loads((BASE/'work/pre_unified_rejection/data.json').read_text(encoding='utf-8'))
    scenes={s['id']:s for s in data['scenes']};previous={s['id']:s for s in old['scenes']}
    assert set(scenes)==set(previous) and len(scenes)==16
    for sid,s in scenes.items():
        for field in ('candidates','map','classification'):assert s[field]==previous[sid][field],(sid,field)
    for f in data['fixtures']:
        s=scenes[f['scene_id']];gs=[dict(candidate_id=c['id'],**c['guards'][f['safety_mode']]) for c in s['candidates']]
        r=fusion.fuse_candidates(s['candidates'],gs,f['command'],f['config']);assert r==f['result']
        verify(s['candidates'],gs,f['command'],f['config'],r)
    for f in old['fixtures']:
        s=scenes[f['scene_id']];gs=[dict(candidate_id=c['id'],**c['guards'][f['safety_mode']]) for c in s['candidates']]
        cfg=dict(f['config'],normalization='softmax',decision_rule='direction_tolerance')
        r=fusion.fuse_candidates(s['candidates'],gs,f['command'],cfg)
        for key,v in f['result'].items():
            if key=='config':assert all(r[key][k]==value for k,value in v.items())
            else:assert r[key]==v,(f['scene_id'],key)
    grid=json.loads((BASE/'results/development_grid.json').read_text(encoding='utf-8'))
    for row in grid['rows']:
        s=scenes[row['scene_id']];gs=[dict(candidate_id=c['id'],**c['guards'][row['safety_mode']]) for c in s['candidates']]
        cfg=dict(data['meta']['defaults'],normalization=row['normalization'],decision_rule=row['decision_rule'])
        cmd=dict(active=True,angle_deg=row['command_angle_deg']);r=fusion.fuse_candidates(s['candidates'],gs,cmd,cfg)
        verify(s['candidates'],gs,cmd,cfg,r)
        for key in ('selected_id','status','selected_error_deg','environment_eligible_count','intent_eligible_count'):assert r[key]==row[key]
    rng=random.Random(20261007)
    for _ in range(300):
        cs=[]
        for i in range(20):
            a=rng.uniform(-math.pi,math.pi);length=rng.choice([.5,rng.uniform(1,35)])
            cs.append(dict(id=i,model_logit=rng.uniform(-30,30),poses=[[length*(j+1)/8*math.cos(a),length*(j+1)/8*math.sin(a),0] for j in range(8)]))
        gs=[dict(candidate_id=i,eligible=rng.random()>.3,reasons=[]) for i in range(20)]
        cfg=dict(fusion.DEFAULTS,normalization=rng.choice(['sigmoid','softmax']),decision_rule=rng.choice(['score_reject','direction_tolerance']),alpha=rng.random(),rejection_threshold=rng.random(),temperature=rng.uniform(.5,5))
        cmd=dict(active=rng.random()>.1,angle_deg=rng.uniform(-400,400));r=fusion.fuse_candidates(cs,gs,cmd,cfg);verify(cs,gs,cmd,cfg,r)
    provenance=json.loads((BASE/'source_provenance.json').read_text(encoding='utf-8'))
    for e in provenance['sources']+provenance['code']:assert sha(Path(e['path']))==e['sha256'],e['path']
    report=dict(status='PASS',scene_count=16,candidate_count=320,candidates_logits_anchors_guards_maps_classification_unchanged=True,
        fixture_count=len(data['fixtures']),legacy_fixture_count=len(old['fixtures']),grid_count=len(grid['rows']),random_cases=300,
        data_sha256=sha(BASE/'data.json'),fusion_sha256=sha(BASE/'fusion.py'),calibration_status=data['meta']['calibration'],
        scope='Engineering and mathematical verification only; not independent human labels, parameter calibration, or task-performance evidence.')
    for name in ('work/unified_rejection_validation.json','work/audit_framework.json'):
        (BASE/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
