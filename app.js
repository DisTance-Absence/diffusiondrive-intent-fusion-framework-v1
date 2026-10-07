'use strict';
(() => {
const DATA=window.FUSION_DATA;
const DEFAULTS={alpha:.7,temperature:2,sigma_deg:15,tolerance_deg:10,min_displacement_m:1,normalization:'sigmoid',decision_rule:'score_reject',rejection_threshold:.75,...(DATA?.meta?.defaults||{})};
const DEFAULT_SAFETY_MODE=DATA?.meta?.default_safety_mode||'current_environment_prediction';
const byId=id=>document.getElementById(id);
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const fmt=(n,d=2)=>Number.isFinite(n)?n.toFixed(d):'—';
const signed=n=>Number.isFinite(n)?(n>0?'+':'')+n.toFixed(2):'—';
const idText=id=>id===null||id===undefined?'空轨迹':'#'+id;
const wrap=x=>((x+180)%360+360)%360-180;
const highest=(rows,key)=>rows.length?[...rows].sort((a,b)=>b[key]-a[key]||a.id-b.id)[0].id:null;
const closest=rows=>{const defined=rows.filter(r=>r.error_deg!==null);return defined.length?[...defined].sort((a,b)=>a.error_deg-b.error_deg||a.id-b.id)[0].id:null;};
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));

// The same decision contract as fusion.py. Geometry checks are supplied by the
// explicit adapter in the data; future NC/DAC/TTC is never read here.
function fuseCandidates(candidates,guards,command={active:false},config={}){
  const cfg={...DEFAULTS,...config},active=command.active===true,angle=wrap(Number(command.angle_deg||0));
  const logits=candidates.map(c=>c.model_logit),maximum=Math.max(...logits);
  const exps=logits.map(z=>Math.exp((z-maximum)/cfg.temperature)),total=exps.reduce((a,b)=>a+b,0);
  const guardMap=new Map(guards.map(g=>[g.candidate_id,g]));
  const rows=candidates.map((candidate,index)=>{
    const last=candidate.poses.at(-1),displacement=Math.hypot(last[0],last[1]);
    const direction=displacement>=cfg.min_displacement_m?Math.atan2(last[1],last[0])*180/Math.PI:null;
    const points=[[0,0],...candidate.poses];
    const length=points.slice(1).reduce((s,p,k)=>s+Math.hypot(p[0]-points[k][0],p[1]-points[k][1]),0);
    const error=active&&direction!==null?Math.abs(wrap(direction-angle)):null;
    const intent=error===null?null:Math.exp(-.5*(error/cfg.sigma_deg)**2);
    const model=cfg.normalization==='sigmoid'?sigmoid(candidate.model_logit):exps[index]/total;
    const fused=active?(1-cfg.alpha)*model+cfg.alpha*(intent||0):model;
    const g=guardMap.get(candidate.id),eligible=g?.eligible===true;
    let reasons=g?[...(g.reasons||[])]:['missing_environment_assessment'];
    if(!eligible&&!reasons.length)reasons=['environment_check_not_passed'];
    const intentEligible=eligible&&error!==null&&(cfg.decision_rule==='score_reject'?fused>cfg.rejection_threshold:error<=cfg.tolerance_deg);
    return {id:candidate.id,model_logit:candidate.model_logit,model_score:model,intent_score:intent,fused_score:fused,error_deg:error,
      eligible,safety_reasons:reasons,intent_eligible:intentEligible,selection_eligible:active?intentEligible:eligible,
      direction_deg:direction,net_displacement_m:displacement,length_m:length,mean_speed_mps:length/4};
  });
  const environment=rows.filter(r=>r.eligible),matched=rows.filter(r=>r.intent_eligible);
  const directional=environment.filter(r=>r.error_deg!==null);
  const proposalPool=active?(cfg.decision_rule==='score_reject'?directional:matched):environment;
  const original=highest(rows,'model_logit'),fallback=highest(environment,'model_logit');
  const proposed=active?highest(proposalPool,'fused_score'):fallback,proposedRow=rows.find(r=>r.id===proposed);
  let status,selected,reason;
  if(!environment.length){status='no_feasible_candidate';selected=null;reason='当前环境检查没有通过的候选；返回空轨迹并请求停车控制器处理，尚未执行停车。';}
  else if(!active){status='no_command';selected=fallback;reason='外部指令未启用，选择环境检查通过候选中模型分数最高者。';}
  else if(cfg.decision_rule==='score_reject'){
    if(proposedRow&&proposedRow.fused_score>cfg.rejection_threshold){status='accepted';selected=proposed;reason='环境检查通过候选中的最高融合分严格高于拒绝阈值；接受外部意图。';}
    else{status='rejected_low_support';selected=fallback;reason='最高融合分未严格高于拒绝阈值，或没有方向可定义的候选；拒绝本次意图，回退到检查通过的模型首选。';}
  }
  else if(!matched.length){status='rejected_unsupported';selected=fallback;reason='当前候选中没有同时通过环境检查且处于方向容差内的轨迹；拒绝本次意图，回退到检查通过的模型首选。';}
  else{status='accepted';selected=highest(matched,'fused_score');reason='在同时通过环境检查和方向容差的候选中，选择融合分数最高者。';}
  let directionStatus,directionProposed,directionOutput;
  if(!environment.length){directionStatus='no_feasible_candidate';directionProposed=null;directionOutput=null;}
  else if(!active){directionStatus='no_command';directionProposed=fallback;directionOutput=fallback;}
  else if(cfg.decision_rule==='score_reject'){
    directionProposed=highest(directional,'intent_score');
    const directionRow=rows.find(r=>r.id===directionProposed),accepts=!!directionRow&&directionRow.intent_score>cfg.rejection_threshold;
    directionStatus=accepts?'accepted':'rejected_low_support';directionOutput=accepts?directionProposed:fallback;
  }else{directionProposed=closest(matched);directionStatus=directionProposed!==null?'accepted':'rejected_unsupported';directionOutput=directionProposed??fallback;}
  const directionRow=rows.find(r=>r.id===directionProposed);
  const selectedRow=rows.find(r=>r.id===selected),candidate=candidates.find(c=>c.id===selected);
  for(const row of rows)row.selected=row.id===selected;
  return {config:cfg,command:{active,angle_deg:angle},status,reason,selected_id:selected,
    proposed_id:proposed,proposed_score:proposedRow?.fused_score??null,proposed_error_deg:proposedRow?.error_deg??null,proposal_pool_count:proposalPool.length,
    normalization:cfg.normalization,decision_rule:cfg.decision_rule,effective_temperature:cfg.normalization==='sigmoid'?1:cfg.temperature,rejection_threshold:cfg.rejection_threshold,direction_tolerance_applied:active&&cfg.decision_rule==='direction_tolerance',
    selected_error_deg:selectedRow?.error_deg??null,selected_score:selectedRow?.fused_score??null,
    selected_trajectory:candidate?candidate.poses.map(p=>[...p]):null,intent_satisfied:status==='accepted',
    fallback_used:['rejected_unsupported','rejected_low_support'].includes(status),stop_requested:status==='no_feasible_candidate',
    original_model_top1:original,safe_model_top1:fallback,pure_intent_top1:active?closest(rows):null,
    safe_pure_intent_top1:active?closest(environment):null,ungated_fused_top1:highest(rows,'fused_score'),
    model_only_output_id:fallback,direction_only_output_id:directionOutput,direction_only_proposed_id:directionProposed,direction_only_score:active?(directionRow?.intent_score??null):null,direction_only_proposed_error_deg:active?(directionRow?.error_deg??null):null,direction_only_status:directionStatus,
    environment_eligible_count:environment.length,intent_eligible_count:matched.length,rows};
}
function evaluateScene(scene,command,config,safetyMode){
  return fuseCandidates(scene.candidates,scene.candidates.map(c=>({candidate_id:c.id,...c.guards[safetyMode]})),command,config);
}
window.FusionFramework={fuseCandidates,evaluateScene,wrap};

const REASONS={footprint_outside_drivable_union:'车体轮廓超出已知可行驶区域',footprint_intersects_frozen_current_actor:'车体轮廓与冻结的当前障碍相交',footprint_intersects_time_aligned_actor:'车体轮廓与同一时刻的预测障碍相交',road_context_not_known:'道路上下文不完整',current_actor_context_missing_or_unknown:'当前障碍上下文缺失',missing_environment_assessment:'缺少环境检查记录',environment_check_not_passed:'环境检查未通过',map_query_domain_missing_or_invalid:'地图查询范围缺失',drivable_polygons_missing:'可行驶区域缺失',shapely_dependency_unavailable:'几何计算依赖不可用'};
const reasonText=reasons=>(reasons||[]).map(r=>{
  if(REASONS[r])return REASONS[r];
  const [key,index]=String(r).split(':');
  const unknown={current_dynamic_actor_velocity_missing_or_invalid:'动态障碍当前速度缺失或无效',current_actor_category_unknown:'障碍类别未知',current_actor_motion_class_missing_or_inconsistent:'障碍动静类别缺失或不一致'};
  return unknown[key]?unknown[key]+(index!==undefined?'（记录索引 '+index+'）':''):r;
}).join('；');
const table=(heads,rows,attributes=[])=>'<table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map((r,i)=>'<tr '+(attributes[i]||'')+'>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
let state={sceneId:DATA?.meta?.default_scene||'S08',command:{active:true,angle_deg:DATA?.meta?.default_command_deg??0},config:{...DEFAULTS},safetyMode:DEFAULT_SAFETY_MODE,candidateId:0};
// Road classification is browsing metadata only. It is never an input to
// fuseCandidates, environment guards, candidate pools, or score calculations.
let classificationFilter='all';
let currentScene,currentResult;
function getScene(){return DATA.scenes.find(s=>s.id===state.sceneId)||DATA.scenes[0];}

function classification(scene){
  const c=scene.classification||{};
  return {primary_type:c.primary_type||'unreviewed',primary_label:c.primary_label||'待复核',
    short_label:c.short_label||c.primary_label||'待复核',secondary_features:c.secondary_features||[],
    evidence:c.evidence||[],uncertainties:c.uncertainties||[],status:c.status||'unreviewed',
    confidence:c.confidence,lane_change_status:c.lane_change_status||'unverified',
    variable_direction_lane_status:c.variable_direction_lane_status||'unverified',
    lane_change_note:c.lane_change_note||'尚未核实变道或可变导向车道属性。',
    context_image:c.context_image,research_use:c.research_use||'待复核'};
}
function classificationCategories(){
  const categories=(DATA.meta?.classification?.categories||[]).map(c=>({id:c.id,label:c.label,count:0}));
  for(const scene of DATA.scenes){const c=classification(scene);let category=categories.find(item=>item.id===c.primary_type);if(!category){category={id:c.primary_type,label:c.primary_label,count:0};categories.push(category);}category.count++;}
  return categories.filter(c=>c.count>0);
}
function refreshSceneChoices(){
  const visible=DATA.scenes.filter(s=>classificationFilter==='all'||classification(s).primary_type===classificationFilter);
  byId('scene').innerHTML=visible.map(s=>`<option value="${esc(s.id)}">${esc(s.id)} · ${esc(classification(s).short_label)}</option>`).join('');
  if(!visible.some(s=>s.id===state.sceneId)&&visible.length)state.sceneId=visible[0].id;
  byId('scene').value=state.sceneId;
  byId('road-type-filter').value=classificationFilter;
}
function renderClassification(){
  const c=classification(currentScene),confidence={high:'较高',medium:'中等'}[c.confidence]||'待复核';
  const list=items=>'<ul>'+items.map(item=>'<li>'+esc(item)+'</li>').join('')+'</ul>';
  const image=c.context_image?`<a href="${esc(c.context_image)}" target="_blank" rel="noopener">查看广视野道路图（不含候选）</a>`:'';
  byId('classification-current').innerHTML=`<p><b>道路类型：</b><span class="classification-label">${esc(c.primary_label)}</span>${c.secondary_features.length?' · '+esc(c.secondary_features.join('；')):''}</p><p class="small muted">道路图初步分类，仅用于浏览。${esc(c.lane_change_note)}${image?'　'+image:''}</p><details class="classification-detail"><summary>本场分类依据、用途与待核实项</summary><p>道路形态分类把握：${confidence}；不表示通行或操作许可。</p><b>依据</b>${list(c.evidence)}<b>研究用途</b><p>${esc(c.research_use)}</p><b>操作条件</b><p>合法变道条件：${c.lane_change_status==='unverified'?'未核实':'需复核'}；可变导向车道：${c.variable_direction_lane_status==='unverified'?'未核实':'需复核'}。两项不由道路宽度或候选轨迹分散程度认定。</p><b>待核实</b>${list(c.uncertainties)}</details>`;
  byId('classification-counts').textContent=classificationCategories().map(c=>`${c.label} ${c.count} 场`).join('；')+'。'+(DATA.meta?.classification?.note||'');
  const text=value=>`<span class="class-text" style="display:block">${esc(value)}</span>`;
  byId('classification-table').innerHTML=table(['场景','道路主类','次级特征','研究用途','操作条件核验','道路广视野图'],DATA.scenes.map(scene=>{const item=classification(scene);return [
    `<button type="button" class="scene-jump" data-scene-id="${esc(scene.id)}" aria-label="跳转 ${esc(scene.id)}">${esc(scene.id)}</button>`,esc(item.primary_label),text(item.secondary_features.join('；')||'—'),text(item.research_use),text(`合法变道：${item.lane_change_status==='unverified'?'未核实':'需复核'}；可变导向车道：${item.variable_direction_lane_status==='unverified'?'未核实':'需复核'}。${item.lane_change_note}`),item.context_image?`<a href="${esc(item.context_image)}" target="_blank" rel="noopener">查看道路图</a>`:'—'];}),DATA.scenes.map(scene=>`data-scene-id="${esc(scene.id)}" class="${scene.id===state.sceneId?'current-scene':''}"`));
}
function jumpToScene(id){
  const scene=DATA.scenes.find(s=>s.id===id);if(!scene)return;
  if(classificationFilter!=='all'&&classification(scene).primary_type!==classificationFilter)classificationFilter='all';
  state.sceneId=id;refreshSceneChoices();render();
  byId('road-type-filter').scrollIntoView({block:'start'});
}

function projection(scene){
  const supplied=scene.map?.bounds,points=[[0,0],...scene.candidates.flatMap(c=>c.poses.map(p=>p.slice(0,2)))];
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  let xmin=Math.min(-4,...xs)-2,xmax=Math.max(15,...xs)+3,ymin=Math.min(-8,...ys)-3,ymax=Math.max(8,...ys)+3;
  if(supplied?.length===4){xmin=Math.min(xmin,supplied[0]);ymin=Math.min(ymin,supplied[1]);xmax=Math.max(xmax,supplied[2]);ymax=Math.max(ymax,supplied[3]);}
  const W=620,H=460,left=45,right=20,top=20,bottom=42;
  const scale=Math.min((W-left-right)/(ymax-ymin),(H-top-bottom)/(xmax-xmin));
  const ox=left+((W-left-right)-(ymax-ymin)*scale)/2,oy=top+((H-top-bottom)-(xmax-xmin)*scale)/2;
  const project=p=>[ox+(ymax-p[1])*scale,oy+(xmax-p[0])*scale];
  return {project,xmin,xmax,ymin,ymax,scale,x0:ox,y0:oy,w:(ymax-ymin)*scale,h:(xmax-xmin)*scale};
}
function pointsPath(points,project,close=false){return points.map((p,i)=>{const [x,y]=project(p);return(i?'L':'M')+x.toFixed(2)+','+y.toFixed(2);}).join('')+(close?'Z':'');}
function mapSvg(scene,transform,view){
  const {project,xmin,xmax,ymin,ymax,x0,y0,w,h}=transform,clip='clip-'+view;
  let s=`<defs><clipPath id="${clip}"><rect x="${x0}" y="${y0}" width="${w}" height="${h}"/></clipPath></defs><rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="#e9eef3" stroke="#c9d5e2"/><g clip-path="url(#${clip})">`;
  for(const poly of scene.map.drivable||[]){let d=pointsPath(poly.exterior,project,true);for(const hole of poly.holes||[])d+=pointsPath(hole,project,true);s+=`<path d="${d}" fill="white" fill-rule="evenodd" stroke="#bac8d6" stroke-width=".75"/>`;}
  for(const path of scene.map.paths||[])s+=`<path d="${pointsPath(path.xy,project)}" fill="none" stroke="#d7e0e9" stroke-width=".8"/>`;
  for(const actor of scene.map.actors||[]){const c=Math.cos(actor.yaw),sn=Math.sin(actor.yaw);const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([sx,sy])=>{const x=sx*actor.length/2,y=sy*actor.width/2;return [actor.x+c*x-sn*y,actor.y+sn*x+c*y];});s+=`<path d="${pointsPath(corners,project,true)}" fill="#a9b4c0" opacity=".55" stroke="#8e9dac" stroke-width=".7"><title>${esc(actor.category||'当前障碍')}</title></path>`;}
  const step=Math.max(xmax-xmin,ymax-ymin)>60?10:5;
  for(let x=Math.ceil(xmin/step)*step;x<=xmax;x+=step){const a=project([x,ymax]),b=project([x,ymin]);s+=`<path d="M${a}L${b}" stroke="#ced8e3" opacity=".55" stroke-width=".6"/>`;}
  for(let y=Math.ceil(ymin/step)*step;y<=ymax;y+=step){const a=project([xmin,y]),b=project([xmax,y]);s+=`<path d="M${a}L${b}" stroke="#ced8e3" opacity=".55" stroke-width=".6"/>`;}
  s+='</g>';
  for(let x=Math.ceil(xmin/step)*step;x<=xmax;x+=step){const p=project([x,ymax]);s+=`<text x="${p[0]-7}" y="${p[1]+3}" text-anchor="end" font-size="10" fill="#72859a">${x}</text>`;}
  for(let y=Math.ceil(ymin/step)*step;y<=ymax;y+=step){const p=project([xmin,y]);s+=`<text x="${p[0]}" y="${p[1]+16}" text-anchor="middle" font-size="10" fill="#72859a">${y}</text>`;}
  s+=`<text x="${x0+w/2}" y="449" text-anchor="middle" font-size="11" fill="#657d93">横向 y / m（左为正）</text><text x="14" y="${y0+h/2}" text-anchor="middle" transform="rotate(-90 14 ${y0+h/2})" font-size="11" fill="#657d93">前向 x / m</text>`;
  return s;
}
function originAndRay(transform){
  const {project,xmax,xmin,ymax,ymin}=transform,o=project([0,0]);let s='';
  if(state.command.active){const theta=state.command.angle_deg*Math.PI/180,len=Math.min(32,Math.max(10,(Math.max(xmax-xmin,ymax-ymin))*.65)),p=project([len*Math.cos(theta),len*Math.sin(theta)]);s+=`<path d="M${o}L${p}" stroke="#25384b" stroke-width="1.5" stroke-dasharray="8 3 2 3" fill="none"/><text x="${p[0]}" y="${p[1]-8}" text-anchor="middle" fill="#25384b" font-size="11">指令 ${state.command.angle_deg}°</text>`;}
  s+=`<path d="M${o[0]},${o[1]-9}L${o[0]-6},${o[1]+6}L${o[0]+6},${o[1]+6}Z" fill="#193d5c"/><text x="${o[0]+8}" y="${o[1]+15}" font-size="10" fill="#4d647a">当前起点</text>`;return s;
}
function drawPlots(){
  const scene=currentScene,result=currentResult,t=projection(scene),project=t.project;
  let all=mapSvg(scene,t,'all');
  for(const candidate of scene.candidates){const row=result.rows.find(r=>r.id===candidate.id),xy=[[0,0],...candidate.poses],p=project(xy.at(-1));all+=`<path class="candidate" data-id="${candidate.id}" d="${pointsPath(xy,project)}" fill="none" stroke="${row.eligible?'#7992ac':'#b0b9c3'}" stroke-width="1.3" opacity=".7" ${row.eligible?'':'stroke-dasharray="4 3"'}><title>#${candidate.id}，${signed(row.direction_deg)}°，${row.eligible?'环境检查通过':'环境检查未通过'}</title></path><circle data-id="${candidate.id}" cx="${p[0]}" cy="${p[1]}" r="3.4" fill="${row.eligible?'#7992ac':'#b0b9c3'}"><title>查看 #${candidate.id}</title></circle>`;}
  const focus=scene.candidates.find(c=>c.id===state.candidateId);
  if(focus){const xy=[[0,0],...focus.poses],end=project(xy.at(-1));all+=`<path d="${pointsPath([xy[0],xy.at(-1)],project)}" fill="none" stroke="#bc872b" stroke-width="1" stroke-dasharray="2 3"/><path data-id="${focus.id}" d="${pointsPath(xy,project)}" fill="none" stroke="#c5963e" stroke-width="3"/>`;for(let i=0;i<focus.poses.length;i++){const p=project(focus.poses[i]);all+=`<circle data-id="${focus.id}" cx="${p[0]}" cy="${p[1]}" r="3.3" fill="white" stroke="#bd8c34" stroke-width="1.6"><title>#${focus.id} · ${(i+1)*.5} 秒 (${fmt(focus.poses[i][0])}, ${fmt(focus.poses[i][1])})</title></circle>`;}all+=`<text x="${end[0]+8}" y="${end[1]-7}" font-size="13" font-weight="bold" fill="#a67320">#${focus.id}</text>`;}
  all+=originAndRay(t);byId('all-plot').innerHTML=all;
  let compared=mapSvg(scene,t,'result');
  const outputs=[{id:result.model_only_output_id,color:'#6b7280',width:7,name:'模型'},{id:result.direction_only_output_id,color:'#138b80',width:4.8,name:'纯方向'},{id:result.selected_id,color:'#2868bb',width:2.7,name:'融合'}];
  const grouped=new Map();for(const out of outputs){if(out.id===null)continue;const candidate=scene.candidates.find(c=>c.id===out.id);compared+=`<path data-id="${out.id}" d="${pointsPath([[0,0],...candidate.poses],project)}" fill="none" stroke="${out.color}" stroke-width="${out.width}" opacity=".88"><title>${out.name}输出 #${out.id}</title></path>`;if(!grouped.has(out.id))grouped.set(out.id,[]);grouped.get(out.id).push(out.name);}
  for(const [id,names] of grouped){const candidate=scene.candidates.find(c=>c.id===id),p=project(candidate.poses.at(-1));compared+=`<circle data-id="${id}" cx="${p[0]}" cy="${p[1]}" r="4" fill="white" stroke="#335778"/><text x="${p[0]+7}" y="${p[1]-8}" font-size="11" fill="#284c6e">#${id} ${esc(names.join('/'))}</text>`;}
  compared+=originAndRay(t);byId('result-plot').innerHTML=compared;
}

function guardDiagnostics(candidate){
  const guard=candidate.guards[state.safetyMode]||{},diagnostic=guard.diagnostics||{},event=diagnostic.first_failure||diagnostic.first_unknown;
  const element=byId('candidate-guard-diagnostic');
  if(state.safetyMode==='offline_pdm'){element.innerHTML='<b>事后指标回放</b><p>使用已保存的 NC / DAC / TTC；此模式不提供本地恒速检查的相交时刻。</p>';return;}
  const predicted=state.safetyMode==='current_environment_prediction';
  let html=`<b>${predicted?'恒速预测检查诊断':'冻结障碍检查诊断'}</b>`;
  if(!event){html+=`<p>${guard.eligible?'本模式未记录失败或相交。':'未提供可定位的首次相交时刻；请查看上方未通过原因。'}</p>`;}
  else{
    const actorIds=event.actor_annotation_indices||[],categories=event.actor_categories||[];
    html+=`<p>${actorIds.length?'首次几何相交':'首次检查失败'}：t = <b>${fmt(event.time_s,3)} s</b>；自车后轴位置 (${fmt(event.x)}, ${fmt(event.y)}) m。</p>`;
    if(event.actor_predictions?.length){
      const xy=value=>Array.isArray(value)&&value.length>=2?`(${fmt(value[0])}, ${fmt(value[1])})`:'未提供';
      html+='<ul>'+event.actor_predictions.map(actor=>`<li>障碍 annotation #${esc(actor.annotation_index)} · ${esc(actor.category)}（${actor.is_dynamic_category?'动态类别':'静态类别'}）<br>当前中心 ${xy(actor.current_center_xy_ego_m)} m → t=${fmt(actor.prediction_time_s,3)} s 的检查中心 ${xy(actor.predicted_center_xy_ego_m)} m。<br>记录速度 ${xy(actor.velocity_xy_ego_mps)} m/s；本模式采用 ${xy(actor.applied_velocity_xy_ego_mps)} m/s。</li>`).join('')+'</ul>';
    }else if(actorIds.length){html+='<p>涉及障碍：'+actorIds.map((id,i)=>`annotation #${esc(id)} · ${esc(categories[i]||'类别未提供')}`).join('；')+'。当前/预测中心及速度未随本记录提供。</p>';}
  }
  html+=`<p class="muted">${predicted?'静态对象固定；动态类别按当前速度外推、保持当前朝向，并与自车同一时刻比较。':'所有对象固定在当前中心，即使源记录包含运动速度。'} 这是该假设下的采样检查，不是官方 TTC 或完整安全保证。</p>`;
  element.innerHTML=html;
}
function inspector(){
  const candidate=currentScene.candidates.find(c=>c.id===state.candidateId),row=currentResult.rows.find(r=>r.id===state.candidateId),end=candidate.poses.at(-1);
  byId('candidate').value=state.candidateId;
  const cfg=state.config;
  const directionLine=row.direction_deg===null?'净位移不足 1 m，不定义 θ。':`θ = atan2(${fmt(end[1])}, ${fmt(end[0])}) = <b>${signed(row.direction_deg)}°</b>`;
  const modelLine=cfg.normalization==='sigmoid'?`M = sigmoid(${fmt(row.model_logit,4)}) = <b>${fmt(row.model_score,6)}</b>（原生，不用 T）`:`M = softmax(z / ${cfg.temperature}) = <b>${fmt(row.model_score,6)}</b>（完整 20 条）`;
  byId('candidate-formula').innerHTML=`<b>#${candidate.id}</b> 的 4 秒终点：(${fmt(end[0])}, ${fmt(end[1])}) m<br>${directionLine}<br>${modelLine}<br>`+(state.command.active?
    (row.error_deg===null?'方向不可定义，意图贡献按 0 处理。':`δ = |wrap(θ − ${state.command.angle_deg}°)| = <b>${fmt(row.error_deg)}°</b><br>I = exp[−½(δ / ${cfg.sigma_deg}°)²] = <b>${fmt(row.intent_score,4)}</b>`)+
    `<br><b>F = (1 − α) × M + α × I</b><br>= ${fmt(1-cfg.alpha,2)} × ${fmt(row.model_score,4)} + ${fmt(cfg.alpha,2)} × ${fmt(row.intent_score||0,4)} = <b>${fmt(row.fused_score,4)}</b><br><span class="small muted">模型项 ${fmt((1-cfg.alpha)*row.model_score,4)} + 意图项 ${fmt(cfg.alpha*(row.intent_score||0),4)}；页面显示值已四舍五入。</span>`:
    `外部指令已关闭：不计算意图分，F = M = <b>${fmt(row.model_score,4)}</b>。`);
  const metric=(label,value)=>`<div class="metric"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
  const modelLabel=cfg.normalization==='sigmoid'?'原生 sigmoid 模型分 M':'20 条 softmax 模型偏好 M';
  byId('candidate-metrics').innerHTML=metric('原始模型 logit',fmt(row.model_logit,4))+metric(modelLabel,fmt(row.model_score,4))+metric('外部意图分 I',fmt(row.intent_score,4))+metric('融合分 F',fmt(row.fused_score,4))+metric('终点净位移',fmt(row.net_displacement_m)+' m')+metric('4 秒轨迹总长',fmt(row.length_m)+' m');
  const reasons=reasonText(row.safety_reasons);
  byId('candidate-eligibility').innerHTML=`<b class="${row.eligible?'good':'bad'}">${row.eligible?'当前检查通过':'当前检查未通过'}</b>`+(reasons?'：'+esc(reasons):'')+'<br>'+(!state.command.active?'未启用指令；选择资格仅由环境检查决定。':row.error_deg===null?'净位移不足 1 m，方向不可定义。':cfg.decision_rule==='score_reject'?`本条 F ${fmt(row.fused_score,6)} ${row.fused_score>cfg.rejection_threshold?'＞':'≤'} ρ ${fmt(cfg.rejection_threshold,4)}：${row.fused_score>cfg.rejection_threshold?'分数过线':'未严格过线'}；${row.eligible?'方向可定义，进入最高融合提议的比较。':'环境检查未通过，不进入提议池。'} 不使用 10° 硬方向门槛，等于 ρ 也拒绝。`:`历史方向容差 ${state.config.tolerance_deg}°：${row.error_deg<=state.config.tolerance_deg?'在容差内':'超出容差'}；${row.selection_eligible?'可参与融合竞争':'不能赢得本次意图选择'}。`);
  guardDiagnostics(candidate);
  const m=candidate.offline_metrics||{};byId('candidate-offline').textContent=`离线已存 NC / DAC / TTC：${m.NC??'—'} / ${m.DAC??'—'} / ${m.TTC??'—'}。`+(state.safetyMode==='offline_pdm'?'当前为事后指标回放模式，不可作为在线接口使用。':'这里只作报告，当前检查不读取这些指标进行选择。');
  byId('points').innerHTML=table(['时间 s','前向 x / m','横向 y / m','保存的 yaw / rad（未用于方向评分）'],candidate.poses.map((p,i)=>[fmt((i+1)*.5,1),fmt(p[0],3),fmt(p[1],3),fmt(p[2],3)]));
}
function acceptanceText(status){return {accepted:'接受请求',no_command:'无外部指令',rejected_low_support:'拒绝：共同支持不足',rejected_unsupported:'拒绝：无容差内候选',no_feasible_candidate:'无环境合格输出'}[status]||status;}
function methodsTable(){
  const r=currentResult,active=state.command.active,scoreMode=state.config.decision_rule==='score_reject';
  const modelRow=r.rows.find(row=>row.id===r.model_only_output_id);
  const methods=[{name:'模型偏好（同检查）',raw:r.original_model_top1,proposed:r.model_only_output_id,score:modelRow?.model_score,id:r.model_only_output_id,note:'忽略外部指令，在检查通过集合 G 内取模型最高分。'},
    {name:'纯方向匹配',raw:r.pure_intent_top1,proposed:r.direction_only_proposed_id,score:r.direction_only_score,id:r.direction_only_output_id,note:!active?'无指令，使用相同的模型选择。':scoreMode?`α = 1 对照，只用意图分 I，与同一 ρ 比较：${acceptanceText(r.direction_only_status)}；未过线则回退。`:`历史规则：${acceptanceText(r.direction_only_status)}；无容差内候选时回退。`},
    {name:'加权融合',raw:r.ungated_fused_top1,proposed:r.proposed_id,score:r.proposed_score,id:r.selected_id,note:r.reason}];
  byId('methods').innerHTML=table(['方法','全20未过滤首选','检查后提议','提议分','实际输出','输出方向°','指令误差°','输出说明'],methods.map(m=>{const row=r.rows.find(row=>row.id===m.id);return [`<span class="methodname">${esc(m.name)}</span>`,m.raw===null?'—':'#'+m.raw,idText(m.proposed),fmt(m.score,6),idText(m.id),signed(row?.direction_deg),fmt(row?.error_deg),`<span class="wrap" style="display:block">${esc(m.note)}</span>`];}));
  byId('method-note').textContent='三种实际输出使用同一环境检查。提议分分别为模型 M、纯方向 I、融合 F。提议被拒绝后，实际输出可能回退到模型首选；三种方法同 ID 时，右图叠画。拒绝请求不等于请求不合理，也不等于停车。';
}
function normalizationComparison(){
  const results=['softmax','sigmoid'].map(normalization=>({normalization,result:evaluateScene(currentScene,state.command,{...state.config,normalization,decision_rule:'score_reject'},state.safetyMode)}));
  byId('normalization-comparison').innerHTML=table(['模型分数方法','最高融合提议','最大融合分 F','提议角误差°','拒绝基准 ρ','接受状态','实际输出'],results.map(({normalization,result:r})=>[
    normalization==='sigmoid'?'原生 sigmoid(z)，固定原生尺度':`softmax(z / ${state.config.temperature})，完整 20 条`,idText(r.proposed_id),fmt(r.proposed_score,6),fmt(r.proposed_error_deg),fmt(state.config.rejection_threshold,4),acceptanceText(r.status),idText(r.selected_id)]),results.map(item=>`data-normalization="${item.normalization}"`));
  byId('normalization-comparison-note').textContent=`此表固定使用“统一融合分＋拒绝”规则，保持同一场景、指令、环境集合、α=${state.config.alpha}、σ=${state.config.sigma_deg}°、ρ=${state.config.rejection_threshold}；要求 F > ρ，等于也拒绝。`+(state.config.decision_rule==='direction_tolerance'?' 上方当前输出使用历史容差规则，故不必与此表一致。':'');
  window.FusionFramework.current.normalizationComparison=results;
}

function sensitivityGrid(){
  const enabled=state.command.active&&state.config.decision_rule==='score_reject';
  const mode=state.config.normalization==='sigmoid'?'原生 sigmoid(z)':`完整 20 条 softmax，T=${state.config.temperature}`;
  const guard={'current_environment_prediction':'当前环境＋恒速预测','current_environment_geometry':'冻结当前障碍对照','offline_pdm':'离线 NC/DAC/TTC 回放'}[state.safetyMode];
  byId('sensitivity-context').textContent=`${state.sceneId} · 当前指令 ${state.command.active?state.command.angle_deg+'°':'未启用'} · ${mode} · ${guard} · σ=${state.config.sigma_deg}°。仅比较这一场景与这一指令，不汇总其他场景。`;
  byId('sensitivity-unavailable').textContent=enabled?'':!state.command.active?'网格已禁用：先启用外部方向指令。无指令时直接使用模型选择，不判断请求接受。':'网格已禁用：当前是历史方向容差对照。切换为“统一融合分＋拒绝”后可使用。';
  if(!enabled){byId('sensitivity-grid').innerHTML='';byId('sensitivity-grid-note').textContent='';window.FusionFramework.current.sensitivity={enabled:false};return;}
  const same=(a,b)=>Math.abs(a-b)<1e-12;
  const axis=current=>[...Array.from({length:11},(_,i)=>i/10),current].sort((a,b)=>a-b).filter((v,i,arr)=>!i||!same(v,arr[i-1]));
  const alphas=axis(state.config.alpha),rhos=axis(state.config.rejection_threshold),cells=[];
  const cellText=r=>r.status==='accepted'?'接受 #'+r.proposed_id:r.status==='no_feasible_candidate'?'无输出':'回退 #'+r.selected_id;
  const className=r=>r.status==='accepted'?'accepted':r.status==='no_feasible_candidate'?'no-output':'rejected';
  let html='<table><thead><tr><th>α \\ ρ</th>'+rhos.map(rho=>`<th class="${same(rho,state.config.rejection_threshold)?'current-axis':''}">${rho}</th>`).join('')+'</tr></thead><tbody>';
  for(const alpha of alphas){
    html+=`<tr><th class="${same(alpha,state.config.alpha)?'current-axis':''}">${alpha}</th>`;
    for(const rho of rhos){
      const result=evaluateScene(currentScene,state.command,{...state.config,alpha,rejection_threshold:rho},state.safetyMode),current=same(alpha,state.config.alpha)&&same(rho,state.config.rejection_threshold);
      const margin=result.proposed_score===null?null:result.proposed_score-rho;
      const title=`α=${alpha}，ρ=${rho}；最高融合提议 ${idText(result.proposed_id)}，F=${fmt(result.proposed_score,6)}，F−ρ=${fmt(margin,6)}；${result.status==='accepted'?'接受请求':result.status==='no_feasible_candidate'?'无环境合格输出':'拒绝请求并回退'}，实际输出 ${idText(result.selected_id)}。`;
      html+=`<td><button type="button" class="sensitivity-cell ${className(result)}${current?' current':''}" data-alpha="${alpha}" data-rho="${rho}" data-status="${result.status}" data-proposed-id="${result.proposed_id??''}" data-output-id="${result.selected_id??''}" aria-pressed="${current}" aria-label="${esc(title)}" title="${esc(title)}">${esc(cellText(result))}</button></td>`;
      cells.push({alpha,rho,proposed_id:result.proposed_id,proposed_score:result.proposed_score,status:result.status,selected_id:result.selected_id,current});
    }
    html+='</tr>';
  }
  html+='</tbody></table>';byId('sensitivity-grid').innerHTML=html;
  const current=cells.find(c=>c.current),margin=current.proposed_score===null?null:current.proposed_score-current.rho;
  byId('sensitivity-grid-note').textContent=`当前格：α=${current.alpha}，ρ=${current.rho}；最高融合分 ${fmt(current.proposed_score,6)}，F−ρ=${fmt(margin,6)}。基础行列步长 0.1；额外保留当前参数对应行列。显示值已四舍五入，严格 F > ρ 的判断使用未舍入数值。`;
  window.FusionFramework.current.sensitivity={enabled:true,alphas,rhos,cells};
}

function ranking(){
  const scoreMode=state.config.decision_rule==='score_reject';
  const sorted=[...currentResult.rows].sort((a,b)=>b.fused_score-a.fused_score||a.id-b.id);
  byId('ranking').innerHTML=table(['ID','4秒方向°','模型 logit','模型 M','角误差 δ°','意图 I','融合 F','环境检查',scoreMode?'分数 > ρ':'历史容差内','达到接受条件','未满足条件'],sorted.map(row=>[
    '#'+row.id+(row.selected?' · 输出':'')+(row.id===currentResult.proposed_id?' · 提议':''),signed(row.direction_deg),fmt(row.model_logit,4),fmt(row.model_score,6),fmt(row.error_deg),fmt(row.intent_score,4),fmt(row.fused_score,6),row.eligible?'通过':'未通过',!state.command.active?'—':row.error_deg===null?'方向未定义':scoreMode?(row.fused_score>state.config.rejection_threshold?'是':'否'):(row.error_deg<=state.config.tolerance_deg?'是':'否'),row.selection_eligible?'是':'否',esc(reasonText(row.safety_reasons)||(!state.command.active?'—':row.error_deg===null?'方向未定义':scoreMode?(row.fused_score<=state.config.rejection_threshold?'融合分未严格高于 ρ':'—'):(row.error_deg>state.config.tolerance_deg?'超出指令容差':'—')))]),sorted.map(row=>`data-id="${row.id}" class="${row.id===state.candidateId?'focus ':''}${!row.eligible?'excluded':''}"`));
}

function explain(){
  const c=state.config,scoreMode=c.decision_rule==='score_reject',native=c.normalization==='sigmoid';
  const modelFormula=native?'Mᵢ = sigmoid(zᵢ) = 1 / (1 + exp(−zᵢ))':`Mᵢ = exp(zᵢ/${c.temperature}) / Σⱼ exp(zⱼ/${c.temperature})`;
  const modelExplanation=native?'原生 sigmoid 逐条计算，固定使用原始 logit，不使用温度 T，也不要求 20 条分数相加等于 1。':'softmax 先对完整 20 条归一化，再施加环境检查；它表示这批候选之间的相对偏好，会受其他候选的分数影响。';
  const ruleExplanation=scoreMode?`<h3>最高融合提议，再决定接受</h3><p>G = 环境检查通过的候选<br>提议池 = G 中方向可定义的候选<br>i* = argmax Fᵢ；仅当 Fᵢ* > ρ 才接受。</p><p>ρ = ${c.rejection_threshold}，等于也拒绝。没有提议或最高分未过线：拒绝本次请求，回退 G 内模型首选。此规则不使用 10° 硬方向门槛；接受不等于满足某个固定角误差，也不等于安全保证。</p>`:`<h3>历史规则：方向容差先筛选</h3><p>G = 环境检查通过的候选<br>I = G 中方向可定义且 δ ≤ ${c.tolerance_deg}° 的候选<br>有指令且 I 非空：在 I 中取 F 最大者。</p><p>I 为空但 G 非空：拒绝意图，回退模型首选。该规则不使用 ρ；与 softmax 配合才是原页面的历史配置。</p>`;
  byId('formula-explanation').innerHTML=`<div><h3>终点方向与意图分</h3><p>θᵢ = atan2(yᵢ(4s), xᵢ(4s))<br>δᵢ = |wrap(θᵢ − θₕ)|，范围 0°…180°<br>Iᵢ = exp[−½(δᵢ / σ)²]</p><p>σ = ${c.sigma_deg}°。净位移不足 1 m 不定义方向；不以最后一段或预测 yaw 替代。意图分不再做归一化。</p></div><div><h3>当前模型分数与融合</h3><p>${modelFormula}<br>Fᵢ = (1 − α) Mᵢ + α Iᵢ</p><p>${modelExplanation} α = ${c.alpha}，不是人的控制权百分比；M、F 均不能直接解释为安全概率。关闭指令时 F = M，并跳过意图接受门槛。仅令 α = 0 不等于关闭指令。</p></div><div>${ruleExplanation}<p>G 为空：返回空轨迹并请求停车处理，页面不执行停车。主判据完全并列时取较小 ID。</p></div>`;
  const calibration=DATA.meta?.calibration||{};
  byId('calibration-note').textContent=`ρ=${c.rejection_threshold} 是演示值，未标定；独立标注 ${calibration.independent_label_count??0} 条。16 场用于开发，不能据此宣称已验证拒绝阈值。`;
  byId('calibration-links').innerHTML=[[calibration.annotation_page,'开始独立标注'],[calibration.template,'下载标注模板'],[calibration.instructions,'标定说明'],[calibration.report,'标定状态']].filter(([href])=>href).map(([href,label])=>`<a href="${esc(href)}">${label}</a>`).join('');
  byId('boundaries-body').innerHTML='<p>全部 16 场、每场 20 条来自已保存的 DiffusionDrive 输出，未来 8 点对应 0.5…4 秒。当前演示只读这些数据，不重训、不重新生成、不修改轨迹。</p><p><b>拒绝的含义</b>：当前评分与演示门槛不足以支持本次请求，就回退到同环境检查下的模型首选。这不证明人的请求不合理、轨迹错误或危险，也不是停车。只有环境合格集合为空时才返回空轨迹，等待独立控制器处理。</p><p><b>标定仍未完成</b>：原生 sigmoid 及 softmax 是两种模型分数定义，统一数值 ρ 便于演示，不能证明二者已经具有同等误接受率。阈值需要独立标注或目标执行结果；不能用模型分数、融合分数或角误差给自身生成“正确”标签。高意图分仍可能补偿低模型分；加权和不强制模型与意图各自过线。</p><p><b>当前环境＋恒速预测检查</b>：静态对象保持当前位置；动态类别依据已知当前速度直线外推，朝向保持当前值。将预测障碍与同一时刻的自车轮廓比较，并检查当前道路区域。它不预测加速、刹车、转弯或交互，不是官方 TTC，也不提供完整安全保证；缺少必要动态速度时不会当作已通过。</p><p><b>冻结当前障碍对照</b>：把所有障碍固定在当前位置，包括动态类别，用于对照该简化假设的影响；这不等于障碍真实静止。</p><p><b>离线 NC/DAC/TTC 模式</b>：使用已有事后评价记录用于回放，不能直接作为在线控制信息。三种检查的结果含义不同。</p><p>灰色地图障碍框仅绘制当前位置。恒速模式的未来检查中心可在首次相交诊断中查看，页面没有把未来障碍全部画在图上。</p><p>16 场是开发场景。更高意图分或更小角误差是选择规则所优化的量，不能单独证明独立道路目标完成、真实执行安全或稳定优势。</p><p>后续接口分开接入：候选生成器提供 poses 与 model_logit；输入模块提供 active 与当前车体坐标下的 angle_deg；环境模块提供每个 candidate_id 的 eligible 与 reasons；选择模块返回提议、接受/回退状态与 selected_trajectory 或空轨迹。停车控制器与闭环执行尚未接入。</p><p>这里的“单条融合结果”是该条轨迹的模型分、意图分与融合分；最终选择仍输出原来的坐标。方向指令没有表达速度目标，自车初速偏差不参与本版评分或筛选；障碍当前速度仅用于恒速环境检查。</p>';
}

function render(){
  currentScene=getScene();currentResult=evaluateScene(currentScene,state.command,state.config,state.safetyMode);
  if(!currentScene.candidates.some(c=>c.id===state.candidateId))state.candidateId=currentResult.selected_id??currentScene.candidates[0].id;
  window.FusionFramework.current={sceneId:state.sceneId,safetyMode:state.safetyMode,command:{...state.command},config:{...state.config},result:currentResult};
  byId('scene-context').textContent=`当前自车速度 ${fmt(currentScene.ego_speed_mps)} m/s；8 个未来点覆盖 4 秒。`;
  byId('guard-note').textContent=state.safetyMode==='offline_pdm'?'事后评价回放，不能用于在线选择。':state.safetyMode==='current_environment_prediction'?'动态类别按当前速度外推、固定朝向；静态对象固定。不是官方 TTC 或完整安全保证。':'所有当前障碍固定不动，包括动态类别；仅作简化假设对照。';
  byId('guard-note').className='small '+(state.safetyMode==='offline_pdm'?'warn':'muted');
  const scoreMode=state.config.decision_rule==='score_reject',native=state.config.normalization==='sigmoid';
  byId('temperature').disabled=native;byId('tolerance').disabled=scoreMode;byId('rejection-threshold').disabled=!scoreMode;
  byId('score-mode-note').textContent=(native?'原生 sigmoid(z) 固定使用原始 logit，不使用 T。':`softmax 在完整 20 条上归一化，T=${state.config.temperature}。`)+(scoreMode?' 当前规则取消硬方向容差：最高融合提议 F > ρ 才接受，等于也拒绝。':' 当前是历史方向容差规则，ρ 不参与上方输出；配合 softmax 可复现历史配置。');
  const labels={accepted:scoreMode?'融合分严格过线，接受本次请求':'容差内接受本次方向指令',no_command:'外部指令未启用',rejected_low_support:'融合分未严格过线，拒绝请求并回退模型偏好',rejected_unsupported:'拒绝本次方向意图，回退模型偏好',no_feasible_candidate:'20 条候选仍在，但当前检查没有合格输出'};
  const result=currentResult,status=byId('decision-status');status.className='status '+(result.status==='no_feasible_candidate'?'no-output':result.status.startsWith('rejected')?'caution':'');
  const proposal=state.command.active?`<span class="pill">提议 ${idText(result.proposed_id)} · F ${fmt(result.proposed_score,6)}</span>`:'';
  const qualified=state.command.active?`<span class="pill">${scoreMode?'达到接受条件':'同时满足方向容差'} ${result.intent_eligible_count}/20</span>`:'';
  status.innerHTML=`<b>${labels[result.status]}${result.selected_id!==null?' · 实际输出 #'+result.selected_id:''}</b><span class="pill">环境检查通过 ${result.environment_eligible_count}/20</span>${proposal}${qualified}<br><span class="small">${result.status==='no_feasible_candidate'?'原模型未过滤首选仍为 #'+result.original_model_top1+'；空轨迹是当前检查后的选择结果，不是模型没有生成候选。 ':''}${esc(result.reason)}${result.selected_error_deg!==null?' 当前实际输出与指令相差 '+fmt(result.selected_error_deg)+'°。':''}${result.fallback_used?' 拒绝不等于请求不合理或停车。':''}</span>`;
  byId('direction-range').disabled=!state.command.active;byId('direction-number').disabled=!state.command.active;
  renderClassification();drawPlots();inspector();methodsTable();ranking();normalizationComparison();sensitivityGrid();explain();
}
function focusCandidate(id){if(!currentScene.candidates.some(c=>c.id===id))return;state.candidateId=id;drawPlots();inspector();ranking();}
function readControl(id,min,max,fallback){const n=Number(byId(id).value);return byId(id).value!==''&&Number.isFinite(n)?Math.min(max,Math.max(min,n)):fallback;}
function initialize(){
  if(!DATA?.scenes?.length)throw new Error('缺少内嵌场景数据，请先运行 build_page.py。');
  byId('road-type-filter').innerHTML=`<option value="all">全部道路类型（${DATA.scenes.length}）</option>`+classificationCategories().map(c=>`<option value="${esc(c.id)}">${esc(c.label)}（${c.count}）</option>`).join('');
  byId('candidate').innerHTML=Array.from({length:20},(_,id)=>`<option value="${id}">#${id}</option>`).join('');
  function reset(){state={sceneId:DATA.meta?.default_scene||'S08',command:{active:true,angle_deg:DATA.meta?.default_command_deg??0},config:{...DEFAULTS},safetyMode:DEFAULT_SAFETY_MODE,candidateId:0};classificationFilter='all';refreshSceneChoices();byId('command-enabled').checked=true;byId('direction-range').value=state.command.angle_deg;byId('direction-number').value=state.command.angle_deg;byId('safety-mode').value=state.safetyMode;for(const [id,key] of [['alpha','alpha'],['temperature','temperature'],['sigma','sigma_deg'],['tolerance','tolerance_deg'],['rejection-threshold','rejection_threshold'],['normalization','normalization'],['decision-rule','decision_rule']])byId(id).value=state.config[key];const scene=getScene(),r=evaluateScene(scene,state.command,state.config,state.safetyMode);state.candidateId=r.proposed_id??r.selected_id??0;render();}
  byId('reset').addEventListener('click',reset);
  byId('scene').addEventListener('change',e=>{state.sceneId=e.target.value;render();});
  byId('road-type-filter').addEventListener('change',e=>{classificationFilter=e.target.value;refreshSceneChoices();render();});
  byId('classification-table').addEventListener('click',e=>{if(e.target.closest('a'))return;const row=e.target.closest('[data-scene-id]');if(row)jumpToScene(row.dataset.sceneId);});
  byId('command-enabled').addEventListener('change',e=>{state.command.active=e.target.checked;render();});
  byId('direction-range').addEventListener('input',e=>{state.command.angle_deg=Number(e.target.value);byId('direction-number').value=e.target.value;render();});
  byId('direction-number').addEventListener('change',()=>{state.command.angle_deg=readControl('direction-number',-90,90,state.command.angle_deg);byId('direction-range').value=state.command.angle_deg;byId('direction-number').value=state.command.angle_deg;render();});
  for(const [id,key,min,max] of [['alpha','alpha',0,1],['temperature','temperature',.1,10],['sigma','sigma_deg',1,90],['tolerance','tolerance_deg',0,90],['rejection-threshold','rejection_threshold',0,1]])byId(id).addEventListener('change',()=>{state.config[key]=readControl(id,min,max,state.config[key]);byId(id).value=state.config[key];render();});
  for(const [id,key] of [['normalization','normalization'],['decision-rule','decision_rule']])byId(id).addEventListener('change',e=>{state.config[key]=e.target.value;render();});
  byId('safety-mode').addEventListener('change',e=>{state.safetyMode=e.target.value;render();});
  byId('candidate').addEventListener('change',e=>focusCandidate(Number(e.target.value)));
  byId('sensitivity-grid').addEventListener('click',e=>{const cell=e.target.closest('button[data-alpha][data-rho]');if(!cell||!state.command.active||state.config.decision_rule!=='score_reject')return;state.config.alpha=Number(cell.dataset.alpha);state.config.rejection_threshold=Number(cell.dataset.rho);byId('alpha').value=state.config.alpha;byId('rejection-threshold').value=state.config.rejection_threshold;render();});
  for(const id of ['all-plot','result-plot','ranking'])byId(id).addEventListener('click',e=>{const node=e.target.closest('[data-id]');if(node)focusCandidate(Number(node.dataset.id));});
  reset();
}
try{initialize();}catch(error){byId('fatal').style.display='block';byId('fatal').textContent='页面无法计算：'+error.message;throw error;}
})();
