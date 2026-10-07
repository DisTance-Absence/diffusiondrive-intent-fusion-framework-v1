import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const base=path.dirname(fileURLToPath(import.meta.url)),require=createRequire(import.meta.url);
const {chromium}=require(require.resolve('playwright',{paths:[base,path.join(base,'../scene-choice-inventory-v3')]}));
const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
const page=await browser.newPage({viewport:{width:1500,height:1100},deviceScaleFactor:1});
const errors=[];page.on('pageerror',error=>errors.push(String(error)));
await page.goto(pathToFileURL(path.join(base,'index.html')).href);
await page.waitForFunction(()=>window.FusionFramework?.current?.result?.rows.length===20);
await fs.mkdir(path.join(base,'work'),{recursive:true});

const legacyData=JSON.parse(await fs.readFile(path.join(base,'work/pre_unified_rejection/data.json'),'utf8'));
const fixtureAudit=await page.evaluate(legacyFixtures=>{
  const mismatches=[];let scalarChecks=0,rowChecks=0;
  function compare(actual,expected,location){
    if(typeof expected==='number'){scalarChecks++;if(!Number.isFinite(actual)||Math.abs(actual-expected)>1e-9*Math.max(1,Math.abs(expected)))mismatches.push({location,actual,expected});}
    else if(expected===null||typeof expected!=='object'){scalarChecks++;if(actual!==expected)mismatches.push({location,actual,expected});}
    else if(Array.isArray(expected)){if(!Array.isArray(actual)||actual.length!==expected.length){mismatches.push({location,reason:'array length'});return;}expected.forEach((v,i)=>compare(actual[i],v,location+'['+i+']'));}
    else for(const [key,value] of Object.entries(expected))compare(actual?.[key],value,location+'.'+key);
  }
  for(const [i,f] of FUSION_DATA.fixtures.entries()){
    const scene=FUSION_DATA.scenes.find(s=>s.id===f.scene_id);
    const result=FusionFramework.evaluateScene(scene,f.command,f.config,f.safety_mode);
    compare(result,f.result,'fixture '+i);rowChecks+=result.rows.length;
  }
  const newScalarChecks=scalarChecks,newRowChecks=rowChecks;
  for(const [i,f] of legacyFixtures.entries()){
    const scene=FUSION_DATA.scenes.find(s=>s.id===f.scene_id);
    const result=FusionFramework.evaluateScene(scene,f.command,{...f.config,normalization:'softmax',decision_rule:'direction_tolerance'},f.safety_mode);
    compare(result,f.result,'legacy fixture '+i);rowChecks+=result.rows.length;
  }
  return {fixtures:FUSION_DATA.fixtures.length,rowChecks:newRowChecks,scalarChecks:newScalarChecks,legacyFixtures:legacyFixtures.length,legacyRowChecks:rowChecks-newRowChecks,legacyScalarChecks:scalarChecks-newScalarChecks,mismatches};
},legacyData.fixtures);
if(fixtureAudit.mismatches.length)throw new Error('Python/JS mismatch: '+JSON.stringify(fixtureAudit.mismatches.slice(0,12)));

const initial=await page.evaluate(()=>({scene:document.getElementById('scene').value,mode:document.getElementById('safety-mode').value,defaultMode:FUSION_DATA.meta.default_safety_mode,...FusionFramework.current}));
if(initial.scene!=='S08'||initial.mode!=='current_environment_prediction'||initial.mode!==initial.defaultMode||!initial.command.active||initial.command.angle_deg!==0)throw new Error('Incorrect starting example');
if(initial.config.normalization!=='sigmoid'||initial.config.decision_rule!=='score_reject'||initial.config.rejection_threshold!==.75)throw new Error('Native sigmoid and uncalibrated score rejection must be defaults');
if(!await page.locator('#temperature').isDisabled()||!await page.locator('#tolerance').isDisabled()||await page.locator('#rejection-threshold').isDisabled())throw new Error('Inactive scoring controls should be disabled');
if(!await page.locator('#calibration-note').textContent().then(t=>t.includes('未标定')&&t.includes('独立标注 0 条')))throw new Error('Missing uncalibrated/zero independent labels status');
if(await page.locator('#normalization-comparison tbody tr').count()!==2)throw new Error('Two normalization comparison rows are required');
const modeChoices=await page.locator('#safety-mode option').evaluateAll(nodes=>nodes.map(n=>n.value));
if(JSON.stringify(modeChoices)!==JSON.stringify(['current_environment_prediction','current_environment_geometry','offline_pdm']))throw new Error('All three environment modes must be available');
const predictionNote=await page.locator('#guard-note').textContent(),plotNote=await page.locator('#plot-environment-note').textContent();
if(!predictionNote.includes('当前速度')||!predictionNote.includes('固定朝向')||!predictionNote.includes('不是官方 TTC')||!plotNote.includes('t=0')||!plotNote.includes('旧框位置不等于碰撞'))throw new Error('Prediction/plot assumptions must be visible');
await page.screenshot({path:path.join(base,'work/framework_S08_desktop.png'),fullPage:true});
await page.locator('#normalization-comparison').screenshot({path:path.join(base,'work/framework_normalization_comparison.png')});
const sensitivityDefault=await page.evaluate(()=>({grid:FusionFramework.current.sensitivity,opened:document.getElementById('sensitivity').open,highlighted:document.querySelectorAll('#sensitivity-grid button.current').length}));
if(!sensitivityDefault.grid.enabled||sensitivityDefault.opened||sensitivityDefault.grid.alphas.length!==11||sensitivityDefault.grid.rhos.length!==12||!sensitivityDefault.grid.rhos.includes(.75)||sensitivityDefault.grid.cells.length!==132||sensitivityDefault.highlighted!==1)throw new Error('Default local sensitivity grid axes/highlight are incorrect');
const sensitivityMonotonic=sensitivityDefault.grid.alphas.every(alpha=>{
  const row=sensitivityDefault.grid.cells.filter(cell=>cell.alpha===alpha);
  return row.every((cell,i)=>cell.proposed_id===row[0].proposed_id&&cell.proposed_score===row[0].proposed_score&&(!i||row[i-1].status==='accepted'||cell.status!=='accepted'));
});
if(!sensitivityMonotonic)throw new Error('At fixed alpha, raising rho must not turn rejection into acceptance or change the highest-scoring proposal');
await page.locator('#sensitivity summary').click();
const nativeCellStatus=await page.locator('#sensitivity-grid button[data-alpha="0"][data-rho="0.5"]').getAttribute('data-status');
await page.selectOption('#normalization','softmax');
const softmaxCellStatus=await page.locator('#sensitivity-grid button[data-alpha="0"][data-rho="0.5"]').getAttribute('data-status');
if(nativeCellStatus!=='accepted'||softmaxCellStatus!=='rejected_low_support')throw new Error('The sensitivity grid must use the selected normalization');
await page.selectOption('#normalization','sigmoid');
const sensitivityBefore=await page.evaluate(()=>({sceneId:FusionFramework.current.sceneId,safetyMode:FusionFramework.current.safetyMode,command:FusionFramework.current.command,config:FusionFramework.current.config,candidate:document.getElementById('candidate').value}));
await page.locator('#sensitivity-grid button[data-alpha="0.9"][data-rho="0.6"]').click();
const sensitivityAfter=await page.evaluate(()=>{
  const c=FusionFramework.current,cell=document.querySelector('#sensitivity-grid button.current');
  return {sceneId:c.sceneId,safetyMode:c.safetyMode,command:c.command,config:c.config,candidate:document.getElementById('candidate').value,alphaInput:document.getElementById('alpha').value,rhoInput:document.getElementById('rejection-threshold').value,highlighted:document.querySelectorAll('#sensitivity-grid button.current').length,cellStatus:cell.dataset.status,cellOutput:cell.dataset.outputId,cellTitle:cell.title,resultStatus:c.result.status,resultOutput:c.result.selected_id};
});
const {alpha:oldAlpha,rejection_threshold:oldRho,...beforeKept}=sensitivityBefore.config,{alpha:newAlpha,rejection_threshold:newRho,...afterKept}=sensitivityAfter.config;
if(newAlpha!==.9||newRho!==.6||sensitivityAfter.alphaInput!=='0.9'||sensitivityAfter.rhoInput!=='0.6'||JSON.stringify(beforeKept)!==JSON.stringify(afterKept)||sensitivityBefore.sceneId!==sensitivityAfter.sceneId||sensitivityBefore.safetyMode!==sensitivityAfter.safetyMode||JSON.stringify(sensitivityBefore.command)!==JSON.stringify(sensitivityAfter.command)||sensitivityBefore.candidate!==sensitivityAfter.candidate||sensitivityAfter.highlighted!==1||sensitivityAfter.cellStatus!==sensitivityAfter.resultStatus||sensitivityAfter.cellOutput!==String(sensitivityAfter.resultOutput)||!sensitivityAfter.cellTitle.includes('F='))throw new Error('Sensitivity click changed unrelated state or disagreed with actual output');
await page.locator('#command-enabled').uncheck();
if(await page.locator('#sensitivity-grid button').count()!==0||!await page.locator('#sensitivity-unavailable').textContent().then(t=>t.includes('先启用')))throw new Error('Sensitivity grid must be disabled without an instruction');
await page.locator('#command-enabled').check();await page.selectOption('#decision-rule','direction_tolerance');
if(await page.locator('#sensitivity-grid button').count()!==0||!await page.locator('#sensitivity-unavailable').textContent().then(t=>t.includes('历史方向容差')))throw new Error('Sensitivity grid must be disabled for the historical rule');
await page.locator('#reset').click();
await page.locator('#alpha').fill('0.25');await page.locator('#alpha').dispatchEvent('change');
if(await page.locator('#sensitivity-grid button[data-alpha="0.25"][data-rho="0.75"].current').count()!==1)throw new Error('Off-axis current alpha must remain visible');
await page.locator('#reset').click();
await page.locator('#sensitivity').screenshot({path:path.join(base,'work/framework_sensitivity_grid.png')});
await page.locator('#sensitivity summary').click();
if(await page.locator('#scene option').count()!==16)throw new Error('All sixteen scenes must remain');
if(await page.locator('#road-type-filter').inputValue()!=='all')throw new Error('Road classification filter must start at all scenes');
const classificationAudit=await page.evaluate(()=>{
  const counts={};for(const scene of FUSION_DATA.scenes){const type=scene.classification?.primary_type;counts[type]=(counts[type]||0)+1;}
  const classificationDoesNotAffectFusion=FUSION_DATA.fixtures.every(f=>{
    const scene=FUSION_DATA.scenes.find(s=>s.id===f.scene_id),changed=structuredClone(scene);
    changed.classification={primary_type:'deliberately_changed_metadata',secondary_features:['not a guard or score']};
    const original=FusionFramework.evaluateScene(scene,f.command,f.config,f.safety_mode);
    const altered=FusionFramework.evaluateScene(changed,f.command,f.config,f.safety_mode);
    return JSON.stringify(original)===JSON.stringify(altered);
  });
  return {counts,allScenes:FUSION_DATA.scenes.length,allCandidates:FUSION_DATA.scenes.reduce((s,x)=>s+x.candidates.length,0),
    categories:FUSION_DATA.meta.classification.categories,classificationDoesNotAffectFusion,
    allProvisional:FUSION_DATA.scenes.every(s=>s.classification.status==='provisional_map_review'),
    allLaneChangeUnverified:FUSION_DATA.scenes.every(s=>s.classification.lane_change_status==='unverified'),
    allVariableDirectionUnverified:FUSION_DATA.scenes.every(s=>s.classification.variable_direction_lane_status==='unverified')};
});
if(!classificationAudit.classificationDoesNotAffectFusion||!classificationAudit.allProvisional||!classificationAudit.allLaneChangeUnverified||!classificationAudit.allVariableDirectionUnverified||classificationAudit.allCandidates!==320)throw new Error('Classification metadata boundary failed');
await page.locator('#direction-number').fill('14.3');await page.locator('#direction-number').dispatchEvent('change');
const controlsBeforeClassification=await page.evaluate(()=>({command:FusionFramework.current.command,config:FusionFramework.current.config,safetyMode:FusionFramework.current.safetyMode}));
const categoryCases=[];
for(const category of classificationAudit.categories){
  await page.selectOption('#road-type-filter',category.id);
  const result=await page.evaluate(id=>{
    const expected=FUSION_DATA.scenes.filter(s=>s.classification.primary_type===id).map(s=>s.id);
    const options=[...document.querySelectorAll('#scene option')];
    return {expected,visible:options.map(o=>o.value),labelsCorrect:options.every(o=>o.textContent===o.value+' · '+FUSION_DATA.scenes.find(s=>s.id===o.value).classification.short_label),
      tableRows:document.querySelectorAll('#classification-table tbody tr').length,selected:document.getElementById('scene').value,
      controls:{command:FusionFramework.current.command,config:FusionFramework.current.config,safetyMode:FusionFramework.current.safetyMode}};
  },category.id);
  if(JSON.stringify(result.expected)!==JSON.stringify(result.visible)||result.visible.length!==category.count||!result.labelsCorrect||result.tableRows!==16||!result.visible.includes(result.selected)||JSON.stringify(result.controls)!==JSON.stringify(controlsBeforeClassification))throw new Error('Road type browsing changed data/controls: '+category.id);
  categoryCases.push({id:category.id,count:result.visible.length});
}
await page.selectOption('#road-type-filter','crossing_junction');
await page.locator('#classification-overview summary').click();
await page.locator('#classification-table button[data-scene-id="S05"]').click();
if(await page.locator('#scene').inputValue()!=='S05'||await page.locator('#road-type-filter').inputValue()!=='all'||await page.locator('#scene option').count()!==16)throw new Error('Cross-category table jump should restore all scenes');
const afterJump=await page.evaluate(()=>({command:FusionFramework.current.command,config:FusionFramework.current.config,safetyMode:FusionFramework.current.safetyMode}));
if(JSON.stringify(afterJump)!==JSON.stringify(controlsBeforeClassification))throw new Error('Classification jump reset fusion controls');
await page.locator('.classification-detail summary').click();
if(!await page.locator('.classification-detail ul').first().isVisible())throw new Error('Per-scene classification evidence cannot be read');
await page.locator('.classification-detail summary').click();
await page.screenshot({path:path.join(base,'work/framework_classification_table.png'),fullPage:true});
await page.locator('#classification-overview summary').click();await page.locator('#reset').click();
let renderCases=0,predictionPassTotal=0,geometryPassTotal=0,offlinePassTotal=0;
const modeCases=[];
for(const normalization of ['sigmoid','softmax']){
await page.selectOption('#normalization',normalization);
for(const decisionRule of ['score_reject','direction_tolerance']){
await page.selectOption('#decision-rule',decisionRule);
for(const mode of modeChoices){
  await page.selectOption('#safety-mode',mode);
  for(let n=1;n<=16;n++){
    const sid='S'+String(n).padStart(2,'0');await page.selectOption('#scene',sid);
    const audit=await page.evaluate(()=>({rows:document.querySelectorAll('#ranking tbody tr').length,
      methods:document.querySelectorAll('#methods tbody tr').length,drawn:document.querySelectorAll('#all-plot path.candidate').length,
      result: FusionFramework.current.result,banner:document.getElementById('decision-status').textContent,paths:[...document.querySelectorAll('svg path')].map(p=>p.getAttribute('d'))}));
    if(audit.rows!==20||audit.methods!==3||audit.drawn!==20||audit.paths.some(d=>/NaN|Infinity/.test(d)))throw new Error('Missing/broken UI content at '+sid+'/'+mode);
    if(normalization==='sigmoid'&&decisionRule==='score_reject'){
      if(mode==='current_environment_prediction')predictionPassTotal+=audit.result.environment_eligible_count;else if(mode==='current_environment_geometry')geometryPassTotal+=audit.result.environment_eligible_count;else offlinePassTotal+=audit.result.environment_eligible_count;
    }
    if(mode==='current_environment_geometry'&&['S13','S14'].includes(sid)&&!(audit.result.status==='no_feasible_candidate'&&audit.result.selected_id===null&&audit.result.stop_requested))throw new Error('Empty geometry pool should have no output');
    if(audit.result.status==='no_feasible_candidate'&&(!audit.banner.includes('20 条候选仍在')||!audit.banner.includes('不是模型没有生成候选')||!audit.banner.includes('#'+audit.result.original_model_top1)))throw new Error('An empty checked pool must preserve the explanation of model candidates');
    modeCases.push({scene:sid,mode,normalization,decision_rule:decisionRule,eligible:audit.result.environment_eligible_count,status:audit.result.status,proposed_id:audit.result.proposed_id,selected_id:audit.result.selected_id,original_model_top1:audit.result.original_model_top1});
    if(normalization==='sigmoid'&&decisionRule==='score_reject'&&mode==='current_environment_prediction'&&['S13','S14'].includes(sid)){
      await page.selectOption('#candidate',String(audit.result.selected_id??audit.result.original_model_top1));
      await page.screenshot({path:path.join(base,'work/framework_'+sid+'_prediction.png'),fullPage:true});
    }
    renderCases++;
  }
}
}
}
await page.locator('#reset').click();
const modeScoreInvariance=await page.evaluate(()=>{
  const keys=['model_logit','model_score','intent_score','fused_score','error_deg','direction_deg','net_displacement_m','length_m','mean_speed_mps'];
  return FUSION_DATA.scenes.every(scene=>{
    const results=['current_environment_prediction','current_environment_geometry','offline_pdm'].map(mode=>FusionFramework.evaluateScene(scene,{active:true,angle_deg:14.3},FUSION_DATA.meta.defaults,mode));
    return results.slice(1).every(result=>result.rows.every((row,i)=>keys.every(key=>row[key]===results[0].rows[i][key])));
  });
});
if(!modeScoreInvariance)throw new Error('Changing environment assumptions must not change candidate scores');
const rejectionAudit=await page.evaluate(()=>{
  const scene=FUSION_DATA.scenes.find(s=>s.id==='S08'),cfg=FUSION_DATA.meta.defaults,mode='current_environment_prediction',command={active:true,angle_deg:0};
  const base=FusionFramework.evaluateScene(scene,command,cfg,mode);
  const changedTolerance=FusionFramework.evaluateScene(scene,command,{...cfg,tolerance_deg:0},mode);
  const changedTemperature=FusionFramework.evaluateScene(scene,command,{...cfg,temperature:8},mode);
  const scoreKeys=['model_score','intent_score','fused_score'];
  const toleranceIgnored=base.selected_id===changedTolerance.selected_id&&base.status===changedTolerance.status&&base.proposed_id===changedTolerance.proposed_id;
  const nativeTemperatureIgnored=base.rows.every((row,i)=>scoreKeys.every(k=>row[k]===changedTemperature.rows[i][k]));
  const candidates=[{id:7,model_logit:0,poses:Array.from({length:8},(_,i)=>[(i+1)*2,0,0])},{id:2,model_logit:0,poses:Array.from({length:8},(_,i)=>[(i+1)*2,0,0])}];
  const guards=candidates.map(c=>({candidate_id:c.id,eligible:true,reasons:[]}));
  const equality=FusionFramework.fuseCandidates(candidates,guards,command,{...cfg,alpha:0,rejection_threshold:.5});
  const strictAbove=FusionFramework.fuseCandidates(candidates,guards,command,{...cfg,alpha:0,rejection_threshold:.49});
  const disabled=FusionFramework.fuseCandidates(candidates,guards,{active:false,angle_deg:0},{...cfg,rejection_threshold:1});
  const undefinedDirection=FusionFramework.fuseCandidates(candidates.map(c=>({...c,poses:Array.from({length:8},()=>[0,0,0])})),guards,command,cfg);
  const displayMatches=FusionFramework.current.normalizationComparison.every(({normalization,result})=>{
    const t=document.querySelector('#normalization-comparison tr[data-normalization="'+normalization+'"]').textContent;
    return t.includes('#'+result.proposed_id)&&t.includes(result.proposed_score.toFixed(6))&&t.includes('#'+result.selected_id);
  });
  return {toleranceIgnored,nativeTemperatureIgnored,strictEqualityRejects:equality.status==='rejected_low_support'&&equality.proposed_score===.5&&equality.proposed_id===2,strictAboveAccepts:strictAbove.status==='accepted'&&strictAbove.selected_id===2,noCommandBypassesThreshold:disabled.status==='no_command'&&disabled.selected_id===2,undefinedDirectionRejects:undefinedDirection.status==='rejected_low_support'&&undefinedDirection.proposed_id===null&&undefinedDirection.selected_id===2,comparisonDisplayMatches:displayMatches,defaultProposed:base.proposed_id,defaultOutput:base.selected_id,defaultStatus:base.status};
});
for(const [key,value] of Object.entries(rejectionAudit))if(typeof value==='boolean'&&!value)throw new Error('Unified rejection check failed: '+key);
const diagnosticExample=await page.evaluate(()=>{
  for(const scene of FUSION_DATA.scenes)for(const candidate of scene.candidates){
    const event=candidate.guards.current_environment_prediction.diagnostics?.first_failure;
    if(event?.actor_predictions?.some(a=>a.motion_applied))return {scene:scene.id,id:candidate.id,event};
  }
  return null;
});
if(!diagnosticExample)throw new Error('Missing moving-actor diagnostic test example');
await page.selectOption('#safety-mode','current_environment_prediction');await page.selectOption('#scene',diagnosticExample.scene);await page.selectOption('#candidate',String(diagnosticExample.id));
const diagnosticText=await page.locator('#candidate-guard-diagnostic').textContent(),event=diagnosticExample.event,actor=event.actor_predictions.find(a=>a.motion_applied);
for(const expected of ['首次几何相交',event.time_s.toFixed(3)+' s','annotation #'+actor.annotation_index,actor.category,actor.current_center_xy_ego_m[0].toFixed(2),actor.predicted_center_xy_ego_m[0].toFixed(2),actor.velocity_xy_ego_mps[0].toFixed(2),'记录速度','本模式采用','同一时刻'])if(!diagnosticText.includes(expected))throw new Error('Prediction diagnostic does not explain '+expected);
await page.screenshot({path:path.join(base,'work/framework_prediction_diagnostic.png'),fullPage:true});
await page.selectOption('#scene','S13');await page.selectOption('#candidate','18');await page.selectOption('#safety-mode','current_environment_geometry');
const frozenText=await page.locator('#candidate-guard-diagnostic').textContent();
if(!frozenText.includes('首次几何相交')||!frozenText.includes('annotation #5')||!frozenText.includes('所有对象固定')||!frozenText.includes('本模式采用 (0.00, 0.00) m/s'))throw new Error('Frozen mode diagnostic must show that actor motion was ignored');
await page.selectOption('#scene','S04');await page.selectOption('#safety-mode','current_environment_geometry');
await page.locator('#direction-number').fill('-20');await page.locator('#direction-number').dispatchEvent('change');
const currentS04=await page.evaluate(()=>({id:FusionFramework.current.result.selected_id,status:FusionFramework.current.result.status,
  excluded10and19:FusionFramework.current.result.rows.filter(r=>[10,19].includes(r.id)).every(r=>!r.eligible)}));
if(!currentS04.excluded10and19)throw new Error('S04 saved geometry distinction missing');
await page.screenshot({path:path.join(base,'work/framework_S04_geometry.png'),fullPage:false});
await page.selectOption('#safety-mode','offline_pdm');
const offlineS04=await page.evaluate(()=>({id:FusionFramework.current.result.selected_id,status:FusionFramework.current.result.status,note:document.getElementById('guard-note').textContent}));
if(!offlineS04.note.includes('事后')||!offlineS04.note.includes('不能'))throw new Error('Offline gate limitation is missing');
await page.selectOption('#candidate','10');
if(!await page.locator('#candidate-formula').textContent().then(t=>t.includes('#10')&&t.includes('atan2')))throw new Error('Candidate inspection failed');
await page.locator('.point-details summary').click();if(await page.locator('#points tbody tr').count()!==8)throw new Error('Eight future points missing');
await page.locator('.point-details summary').click();
await page.locator('#ranking tr[data-id="7"]').click();if(await page.locator('#candidate').inputValue()!=='7')throw new Error('Row-to-inspector selection failed');
await page.locator('#command-enabled').uncheck();
const disabled=await page.evaluate(()=>({status:FusionFramework.current.result.status,allNoIntent:FusionFramework.current.result.rows.every(r=>r.error_deg===null&&r.intent_score===null&&r.fused_score===r.model_score),angleDisabled:document.getElementById('direction-number').disabled}));
if(disabled.status!=='no_command'||!disabled.allNoIntent||!disabled.angleDisabled)throw new Error('Disabling intent should select the model only');
await page.locator('#reset').click();
await page.selectOption('#normalization','softmax');await page.selectOption('#decision-rule','direction_tolerance');
if(await page.locator('#temperature').isDisabled()||await page.locator('#tolerance').isDisabled()||!await page.locator('#rejection-threshold').isDisabled())throw new Error('Historical rule controls do not match applicability');
for(const [id,value] of [['alpha','0.25'],['temperature','3'],['sigma','25'],['tolerance','5']]){await page.locator('#'+id).fill(value);await page.locator('#'+id).dispatchEvent('change');}
await page.selectOption('#decision-rule','score_reject');await page.locator('#rejection-threshold').fill('0.62');await page.locator('#rejection-threshold').dispatchEvent('change');
const changed=await page.evaluate(()=>FusionFramework.current.config);
if(changed.alpha!==.25||changed.temperature!==3||changed.sigma_deg!==25||changed.tolerance_deg!==5||changed.rejection_threshold!==.62||changed.normalization!=='softmax'||changed.decision_rule!=='score_reject')throw new Error('Parameter controls did not update');
await page.locator('#direction-range').fill('30');await page.locator('#direction-range').dispatchEvent('input');
if(await page.locator('#direction-number').inputValue()!=='30')throw new Error('Slider/number not synchronized');
await page.locator('#formulas summary').click();if(!await page.locator('#formula-explanation').isVisible())throw new Error('Formula explanation unavailable');
await page.locator('#formulas summary').click();await page.locator('#reset').click();
const links=await page.locator('a[href]').evaluateAll(nodes=>[...new Set(nodes.map(n=>n.getAttribute('href')))]),linkStatus=[];
for(const target of links){let exists=true;try{await fs.access(path.join(base,target));}catch{exists=false;}const pending=!exists&&target==='README.md'&&process.argv.includes('--allow-pending-docs');linkStatus.push({path:target,exists,pending});if(!exists&&!pending)throw new Error('Broken file link: '+target);}
await page.setViewportSize({width:430,height:932});await page.evaluate(()=>window.scrollTo(0,0));
await page.screenshot({path:path.join(base,'work/framework_mobile.png'),fullPage:true});
await page.screenshot({path:path.join(base,'work/framework_mobile_top.png'),fullPage:false});
await page.locator('#sensitivity summary').click();
await page.locator('#sensitivity').screenshot({path:path.join(base,'work/framework_sensitivity_mobile.png')});
const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
await browser.close();
if(errors.length||mobileOverflow)throw new Error(JSON.stringify({errors,mobileOverflow}));
const result={result:'PASS',defaultSafetyMode:initial.mode,defaultConfig:initial.config,modeChoices,fixtureAudit,rejectionAudit,sensitivityAudit:{defaultRows:sensitivityDefault.grid.alphas.length,defaultColumns:sensitivityDefault.grid.rhos.length,defaultCells:sensitivityDefault.grid.cells.length,currentRhoExtraColumn:true,raisingRhoCannotRestoreAcceptance:sensitivityMonotonic,normalizationSwitchRecomputesGrid:true,clickPreservesOtherState:true,clickedAlpha:newAlpha,clickedRho:newRho,offAxisAlphaVisible:true,disabledWithoutCommand:true,disabledForLegacyRule:true},classificationAudit,classificationCategoryCases:categoryCases,classificationJump:'passed; cross-category restores all and preserves controls',renderCases,modeCases,predictionPassTotal,geometryPassTotal,offlinePassTotal,modeScoreInvariance,predictionDiagnostic:{scene:diagnosticExample.scene,id:diagnosticExample.id,time_s:event.time_s,actor_annotation_index:actor.annotation_index,display:'passed; same-time center and current/applied velocity'},frozenDiagnostic:'passed; current velocity shown, applied velocity zero',currentS04,offlineS04,noCommand:disabled,parameterControls:changed,
  candidateSelection:'passed',eightFuturePoints:true,formulaPanel:'passed',downloadLinks:linkStatus,javascriptErrors:errors,mobilePageOverflow:mobileOverflow};
await fs.writeFile(path.join(base,'work/page_validation.json'),JSON.stringify(result,null,2));
console.log('Framework UI PASS: '+fixtureAudit.fixtures+' Python fixtures, '+renderCases+' scene/mode renders, controls and responsive layout');
