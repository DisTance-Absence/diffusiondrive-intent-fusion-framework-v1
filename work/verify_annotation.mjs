// Isolated UI-only smoke test. Dummy edits live in a disposable browser context.
// No exported labels are saved to the project or passed to parameter fitting.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';

const work=path.dirname(fileURLToPath(import.meta.url)),base=path.dirname(work);
const require=createRequire(import.meta.url);
const {chromium}=require(require.resolve('playwright',{paths:[base,path.join(base,'../scene-choice-inventory-v3')]}));
const report={status:'RUNNING',isolatedBrowserContext:true,realAnnotationsWritten:false,
  calibrationFitCalled:false,syntheticFilesPersisted:false,javascriptErrors:[],validationFindings:[],checks:{}};
const sourceNames=['annotation.html','annotation_template.json','annotation_page_template.html','build_annotation_page.py','calibration.py'];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const before={};for(const name of sourceNames)before[name]=sha(await fs.readFile(path.join(base,name)));
let browser,context;
const assert=(condition,message)=>{if(!condition)throw Error(message);};
function validateOnly(document){
  const synthetic=structuredClone(document);synthetic.synthetic=true;
  const code=`import json,sys\nfrom pathlib import Path\nfrom calibration import validate\nd=json.load(sys.stdin)\ne=json.loads(Path('annotation_template.json').read_text(encoding='utf-8'))\nprint(json.dumps(validate(d,e,allow_synthetic=True)))`;
  const result=spawnSync('D:/Software/anaconda/python.exe',['-X','utf8','-c',code],
    {cwd:base,input:JSON.stringify(synthetic),encoding:'utf8',env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
  assert(result.status===0,'Validation-only Python failed: '+result.stderr);
  return JSON.parse(result.stdout);
}
try{
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  context=await browser.newContext({viewport:{width:1500,height:1100},deviceScaleFactor:1,acceptDownloads:true});
  const page=await context.newPage();page.on('pageerror',error=>report.javascriptErrors.push(String(error)));
  await page.goto(pathToFileURL(path.join(base,'annotation.html')).href);
  await page.waitForFunction(()=>document.querySelector('#scene option')?.textContent==='S01');
  const initial=await page.evaluate(()=>({records:doc.records.length,contexts:Object.keys(doc.contexts).length,
    blank:doc.records.filter(r=>r.label==='').length,independenceNull:doc.records.every(r=>r.independent_of_model_outputs===null),
    metadataBlank:doc.records.every(r=>['goal','reason','evidence_source','annotator_id','reviewed_at','label_source','split'].every(k=>r[k]==='')),
    localStorageInitiallyEmpty:localStorage.getItem(key)===null,document:structuredClone(doc),
    formIndependent:document.getElementById('independent').checked,
    displayProtocol:document.body.innerText,
    overflow:document.documentElement.scrollWidth>innerWidth}));
  assert(initial.records===112&&initial.contexts===16&&initial.blank===112,'Template must contain exactly 112 blank requests and 16 contexts');
  assert(initial.independenceNull&&initial.metadataBlank&&!initial.formIndependent&&initial.localStorageInitiallyEmpty,'Labels/independence/metadata must not be prefilled');
  assert(!initial.overflow,'Desktop horizontal overflow');
  report.checks.initial={records:initial.records,contexts:initial.contexts,blank:initial.blank,
    independenceNull:true,metadataBlank:true,independenceCheckboxUnchecked:true,localStorageInitiallyEmpty:true};
  const forbidden=new Set(['candidates','candidate_scores','model_score','fused_score','model_logit','guards',
    'offline_metrics','NC','DAC','TTC','selected_id','proposed_id','rejection_threshold','predicted_trajectory']);
  const leaked=[];
  function inspect(value,location='document'){
    if(!value||typeof value!=='object')return;
    for(const [k,v] of Object.entries(value)){if(forbidden.has(k))leaked.push(location+'.'+k);inspect(v,location+'.'+k);}
  }
  inspect(initial.document);assert(leaked.length===0,'Model/candidate data leaked into annotation page: '+leaked.join(','));
  assert(await page.locator('#map .candidate').count()===0,'Candidate trajectory drawn in annotation map');
  report.checks.leakage={forbiddenEmbeddedFields:leaked,candidateElements:0};
  report.checks.originalBlankValidation=validateOnly(initial.document);
  if(report.checks.originalBlankValidation.status!=='valid')report.validationFindings.push('Browser-roundtripped blank template fails calibration.validate');
  await page.screenshot({path:path.join(work,'annotation_desktop.png'),fullPage:true});
  await page.setViewportSize({width:430,height:932});
  await page.screenshot({path:path.join(work,'annotation_mobile.png'),fullPage:true});
  report.checks.mobileHorizontalOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  assert(!report.checks.mobileHorizontalOverflow,'Mobile horizontal overflow');
  await page.setViewportSize({width:1500,height:1100});

  let switched=0;
  for(let n=1;n<=16;n++){
    const sid='S'+String(n).padStart(2,'0');await page.selectOption('#scene',sid);
    const options=await page.locator('#command option').evaluateAll(nodes=>nodes.map(n=>({id:n.value,angle:Number(n.textContent.replace('°',''))})));
    assert(JSON.stringify(options.map(o=>o.angle))===JSON.stringify([-45,-30,-15,0,15,30,45]),'Missing seven requests at '+sid);
    for(const option of options){
      await page.selectOption('#command',option.id);
      const state=await page.evaluate(()=>({id:doc.records[index].record_id,title:document.getElementById('record-title').textContent,
        value:document.getElementById('label').value,independent:document.getElementById('independent').checked,
        paths:[...document.querySelectorAll('#map path,#map polygon,#map polyline')].map(n=>n.getAttribute('d')||n.getAttribute('points'))}));
      assert(state.id===option.id&&state.title.includes(sid)&&state.title.includes(option.angle+'°'),'Scene/command selection desynchronised');
      assert(state.value===''&&!state.independent,'Switching prefilled labels');
      assert(state.paths.length>2&&!state.paths.some(d=>/NaN|Infinity/.test(d)),'Broken evidence drawing at '+option.id);
      switched++;
    }
  }
  report.checks.recordSwitches=switched;
  await page.selectOption('#scene','S01');
  await page.locator('#record-list button').last().click();
  assert(await page.locator('#command').inputValue()==='S01_direction_+45','Record list navigation failed');
  await page.locator('#next').click();
  assert(await page.locator('#scene').inputValue()==='S02'&&await page.locator('#command').inputValue()==='S02_direction_-45','Next does not cross scene boundary');
  await page.selectOption('#scene','S01');

  async function fillDummy(){
    await page.selectOption('#label','uncertain');
    for(const id of ['goal','reason','evidence','annotator'])await page.locator('#'+id).fill('SYNTHETIC UI TEST ONLY - '+id+' - NOT A HUMAN LABEL');
    await page.locator('#independent').check();
  }
  const required=[];
  for(const missing of ['label','goal','reason','evidence','annotator','independent']){
    await fillDummy();
    if(missing==='label')await page.selectOption('#label','');
    else if(missing==='independent')await page.locator('#independent').uncheck();
    else await page.locator('#'+missing).fill('   ');
    await page.locator('button[type=submit]').click();
    assert((await page.locator('#status').textContent()).includes('请填写'),'Missing '+missing+' was not rejected');
    assert(await page.evaluate(()=>doc.records.every(r=>r.label==='')),'Incomplete form saved a label');
    required.push(missing);
  }
  report.checks.requiredFields=required;
  await fillDummy();await page.locator('button[type=submit]').click();
  assert((await page.locator('#status').textContent()).includes('本条已记录'),'Dummy smoke save failed');
  const saved=await page.evaluate(()=>structuredClone(doc.records[index]));
  assert(saved.label==='uncertain'&&saved.independent_of_model_outputs===true&&saved.label_source==='human_review','Saved form contract incorrect');
  assert(!Number.isNaN(Date.parse(saved.reviewed_at))&&saved.reviewed_at.endsWith('Z'),'Save timestamp missing timezone');
  await page.reload();
  assert(await page.locator('#label').inputValue()==='uncertain'&&await page.locator('#independent').isChecked(),'Disposable localStorage restore failed');
  report.checks.disposableLocalStorageRestore=true;
  const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();
  const download=await downloadPromise,downloadPath=await download.path();
  const exported=JSON.parse(await fs.readFile(downloadPath,'utf8'));
  assert(download.suggestedFilename()==='independent_intent_annotations.json','Download filename wrong');
  assert(exported.records.filter(r=>r.label).length===1,'Download record counts wrong');
  assert(JSON.stringify(exported.contexts)===JSON.stringify(initial.document.contexts),'Export changed source contexts');
  report.checks.syntheticExportValidation=validateOnly(exported);
  if(report.checks.syntheticExportValidation.status!=='valid')report.validationFindings.push('Browser-exported synthetic smoke record fails calibration.validate');
  assert(report.checks.syntheticExportValidation.blank===111&&report.checks.syntheticExportValidation.completed.uncertain===1,'Synthetic export summary incorrect');

  await page.locator('#clear').click();
  assert(await page.locator('#label').inputValue()===''&&!await page.locator('#independent').isChecked(),'Clear must restore blank and unattested state');
  async function importBuffer(document){
    await page.locator('#io-status').evaluate(node=>{node.textContent='';});
    await page.locator('#import').setInputFiles({name:'synthetic_ui_smoke_NOT_HUMAN.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(document))});
    await page.waitForFunction(()=>document.getElementById('io-status').textContent.startsWith('已导入')||document.getElementById('io-status').textContent.startsWith('导入失败'));
  }
  await importBuffer(exported);
  assert((await page.locator('#io-status').textContent()).startsWith('已导入'),'Roundtrip import rejected');
  assert(await page.locator('#label').inputValue()==='uncertain','Roundtrip import lost label');
  const altered=structuredClone(exported);altered.contexts.S01.current_ego_speed_mps=99999;
  await importBuffer(altered);
  assert(await page.evaluate(()=>doc.contexts.S01.current_ego_speed_mps===original.contexts.S01.current_ego_speed_mps),'Import overwrote source evidence');
  const badIdentity=structuredClone(exported);badIdentity.identity_sha256='0'.repeat(64);
  await importBuffer(badIdentity);
  assert((await page.locator('#io-status').textContent()).startsWith('导入失败'),'Foreign identity was accepted');
  assert(await page.locator('#label').inputValue()==='uncertain','Failed import modified existing record');
  const badLabel=structuredClone(exported);badLabel.records[0].label='model_generated';
  await importBuffer(badLabel);
  assert((await page.locator('#io-status').textContent()).startsWith('导入失败'),'Unsupported label imported');
  report.checks.importRoundtrip=true;report.checks.sourceContextImmutableOnImport=true;
  report.checks.invalidIdentityAndLabelRejected=true;
  await page.locator('#clear').click();
  assert(await page.evaluate(()=>doc.records.every(r=>r.label===''&&r.independent_of_model_outputs===null)),'Final disposable page state not blank');
  report.checks.protocolVisible={horizon4s:/4\s*秒/.test(initial.displayProtocol),
    endpointBearing:/终点/.test(initial.displayProtocol),
    explicitTargetOrBehaviorScope:/目标区域|行为范围/.test(initial.displayProtocol),
    noAutomaticAngularTolerance:initial.document.instruction.angular_interpretation_tolerance_deg===undefined};
  assert(Object.values(report.checks.protocolVisible).every(Boolean),'Annotation protocol missing visible horizon/direction/target scope or retains an automatic angle tolerance');
  report.checks.finalBrowserRecordsBlank=112;
  assert(report.javascriptErrors.length===0,'Browser JavaScript errors');
  assert(report.validationFindings.length===0,report.validationFindings.join('; '));
  report.status='PASS';
}catch(error){report.status='FAIL';report.error=String(error);}
finally{
  if(context)await context.close();if(browser)await browser.close();
  report.browserContextClosed=true;
  const after={};for(const name of sourceNames)after[name]=sha(await fs.readFile(path.join(base,name)));
  report.sourceHashes=after;report.sourceUnchanged=sourceNames.every(name=>before[name]===after[name]);
  if(!report.sourceUnchanged){report.status='FAIL';report.sourceChange='Sources changed during test; rerun after root freezes edits.';}
  await fs.writeFile(path.join(work,'annotation_page_validation.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({status:report.status,error:report.error,checks:report.checks,sourceUnchanged:report.sourceUnchanged}));
}
if(report.status!=='PASS')process.exitCode=1;
