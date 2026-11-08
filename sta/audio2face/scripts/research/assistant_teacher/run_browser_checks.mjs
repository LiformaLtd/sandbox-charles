// Three isolated trials; training/renderer RPC must have finished first.
import {spawn} from 'node:child_process';
import {mkdir,copyFile,readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
const root=resolve('.local/compact-assistant-v1');
if(!process.env.DIAGNOSTIC_URL||!process.env.PLAYWRIGHT_MODULE||!process.env.WEBKIT_EXECUTABLE)throw Error('Set actual local URL and Playwright/WebKit paths.');
const env={...process.env,A2F_LOSS_OUTPUT:root,A2F_ACTIVATION_OUTPUT:root+'/browser/parity',A2F_ACTIVATION_ASSET_PATH:'/.local/compact-assistant-v1/browser/parity/',A2F_PARITY_MODELS:JSON.stringify({v4:'/.local/compact-feature-v1/control/model.onnx',control:'/.local/compact-assistant-v1/control/model.onnx',distilled:'/.local/compact-assistant-v1/distilled/model.onnx'}),A2F_PARITY_RUNNERS:'{}',COMPARE_ORIGINAL:'0',A2F_MEMORY_SAMPLER:root+'/memory/sample',LOSS_REVIEW_URL:process.env.DIAGNOSTIC_URL+'/.local/compact-assistant-v1/review.html'};
async function run(file){await new Promise((done,fail)=>{const p=spawn(process.execPath,['scripts/research/assistant_teacher/'+file],{env,stdio:'inherit'});p.once('error',fail);p.once('exit',code=>code===0?done():fail(Error(file+' exited '+code)));});}
const trials=[];
for(let trial=1;trial<=3;trial++){
 await run('verify_browser.mjs');
 for(const model of ['v4','control','distilled'])assert.ok((await readFile(root+'/browser/parity/phonon-original-fixed-'+model+'.f32')).equals(await readFile(root+'/browser/parity/phonon-original-repeat-fixed-'+model+'.f32')),'Reset repeat differs: '+model);
 await copyFile(root+'/browser/parity/activation-parity.json',root+'/browser/parity/run-'+trial+'.json');
 await run('measure_memory.mjs');await mkdir(root+'/memory/run-'+trial,{recursive:true});
 for(const model of ['v4','control','distilled'])for(const extension of ['json','png'])await copyFile(root+'/memory/'+model+'.'+extension,root+'/memory/run-'+trial+'/'+model+'.'+extension);
 trials.push({trial,reset_repeats_identical:true});
}

await writeFile(root+'/browser/complete.json',JSON.stringify({trials,node:process.version,scope:'Actual animated James, accelerated prerecorded STA, TTS and initial rig loading excluded; desktop WebKit, not physical iPhone.'},null,2));
