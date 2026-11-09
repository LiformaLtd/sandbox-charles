// Explicit read-only inputs. Never download or change another app's artifacts.
import {readFile,writeFile,mkdir,copyFile,cp} from 'node:fs/promises';
import {prepareVoices} from './prepare-voices.mjs';
import {resolve,join} from 'node:path';import {createHash} from 'node:crypto';
const source=process.argv[2],faceModel=process.argv[3],voiceBundle=process.argv[4];
if(!source||!faceModel||!voiceBundle)throw Error('Usage: npm run assets -- /path/to/audio2face-2d/static/assets /path/to/face_landmarker.task /path/to/phonon-31m-en');
const output=resolve('static/assets');await mkdir(join(output,'models'),{recursive:true});
const catalog=JSON.parse(await readFile(join(source,'catalog.json'),'utf8'));
const model=await readFile(join(source,catalog.model.url.replace('/assets/','')));
const hash=createHash('sha256').update(model).digest('hex');
if(hash!=='5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd')throw Error('Unexpected compact model checkpoint');
await writeFile(join(output,'models/lf-sta.onnx'),model);
await cp(join(source,'lf-tts'),join(output,'lf-tts'),{recursive:true});
const avatar=catalog.avatars[0];const folder=join(source,avatar.url.replace('/assets/','').replace('/mesh.json',''));
await copyFile(join(folder,'mesh.json'),join(output,'template-mesh.json'));
await copyFile(join(folder,'linear.json'),join(output,'template-map.json'));
await copyFile(faceModel,join(output,'face_landmarker.task'));
await cp('node_modules/@mediapipe/tasks-vision/wasm',join(output,'vision-wasm'),{recursive:true});
for(const name of ['vision_wasm_internal.js','vision_wasm_nosimd_internal.js']){
 const path=join(output,'vision-wasm',name),text=await readFile(path,'utf8');
 await writeFile(path,text+'\nglobalThis.ModuleFactory = ModuleFactory;\n');
}
await writeFile(join(output,'runtime.json'),JSON.stringify({model:{...catalog.model,url:'/assets/models/lf-sta.onnx'},contact:catalog.contact},null,2));
await writeFile(join(output,'provenance.json'),JSON.stringify({source:resolve(source),template:avatar.id,modelSHA256:hash,faceModelSHA256:createHash('sha256').update(await readFile(faceModel)).digest('hex'),copied:new Date().toISOString()},null,2));
console.log('Prepared local speech, mapping, compact model and MediaPipe assets.');

await prepareVoices(voiceBundle);
