import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';import {pathToFileURL} from 'node:url';import {createHash} from 'node:crypto';
import {VOICES,VOICE_CATALOGUE_REVISION} from '../src/lib/voices.ts';
export async function prepareVoices(source){
 const bundle=resolve(source),output=resolve('static/assets/lf-tts'),manifest=JSON.parse(await readFile(join(bundle,'manifest.json'),'utf8'));
 if(manifest.voice_revision!==VOICE_CATALOGUE_REVISION)throw new Error('Prepare the pinned Phonon voice bundle before copying voices.');
 await mkdir(output,{recursive:true});const voices=[];
 for(const voice of VOICES){
  const path=join(bundle,'voices',voice.name+'.safetensors'),bytes=await readFile(path),sha256=createHash('sha256').update(bytes).digest('hex');
  if(sha256!==manifest.files[`voices/${voice.name}.safetensors`])throw new Error(`Voice checksum mismatch: ${voice.name}`);
  await copyFile(path,join(output,voice.id+'.safetensors'));voices.push({...voice,sha256,bytes:bytes.length});
 }
 await writeFile(join(output,'voice-provenance.json'),JSON.stringify({revision:VOICE_CATALOGUE_REVISION,source:bundle,voices},null,2)+'\n');
 console.log(`Prepared ${voices.length} verified local Phonon voices.`);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
 if(!process.argv[2])throw new Error('Usage: node scripts/prepare-voices.mjs /path/to/phonon-31m-en');
 await prepareVoices(process.argv[2]);
}
