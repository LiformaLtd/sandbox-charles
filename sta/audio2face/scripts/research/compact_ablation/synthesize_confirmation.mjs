import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
if(!process.env.A2F_DICT_ROOT || !process.env.A2F_ABLATION_OUTPUT)throw Error('Set A2F_DICT_ROOT and A2F_ABLATION_OUTPUT.');
const {createSpeech,createAligner}=await import(pathToFileURL(join(process.env.A2F_DICT_ROOT,'scripts/speech.mjs')).href);
const root=pathToFileURL(join(process.env.A2F_ABLATION_OUTPUT,'/'));
const rows=JSON.parse(await readFile(new URL('confirmation-prompts.json',import.meta.url)));
await mkdir(new URL('confirmation-audio/',root),{recursive:true});
await mkdir(new URL('confirmation/labels/',root),{recursive:true});
const speech=await createSpeech(),aligner=await createAligner();
const manifest=[];
try{
 for(const [i,row] of rows.entries()){
  const path=new URL('confirmation-audio/'+row.stem+'.pcm',root);
  let pcm;
  try{const b=await readFile(path);pcm=new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
  catch(error){if(error.code!=='ENOENT')throw error;pcm=await speech.speak(row.text);await writeFile(path,new Uint8Array(pcm.buffer,pcm.byteOffset,pcm.byteLength));}
  const phones=await aligner.phones(pcm,row.text);
  manifest.push({...row,audio:path.pathname,samples:pcm.length,phones,sha256_audio:createHash('sha256').update(new Uint8Array(pcm.buffer,pcm.byteOffset,pcm.byteLength)).digest('hex')});
  await writeFile(new URL('confirmation/manifest.json',root),JSON.stringify(manifest,null,2));
  if(i%8===0)console.log('Synthesized and aligned',i+1,'of',rows.length);
 }
}finally{speech.close();await aligner.close();}
