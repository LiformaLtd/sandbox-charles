import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createSpeech,createAligner} from './speech.mjs';
import {extendedVisemePrompts} from './extended-viseme-prompts.mjs';

const output=new URL('../.local/viseme-challenge/',import.meta.url);
await mkdir(output,{recursive:true});
const normalize=text=>text.toLowerCase().replace(/[^a-z]+/g,' ').trim();
const seen=new Set();
for(const corpus of ['full','targeted','subtitles','fresh'])for(const row of JSON.parse(await readFile(new URL(`../.local/${corpus}/clips.json`,import.meta.url))))seen.add(normalize(row.text));
const aligner=await createAligner();let speech;
try {
  const rows=[];
  for(const [intendedPair,texts] of Object.entries(extendedVisemePrompts))for(const text of texts) {
    assert.ok(!seen.has(normalize(text)),'Challenge transcript overlaps another set');seen.add(normalize(text));
    const id=createHash('sha256').update(text).digest('hex').slice(0,16);
    let bytes;
    try{bytes=await readFile(new URL(id+'.pcm',output));}catch(error){
      if(error.code!=='ENOENT')throw error;speech??=await createSpeech();
      const pcm=await speech.speak(text);bytes=Buffer.from(pcm.buffer,pcm.byteOffset,pcm.byteLength);await writeFile(new URL(id+'.pcm',output),bytes);
    }
    const pcm=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
    let phones;
    try{phones=JSON.parse(await readFile(new URL(id+'.phones.json',output)));}catch(error){
      if(error.code!=='ENOENT')throw error;phones=await aligner.phones(pcm,text);await writeFile(new URL(id+'.phones.json',output),JSON.stringify(phones));
    }
    rows.push({id,text,split:'challenge',intendedPair,samples:pcm.length,phones});
    console.log(intendedPair,text,(pcm.length/16000).toFixed(2)+'s');
  }
  await writeFile(new URL('clips.json',output),JSON.stringify(rows,null,2));
}finally{speech?.close();await aligner.close();}
