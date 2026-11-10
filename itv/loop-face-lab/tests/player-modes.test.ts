import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';

// Exercise the actual dual-deck scheduler with deterministic media clocks.
const compiled=ts.transpileModule(readFileSync(new URL('../src/lib/player.ts',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText;
const exports:Record<string,any>={};new Function('exports',compiled)(exports);
const {LoopPlayer}=exports;
class Video extends EventTarget{
 currentTime=0;duration=10;readyState=0;paused=true;src='';loop=false;muted=true;
 load(){if(this.src)queueMicrotask(()=>{this.readyState=4;this.dispatchEvent(new Event('loadeddata'))})}
 async play(){this.paused=false}pause(){this.paused=true}removeAttribute(){this.src=''}
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('speech interrupts idle, drains the current talking clip, then returns to randomized idle',async()=>{
 const saved=Object.getOwnPropertyDescriptors(globalThis),decks:Video[]=[];
 const ctx={globalAlpha:1,drawImage(){}};
 const canvas={width:512,height:512,getContext:()=>ctx};
 let frame:((time:number)=>void)|undefined,state:any;
 Object.assign(globalThis,{document:{createElement:(tag:string)=>{if(tag==='video'){const v=new Video();decks.push(v);return v}return canvas}},requestAnimationFrame:(cb:(time:number)=>void)=>{frame=cb;return 1},cancelAnimationFrame:()=>{frame=undefined}});
 const errors:string[]=[];
 const player=new LoopPlayer(canvas,(value:any)=>state=value,(error:string)=>errors.push(error),()=>{});
 try{
  player.setClips(['idle1','idle2','talk1','talk2'].map((id,slot)=>({id,slot,url:id,seed:0,kind:id.startsWith('idle')?'idle':'talking'})));
  await player.start();await flush();assert.match(state.current,/^idle/);
  decks.find(v=>!v.paused)!.currentTime=3;
  await player.setSpeechActive(true);assert.match(state.current,/^talk/);assert.equal(state.elapsed,0);
  const talking=state.current;
  decks.find(v=>!v.paused)!.currentTime=2;
  await player.setSpeechActive(false);assert.equal(state.current,talking,'speech ending must not cut the talking clip');
  frame!(performance.now());await flush();assert.equal(state.current,talking);
  decks.find(v=>!v.paused)!.currentTime=9.99;
  frame!(performance.now());await flush();assert.match(state.current,/^idle/);
  const firstIdle=state.current;decks.find(v=>!v.paused)!.currentTime=9.99;
  frame!(performance.now());await flush();assert.match(state.current,/^idle/);assert.notEqual(state.current,firstIdle);
  await player.setSpeechActive(true);const firstTalk=state.current;
  decks.find(v=>!v.paused)!.currentTime=9.99;frame!(performance.now());await flush();
  assert.match(state.current,/^talk/);assert.notEqual(state.current,firstTalk);
  await player.setSpeechActive(false);player.stop();const stopped=state.current;
  await player.start();assert.equal(state.current,stopped,'Stop/Start resumes the draining talking clip');
  decks.find(v=>!v.paused)!.currentTime=9.99;frame!(performance.now());await flush();assert.match(state.current,/^idle/);
  player.setClips([{id:'idle',slot:0,url:'idle',seed:0,kind:'idle',seconds:5},{id:'talk3',slot:1,url:'talk3',seed:0,kind:'talking',seconds:3},{id:'talk7',slot:2,url:'talk7',seed:0,kind:'talking',seconds:7}]);
  player.setSpeechPlan(['talk3','talk7','talk7']);
  await player.setSpeechActive(true);assert.equal(state.current,'talk3');
  decks.find(v=>!v.paused)!.currentTime=3;frame!(performance.now());await flush();assert.equal(state.current,'talk7');
  decks.find(v=>!v.paused)!.currentTime=7;frame!(performance.now());await flush();assert.equal(state.current,'talk7');assert.equal(state.elapsed,0,'a repeated planned duration restarts in the other deck');
  await player.setSpeechActive(false);const joins=state.transitions;
  decks.find(v=>!v.paused)!.currentTime=4;frame!(performance.now());await flush();assert.equal(state.transitions,joins);
  decks.find(v=>!v.paused)!.currentTime=7;frame!(performance.now());await flush();assert.equal(state.current,'idle');
  player.rewindSpeechPlan();await player.setSpeechActive(true);assert.equal(state.current,'talk3','replay starts the original sequence');
  assert.deepEqual(errors,[]);assert.ok(decks.every(v=>!v.loop),'the browser must hold the endpoint while the other deck buffers');
 }finally{player.destroy();for(const key of ['document','requestAnimationFrame','cancelAnimationFrame']){if(saved[key])Object.defineProperty(globalThis,key,saved[key]);else Reflect.deleteProperty(globalThis,key)}}
});
