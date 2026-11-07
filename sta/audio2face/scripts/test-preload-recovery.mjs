import {test} from 'node:test';
import assert from 'node:assert/strict';
import {installPreloadRecovery} from '../src/preload-recovery.js';
function fixture({status=404,saveFails=false,noStorage=false,message='Failed to fetch dynamically imported module: https://demo.test/assets/voice-old.js'}={}){
 const target=new EventTarget(),data=new Map(),calls={reloads:0,saved:0,fetches:0};
 target.location={href:'https://demo.test/?model=compact-v4',origin:'https://demo.test',reload(){calls.reloads++;}};
 target.sessionStorage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v)};
 target.document={querySelector:()=>({src:'https://demo.test/assets/index-old.js'})};
 if(noStorage)Object.defineProperty(target,'sessionStorage',{get(){throw Error('Unavailable');}});
 installPreloadRecovery({target,now:()=>100_000,fetchAsset:async()=>{calls.fetches++;return {status};},saveDraft(){if(saveFails)throw Error('Cannot preserve input');calls.saved++;}});
 const fire=async()=>{const event=new Event('vite:preloadError');event.payload={message};target.dispatchEvent(event);await new Promise(r=>setImmediate(r));};
 return {calls,fire};
}
test('missing same-origin chunk preserves draft and refreshes once',async()=>{const f=fixture();await f.fire();assert.deepEqual(f.calls,{reloads:1,saved:1,fetches:1});await f.fire();assert.equal(f.calls.reloads,1);});
test('existing assets or server errors do not trigger a refresh',async()=>{for(const status of [200,503]){const f=fixture({status});await f.fire();assert.equal(f.calls.reloads,0);}});
test('cross-origin errors are ignored',async()=>{const f=fixture({message:'Failed to fetch https://elsewhere.test/assets/voice.js'});await f.fire();assert.equal(f.calls.fetches,0);});
test('input/storage failure never causes data-losing refresh',async()=>{for(const option of [{saveFails:true},{noStorage:true}]){const f=fixture(option);await f.fire();assert.equal(f.calls.reloads,0);}});
test('URL-less WebKit error checks the old entry asset',async()=>{const f=fixture({message:'Importing a module script failed.'});await f.fire();assert.equal(f.calls.reloads,1);});
