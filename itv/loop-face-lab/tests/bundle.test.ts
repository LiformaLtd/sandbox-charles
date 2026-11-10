import {test} from 'node:test';
import assert from 'node:assert/strict';
import {zipSync,strToU8} from 'fflate';
import {readBundle} from '../src/lib/bundle.ts';
const manifest={format:'loop-face-bundle',version:1,image:{path:'source.png',type:'image/png'},clips:[{path:'talking/7s.mp4',kind:'talking',seconds:7},{path:'idle/5s.mp4',kind:'idle',seconds:5}]};
function bundle(value:unknown=manifest,extra:Record<string,Uint8Array>={}){return new File([new Uint8Array(zipSync({'manifest.json':strToU8(JSON.stringify(value)),'source.png':new Uint8Array([1]),'talking/7s.mp4':new Uint8Array([2]),'idle/5s.mp4':new Uint8Array([3]),...extra},{level:0}))],'loops.zip')}
test('one portable ZIP restores source image, groups and durations',async()=>{
 const scene=await readBundle(bundle(),'Keep this speech');assert.equal(scene.text,'Keep this speech');assert.equal(scene.image?.type,'image/png');
 assert.deepEqual(scene.videos.map(v=>[v.kind,v.seconds]),[['talking',7],['idle',5]]);assert.notEqual(scene.videos[0].id,scene.videos[1].id);
});
test('incomplete, unsupported or unsafe ZIPs fail before replacing the scene',async()=>{
 await assert.rejects(readBundle(bundle({...manifest,version:2}),''),/Unsupported/);
 await assert.rejects(readBundle(bundle({...manifest,clips:[{path:'missing.mp4',kind:'talking',seconds:7}]}),''),/missing or invalid/);
 await assert.rejects(readBundle(bundle(manifest,{'../escape':new Uint8Array()}),''),/Invalid/);
 await assert.rejects(readBundle(bundle({...manifest,clips:[{...manifest.clips[0],seconds:-1}]}),''),/missing or invalid/);
});
