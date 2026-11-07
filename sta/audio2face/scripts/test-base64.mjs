import {test} from 'node:test';
import assert from 'node:assert/strict';
import {bytesToBase64} from '../src/base64.js';
test('large tokenizer bytes encode without an unbounded argument list',()=>{
 const bytes=Uint8Array.from({length:1_000_003},(_,i)=>i%256);
 assert.equal(bytesToBase64(bytes),Buffer.from(bytes).toString('base64'));
});
test('empty and partial final chunks preserve the exact byte content',()=>{
 for(const length of [0,1,8191,8192,8193]){
  const bytes=Uint8Array.from({length},(_,i)=>(i*17)%256);
  assert.equal(bytesToBase64(bytes),Buffer.from(bytes).toString('base64'));
 }
});
