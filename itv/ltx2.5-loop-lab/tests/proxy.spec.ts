import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';

test('real browser multipart POST reaches upstream without forwarding Origin; foreign origins stay blocked',async({page,request})=>{
 await page.goto('/');
 const image=(await readFile(new URL('./fixtures/start.png',import.meta.url))).toString('base64');
 const result=await page.evaluate(async(encoded)=>{
  const image=Uint8Array.from(atob(encoded),character=>character.charCodeAt(0));
  const form=new FormData();form.set('image',new Blob([image],{type:'image/png'}),'first.png');form.set('prompt','A calm repeating scene');form.set('width','512');form.set('height','512');
  const response=await fetch('/api/v1/loop-sets',{method:'POST',body:form});
  return {status:response.status,body:await response.json()};
 },image);
 expect(result.status).toBe(202);
 expect(result.body.received_origin).toBeNull();
 expect(result.body.bytes).toBeGreaterThan(100);
 expect(result.body.content_type).toContain('multipart/form-data; boundary=');
 const before=await(await request.get(process.env.LOOPLAB_STUB_URL+'/stats')).json();
 const forbidden=await request.post('/api/v1/loop-sets',{headers:{Origin:'https://untrusted.example'},multipart:{prompt:'Disallowed'}});
 expect(forbidden.status()).toBe(403);
 const after=await(await request.get(process.env.LOOPLAB_STUB_URL+'/stats')).json();
 expect(after.posts).toBe(before.posts);
});
