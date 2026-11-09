import type {RequestHandler} from './$types';
import {LTX25_API_URL} from '$app/env/private';
export const GET:RequestHandler=async({params,request})=>{
 const path=params.path;
 if(!/^(health|v1\/loop-sets(?:\/[a-f0-9]{64}(?:\/image)?)?|v1\/jobs\/[a-f0-9-]{36}\/video)$/.test(path))return new Response('Unknown library endpoint',{status:404});
 try{
  const headers=new Headers();const range=request.headers.get('range');if(range)headers.set('range',range);
  const response=await globalThis.fetch(`${LTX25_API_URL}/${path}`,{headers,signal:request.signal});
  const out=new Headers();for(const key of ['content-type','content-length','content-range','accept-ranges']){const value=response.headers.get(key);if(value)out.set(key,value)}
  return new Response(response.body,{status:response.status,headers:out});
 }catch(error){console.error('Saved-loop library unavailable',error);return Response.json({detail:'The saved-loop server is offline. You can still upload files from disk.'},{status:503})}
};
