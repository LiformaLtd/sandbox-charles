import {Input,BlobSource,ALL_FORMATS,VideoSampleSink} from 'mediabunny';
import {read,write} from './cache';
import {TRACK_REVISION} from './media-types';
import type {FaceFit,FaceFrame,FaceTrack} from './media-types';

export class FaceTracker{
 private worker:Worker|undefined;private next=0;
 private pending=new Map<number,{resolve:(fit:FaceFit|null)=>void;reject:(error:Error)=>void}>();
 async detect(image:ImageBitmap):Promise<FaceFit|null>{
  if(!this.worker){
   this.worker=new Worker(new URL('./tracking.worker.ts',import.meta.url),{type:'module'});
   this.worker.onmessage=({data})=>{const request=this.pending.get(data.id);this.pending.delete(data.id);if(data.error)request?.reject(new Error(data.error));else request?.resolve(data.fit)};
   this.worker.onerror=event=>{for(const request of this.pending.values())request.reject(new Error(event.message||'Face tracking worker failed'));this.pending.clear();this.worker?.terminate();this.worker=undefined};
  }
  const id=++this.next;
  return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});this.worker!.postMessage({id,image},[image])});
 }
 async fitImage(image:File,hash:string,signal:AbortSignal):Promise<FaceFit>{
  const key=`fit:${TRACK_REVISION}:${hash}`,cached=await read<FaceFit>(key);if(cached)return cached;
  signal.throwIfAborted();const fit=await this.detect(await createImageBitmap(image));signal.throwIfAborted();
  if(!fit)throw new Error('No face found in the source image. Choose a clear image with one visible face.');
  await write(key,fit);return fit;
 }
 async video(file:File,hash:string,signal:AbortSignal,progress:(message:string)=>void):Promise<FaceTrack>{
  const key=`track:${TRACK_REVISION}:${hash}`,cached=await read<FaceTrack>(key);
  if(cached){progress(`Reused ${cached.frames.length} cached frames`);return cached}
  const input=new Input({source:new BlobSource(file),formats:ALL_FORMATS});
  try{
   const track=await input.getPrimaryVideoTrack();if(!track)throw new Error('This file has no video track.');
   if(!await track.canDecode())throw new Error('This browser cannot decode the video. Use Chrome with an H.264 MP4.');
   const duration=await input.computeDuration();if(!Number.isFinite(duration)||duration<=0||duration>30)throw new Error('Choose video loops up to 30 seconds long.');
   const sink=new VideoSampleSink(track),frames:(FaceFrame|null)[]=[],times:number[]=[];let missing=0,width=0,height=0;
   // Decode every presentation frame in order; never sample a playing video.
   for await(const sample of sink.samples()){
    try{
     signal.throwIfAborted();width=sample.displayWidth;height=sample.displayHeight;
     const canvas=new OffscreenCanvas(width,height);sample.draw(canvas.getContext('2d')!,0,0);
     const fit=await this.detect(canvas.transferToImageBitmap());signal.throwIfAborted();
     times.push(sample.timestamp);frames.push(fit?{...fit,time:sample.timestamp}:null);if(!fit)missing++;
     progress(`Tracking frame ${frames.length} · ${Math.min(100,Math.round(sample.timestamp/duration*100))}%`);
    }finally{sample.close()}
   }
   if(!frames.length||missing===frames.length)throw new Error('No face found in this loop. Choose a clip with a visible face.');
   const result={revision:TRACK_REVISION,width,height,duration,frames,times,missing};await write(key,result);return result;
  }finally{input.dispose()}
 }
 dispose(){this.worker?.terminate();this.worker=undefined;for(const p of this.pending.values())p.reject(new DOMException('Tracking cancelled','AbortError'));this.pending.clear()}
}
