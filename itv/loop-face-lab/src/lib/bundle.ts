import {unzipSync,strFromU8} from 'fflate';
import {fingerprint} from './media-types.ts';
import type {Workspace,SavedVideo} from './media-types.ts';

export async function readBundle(file:File,text:string):Promise<Workspace>{
 if(file.size>750*1024*1024)throw new Error('Choose a loop ZIP smaller than 750MB.');
 let total=0;const names=new Set<string>();
 const entries=unzipSync(new Uint8Array(await file.arrayBuffer()),{filter:entry=>{
  if(names.has(entry.name)||entry.name.includes('..')||entry.name.startsWith('/')||entry.name.includes('\\'))throw new Error('Invalid or repeated ZIP entry.');
  names.add(entry.name);total+=entry.originalSize;
  if(entry.originalSize>150*1024*1024||total>750*1024*1024||names.size>32)throw new Error('The ZIP expands beyond the supported size.');
  return true;
 }});
 const metadata=entries['manifest.json'];if(!metadata||metadata.length>65536)throw new Error('This ZIP is missing a valid loop manifest.');
 const manifest=JSON.parse(strFromU8(metadata));
 if(manifest.format!=='loop-face-bundle'||manifest.version!==1||!Array.isArray(manifest.clips)||!manifest.clips.length||manifest.clips.length>20)throw new Error('Unsupported loop ZIP format. Export a bundle from Loop Lab.');
 const image=manifest.image;
 if(!image||!['image/png','image/jpeg','image/webp'].includes(image.type)||typeof image.path!=='string'||!Object.hasOwn(entries,image.path))throw new Error('The source image is missing from the ZIP.');
 if(entries[image.path].length>20*1024*1024)throw new Error('The source image is larger than 20MB.');
 const videos:SavedVideo[]=[];const used=new Set<string>([image.path,'manifest.json']);
 for(const clip of manifest.clips){
  if(!clip||!['idle','talking'].includes(clip.kind)||!Number.isFinite(clip.seconds)||clip.seconds<1||clip.seconds>30||typeof clip.path!=='string'||!clip.path.endsWith('.mp4')||!Object.hasOwn(entries,clip.path)||used.has(clip.path))throw new Error('A loop entry is missing or invalid.');
  used.add(clip.path);
  const video=new File([new Uint8Array(entries[clip.path])],clip.path.split('/').pop()!,{type:'video/mp4'});
  const id=await fingerprint(video);
  if(videos.some(v=>v.id===id))throw new Error('The bundle contains duplicate video files.');
  videos.push({id,file:video,name:video.name,kind:clip.kind,seconds:clip.seconds});
 }
 return {image:new File([new Uint8Array(entries[image.path])],image.path,{type:image.type}),videos,text};
}
