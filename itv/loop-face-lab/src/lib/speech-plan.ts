export interface MotionClip{id:string;kind?:'idle'|'talking';seconds?:number}
/** Balance the rounded-up audio duration across the fewest available talking clips. */
export function planSpeech<T extends MotionClip>(duration:number,clips:T[],random:()=>number=Math.random):T[]{
 if(!Number.isFinite(duration)||duration<=0)throw new Error('Speech duration must be positive.');
 const talking=clips.filter(c=>c.kind!=='idle'&&Number.isFinite(c.seconds)&&c.seconds!>0);
 if(!talking.length)throw new Error('Load at least one talking video with a known duration.');
 const maximum=Math.max(...talking.map(c=>c.seconds!));
 const count=Math.ceil(Math.ceil(duration)/maximum),result:T[]=[];
 let remaining=Math.ceil(duration);
 for(let index=0;index<count;index++){
  const desired=Math.floor(remaining/(count-index));
  const lengths=talking.filter(c=>c.seconds!>=desired);
  const shortest=Math.min(...lengths.map(c=>c.seconds!));
  const candidates=lengths.filter(c=>c.seconds===shortest);
  const selected=candidates[Math.min(candidates.length-1,Math.floor(random()*candidates.length))];
  result.push(selected);remaining=Math.max(0,remaining-selected.seconds!);
 }
 return result;
}
