import channels from '../studio/channels.json' with { type: 'json' };
import closureProfile from '../../../data/closure-windows.json' with { type: 'json' };
import { CHANNELS, FPS } from './index.mjs';

const mouthChannels = channels.flatMap((name,i) => /^(mouth|jaw)/.test(name) ? [i] : []);
const ease = x => { x = Math.max(0, Math.min(1,x)); return x*x*(3-2*x); };
const clamp = (x,lo,hi) => Math.max(lo,Math.min(hi,x));
const indexes = new WeakMap();

function pauseIndex(dictionary) {
  if (indexes.has(dictionary)) return indexes.get(dictionary);
  const entries = [];
  const center=Math.floor(dictionary.lipRig.neutral.length/2);
  const gap=(clip,f)=>dictionary.lipRig.deltas.reduce((v,d,c)=>v+d[center]*dictionary.weights[(clip.frameOffset+f)*CHANNELS+c],dictionary.lipRig.neutral[center]);
  for (const clip of dictionary.clips) for (const run of clip.runs ?? []) {
    if (!run.pause || run.to >= clip.phones.length) continue;
    const last = clip.phones[run.to-1], next = clip.phones[run.to];
    if (!(last.end <= run.pause.start+1e-6 && run.pause.end <= next.start+1e-6 && next.start <= (clip.frameCount-1)/FPS)) continue;
    // Find a recorded rest before the following sound's preparation, rather
    // than assuming the middle of a quiet waveform has a neutral mouth.
    const advance=(closureProfile.phones[next.phone]?.onsetRangeMs[0] ?? -40)/1000;
    const lo=Math.ceil(run.pause.start*FPS), hi=Math.floor(Math.min(run.pause.end,next.start+advance)*FPS);
    if (hi<lo) continue;
    let rest=lo,best=Infinity;
    for(let f=lo;f<=hi;f++) {
      let activity=0;
      for(const c of mouthChannels) activity+=dictionary.weights[(clip.frameOffset+f)*CHANNELS+c]**2;
      const open=Math.max(0,gap(clip,f)-.1);
      const cost=activity/mouthChannels.length+open**2;
      if(cost<best){best=cost;rest=f;}
    }
    const middle=rest/FPS;
    if(middle<=last.end || middle>=next.start) continue;
    if(gap(clip,rest)>.3) continue;
    const nextRun=clip.runs[clip.runs.indexOf(run)+1];
    entries.push({ clip, run, last, next, middle,
      before:run.to-run.from>1?clip.phones[run.to-2]:undefined,
      after:nextRun&&nextRun.to-nextRun.from>1?clip.phones[run.to+1]:undefined });
  }
  indexes.set(dictionary,entries);
  return entries;
}

function sample(weights, offset, count, time, out) {
  const frame = clamp(time*FPS,0,count-1), a = Math.floor(frame), b = Math.min(count-1,a+1);
  for (let c=0;c<CHANNELS;c++) out[c] = weights[(offset+a)*CHANNELS+c] + (weights[(offset+b)*CHANNELS+c]-weights[(offset+a)*CHANNELS+c])*(frame-a);
}

function choose(dictionary, entries, side, phone, adjacent, boundary, half) {
  let pool = entries.filter(e => (side==='outgoing' ? e.last : e.next).phone === phone.phone);
  const center=Math.floor(dictionary.lipRig.neutral.length/2);
  const gap=(e,f)=>dictionary.lipRig.deltas.reduce((v,d,c)=>v+d[center]*dictionary.weights[(e.clip.frameOffset+f)*CHANNELS+c],dictionary.lipRig.neutral[center]);
  // Reject a release/preparation containing an unrelated large opening excursion.
  pool=pool.filter(e=>{
    const a=Math.round((side==='outgoing'?e.last.end:e.middle)*FPS), b=Math.round((side==='outgoing'?e.middle:e.next.start)*FPS);
    const limit=Math.max(gap(e,a),gap(e,b))+.35;
    for(let f=a;f<=b;f++)if(gap(e,f)>limit)return false;
    return true;
  });
  const pair = pool.filter(e => (side==='outgoing' ? e.before : e.after)?.phone === adjacent?.phone);
  if (pair.length && adjacent) pool = pair;
  let best; const pose = new Float32Array(CHANNELS);
  for (const entry of pool) {
    const anchor = side==='outgoing' ? entry.last.end : entry.next.start;
    sample(dictionary.weights,entry.clip.frameOffset,entry.clip.frameCount,anchor,pose);
    const span = side==='outgoing' ? entry.middle-anchor : anchor-entry.middle;
    const cost = mouthChannels.reduce((s,c)=>s+(pose[c]-boundary[c])**2,0)/mouthChannels.length + .01*Math.log(span/half)**2;
    if (!best || cost<best.cost) best = { ...entry, anchor, slope:clamp(span/half,.5,2), cost, pair: Boolean(adjacent && pair.length) };
  }
  return best;
}

function protectionDistance(phones,time) {
  return Math.min(...phones.map(p => {
    const profile = closureProfile.phones[p.phone];
    const from=p.start + (profile ? profile.onsetRangeMs[0]/1000 - 1/FPS : 0), to=p.end+(profile?.04:0);
    return Math.max(from-time,time-to,0);
  }));
}
export const speechContactProtected=(phones,time)=>protectionDistance(phones,time)===0;

export function applyRecordedPauses(dictionary,frames,plan) {
  const entries = dictionary.lipRig ? pauseIndex(dictionary) : [], transitions = [];
  const diagnostics = { availableRecordings: entries.length, detectedPauses: plan.runs.filter(r=>r.pause).length, usedPauses: 0, unsupportedPauses: 0, rejectedContactCycles: 0, contactLimitedFrames: 0, changedFrames: 0, transitions };
  if (!dictionary.lipRig || !entries.length) { diagnostics.unsupportedPauses=diagnostics.detectedPauses; return diagnostics; }
  const center = Math.floor(dictionary.lipRig.neutral.length/2);
  const gap = pose => dictionary.lipRig.deltas.reduce((v,d,c)=>v+d[center]*pose[c],dictionary.lipRig.neutral[center]);
  const base = new Float32Array(CHANNELS), left = new Float32Array(CHANNELS), right = new Float32Array(CHANNELS), result = new Float32Array(CHANNELS);
  for (let i=0;i+1<plan.runs.length;i++) {
    const run = plan.runs[i], nextRun = plan.runs[i+1];
    if (!run.pause) continue;
    const last = plan.phones[run.to-1], next = plan.phones[nextRun.from];
    const start = last.end, end = next.start, middle = (start+end)/2, half = (end-start)/2;
    if (!(half>0)) continue;
    const startPose = new Float32Array(CHANNELS), endPose = new Float32Array(CHANNELS);
    sample(frames,0,frames.length/CHANNELS,start,startPose);
    sample(frames,0,frames.length/CHANNELS,end,endPose);
    const outgoing = choose(dictionary,entries,'outgoing',last,run.to-run.from>1?plan.phones[run.to-2]:null,startPose,half);
    const incoming = choose(dictionary,entries,'incoming',next,nextRun.to-nextRun.from>1?plan.phones[nextRun.from+1]:null,endPose,half);
    if (!outgoing || !incoming) { diagnostics.unsupportedPauses++; continue; }
    const firstFrame=Math.max(0,Math.ceil(start*FPS)-1), finalFrame=Math.min(frames.length/CHANNELS,Math.ceil(end*FPS)+1);
    const original=frames.slice(firstFrame*CHANNELS,finalFrame*CHANNELS);
    const quietCycles=track=>{
      let count=0,active=false,protectedContact=false,clipped=false;
      for(let f=0;f<track.length/CHANNELS;f++) {
        const closed=gap(track.subarray(f*CHANNELS,(f+1)*CHANNELS))<=.1;
        if(closed){if(!active){active=true;protectedContact=false;clipped=f===0;}
          protectedContact ||= speechContactProtected(plan.phones,(firstFrame+f)/FPS);
        }else if(active){if(!protectedContact&&!clipped)count++;active=false;}
      }
      return count; // A contact reaching the right boundary is not an isolated cycle.
    };
    const leftAnchor = new Float32Array(CHANNELS), rightAnchor = new Float32Array(CHANNELS);
    sample(dictionary.weights,outgoing.clip.frameOffset,outgoing.clip.frameCount,outgoing.anchor,leftAnchor);
    sample(dictionary.weights,incoming.clip.frameOffset,incoming.clip.frameCount,incoming.anchor,rightAnchor);
    const transition={ start,end,quiet:run.pause,leftPhone:last.phone,rightPhone:next.phone,
      outgoing:{ clip:outgoing.clip.id,from:outgoing.anchor,to:outgoing.middle,pair:outgoing.pair },
      incoming:{ clip:incoming.clip.id,from:incoming.middle,to:incoming.anchor,pair:incoming.pair } };
    let changed=0;
    for (let f=Math.ceil(start*FPS);f<Math.min(frames.length/CHANNELS,Math.ceil(end*FPS));f++) {
      const time=f/FPS, at=f*CHANNELS, out=frames.subarray(at,at+CHANNELS);
      base.set(out); result.set(base);
      const lt=Math.min(outgoing.middle,outgoing.anchor+(time-start)*outgoing.slope);
      const rt=Math.max(incoming.middle,incoming.anchor-(end-time)*incoming.slope);
      sample(dictionary.weights,outgoing.clip.frameOffset,outgoing.clip.frameCount,lt,left);
      sample(dictionary.weights,incoming.clip.frameOffset,incoming.clip.frameCount,rt,right);
      const blend=ease((time-start)/(end-start)), margin=Math.min(.15,half);
      const leftCorrection=1-ease((time-start)/margin), rightCorrection=1-ease((end-time)/margin);
      for (const c of mouthChannels) {
        const a=clamp(left[c]+(startPose[c]-leftAnchor[c])*leftCorrection,0,1);
        const b=clamp(right[c]+(endPose[c]-rightAnchor[c])*rightCorrection,0,1);
        result[c]=a+(b-a)*blend;
      }
      let amount=1;
      const distance=protectionDistance(plan.phones,time), influence=1-ease(distance/.08);
      const before=gap(base),after=gap(result);
      const limit=before<=.1 ? .1-1e-6 : .1+1e-6;
      if ((before<=.1)!==(after<=.1)) {
        const safe=clamp((limit-before)/(after-before),0,1);
        amount=1-influence*(1-safe);
        if (amount<1) diagnostics.contactLimitedFrames++;
      }
      for (const c of mouthChannels) out[c]=base[c]+amount*(result[c]-base[c]);
      if (speechContactProtected(plan.phones,time) && (gap(base)<=.1)!==(gap(out)<=.1)) out.set(base);
      if (out.some((v,c)=>v!==base[c])) changed++;
    }
    if(quietCycles(frames.subarray(firstFrame*CHANNELS,finalFrame*CHANNELS))>quietCycles(original)) {
      frames.set(original,firstFrame*CHANNELS);diagnostics.rejectedContactCycles++;
      transitions.push({...transition,rejected:'extra_isolated_quiet_contact'});
    } else {
      diagnostics.usedPauses++;diagnostics.changedFrames+=changed;transitions.push(transition);
    }
  }
  return diagnostics;
}
