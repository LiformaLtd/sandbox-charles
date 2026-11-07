import { CHANNELS, FPS } from './index.mjs';
import closureProfile from '../../../data/closure-windows.json' with { type:'json' };

// Quintic ease: zero velocity and acceleration at both ends.
export function restEase(progress) {
  const x=Math.max(0,Math.min(1,progress));
  return Math.max(0,Math.min(1,x*x*x*(10+x*(-15+6*x))));
}

export function neutralContactProtected(phones,time) {
  return phones.some(p=>time>=p.start+(closureProfile.phones[p.phone]?closureProfile.phones[p.phone].onsetRangeMs[0]/1000-1/FPS:0)&&time<=p.end);
}

function protectedDistance(phones,time) {
  // Ease into upcoming speech protection. Do not drag the old resting-contact
  // timing forwards after a completed sound, which would retain its fast snap.
  return Math.min(...phones.map(p=>time>p.end?Infinity:Math.max(p.start+(closureProfile.phones[p.phone]?closureProfile.phones[p.phone].onsetRangeMs[0]/1000-1/FPS:0)-time,0)));
}

export function applySmoothNeutral(dictionary,frames,plan,{seconds=.2}={}) {
  if(!Number.isFinite(seconds)||seconds<=0)throw new RangeError('Neutral transition duration must be positive');
  const diagnostics={changedFrames:0,contactLimitedFrames:0,intervals:[],recordedPausesSkipped:0,rejectedContactEpisodes:0};
  const accepted=plan.pauseDiagnostics?.transitions.filter(t=>!t.rejected)??[];
  const center=dictionary.lipRig?Math.floor(dictionary.lipRig.neutral.length/2):0;
  const gap=pose=>dictionary.lipRig.deltas.reduce((v,d,c)=>v+d[center]*pose[c],dictionary.lipRig.neutral[center]);
  const leftPose=new Float32Array(CHANNELS),rightPose=new Float32Array(CHANNELS),base=new Float32Array(CHANNELS),candidate=new Float32Array(CHANNELS);
  for(let i=0;i<plan.segments.length;i++) {
    const segment=plan.segments[i];
    if(segment.pieces.length||segment.kind)continue; // Unmatched voiced intervals are not silence.
    const left=plan.segments[i-1]?.pieces.at(-1),right=plan.segments[i+1]?.pieces[0];
    if(!left)continue; // Preserve initial speech preparation and opening-context fix.
    if(accepted.some(t=>Math.abs(t.start-segment.start)<1e-6&&Math.abs(t.end-segment.end)<1e-6)) {
      diagnostics.recordedPausesSkipped++;continue;
    }
    const from=segment.pause?.start??segment.start,to=segment.pause?.end??segment.end;
    if(!(to>from))continue;
    const fade=Math.min(seconds,(to-from)/(right?2:1));
    // Relax from the actual last baked speech pose, including any closure
    // correction, rather than jumping back to its uncorrected donor endpoint.
    const leftFrame=Math.max(0,Math.floor(segment.start*FPS));
    leftPose.set(frames.subarray(leftFrame*CHANNELS,(leftFrame+1)*CHANNELS));
    if(right) {
      const rightFrame=Math.min(frames.length/CHANNELS-1,Math.ceil(segment.end*FPS));
      rightPose.set(frames.subarray(rightFrame*CHANNELS,(rightFrame+1)*CHANNELS));
    }
    const details={kind:right?'pause':'tail',start:segment.start,end:segment.end,quietStart:from,quietEnd:to,fadeSeconds:fade};
    const firstFrame=Math.max(0,Math.ceil(segment.start*FPS)-1),lastFrame=Math.min(frames.length/CHANNELS,Math.ceil(segment.end*FPS)+1);
    const original=frames.slice(firstFrame*CHANNELS,lastFrame*CHANNELS);
    let changed=0;
    for(let f=Math.floor(segment.start*FPS)+1;f<Math.min(frames.length/CHANNELS,Math.ceil(segment.end*FPS));f++) {
      const time=f/FPS,out=frames.subarray(f*CHANNELS,(f+1)*CHANNELS);
      base.set(out);candidate.fill(0);
      const outgoing=1-restEase((time-from)/fade),incoming=right?restEase((time-(to-fade))/fade):0;
      for(let c=0;c<CHANNELS;c++)candidate[c]=leftPose[c]*outgoing+(right?rightPose[c]*incoming:0);
      let amount=1;
      // Preserve speech contacts, including anticipation inside silence, but
      // let ordinary resting contact move later with the slower relaxation.
      if(dictionary.lipRig) {
        const before=gap(base),after=gap(candidate);
        if((before<=.1)!==(after<=.1)) {
          const limit=before<=.1?.1-1e-6:.1+1e-6;
          const safe=Math.max(0,Math.min(1,(limit-before)/(after-before)));
          const influence=1-restEase(protectedDistance(plan.phones,time)/.08);
          amount=1-influence*(1-safe);
          if(amount<1)diagnostics.contactLimitedFrames++;
        }
      }
      for(let c=0;c<CHANNELS;c++)out[c]=base[c]+amount*(candidate[c]-base[c]);
      if(dictionary.lipRig&&neutralContactProtected(plan.phones,time)&&(gap(base)<=.1)!==(gap(out)<=.1))out.set(base);
      if(out.some((v,c)=>v!==base[c]))changed++;
    }
    const episodes=track=>{let count=0,closed=false;for(let f=0;f<track.length/CHANNELS;f++){
      const now=gap(track.subarray(f*CHANNELS,(f+1)*CHANNELS))<=.1;if(now&&!closed)count++;closed=now;
    }return count;};
    if(dictionary.lipRig&&episodes(frames.subarray(firstFrame*CHANNELS,lastFrame*CHANNELS))>episodes(original)) {
      frames.set(original,firstFrame*CHANNELS);diagnostics.rejectedContactEpisodes++;details.rejected='extra_contact_episode';
    }else diagnostics.changedFrames+=changed;
    diagnostics.intervals.push(details);
  }
  return diagnostics;
}
