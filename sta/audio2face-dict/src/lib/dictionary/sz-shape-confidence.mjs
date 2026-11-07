import { CHANNELS, FPS } from './index.mjs';
import { matchShapeDistance } from './select-joins.mjs';

function selectedMatches(plan) {
  const matches=new Map();
  for(const segment of plan.segments)if(segment.first&&segment.kind!=='fallback')for(const piece of segment.pieces) {
    const match=piece.match,at=plan.phones.indexOf(segment.first)-match.source.indexOf(piece.first);
    if(at>=0)matches.set(match,at);
  }
  return matches;
}
const identity=match=>match.clip.id+':'+match.source[0].start+':'+match.source.at(-1).end;

export function makeSZShapeRegions(dictionary,plan,referencePlan) {
  const selected=selectedMatches(plan),reference=new Map();
  for(const [match,at] of selectedMatches(referencePlan))if(!match.approximate&&match.count===3)reference.set(at,match);
  const regions=[...selected].filter(([m])=>m.approximate?.pair==='S/Z').map(([match,at])=>({
    targetStart:plan.phones[at].start,pair:'S/Z',at,start:plan.phones[at].start,end:plan.phones[at+2].end,
    confidence:1,shapeDistance:0,changedExactTriples:[],
  }));
  for(const [match,at] of selected) {
    const anchor=reference.get(at);
    if(match.approximate||match.count!==3||!anchor||identity(match)===identity(anchor))continue;
    const run=plan.runs.find(r=>at>=r.from&&at<r.to);
    if(!run)continue;
    const nearby=regions.filter(r=>r.at>=run.from&&r.at<run.to).sort((a,b)=>Math.abs(a.at-at)-Math.abs(b.at-at));
    const region=nearby[0];if(!region)continue;
    const distance=matchShapeDistance(dictionary.weights,dictionary.lipRig,match,anchor);
    region.shapeDistance=Math.max(region.shapeDistance,distance);
    region.start=Math.min(region.start,plan.phones[at].start);
    region.end=Math.max(region.end,plan.phones[at+2].end);
    region.changedExactTriples.push({at,selectedClip:match.clip.id,referenceClip:anchor.clip.id,distance});
  }
  // A substituted clip must not dominate an already exact context solely to
  // obtain a cheaper join. Exact trajectories retain at least three quarters
  // of the contribution; larger shape disagreements reduce confidence more.
  for(const region of regions)if(region.changedExactTriples.length)region.confidence=.25/(1+region.shapeDistance);
  return regions;
}

export function applySZShapeConfidence(dictionary,frames,reference,regions) {
  if(frames.length!==reference.length)throw new Error('S/Z shape reference has a different duration');
  const diagnostics={regions,changedFrames:0,contactLimitedFrames:0};
  const center=dictionary.lipRig?Math.floor(dictionary.lipRig.neutral.length/2):0;
  const gap=pose=>dictionary.lipRig.deltas.reduce((sum,row,c)=>sum+row[center]*pose[c],dictionary.lipRig.neutral[center]);
  const original=new Float32Array(CHANNELS),candidate=new Float32Array(CHANNELS);
  const ease=x=>{x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);};
  const margin=2/FPS;
  for(let f=0;f<frames.length/CHANNELS;f++) {
    const time=f/FPS;
    const confidence=Math.min(1,...regions.filter(r=>time>=r.start-margin&&time<=r.end+margin).map(r=>
      1-(1-r.confidence)*ease((time-r.start+margin)/margin)*ease((r.end+margin-time)/margin)));
    if(confidence===1)continue;
    const out=frames.subarray(f*CHANNELS,(f+1)*CHANNELS),base=reference.subarray(f*CHANNELS,(f+1)*CHANNELS);
    original.set(out);
    for(let c=0;c<CHANNELS;c++)candidate[c]=base[c]+confidence*(original[c]-base[c]);
    let amount=1;
    if(dictionary.lipRig) {
      const before=gap(original),after=gap(candidate);
      if((before<=.1)!==(after<=.1)) {
        const limit=before<=.1?.1-1e-6:.1+1e-6;
        amount=Math.max(0,Math.min(1,(limit-before)/(after-before)));diagnostics.contactLimitedFrames++;
      }
    }
    for(let c=0;c<CHANNELS;c++)out[c]=original[c]+amount*(candidate[c]-original[c]);
    if(dictionary.lipRig&&(gap(original)<=.1)!==(gap(out)<=.1))out.set(original);
    if(out.some((v,c)=>v!==original[c]))diagnostics.changedFrames++;
  }
  return diagnostics;
}
