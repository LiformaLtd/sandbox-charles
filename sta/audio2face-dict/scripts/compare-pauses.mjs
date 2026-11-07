import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { planAnimation, bakeAnimation } from '../src/lib/dictionary/animate.mjs';
import { CHANNELS, FPS } from '../src/lib/dictionary/index.mjs';
import { speechContactProtected } from '../src/lib/dictionary/recorded-pauses.mjs';
import { contactEvents } from './lib/closure-events.mjs';
import closureProfile from '../data/closure-windows.json' with { type:'json' };
import { neutralContactProtected } from '../src/lib/dictionary/neutral-transition.mjs';
import { checkAdvanceClosures, summarizeAdvanceClosures } from './lib/closure-quality.mjs';

const json=async p=>JSON.parse(await readFile(p));
const floats=async p=>{ const b=await readFile(p);return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)); };
const dictionary=await json('static/dictionary/index.json');
dictionary.weights=await floats('static/dictionary/'+dictionary.weightsFile);
const rig=dictionary.lipRig, n=rig.neutral.length, center=Math.floor(n/2);
const gaps=track=>Float32Array.from({length:track.length/CHANNELS*n},(_,i)=>{
  const f=Math.floor(i/n),j=i%n;return rig.deltas.reduce((v,d,c)=>v+d[j]*track[f*CHANNELS+c],rig.neutral[j]);
});
const full=process.argv.includes('--all');
const smoothComparison=process.argv.includes('--smooth-neutral');
const withRecorded=smoothComparison&&process.argv.includes('--with-recorded-pauses');
const variants=['baseline',smoothComparison?'smoothed':'recorded'];
const contactProtected=smoothComparison?neutralContactProtected:speechContactProtected;
const reviewDirectory=smoothComparison?`.local/neutral-review${withRecorded?'-recorded':''}`:'.local/pause-review';
const sets=full?['development','fresh','test']:['development'];
const reports=[];
for(const split of sets) {
  for(const paired of [false,true]) {
    const directory=paired?`.local/pause-pilot/${split}`:`.local/${split==='fresh'?'fresh':'full'}`;
    const rows=(await json(directory+'/clips.json')).filter(r=>paired||split==='fresh'||r.split===split);
    const aggregate={sentences:rows.length,pausesDetected:0,pausesUsed:0,changedFrames:0,speechContactFramesChanged:0,
      baseline:{squares:0,values:0,pauseSquares:0,pauseValues:0,stepSquares:0,steps:0,accelerationSquares:0,accelerations:0,peakGapStep:0,neutralFrames:0},
      [variants[1]]:{squares:0,values:0,pauseSquares:0,pauseValues:0,stepSquares:0,steps:0,accelerationSquares:0,accelerations:0,peakGapStep:0,neutralFrames:0}};
    for(const variant of variants)Object.assign(aggregate[variant],{tailSquares:0,tailValues:0,internalSquares:0,internalValues:0,coefficientStepSquares:0,coefficientSteps:0,peakCoefficientStep:0,coefficientAccelerationSquares:0,coefficientAccelerations:0});
    const clips=[];
    const advanceChecks=Object.fromEntries(variants.map(name=>[name,[]]));
    for(const row of rows) {
      const [pcm,native,fitted]=await Promise.all(['pcm','geometry-gaps','lip.arkit'].map(ext=>floats(`${directory}/${row.id}.${ext}`)));
      const plans=[false,true].map(enabled=>planAnimation(dictionary,row.phones,pcm,{boundedContext:false,
        recordedPauses:smoothComparison?withRecorded:enabled,smoothNeutral:smoothComparison&&enabled}));
      assert.deepEqual(plans[0].coverage,plans[1].coverage);assert.deepEqual(plans[0].usage,plans[1].usage);
      const frames=plans.map(p=>bakeAnimation(dictionary,p)), tracks=frames.map(gaps);
      if(withRecorded)for(const transition of plans[0].pauseDiagnostics.transitions.filter(t=>!t.rejected)) {
        const from=Math.ceil(transition.start*FPS)*CHANNELS,to=Math.ceil(transition.end*FPS)*CHANNELS;
        assert.deepEqual(frames[0].subarray(from,to),frames[1].subarray(from,to),'Changed an accepted recorded pause: '+row.text);
      }
      assert.equal(tracks[0].length,native.length);assert.equal(tracks[1].length,native.length);
      assert.ok(frames[1].every(v=>Number.isFinite(v)&&v>=0&&v<=1));
      const pauses=plans[1].pauseDiagnostics??{usedPauses:0,changedFrames:0,transitions:[]};
      aggregate.pausesDetected+=plans[0].runs.filter(r=>r.pause).length;
      aggregate.pausesUsed+=pauses.usedPauses;aggregate.changedFrames+=smoothComparison?plans[1].neutralDiagnostics.changedFrames:pauses.changedFrames;
      const intervals=plans[0].runs.slice(0,-1).map((r,i)=>[row.phones[r.to-1].end,row.phones[plans[0].runs[i+1].from].start]);
      const tailStart=row.phones.at(-1).end;
      if(smoothComparison&&tailStart<plans[0].duration)intervals.push([tailStart,plans[0].duration]);
      const inside=t=>intervals.some(([a,b])=>t>=a&&t<=b);
      const near=t=>intervals.some(([a,b])=>t>=a-.05&&t<=b+.05);
      let changedSpeech=0,changedOutsidePause=0,changedContacts=0;
      for(let f=0;f<frames[0].length/CHANNELS;f++) {
        const time=f/FPS;
        if((tracks[0][f*n+center]<=.1)!==(tracks[1][f*n+center]<=.1))changedContacts++;
        if(contactProtected(row.phones,time)) {
          if((tracks[0][f*n+center]<=.1)!==(tracks[1][f*n+center]<=.1)) changedSpeech++;
        }
        if(!inside(time)) for(let c=0;c<CHANNELS;c++) if(frames[0][f*CHANNELS+c]!==frames[1][f*CHANNELS+c])changedOutsidePause++;
        for(const [i,name] of variants.entries()) {
          const tally=aggregate[name],track=tracks[i];
          for(let j=0;j<n;j++) {
            const at=f*n+j,error=track[at]-native[at];tally.squares+=error**2;tally.values++;
            if(inside(time)){tally.pauseSquares+=error**2;tally.pauseValues++;}
            if(smoothComparison&&time>=tailStart){tally.tailSquares+=error**2;tally.tailValues++;}
            if(smoothComparison&&time<tailStart&&inside(time)){tally.internalSquares+=error**2;tally.internalValues++;}
            if(f&&near((f-.5)/FPS)) {
              const step=track[at]-track[at-n],refStep=native[at]-native[at-n];
              tally.stepSquares+=(step-refStep)**2;tally.steps++;tally.peakGapStep=Math.max(tally.peakGapStep,Math.abs(step));
            }
            if(f>=2&&near((f-1)/FPS)){
              const acc=track[at]-2*track[at-n]+track[at-2*n];
              const refAcc=native[at]-2*native[at-n]+native[at-2*n];
              tally.accelerationSquares+=(acc-refAcc)**2;tally.accelerations++;
            }
          }
          if(f&&near((f-.5)/FPS))for(let c=0;c<CHANNELS;c++){
            const delta=frames[i][f*CHANNELS+c]-frames[i][(f-1)*CHANNELS+c];
            tally.coefficientStepSquares+=delta**2;tally.coefficientSteps++;tally.peakCoefficientStep=Math.max(tally.peakCoefficientStep,Math.abs(delta));
          }
          if(f>=2&&near((f-1)/FPS))for(let c=0;c<CHANNELS;c++){
            const acc=frames[i][f*CHANNELS+c]-2*frames[i][(f-1)*CHANNELS+c]+frames[i][(f-2)*CHANNELS+c];
            tally.coefficientAccelerationSquares+=acc**2;tally.coefficientAccelerations++;
          }
          if(inside(time)&&frames[i].subarray(f*CHANNELS,(f+1)*CHANNELS).every(v=>Math.abs(v)<1e-6))tally.neutralFrames++;
        }
      }
      assert.equal(changedSpeech,0,'Changed speech contact mask: '+row.text);
      assert.equal(changedOutsidePause,0,'Changed speech animation: '+row.text);
      // Compare episode masks on every speech run, including advance windows
      // for a first P/B/M. Quiet-rest contact episodes are reported separately.
      const speechRuns=plans[0].runs.map(r=>{
        const phones=row.phones.slice(r.from,r.to);
        const from=Math.max(0,Math.ceil(Math.min(...phones.map(p=>p.start+(closureProfile.phones[p.phone]?closureProfile.phones[p.phone].onsetRangeMs[0]/1000-1/FPS:0)))*FPS));
        const to=Math.min(native.length/n,Math.ceil(Math.max(...phones.map(p=>p.end+(!smoothComparison&&closureProfile.phones[p.phone]?.04:0)))*FPS));
        const masks=tracks.map(t=>Array.from({length:to-from},(_,i)=>t[(from+i)*n+center]<=.1));
        assert.deepEqual(masks[0],masks[1],'Changed a speech closure episode: '+row.text);
        const count=mask=>mask.reduce((s,v,i)=>s+Number(v&&!mask[i-1]),0);
        return {from,to,baselineEpisodes:count(masks[0]),recordedEpisodes:count(masks[1]),identicalMask:true};
      });
      const contactDetails=Object.fromEntries([...variants,'native'].map((name,i)=>{
        const track=i===2?native:tracks[i];const centers=Float32Array.from({length:track.length/n},(_,f)=>track[f*n+center]);
        return [name,contactEvents(centers).map(e=>({...e,quietOnly:!Array.from({length:Math.max(0,Math.floor(e.end*FPS)-Math.ceil(e.start*FPS)+1)},(_,i)=>contactProtected(row.phones,(Math.ceil(e.start*FPS)+i)/FPS)).some(Boolean)}))];
      }));
      const nativeCenter=Float32Array.from({length:native.length/n},(_,f)=>native[f*n+center]);
      const closureChecks=Object.fromEntries(variants.map((name,i)=>{
        const actual=Float32Array.from({length:tracks[i].length/n},(_,f)=>tracks[i][f*n+center]);
        const checks=checkAdvanceClosures(row.phones,actual,nativeCenter,closureProfile);
        advanceChecks[name].push(...checks);return[name,checks];
      }));
      if(smoothComparison)for(let i=0;i<closureChecks.baseline.length;i++)assert.equal(
        closureChecks[variants[1]][i].classification,closureChecks.baseline[i].classification,
        'Changed a P/B/M advance-window closure result: '+row.text);
      const quietContacts=Object.fromEntries(Object.entries(contactDetails).map(([name,events])=>[name,events.filter(e=>e.quietOnly).length]));
      assert.ok(quietContacts[variants[1]]<=quietContacts.baseline,'Added an isolated quiet contact episode: '+row.text);
      if(smoothComparison)assert.ok(contactDetails[variants[1]].length<=contactDetails.baseline.length,'Added a contact episode: '+row.text);
      const clip={id:row.id,text:row.text,intervals,diagnostics:pauses,speechContactFramesChanged:changedSpeech,changedOutsidePause,speechRuns,contacts:contactDetails,
        ...(smoothComparison?{neutralDiagnostics:plans[1].neutralDiagnostics,contactFramesChanged:changedContacts}:{}),
        advanceClosureChecks:closureChecks,
        quietContacts,
        lipGapRms:Object.fromEntries(variants.map((name,i)=>[name,Math.sqrt(tracks[i].reduce((s,v,j)=>s+(v-native[j])**2,0)/native.length)]))};
      clips.push(clip);
      if(pauses.usedPauses||smoothComparison) {
        await mkdir(reviewDirectory,{recursive:true});
        for(const [i,name] of variants.entries())await writeFile(`${reviewDirectory}/${row.id}.${name}.f32`,Buffer.from(frames[i].buffer));
        await writeFile(`${reviewDirectory}/${row.id}.native-fit.f32`,Buffer.from(fitted.buffer));
        await writeFile(`${reviewDirectory}/${row.id}.pcm`,Buffer.from(pcm.buffer));
      }
    }
    for(const name of variants) {
      const x=aggregate[name];x.lipGapRms=Math.sqrt(x.squares/x.values);
      x.pauseLipGapRms=x.pauseValues?Math.sqrt(x.pauseSquares/x.pauseValues):null;
      x.transitionStepErrorRms=x.steps?Math.sqrt(x.stepSquares/x.steps):null;
      x.transitionAccelerationErrorRms=x.accelerations?Math.sqrt(x.accelerationSquares/x.accelerations):null;
      x.coefficientStepRms=x.coefficientSteps?Math.sqrt(x.coefficientStepSquares/x.coefficientSteps):null;
      x.coefficientAccelerationRms=x.coefficientAccelerations?Math.sqrt(x.coefficientAccelerationSquares/x.coefficientAccelerations):null;
      x.tailLipGapRms=x.tailValues?Math.sqrt(x.tailSquares/x.tailValues):null;
      x.internalPauseLipGapRms=x.internalValues?Math.sqrt(x.internalSquares/x.internalValues):null;
    }
    reports.push({split,paired,aggregate,advanceWindowClosures:Object.fromEntries(variants.map(name=>[name,summarizeAdvanceClosures(advanceChecks[name])])),clips});
  }
}
const report={version:1,method:smoothComparison?'Identical recorded speech and full native references. Previous 25ms neutral fade vs smooth neutral transition; bounded feature-context experiment off in both.':'Identical recorded speech; paired audio receives new full native Audio2Face inference on the entire waveform. Current vs internal-recorded-pause pilot. Bounded context experiment off in both.',
  ...(smoothComparison?{comparison:'25ms linear fade vs 200ms quintic neutral transition, capped to half the quiet interval for internal pauses and available trailing duration for tails. Speech-contact masks protected; resting contact timing may move.',recordedPauses:withRecorded}:{}),
  dictionaryWeights:dictionary.weightsFile,trainingPauseCount:dictionary.clips.reduce((s,c)=>s+c.runs.filter(r=>r.pause).length,0),
  indexSha256:createHash('sha256').update(await readFile('static/dictionary/index.json')).digest('hex'),reports};
await writeFile(smoothComparison?`.local/neutral-transition-comparison${withRecorded?'-recorded':''}.json`:'.local/pause-pilot-comparison.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(reports.map(({split,paired,aggregate})=>({split,paired,...aggregate})),null,2));
