import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ShapeSampler,parseMapping} from '../src/lib/shape.ts';
import {LipContactConstraint,INNER_UP,INNER_DOWN} from '../src/lib/contact.ts';
const read=(path:string)=>JSON.parse(readFileSync(new URL(path,import.meta.url),'utf8'));
const catalog=read('../static/assets/catalog.json');
const fixture=read('./fixtures/source-seal.json');
const nonSeal=Float32Array.from(fixture.nonSealWeights);
const sealCases=[['appointment-p',Float32Array.from(fixture.weights)],['matched-3d-am',Float32Array.from(read('./fixtures/am-3d-reference.json').weights)]] as const;
const outerUp=[61,185,40,39,37,0,267,269,270,409,291],outerDown=[61,146,91,181,84,17,314,405,321,375,291];
const area=(p:Float32Array,a:number,b:number,c:number)=>(p[b*2]-p[a*2])*(p[c*2+1]-p[a*2+1])-(p[b*2+1]-p[a*2+1])*(p[c*2]-p[a*2]);
for(const [caseName,seal] of sealCases)for(const kind of ['linear','local'])test(`${caseName} ${kind}: zero-gap source seals preserve lip thickness and triangle orientation across all fits`,()=>{
 for(const avatar of catalog.avatars){
  const folder=`../static/assets/avatars/${avatar.id}`;const mesh=read(folder+'/mesh.json');const map=parseMapping(read(folder+`/${kind}.json`));
  const sampler=new ShapeSampler(mesh,map,catalog.contact);sampler.sample(seal,1,false,1);const base=sampler.local.slice();
  const constraint=new LipContactConstraint(mesh,map.reference);const next=base.slice();constraint.apply(next,1);
  for(let p=1;p<10;p++)for(let axis=0;axis<2;axis++){
   assert.equal(next[INNER_UP[p]*2+axis],next[INNER_DOWN[p]*2+axis],`${avatar.id} exact contact`);
   for(const [inner,outer] of [[INNER_UP,outerUp],[INNER_DOWN,outerDown]])assert.ok(Math.abs(Math.hypot(next[outer[p]*2]-next[inner[p]*2],next[outer[p]*2+1]-next[inner[p]*2+1])-Math.hypot(base[outer[p]*2]-base[inner[p]*2],base[outer[p]*2+1]-base[inner[p]*2+1]))<1e-6,`${avatar.id} rim thickness`);
  }
  assert.equal(constraint.unresolvedTriangles,0,`${avatar.id} unresolved folds`);
  for(let t=0;t<mesh.triangles.length;t+=3){const [a,b,c]=mesh.triangles.slice(t,t+3);if(Math.max(a,b,c)>=478)continue;
   const before=area(base,a,b,c),after=area(next,a,b,c),rest=area(map.reference,a,b,c);if(Math.abs(rest)>1e-8&&before*rest>0&&Math.abs(before)>1e-9)assert.ok(after*Math.sign(rest)>=-1e-8,`${avatar.id} new fold ${a},${b},${c}`);
  }
  for(const strength of [.5,1,1.5]){
   sampler.sample(seal,strength,true,1);for(let p=1;p<10;p++)for(let axis=0;axis<2;axis++)assert.equal(sampler.local[INNER_UP[p]*2+axis],sampler.local[INNER_DOWN[p]*2+axis]);
  }
  sampler.sample(nonSeal,1,false,1);const open=sampler.local.slice();sampler.sample(nonSeal,1,true,1);assert.deepEqual(sampler.local,open,`${avatar.id} no invented source closure`);
  const untouched=base.slice();constraint.apply(untouched,0);assert.deepEqual(untouched,base);
 }
});

test('contact capture preserves the calibrated open visemes',()=>{
 const calibration=read('../static/assets/contact-calibration.json');const mesh=read('../static/assets/avatars/0c72c586/mesh.json');
 const sampler=new ShapeSampler(mesh,parseMapping(read('../static/assets/avatars/0c72c586/linear.json')),catalog.contact);
 assert.ok(calibration.capture.full<calibration.capture.release);
 for(const pose of calibration.poses){const weights=Float32Array.from(pose.weights);
  sampler.sample(weights,1,false,1);const raw=sampler.local.slice();sampler.sample(weights,1,true,1);
  if(!pose.closed)assert.deepEqual(sampler.local,raw,pose.role+' stays open/unmodified');
  else for(let p=1;p<10;p++)for(let axis=0;axis<2;axis++)assert.equal(sampler.local[INNER_UP[p]*2+axis],sampler.local[INNER_DOWN[p]*2+axis],pose.role+' seals');
 }
});
