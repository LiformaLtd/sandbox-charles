struct Params { offset:u32, first:u32, count:u32, mode:u32, project:u32 }
@group(0) @binding(0) var<storage,read> encoded:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> result:array<f32>;
@group(0) @binding(3) var<storage,read_write> state:array<f32>;
@group(0) @binding(4) var<uniform> params:Params;
var<workgroup> feature:array<f32,96>;
var<workgroup> old:array<f32,32>;
var<workgroup> next:array<f32,32>;
fn sigmoid(x:f32)->f32 {return 1.0/(1.0+exp(-x));}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) id:u32) {
 if(id<32u){old[id]=state[id];}
 workgroupBarrier();
 for(var frame=0u;frame<params.count;frame++){
  let t=params.first+frame;let pos=f32(t*16000u/60u)/160.0;let low=u32(pos);let fraction=pos-f32(low);
  for(var i=id;i<96u;i+=64u){let a=encoded[i*101u+low];feature[i]=a+fraction*(encoded[i*101u+low+1u]-a);}
  workgroupBarrier();
  if(id<32u){
   var r=weights[params.offset+12288u+id];var z=weights[params.offset+12288u+32u+id];var n=weights[params.offset+12288u+64u+id];var hn=weights[params.offset+12384u+id];
   for(var i=0u;i<96u;i++){let x=tanh(feature[i]/4.0);r+=x*weights[params.offset+id*96u+i];z+=x*weights[params.offset+(id+32u)*96u+i];n+=x*weights[params.offset+(id+64u)*96u+i];}
   for(var i=0u;i<32u;i++){let h=select(0.0,old[i],params.mode==1u);r+=h*weights[params.offset+9216u+id*32u+i];z+=h*weights[params.offset+9216u+(id+32u)*32u+i];hn+=h*weights[params.offset+9216u+(id+64u)*32u+i];}
   let gate=sigmoid(z);next[id]=(1.0-gate)*tanh(n+sigmoid(r)*hn)+gate*select(0.0,old[id],params.mode==1u);
  }
  workgroupBarrier();
  if(id<39u){
   let bias=params.project+3744u+id;
   var raw=weights[bias]+weights[params.offset+12416u+1248u+id];
   for(var i=0u;i<96u;i++){raw+=feature[i]*weights[params.project+id*96u+i];}
   for(var i=0u;i<32u;i++){raw+=next[i]*weights[params.offset+12416u+id*32u+i];}
   let mean=weights[bias+39u];let scale=weights[bias+78u];let channel=u32(weights[bias+117u]);
   result[t*52u+channel]=clamp(raw*scale+mean,0.0,1.0);
  }
  workgroupBarrier();
  if(id<32u){old[id]=next[id];}
  workgroupBarrier();
 }
 if(id<32u){state[id]=old[id];}
}
