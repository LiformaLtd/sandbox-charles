struct Params { offset:u32, residual:u32, early:u32 }
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read_write> result:array<f32>;
@group(0) @binding(4) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> early_input:array<f32>;
var<workgroup> branch:array<f32,96>;
var<workgroup> audio:array<f32,96>;
var<workgroup> prediction:array<f32,39>;
var<workgroup> head:array<f32,96>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
  let t=group.x;
  let o=params.offset;
  let pos=f32(t*16000u/60u)/160.0;
  let low=u32(pos);
  let fraction=pos-f32(low);
  for(var i=lane;i<96u;i+=64u){
    let first=input[i*101u+low];
    audio[i]=first+fraction*(input[i*101u+low+1u]-first);
    if(params.early>0u){
      let e=early_input[i*101u+low];
      branch[i]=e+fraction*(early_input[i*101u+low+1u]-e);
    }else{branch[i]=audio[i];}
  }
  workgroupBarrier();
  if(lane<39u){
    var acc=weights[o+3744u+lane];
    for(var i=0u;i<96u;i++){acc+=audio[i]*weights[o+lane*96u+i];}
    prediction[lane]=acc;
  }
  workgroupBarrier();
  // Extra tensors follow the unchanged v4 output/bias/mean/scale/channel map.
  for(var c=lane;c<96u;c+=64u){
    var acc=weights[o+16956u+c];
    for(var i=0u;i<135u;i++){
      var feature:f32;
      if(i<96u){feature=branch[i]/weights[o+3900u+i];}
      else{feature=prediction[i-96u];}
      acc+=feature*weights[o+3996u+c*135u+i];
    }
    let argument=clamp(.7978845608*(acc+.044715*acc*acc*acc),-10.0,10.0);
    head[c]=.5*acc*(1.0+tanh(argument));
  }
  workgroupBarrier();
  if(lane<39u){
    var acc=weights[o+20796u+lane];
    for(var i=0u;i<96u;i++){acc+=head[i]*weights[o+17052u+lane*96u+i];}
    acc+=f32(params.residual)*prediction[lane];
    let channel=u32(weights[o+3861u+lane]);
    result[t*52u+channel]=clamp(acc*weights[o+3822u+lane]+weights[o+3783u+lane],0.0,1.0);
  }
}
