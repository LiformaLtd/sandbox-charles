struct Params { offset: u32, first: u32, count: u32, mode: u32, project: u32 }
@group(0) @binding(0) var<storage,read> encoded: array<f32>;
@group(0) @binding(1) var<storage,read> weights: array<f32>;
@group(0) @binding(2) var<storage,read> baseline: array<f32>;
@group(0) @binding(3) var<storage,read_write> result: array<f32>;
@group(0) @binding(4) var<storage,read_write> history: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> current: array<f32,96>;
var<workgroup> past: array<f32,672>;
var<workgroup> hidden: array<f32,16>;

fn gelu(x:f32) -> f32 {
  let argument = clamp(0.7978845608028654 * (x + 0.044715*x*x*x), -10.0, 10.0);
  return 0.5*x*(1.0+tanh(argument));
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) id:u32) {
  let offset=params.offset;
  for(var i=id;i<672u;i+=64u) { past[i]=history[i]; }
  workgroupBarrier();
  for(var frame=0u;frame<params.count;frame++) {
    let t=params.first+frame;
    let pos=f32(t*16000u/60u)/160.0;
    let low=u32(pos); let fraction=pos-f32(low);
    for(var i=id;i<96u;i+=64u) {
      let first=encoded[i*101u+low];
      let feature=first+fraction*(encoded[i*101u+low+1u]-first);
      current[i]=(feature-weights[offset+i])/weights[offset+96u+i];
    }
    workgroupBarrier();
    if(id<16u) {
      var acc=weights[offset+192u+12288u+id];
      for(var lag=0u;lag<8u;lag++) {
        for(var i=0u;i<96u;i++) {
          var x=current[i];
          if(params.mode==1u && lag<7u) { x=past[lag*96u+i]; }
          acc+=x*weights[offset+192u+id*768u+lag*96u+i];
        }
      }
      hidden[id]=gelu(acc);
    }
    workgroupBarrier();
    if(id<39u) {
      var delta=weights[offset+192u+12288u+16u+624u+id];
      for(var h=0u;h<16u;h++) {
        delta+=hidden[h]*weights[offset+192u+12288u+16u+id*16u+h];
      }
      let bias=params.project+39u*96u+id;
      let scale=weights[bias+2u*39u];
      let channel=u32(weights[bias+3u*39u]);
      result[t*52u+channel]=clamp(baseline[t*52u+channel]+delta*scale,0.0,1.0);
    }
    for(var i=id;i<672u;i+=64u) {
      var value=current[i%96u];
      if(i<576u) { value=past[i+96u]; }
      history[i]=value;
    }
    storageBarrier();
    for(var i=id;i<672u;i+=64u) { past[i]=history[i]; }
    workgroupBarrier();
  }
}
