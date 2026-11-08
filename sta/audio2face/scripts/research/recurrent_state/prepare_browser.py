"""Research-only third pipeline; production FFT, convolution and projection unchanged."""
from common import *
def main():
 root=OUT/'browser';root.mkdir(exist_ok=True);source=(APP/'src/compact-webgpu.js').read_text()
 for name in ['compact-conv.wgsl','compact-project.wgsl','compact-mel.js']:source=source.replace("'./"+name,"'/src/"+name)
 source="import recurrent from './recurrent.wgsl?raw';\n"+source
 source=source.replace('createCompactWebGpu(adapter, data, onLost)','createCompactWebGpu(adapter, data, onLost, mode = 1)')
 source=source.replace('const [convPipeline, projectPipeline]','const [convPipeline, projectPipeline, recurrentPipeline]').replace('createPipeline(conv), createPipeline(project)','createPipeline(conv), createPipeline(project), createPipeline(recurrent)')
 source=source.replace('39 * (hidden + 4);','39 * (hidden + 4) + 13703;')
 source=source.replace('const projection = bind(', 'const projectOffset = offset;\n    const projection = bind(')
 source=source.replace('device.lost.then(onLost);', '''const decoderOffset = offset;
    for (const [name, length] of [['cell_Wx',9216],['cell_Wh',3072],['cell_b',96],['cell_bhn',32],['head_W',1248],['head_b',39]]) append(read(name,length));
    if(offset!==weightCount) throw Error('Recurrent weight packing mismatch');
    const history=make(128,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const params=uniform([decoderOffset,15,30,mode,projectOffset]);
    const recurrentGroup=bind(recurrentPipeline,[a,model,result,history,params]);
    device.lost.then(onLost);''')
 source=source.replace('return async function runCompact(pcm) {','''async function runCompact(pcm,{first=15,count=30}={}) {
      if(!Number.isInteger(first)||!Number.isInteger(count)||first<15||count<0||count>30||first+count>45) throw Error('Invalid emitted crop');
      device.queue.writeBuffer(params,0,new Uint32Array([decoderOffset,first,count,mode,projectOffset]));''')
 source=source.replace('pass.end();','''pass.setPipeline(recurrentPipeline);pass.setBindGroup(0,recurrentGroup);pass.dispatchWorkgroups(1);
      pass.end();''')
 source=source.replace('''    };
  } catch (error)''','''    }
    runCompact.reset=()=>device.queue.writeBuffer(history,0,new Float32Array(32));
    runCompact.dispose=()=>device.destroy();
    return runCompact;
  } catch (error)''')
 freeze(root/'recurrent-runner.js',source.encode());freeze(root/'control-runner.js',b"import {createCompactWebGpu as create} from './recurrent-runner.js';\nexport const createCompactWebGpu=(a,b,c)=>create(a,b,c,0);\n")
 freeze(root/'recurrent.wgsl',(HERE/'recurrent.wgsl').read_bytes());freeze(root/'source-lock.json',dict(source_sha256=sha(__file__),shader_sha256=sha(HERE/'recurrent.wgsl'),runner_sha256=sha(root/'recurrent-runner.js'),frontend_sha256=sha(APP/'src/compact-mel.js'),conv_sha256=sha(APP/'src/compact-conv.wgsl'),project_sha256=sha(APP/'src/compact-project.wgsl')))
if __name__=='__main__':main()
