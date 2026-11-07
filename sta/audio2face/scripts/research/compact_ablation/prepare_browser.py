from pathlib import Path
import shutil,json,os
APP=Path(__file__).resolve().parents[3]
H=Path(os.environ.get('A2F_ABLATION_OUTPUT',APP/'.local/compact-ablation')).resolve()
WORK=Path(os.environ.get('A2F_RESEARCH_WORK',APP/'.local/research-corpora')).resolve()
WEB=H/'web';WEB.mkdir(parents=True,exist_ok=True)
for name in ('src','public'):
    shutil.copytree(APP/name,WEB/name,dirs_exist_ok=True)
for name in ('index.html','vercel.json','package.json'):shutil.copy(APP/name,WEB/name)
if not (WEB/'node_modules').exists():(WEB/'node_modules').symlink_to(APP/'node_modules',target_is_directory=True)
configuration='''const candidate = new URLSearchParams(location.search).get('candidate') ?? 'baseline';
const hidden = candidate === 'width128' ? 128 : candidate === 'width192' ? 192 : 96;
const frames = candidate.startsWith('context') ? 151 : 101;
const outputFrames = candidate.startsWith('context') ? 90 : 60;
'''
p=WEB/'src/compact-webgpu.js';s=p.read_text().replace('const frames = 101;\nconst hidden = 96;',configuration)
s=s.replace('createLogMel(read(', 'createLogMel(read(').replace("read('mel', 64 * 257));", "read('mel', 64 * 257), frames);")
s=s.replace('make(60 * 52 * 4','make(outputFrames * 52 * 4')
s=s.replace('createPipeline(conv), createPipeline(project)', "createPipeline(conv.replaceAll('101', String(frames))), createPipeline(project.replaceAll('101u', frames+'u').replaceAll('96u',hidden+'u').replace('t >= 60', 't >= '+outputFrames))")
s=s.replace('pass.dispatchWorkgroups(7, 12);','pass.dispatchWorkgroups(Math.ceil(frames / 16), Math.ceil(hidden / 8));')
s=s.replace('pass.dispatchWorkgroups(8, 5);','pass.dispatchWorkgroups(Math.ceil(outputFrames / 8), 5);')
s=s.replace('const dilations = [1, 2, 4, 8, 16, 1, 2, 4];', "const dilations = candidate === 'contextwide' ? [1,2,4,8,16,32,2,4] : [1,2,4,8,16,1,2,4];")
s=s.replace('device.queue.writeBuffer(input, 0, logMel(pcm));', 'const __bench = performance.now();\n      device.queue.writeBuffer(input, 0, logMel(pcm));')
s=s.replace('return new Float32Array(readback.getMappedRange()).slice();','const value = new Float32Array(readback.getMappedRange()).slice();\n        (window.staMs ??= []).push(performance.now() - __bench);\n        return value;')
p.write_text(s)
p=WEB/'src/compact-mel.js';s=p.read_text().replace('const frames = 101;\n','').replace('createLogMel(window, mel)', 'createLogMel(window, mel, frames = 101)');p.write_text(s)
p=WEB/'src/inference.js';s=p.read_text().replace('const url = modelAssetUrl(modelPaths[model]);', "const url = '/candidate.onnx?candidate=' + (new URLSearchParams(location.search).get('candidate') ?? 'baseline');");p.write_text(s)
p=WEB/'src/geometry.mjs';s=p.read_text().replace('export async function* audioWindows(source) {', '''export async function* audioWindows(source) {
  if (new URLSearchParams(location.search).get('candidate')?.startsWith('context')) {
    yield* longWindows(source); return;
  }''')
s+='''
async function* longWindows(source) {
  const chunks = (source[Symbol.asyncIterator] ?? source[Symbol.iterator]).call(source);
  let audio = new Float32Array(24000), length = 0, ended = false;
  try {
    for (let start = -20000; ; start += 8000) {
      while (!ended && length < start + 24000) {
        const {value,done} = await chunks.next(); ended = done;
        if (done) break;
        if (!(value instanceof Float32Array) || !value.every(Number.isFinite)) throw Error('Invalid PCM');
        if (length + value.length > audio.length) {
          const next = new Float32Array(Math.max(audio.length * 2, length + value.length)); next.set(audio.subarray(0,length)); audio=next;
        }
        audio.set(value,length); length += value.length;
      }
      if (ended && start >= length - 8000) return;
      const samples = new Float32Array(24000), left = Math.max(0,start);
      samples.set(audio.subarray(left,Math.min(start+24000,length)),left-start);
      const sampleIndices=[]; let first=0;
      for(let frame=30;frame<60;frame++) {
        const sample=start+Math.floor(frame*16000/60);
        if(sample<0 || sample>=length)continue;
        if(!sampleIndices.length)first=frame;
        sampleIndices.push(sample);
      }
      const pcm=sampleIndices.length ? audio.slice(Math.max(0,start+8000),Math.min(length,start+16000)) : undefined;
      yield {samples,sampleIndices,first,pcm};
    }
  } finally {if(!ended)await chunks.return?.();}
}
''';p.write_text(s)
# Only local model serving differs from the hosted build. Other model URLs remain pinned.
config=f'''import {{ createReadStream }} from 'node:fs';
import {{ stat }} from 'node:fs/promises';
import {{ defineConfig }} from {json.dumps(str(APP/'node_modules/vite/dist/node/index.js'))};
import {{ join }} from 'node:path';
const root={json.dumps(str(H))};
export default defineConfig({{
 root: {json.dumps(str(WEB))},
 cacheDir:join(root,'vite-cache'),
 server:{{host:'127.0.0.1',headers:{{'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'}}}},
 worker:{{format:'es'}},
 optimizeDeps:{{include:['phonon-tts/worker.js'],exclude:['onnxruntime-web/webgpu','onnxruntime-web/wasm']}},
 plugins:[{{name:'candidate-assets',configureServer(server){{server.middlewares.use(async(req,res,next)=>{{
 const url=new URL(req.url,'http://localhost');
 if(!['/candidate.onnx','/fixtures.json'].includes(url.pathname))return next();
 const candidate=url.searchParams.get('candidate')??'baseline';
 const allowed=['baseline','control','width128','width192','context1500','contextwide','data','folded'];
 if(!allowed.includes(candidate)){{res.writeHead(400);return res.end();}}
 const path=candidate==='baseline'?{json.dumps(str(WORK/'arkit-student-v3/james-arkit-student-v3.onnx'))}:join(root,candidate+'.onnx');
 const file=url.pathname==='/fixtures.json'?join(root,candidate+'-parity.json'):path;
 try{{const info=await stat(file);res.writeHead(200,{{'Content-Type':'application/octet-stream','Content-Length':info.size}});createReadStream(file).pipe(res);}}
 catch(e){{res.writeHead(404);res.end(e.message);}}
 }});}}}}]
}});
'''
(H/'vite.config.mjs').write_text(config)
shutil.copy(Path(__file__).with_name('parity.html'),WEB/'parity.html')
for name in ('review.html','review.js'):
    source=Path(__file__).with_name(name)
    if source.exists():shutil.copy(source,WEB/name)
print('Prepared isolated benchmark app:',WEB)
print('Start Vite with --config',H/'vite.config.mjs','--mode hosted; use its printed URL.')
