"""Synthesize/align a frozen confirmation set, then teacher-label every changed waveform."""
from concurrent.futures import ThreadPoolExecutor
import subprocess,numpy as np
from common import *
def main():
 base=OUT/'confirmation/base';target=OUT/'confirmation/variants';logs=OUT/'confirmation/logs';logs.mkdir(exist_ok=True)
 env=dict(os.environ,A2F_MULTIVOICE_OUTPUT=str(base));node=os.environ['NODE22'];python=os.environ['EVAL_PYTHON']
 def command(name,args,e=env):
  with (logs/(name+'.log')).open('w') as f:subprocess.run(args,cwd=APP,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(name,'complete',flush=True)
 with ThreadPoolExecutor(2) as pool:
  list(pool.map(lambda engine:command(engine,[node,'scripts/research/multivoice/synthesize.mjs',engine]),['phonon','supertonic']))
 command('align',[node,'--experimental-strip-types','scripts/research/multivoice/align.mjs'])
 command('verify',[python,'scripts/research/multivoice/verify.py'])
 rows=[]
 for r in json.loads((base/'manifest.json').read_text()):
  pcm=np.fromfile(r['audio'],'<f4');boost=min(2.,.98/max(1e-8,float(np.abs(pcm).max())))
  for condition,gain in [('original',1.),('half',.5),('boost',boost),('quarter',.25)]:
   row=dict(r,condition=condition,stem=r['stem']+'-'+condition,gain=gain)
   audio=target/'audio'/(row['stem']+'.f32');freeze(audio,(pcm*np.float32(gain)).astype('<f4').tobytes());row.update(audio=str(audio),sha256_audio=sha(audio));rows.append(row)
 freeze(target/'manifest.json',rows)
 for p in (base/'sources').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:freeze(target/'sources'/p.relative_to(base/'sources'),p.read_bytes())
 freeze(target/'instrumentation-parity.json',(base/'instrumentation-parity.json').read_bytes())
 command('label',[python,'scripts/research/multivoice/label.py'],dict(env,A2F_MULTIVOICE_OUTPUT=str(target)))
 freeze(OUT/'confirmation/complete.json',dict(cases=len(rows),manifest_sha256=sha(target/'manifest.json'),recipe_sha256=sha(target/'label-recipe.json'),pipeline_sha256=sha(__file__)))
 print('All384 confirmation cases complete.',flush=True)
if __name__=='__main__':main()
