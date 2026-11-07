"""Wait for bounded data jobs, verify/cache/freeze inputs, run the three fixed arms."""
import subprocess,time
from common import *
def command(name,args):
 with (OUT/(name+'.log')).open('w') as f:subprocess.run(args,cwd=APP,env=os.environ,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
def main():
 while not(OUT/'corpus/complete.json').exists():
  p=OUT/'corpus/progress.json'
  if p.exists() and any(r['stage']=='failed' for r in json.loads(p.read_text()).values()):raise ValueError('Corpus failed; inspect shard logs')
  time.sleep(5)
 python=os.environ['EVAL_PYTHON'];train=os.environ['TRAIN_PYTHON']
 command('assemble',[python,'scripts/research/targeted_speech/assemble.py'])
 command('build-features',[train,'scripts/research/targeted_speech/build_features.py'])
 command('freeze-training',[python,'scripts/research/targeted_speech/freeze_training.py'])
 for arm in ['control','broad','targeted']:
  if not(OUT/'trajectories'/arm/'complete.json').exists():command('train-'+arm,[train,'scripts/research/targeted_speech/train.py','--arm',arm])
if __name__=='__main__':main()
