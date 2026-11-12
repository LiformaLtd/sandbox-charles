"""Wait for teacher targets, calibrate/freeze, then train both arms without test scoring."""
import subprocess,time
from common import *
def command(name,args):
 with (OUT/(name+'.log')).open('w') as f:subprocess.run(args,cwd=APP,env=os.environ,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
def main():
 required=[OUT/'surface-primary-lock.json']+[OUT/'surface-maps'/n/'complete.json' for n in ['old-train','new-train','common-dev']]
 while not all(p.exists() for p in required):time.sleep(5)
 train=os.environ['TRAIN_PYTHON'];python=os.environ['EVAL_PYTHON']
 command('calibration',[train,'scripts/research/lip_surface/calibrate.py'])
 command('freeze-training',[python,'scripts/research/lip_surface/freeze_training.py'])
 for arm in ['control','surface']:
  if not(OUT/'trajectories'/arm/'complete.json').exists():command('train-'+arm,[train,'scripts/research/lip_surface/train.py','--arm',arm])
if __name__=='__main__':main()
