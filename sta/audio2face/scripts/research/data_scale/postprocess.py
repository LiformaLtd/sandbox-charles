"""Wait for the corpus, build locked caches, then execute both matched trajectories."""
from pathlib import Path
import json,os,subprocess,time
from prepare_corpus import APP,OUT
PYTHON=os.environ['EVAL_PYTHON'];TRAIN=os.environ['TRAIN_PYTHON']


def command(name,args):
    log=OUT/(name+'.log')
    with log.open('w') as f:subprocess.run(args,cwd=APP,env=os.environ,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(name,'complete',flush=True)


def main():
    while not (OUT/'corpus/complete.json').exists():
        progress=OUT/'corpus/progress.json'
        if progress.exists() and any(v['stage']=='failed' for v in json.loads(progress.read_text()).values()):raise ValueError('Corpus shard failed; inspect pipeline logs')
        time.sleep(10)
    command('assemble',[PYTHON,'scripts/research/data_scale/assemble.py'])
    command('feature-build',[TRAIN,'scripts/research/data_scale/features.py'])
    command('freeze-training',[PYTHON,'scripts/research/data_scale/freeze_training.py'])
    for data in ['original','expanded']:
        if not (OUT/'trajectories'/data/'complete.json').exists():command('train-'+data,[TRAIN,'scripts/research/data_scale/train.py','--data',data])


if __name__=='__main__':main()
