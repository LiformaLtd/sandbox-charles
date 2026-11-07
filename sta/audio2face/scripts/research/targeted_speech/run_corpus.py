"""Resumable independent synthesis/alignment jobs with bounded teacher concurrency."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os,subprocess,threading,time
from common import APP,OUT,HERE,sha,freeze
ROOT=OUT/'corpus';status={};mutex=threading.Lock();teacher_slots=threading.Semaphore(2)
NODE=os.environ['NODE22'];PYTHON=os.environ['EVAL_PYTHON']


def mark(shard,stage):
    with mutex:
        status[shard]=dict(stage=stage,at=time.time())
        temp=ROOT/'progress.part.json';temp.write_text(json.dumps(status,indent=2));temp.replace(ROOT/'progress.json')
        print(shard,stage,flush=True)


def run(shard):
    base=ROOT/'base-shards'/shard;label=ROOT/'label-shards'/shard;done=ROOT/'done'/(shard+'.json')
    signature={name:sha(APP/name) for name in ['scripts/research/multivoice/synthesize.mjs','scripts/research/multivoice/align.mjs','scripts/research/multivoice/label.py','scripts/research/data_scale/prepare_shard.py']}
    if done.exists():
        if json.loads(done.read_text())['source_sha256']!=signature:raise ValueError('Changed completed pipeline')
        mark(shard,'complete');return
    env=dict(os.environ,A2F_MULTIVOICE_OUTPUT=str(base))
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    def command(name,args,output=None):
        with (logs/(shard+'-'+name+'.log')).open('w') as f:subprocess.run(args,cwd=APP,env=env if output is None else dict(env,A2F_MULTIVOICE_OUTPUT=str(output)),stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        mark(shard,'synthesizing');command('synthesize',[NODE,'scripts/research/multivoice/synthesize.mjs',shard.split('-')[0]])
        mark(shard,'aligning');command('align',[NODE,'--experimental-strip-types','scripts/research/multivoice/align.mjs'])
        mark(shard,'waiting-for-teacher')
        with teacher_slots:
            mark(shard,'verifying');command('verify',[PYTHON,'scripts/research/multivoice/verify.py'])
            command('prepare-gains',[PYTHON,'scripts/research/data_scale/prepare_shard.py',shard])
            mark(shard,'labelling');command('label',[PYTHON,'scripts/research/multivoice/label.py'],label)
        rows=json.loads((label/'manifest.json').read_text())
        freeze(done,dict(source_sha256=signature,manifest_sha256=sha(label/'manifest.json'),recipe_sha256=sha(label/'label-recipe.json'),examples=len(rows)))
        mark(shard,'complete')
    except Exception:
        mark(shard,'failed');raise


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen corpus source changed')
    lock=ROOT/'pipeline.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        shards=sorted(p.name for p in (ROOT/'base-shards').iterdir())
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed([pool.submit(run,s) for s in shards]):f.result()
        freeze(ROOT/'complete.json',dict(shards=shards,source_sha256=sha(__file__),independent_originals=704))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
