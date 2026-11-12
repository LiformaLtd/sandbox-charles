"""Reconstruct sampler/reset recipe and validate every locked data/source artifact."""
from common import *
import numpy as np
from features import Dataset

def main():
 p=json.loads((OUT/'training-protocol.json').read_text());assert sha(BASELINE)==EXPECTED
 for name,digest in p['source_sha256'].items():assert sha(APP/name)==digest,name
 for name,digest in p['array_sha256'].items():assert sha(SCALE/name)==digest,name
 assert sha(OUT/'confirmation/audit.json')==p['confirmation_audit_sha256']
 train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);voices=sorted(set(train.voices));groups=[np.flatnonzero(train.voices==v) for v in voices];rng=np.random.default_rng(1037);sampler=hashlib.sha256()
 for step in range(3720):
  ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);sampler.update(ix.tobytes()+old.tobytes())
 result={}
 for arm in ['control','protected']:
  run=OUT/'trajectories'/arm;complete=json.loads((run/'complete.json').read_text());assert complete['sampler_sha256']==sampler.hexdigest();assert complete['steps']==3720;selection=json.loads((OUT/arm/'selection.json').read_text());assert selection['best_step']==complete['best_step'] and sha(OUT/arm/'model.onnx')==selection['model_sha256'];assert selection['bytes']==2251827 and selection['parameters']==331047
  ck=json.loads((run/'checkpoint.json').read_text());assert ck['step']==3720 and sha(run/ck['checkpoint_file'])==ck['checkpoint_sha256'];assert ck['rng']==rng.bit_generator.state
  result[arm]=dict(selected_step=selection['best_step'],sampler_sha256=sampler.hexdigest(),checkpoint_sha256=ck['checkpoint_sha256'],parameters=selection['parameters'])
 guards=json.loads((OUT/'trajectories/protected/checkpoint.json').read_text())['guard_history'];assert len(guards)==3720
 assert [r['step'] for r in guards]==list(range(1,3721))
 for r in guards:
  assert all(v<=limit for v,limit in zip(r['new_guards'],r['guard_limits']))
  assert r['accepted_scale'] in [0.,1.,.5,.25,.125,.0625]
  if r['accepted_scale']==0:assert r['actual_step_norm']<=1e-10
 result['protection']=dict(scheduled_steps=len(guards),nonzero_steps=sum(r['actual_step_norm']>1e-10 for r in guards),projected_steps=sum(r['projected'] for r in guards),backtracked_steps=sum(0<r['accepted_scale']<1 for r in guards),skipped_steps=sum(r['accepted_scale']==0 for r in guards),all_batch_guards_pass=True)
 a=json.loads((OUT/'trajectories/control/initial-dev.json').read_text());b=json.loads((OUT/'trajectories/protected/initial-dev.json').read_text());assert a==b
 freeze(OUT/'training-audit.json',dict(arms=result,initial_dev_identical=True,source_sha256=sha(__file__),frozen_data=True));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
