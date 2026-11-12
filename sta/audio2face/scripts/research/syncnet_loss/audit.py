"""Reconstruct sampler/reset recipe and validate every locked data/source artifact."""
from common import *
import numpy as np
from features import Dataset

def main():
 p=json.loads((OUT/'training-protocol.json').read_text());assert sha(BASELINE)==EXPECTED
 cache=json.loads((OUT/'cache-lock.json').read_text());assert sha(OUT/'audio-features.npy')==cache['audio_sha256'];assert sha(OUT/'aux-map.json')==cache['map_sha256'];assert sha(EXPERT/'vendor/syncnet_v2.model')==cache['sync_checkpoint_sha256']
 expert_protocol=json.loads((EXPERT/'protocol.json').read_text())
 for name,digest in expert_protocol['rig_sha256'].items():assert sha(APP/'.local'/name)==digest,name
 for name in ['src/rig.js','src/blendshape-rig.js']:assert sha(APP/name)==expert_protocol['sources'][name],name
 for name,digest in p['source_sha256'].items():assert sha(APP/name)==digest,name
 for name,digest in p['array_sha256'].items():assert sha(SCALE/name)==digest,name
 assert sha(OUT/'confirmation/audit.json')==p['confirmation_audit_sha256']
 train=Dataset(['old-train','new-train']);replay=Dataset(['replay-train','extra-train']);voices=sorted(set(train.voices));groups=[np.flatnonzero(train.voices==v) for v in voices];rng=np.random.default_rng(1037);sampler=hashlib.sha256()
 for step in range(3720):
  ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);old=rng.integers(replay.length,size=16);sampler.update(ix.tobytes()+old.tobytes())
 result={}
 for arm in ['control','syncnet']:
  run=OUT/'trajectories'/arm;complete=json.loads((run/'complete.json').read_text());assert complete['sampler_sha256']==sampler.hexdigest();assert complete['steps']==3720;selection=json.loads((OUT/arm/'selection.json').read_text());assert selection['best_step']==complete['best_step'] and sha(OUT/arm/'model.onnx')==selection['model_sha256'];assert selection['bytes']==2251827 and selection['parameters']==331047
  ck=json.loads((run/'checkpoint.json').read_text());assert ck['step']==3720 and sha(run/ck['checkpoint_file'])==ck['checkpoint_sha256'];assert ck['rng']==rng.bit_generator.state
  result[arm]=dict(selected_step=selection['best_step'],sampler_sha256=sampler.hexdigest(),checkpoint_sha256=ck['checkpoint_sha256'],parameters=selection['parameters'])
 history=json.loads((OUT/'trajectories/syncnet/auxiliary-history.json').read_text());mapping=json.loads((OUT/'aux-map.json').read_text());expected=[];rng=np.random.default_rng(1037)
 for step in range(1,3721):
  ix=np.concatenate([rng.choice(g,6,replace=True) for g in groups]);rng.integers(replay.length,size=16)
  if step%4==0:
   voice=(step//4-1)%8;eligible=[int(i) for i in ix[voice*6:voice*6+6] if mapping[int(i)]['eligible']]
   if eligible:expected.append((step,eligible[0]))
 assert [(r['step'],r['index']) for r in history]==expected
 assert all(np.isfinite(r['loss']) and np.isfinite(r['coefficient_gradient_norm']) for r in history)
 assert all(not r['active'] or (r['directions']==16 and r['probe_epsilon']==.003) for r in history)
 result['auxiliary']=dict(updates=len(history),active_updates=sum(r['active'] for r in history),schedule_identical=True,voice_counts={str(v):int(sum(train.voices[r['index']]==v for r in history)) for v in voices})
 a=json.loads((OUT/'trajectories/control/initial-dev.json').read_text());b=json.loads((OUT/'trajectories/syncnet/initial-dev.json').read_text());assert a==b
 freeze(OUT/'training-audit.json',dict(arms=result,initial_dev_identical=True,source_sha256=sha(__file__),frozen_data=True));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
