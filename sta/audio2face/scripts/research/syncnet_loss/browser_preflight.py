"""Actual-image parity and predeclared numerical-gradient descent checks."""
from common import *
if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
from browser_auxiliary import BrowserExpert
from features import Dataset
from mlx_backend import Student
import numpy as np,torch,onnx,cv2,time
import mlx.core as mx
from onnx import numpy_helper

def main():
 expert=BrowserExpert();w=np.fromfile(EXPERT/'tracks/case-00-correct.f32','<f4').reshape(-1,52);indices=[25,30,35,40];actual=expert.images(w[indices]);expected=np.stack([cv2.resize(cv2.imread(str(EXPERT/f'frames/case-00/correct/face/{i:04}.png'))[16:176,32:192],(224,224)) for i in indices]);delta=np.abs(actual.astype(float)-expected.astype(float));assert delta.max()<=2 and delta.mean()<.01,(delta.max(),delta.mean())
 freeze(OUT/'browser-gradient-protocol.json',dict(source_sha256=sha(__file__),renderer_sha256=sha(HERE/'browser_renderer.mjs'),page_sha256=sha(HERE/'browser-render.html'),auxiliary_sha256=sha(HERE/'browser_auxiliary.py'),directions=16,epsilon=.003,descent_max_coefficient_step=.001,minimum_descent_wins=6,examples='First example from each voice in seed37 first main batch; training data only; raw distance gradients regardless of hinge activity'))
 data=Dataset(['old-train','new-train']);voices=sorted(set(data.voices));rng=np.random.default_rng(1037);ix=np.concatenate([rng.choice(np.flatnonzero(data.voices==voice),6,replace=True) for voice in voices]);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};net=Student(values);rows=[];began=time.monotonic()
 for voice in range(8):
  index=next(int(i) for i in ix[voice*6:voice*6+6] if expert.mapping[int(i)]['eligible']);batch=data.batch([index]);pred=np.array(net(mx.array(batch['mel']))[:,15:45])[0];weights=expert.coefficients(torch.tensor(pred),index).numpy();base=float(expert.scores(weights[None],index)[0]);g,detail=expert.estimate(weights,index,37000+voice);direction=g/max(float(np.abs(g).max()),1e-8)*.001;scores=expert.scores(np.stack([np.clip(weights-direction,0,1),np.clip(weights+direction,0,1)]),index);rows.append(dict(voice=voices[voice],index=index,base=base,descent=float(scores[0]),ascent=float(scores[1]),descent_win=bool(scores[0]<base),direction_win=bool(scores[0]<scores[1]),gradient_norm=float(np.linalg.norm(g)),**detail));print(voices[voice],rows[-1],round(time.monotonic()-began),flush=True)
 wins=sum(r['descent_win'] and r['direction_win'] for r in rows);passes=wins>=6 and all(sum(r['descent_win'] and r['direction_win'] for r in rows if r['voice'].startswith(e))>=2 for e in ['phonon','supertonic']);freeze(OUT/'browser-gradient.json',dict(rows=rows,wins=wins,passes=bool(passes),pixel_max_delta=float(delta.max()),pixel_mean_delta=float(delta.mean()),seconds=time.monotonic()-began));print('passes',passes,'wins',wins,flush=True)
if __name__=='__main__':main()
