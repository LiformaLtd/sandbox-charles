"""One frozen-feature, normalized coefficient least-squares readout diagnostic."""
from common import *
import numpy as np,onnx,mlx.core as mx
from onnx import numpy_helper
from mlx_backend import Student,gelu
from features import Dataset

def encoded(net,mel):
 x=gelu(net.encoder(mel.transpose(0,2,1)))
 for block in net.blocks:x=block(x)
 lo=mx.take(x,net.left,axis=1);hi=mx.take(x,net.left+1,axis=1)
 return (lo+net.fraction*(hi-lo))[:,15:45]

def main():
 path=MODELS['control'];assert sha(path)=='63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867'
 freeze(OUT/'head-protocol.json',dict(source_sha256=sha(__file__),baseline_sha256=sha(path),method='Frozen encoder/blocks/interpolation; weighted normalized coefficient MSE optimum by eigensolve; no temporal/contact penalties, no fine-tuning backbone',primary_frame_weight=.75,replay_frame_weight=.25,primary_voice_weights='equal eight voices, frame means within voice',eigen_relative_cutoff=1e-9,selection='One fixed solution; no development/test selection or weight sweep',interpretation='Readout optimality diagnostic, not a matched SGD-budget release study'))
 values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(path).graph.initializer};net=Student(values);mx.set_default_device(mx.gpu);groups={};parity=[]
 for cohort,ds in [('primary',Dataset(['old-train','new-train'])),('replay',Dataset(['replay-train','extra-train']))]:
  for start in range(0,ds.length,128):
   ix=np.arange(start,min(start+128,ds.length));batch=ds.batch(ix);mel=mx.array(batch['mel']);x=np.array(encoded(net,mel));raw=np.array(net(mel)[:,15:45]);rebuilt=np.einsum('ntc,oc->nto',x,values['student.output.weight'][:,:,0],optimize=False)+values['student.output.bias'];rebuilt=rebuilt*values['student.scale']+values['student.mean'];delta=float(np.abs(raw-rebuilt).max());assert delta<5e-5;parity.append(delta)
   valid=batch['valid'];target=(batch['weights']-values['student.mean'])/values['student.scale'];voices=ds.voices[ix]
   for voice in sorted(set(voices)):
    take=voices==voice;mask=valid[take];features=np.concatenate([x[take][mask].astype(np.float64),np.ones((int(mask.sum()),1))],1);y=target[take][mask].astype(np.float64);key=(cohort,str(voice));r=groups.setdefault(key,dict(H=np.zeros((97,97)),J=np.zeros((97,39)),Y=np.zeros(39),n=0));r['H']+=np.einsum('ni,nj->ij',features,features,optimize=False);r['J']+=np.einsum('ni,nj->ij',features,y,optimize=False);r['Y']+=np.square(y).sum(0);r['n']+=len(features)
  print('accumulated',cohort,ds.length,'windows',flush=True)
 H=np.zeros((97,97));J=np.zeros((97,39));Y=np.zeros(39)
 for (cohort,voice),r in groups.items():
  weight=(.75/8 if cohort=='primary' else .25)/r['n'];H+=weight*r['H'];J+=weight*r['J'];Y+=weight*r['Y']
 mu=H[:-1,-1];scale=np.sqrt(np.maximum(np.diag(H)[:-1]-mu*mu,1e-20));cov=(H[:-1,:-1]-np.outer(mu,mu))/np.outer(scale,scale);cross=(J[:-1]-mu[:,None]*J[-1])/scale[:,None];eigen,V=np.linalg.eigh(cov);take=eigen>1e-9*eigen.max();beta=np.einsum('ij,jk->ik',V[:,take],np.einsum('ij,ik->jk',V[:,take],cross,optimize=False)/eigen[take,None],optimize=False);slope=beta/scale[:,None];bias=J[-1]-np.einsum('i,ij->j',mu,slope,optimize=False);solution=np.concatenate([slope,bias[None]],0)
 previous=np.concatenate([values['student.output.weight'][:,:,0].T,values['student.output.bias'][None]],0).astype(np.float64)
 def mse(w):return np.einsum('ic,ij,jc->c',w,H,w,optimize=False)-2*np.einsum('ic,ic->c',w,J,optimize=False)+Y
 before,after=mse(previous),mse(solution);assert after.mean()<=before.mean()+1e-8;normal_residual=np.linalg.norm(H@solution-J)/max(np.linalg.norm(J),1e-20)
 graph=onnx.load(path);newvalues={'student.output.weight':slope.T[:,:,None].astype(np.float32),'student.output.bias':bias.astype(np.float32)}
 for i,v in enumerate(graph.graph.initializer):
  if v.name in newvalues:graph.graph.initializer[i].CopyFrom(numpy_helper.from_array(newvalues[v.name],v.name))
 target=OUT/'head-refit/model.onnx';target.parent.mkdir(exist_ok=True);onnx.checker.check_model(graph);onnx.save(graph,target);assert target.stat().st_size==path.stat().st_size
 freeze(OUT/'head-refit/fit.json',dict(model_sha256=sha(target),bytes=target.stat().st_size,rank=int(take.sum()),eigenvalues=eigen.tolist(),normalized_mse_before=float(before.mean()),normalized_mse_after=float(after.mean()),normal_equation_relative_residual=float(normal_residual),max_feature_reconstruction_delta=max(parity),max_abs_head_weight=float(np.abs(slope).max()),groups={':'.join(k):r['n'] for k,r in groups.items()},note='Unclipped normalized coefficient MSE; playback clamps outputs to[0,1]. Raw-gap/contact gains must be measured separately.'))
 print((OUT/'head-refit/fit.json').read_text(),flush=True)
if __name__=='__main__':main()
