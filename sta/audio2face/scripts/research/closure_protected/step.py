"""Guard the actual adaptive/weight-decay step, then check nonlinear batch losses."""
from common import *
import mlx.core as mx
from protection import pack,unpack,project,dot
from loss import components

def protected_step(net,optimizer,grads,guard_grads,guard_values,args):
 template=net.trainable_parameters();before=pack(template);mx.eval(before)
 optimizer.update(net,grads);delta=pack(net.trainable_parameters())-before
 corrected,info=project(delta,[pack(g) for g in guard_grads]);limits=[float(v)+1e-7*max(1.,abs(float(v))) for v in guard_values];accepted=0.;new=list(map(float,guard_values))
 for attempt in range(5):
  scale=2.**(-attempt);net.update(unpack(before+scale*corrected,template));parts=components(net,*args);values=[float(.22*parts[3]),float(.45*parts[4])]
  if all(v<=limit for v,limit in zip(values,limits)):
   accepted=scale;new=values;break
 if not accepted:net.update(unpack(before,template))
 actual=pack(net.trainable_parameters())-before;dots=[dot(pack(g),actual) for g in guard_grads]
 return dict(**info,accepted_scale=accepted,old_guards=list(map(float,guard_values)),new_guards=new,guard_limits=limits,actual_guard_dots=dots,actual_step_norm=float(max(0,dot(actual,actual))**.5))
