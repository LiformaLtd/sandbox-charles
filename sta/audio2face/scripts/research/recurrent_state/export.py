"""Export explicit emitted-frame state; no state update for overlapping/padded rows."""
from common import *
import importlib.util,numpy as np,torch
from torch import nn
from torch.nn import functional as F
import onnx
from onnx import numpy_helper

def base_module():
 spec=importlib.util.spec_from_file_location('base',SCALE/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);return base

def make(arm):
 base=base_module();values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};rig=np.load(WORK/'arkit-student/rig-metrics.npz');student=base.Student(96,values['student.mean'],values['student.scale'],rig['active']);student.load_state_dict({k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()})
 with np.load(OUT/arm/'best.npz') as z:weights={k:z[k].copy() for k in z.files}
 for key in student.state_dict():
  if 'student.'+key in weights:
   value=weights['student.'+key];student.state_dict()[key].copy_(torch.from_numpy(value.transpose(0,2,1).copy() if value.ndim==3 else value))
 class Waveform(base.WaveformModel):
  def __init__(self):
   super().__init__(student)
   for name,key in [('cell_Wx','cell.Wx'),('cell_Wh','cell.Wh'),('cell_b','cell.b'),('cell_bhn','cell.bhn'),('head_W','head.weight'),('head_b','head.bias')]:self.register_parameter(name,nn.Parameter(torch.from_numpy(weights[key])))
  def forward(self,pcm,state,first,count):
   spec=F.conv1d(pcm.unsqueeze(1),self.filters,stride=160,padding=200);real,imag=spec.chunk(2,1);mel=(1+F.conv1d(real.square()+imag.square(),self.mel)*10).log()
   s=self.student;encoded=s.blocks(F.gelu(s.encoder(mel),approximate='tanh'));lo=torch.index_select(encoded,2,s.left);hi=torch.index_select(encoded,2,s.left+1);feature=(lo+s.fraction*(hi-lo)).transpose(1,2);rows=[]
   for t in range(30):
    x=feature[:,torch.clamp(first[0]+t,max=59)];h=state if arm=='recurrent' else state*0
    xp=F.linear(torch.tanh(x/4),self.cell_Wx,self.cell_b);hp=F.linear(h,self.cell_Wh);r=torch.sigmoid(xp[:,:32]+hp[:,:32]);z=torch.sigmoid(xp[:,32:64]+hp[:,32:64]);v=torch.tanh(xp[:,64:]+r*(hp[:,64:]+self.cell_bhn));new=(1-z)*v+z*h
    state=torch.where(t<count[0],new,state)
    raw=F.linear(x,s.output.weight[:,:,0],s.output.bias)+F.linear(state,self.head_W,self.head_b)
    rows.append((raw*s.scale+s.mean).clamp(0,1)@s.channel_map)
   return torch.stack(rows,1),state
 return Waveform().eval()

def main():
 torch.set_num_threads(2)
 for arm in ['control','recurrent']:
  report=json.loads((OUT/arm/'complete.json').read_text());assert sha(OUT/arm/'best.npz')==report['best_sha256']
  model=make(arm);args=(torch.zeros(1,16000),torch.zeros(1,32),torch.tensor([30]),torch.tensor([15]));path=OUT/arm/'model.onnx'
  torch.onnx.export(model,args,str(path),input_names=['pcm','state','first','count'],output_names=['blendshapes','next_state'],opset_version=17,do_constant_folding=False,dynamo=False)
  onnx.checker.check_model(onnx.load(path));freeze(OUT/arm/'selection.json',dict(**report,model_sha256=sha(path),bytes=path.stat().st_size,export_source_sha256=sha(__file__)))
  print(arm,path.stat().st_size,'bytes',report['parameters'],'params',flush=True)
if __name__=='__main__':main()
