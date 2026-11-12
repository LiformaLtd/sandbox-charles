"""Offline wider assistant; the exported tiny student remains unchanged."""
from common import *
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import gelu, Student

def features(net, mel):
    hidden=gelu(net.encoder(mel.transpose(0,2,1)))
    for block in net.blocks: hidden=block(hidden)
    lo=mx.take(hidden,net.left,axis=1);hi=mx.take(hidden,net.left+1,axis=1)
    return lo+net.fraction*(hi-lo)

class WideBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.conv=nn.Conv1d(192,192,3,padding=dilation,dilation=dilation);self.mix=nn.Conv1d(192,192,1)
    def __call__(self,x):return x+.5*self.mix(gelu(self.conv(x)))

class Assistant(nn.Module):
    def __init__(self,values):
        super().__init__();tiny=Student(values)
        self.encoder=nn.Conv1d(64,192,5,padding=2);self.blocks=[WideBlock(d) for d in [1,2,4,8,16,1,2,4]]
        self.bottleneck=nn.Conv1d(192,96,1);self.output=nn.Conv1d(96,39,1)
        for key in ['mean','scale','left','fraction']:setattr(self,key,getattr(tiny,key))
        weights=[]
        for name,tensor in tree_flatten(tiny.trainable_parameters()):
            arr=np.array(tensor)
            if name.startswith('encoder.'):arr=np.tile(arr,(2,1,1) if arr.ndim==3 else (2,))
            elif name.startswith('blocks.'):arr=np.tile(arr,(2,1,2) if arr.ndim==3 else (2,))/(2 if arr.ndim==3 else 1)
            weights.append((name,mx.array(arr)))
        # Complementary columns cancel on duplicated features, preserving the
        # function while allowing the two widened halves to learn differently.
        noise=np.random.default_rng(37).normal(0,.01/np.sqrt(96),(96,96))
        bottleneck=np.concatenate([.5*np.eye(96)+noise,.5*np.eye(96)-noise],axis=1)
        weights += [('bottleneck.weight',mx.array(bottleneck[:,None,:].astype(np.float32))),('bottleneck.bias',mx.zeros(96))]
        self.load_weights(weights,strict=False);self.freeze(keys=['mean','scale','left','fraction'],recurse=False)
    def feature(self,mel):return self.bottleneck(features(self,mel))
    def __call__(self,mel):return self.output(self.feature(mel))*self.scale+self.mean
    def torch_weights(self):
        return {n:(np.array(v).transpose(0,2,1).copy() if v.ndim==3 else np.array(v).copy()) for n,v in tree_flatten(self.trainable_parameters())}

def load_selected(net,path):
    with np.load(path) as z:
        net.load_weights([(n,mx.array(z[n].transpose(0,2,1) if z[n].ndim==3 else z[n])) for n,_ in tree_flatten(net.trainable_parameters())],strict=False)

def export(net,values,active,path):
    """Same frozen PyTorch exporter; assistant adds one offline-only bottleneck."""
    if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
    import torch, importlib.util, onnx
    spec=importlib.util.spec_from_file_location('assistant_frozen_export',SCALE/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    if isinstance(net,Assistant):
        class TorchAssistant(base.Student):
            def __init__(self):
                super().__init__(192,values['student.mean'],values['student.scale'],active)
                self.bottleneck=torch.nn.Conv1d(192,96,1);self.output=torch.nn.Conv1d(96,39,1)
            def forward(self,mel):
                encoded=self.blocks(torch.nn.functional.gelu(self.encoder(mel),approximate='tanh'));lo=torch.index_select(encoded,2,self.left);hi=torch.index_select(encoded,2,self.left+1)
                return self.output(self.bottleneck(lo+self.fraction*(hi-lo))).transpose(1,2)*self.scale+self.mean
        student=TorchAssistant()
    else:student=base.Student(96,values['student.mean'],values['student.scale'],active)
    state={k:torch.from_numpy(v.copy()) for k,v in net.torch_weights().items()};student.load_state_dict(state,strict=False);student.eval();path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(path),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    graph=onnx.load(path);onnx.checker.check_model(graph)
    if not isinstance(net,Assistant):
        original=onnx.load(BASELINE);ops=lambda m:[(n.op_type,list(n.input),list(n.output)) for n in m.graph.node]
        if ops(graph)!=ops(original) or path.stat().st_size!=2251827:raise ValueError('Tiny browser model changed')
    return dict(bytes=path.stat().st_size,parameters=sum(v.size for _,v in tree_flatten(net.trainable_parameters())),sha256=sha(path))
