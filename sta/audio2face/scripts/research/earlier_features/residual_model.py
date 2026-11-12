"""Equal residual heads: first-encoder versus final features, fixed v4 predictions."""
from common import *
import importlib.util,numpy as np
import mlx.core as mx,mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_backend import Student,gelu

def features(net,mel):
    h=gelu(net.encoder(mel.transpose(0,2,1)))
    for block in net.blocks:h=block(h)
    lo=mx.take(h,net.left,axis=1);hi=mx.take(h,net.left+1,axis=1)
    return lo+net.fraction*(hi-lo)

class Head(nn.Module):
    def __init__(self,values,feature_scale,arm):
        super().__init__();self.head_in=nn.Conv1d(135,96,1);self.head_out=nn.Conv1d(96,39,1)
        self.mean=mx.array(values['student.mean']);self.scale=mx.array(values['student.scale'])
        self.base_weight=mx.array(values['student.output.weight'][:,:,0]);self.base_bias=mx.array(values['student.output.bias'])
        self.feature_scale=mx.array(np.asarray(feature_scale).reshape(1,1,96));self.residual_kind=mx.array([1.]);self.feature_source_kind=mx.array([float(arm=='early')])
        weight=np.zeros((96,1,135),np.float32);weight[:39,0,96:]=np.eye(39);weight[39:78,0,96:]=-np.eye(39)
        weight[78:,0,:96]=np.random.default_rng(37).normal(0,.01/np.sqrt(96),(18,96))
        output=np.zeros((39,1,96),np.float32)
        self.load_weights([('head_in.weight',mx.array(weight)),('head_in.bias',mx.zeros(96)),('head_out.weight',mx.array(output)),('head_out.bias',mx.zeros(39))],strict=False)
        self.freeze(keys=['mean','scale','base_weight','base_bias','feature_scale','residual_kind','feature_source_kind'],recurse=False)
        assert sum(v.size for _,v in tree_flatten(self.trainable_parameters()))==16839
    def __call__(self,h):
        p=h[:,:,96:];inp=mx.concatenate([h[:,:,:96]/self.feature_scale,p],axis=-1)
        y=self.head_out(gelu(self.head_in(inp)))+self.residual_kind*p
        return y*self.scale+self.mean
    def save(self,path):np.savez(path,**{n:np.array(v) for n,v in tree_flatten(self.trainable_parameters())})
    def restore(self,path):
        with np.load(path) as z:
            expected={n for n,_ in tree_flatten(self.trainable_parameters())};assert set(z.files)==expected
            self.load_weights([(n,mx.array(z[n])) for n in z.files],strict=False)

def export(head,values,active,path):
    if os.environ.get('A2F_TORCH_SITE_PACKAGES'):sys.path.append(os.environ['A2F_TORCH_SITE_PACKAGES'])
    import torch,onnx
    from onnx import numpy_helper
    spec=importlib.util.spec_from_file_location('residual_frozen_export',SCALE/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    class Combined(base.Student):
        def __init__(self):
            super().__init__(96,values['student.mean'],values['student.scale'],active)
            self.head_in=torch.nn.Conv1d(135,96,1);self.head_out=torch.nn.Conv1d(96,39,1)
            self.register_buffer('feature_scale',torch.from_numpy(np.array(head.feature_scale).transpose(0,2,1).copy()))
            self.register_buffer('residual_kind',torch.from_numpy(np.array(head.residual_kind).copy()))
            self.register_buffer('feature_source_kind',torch.from_numpy(np.array(head.feature_source_kind).copy()))
        def forward(self,mel):
            early=torch.nn.functional.gelu(self.encoder(mel),approximate='tanh');h=self.blocks(early);lo=torch.index_select(h,2,self.left);hi=torch.index_select(h,2,self.left+1);h=lo+self.fraction*(hi-lo)
            elo=torch.index_select(early,2,self.left);ehi=torch.index_select(early,2,self.left+1);early=elo+self.fraction*(ehi-elo)
            p=self.output(h);branch=h+self.feature_source_kind*(early-h);x=torch.cat([branch/self.feature_scale,p],dim=1)
            y=self.head_out(torch.nn.functional.gelu(self.head_in(x),approximate='tanh'))+self.residual_kind*p
            return y.transpose(1,2)*self.scale+self.mean
    student=Combined();state=student.state_dict()
    learned={n:np.array(v).transpose(0,2,1).copy() for n,v in tree_flatten(head.trainable_parameters()) if v.ndim==3}
    learned.update({n:np.array(v).copy() for n,v in tree_flatten(head.trainable_parameters()) if v.ndim!=3})
    for key in state:
        if key in learned:state[key]=torch.from_numpy(learned[key])
        elif 'student.'+key in values:state[key]=torch.from_numpy(values['student.'+key].copy())
        elif key not in ['feature_scale','residual_kind','feature_source_kind']:raise ValueError('Uninitialized export parameter '+key)
    student.load_state_dict(state,strict=True);student.eval();path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(path),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    graph=onnx.load(path)
    # PyTorch aliases the two equal kind flags in the early arm. Materialize
    # this one constant so both arms have the same graph and browser contract.
    for node in list(graph.graph.node):
        if node.op_type=='Identity' and list(node.output)==['student.feature_source_kind']:
            original=next(t for t in graph.graph.initializer if t.name==node.input[0])
            graph.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(original).copy(),node.output[0]));graph.graph.node.remove(node)
    onnx.checker.check_model(graph);onnx.save(graph,path);tensors={t.name:numpy_helper.to_array(t) for t in graph.graph.initializer}
    for key,value in values.items():
        if key.startswith('onnx::'):
            assert any(v.dtype==value.dtype and v.shape==value.shape and np.array_equal(v,value) for k,v in tensors.items() if k.startswith('onnx::')),key
        else:np.testing.assert_array_equal(tensors[key],value,err_msg=key)
    for key,n in [('student.feature_scale',96),('student.residual_kind',1),('student.feature_source_kind',1),('student.head_in.weight',12960),('student.head_out.weight',3744)]:assert tensors[key].size==n
    assert path.stat().st_size<2350000
    return dict(bytes=path.stat().st_size,parameters=sum(p.numel() for p in student.parameters()),head_parameters=16839,sha256=sha(path))
