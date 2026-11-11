"""Train only 375 brow decoder parameters; append them to the frozen ONNX graph."""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
import onnx
from onnx import helper,numpy_helper
from common import *
from gaze import NEW
from model import ExpressionAdapter
from train import load
from brow_model import BrowDecoder,BROW_IDS

FIX=Path(os.environ.get('A2F_BROW_OUTPUT',APP/'.local/expressive-brow-stable-v2')).resolve()
PARENT=NEW/'fullface.onnx'


def prepare(split,adapter):
    data=load(split)
    # Use the teacher's absolute brow weights in every condition, including
    # neutral. No neutral gate or frozen-v4 eyebrow target is retained.
    teacher_target=np.zeros((len(data['cases']),30,5),np.float32)
    labels={};cursor={}
    recipe_sha=sha(OUT/'label-recipe.json')
    for i,case in enumerate(data['cases']):
        key=str(case)
        if key not in labels:
            with np.load(OUT/'labels'/f'{key}.npz') as z:
                if str(z['recipe_sha256'])!=recipe_sha:raise ValueError('Changed teacher brow labels')
                labels[key]=z['weights'][:,BROW_IDS].copy()
            cursor[key]=0
        count=int(data['valid'][i].sum());start=cursor[key]
        teacher_target[i,data['valid'][i]]=labels[key][start:start+count]
        cursor[key]+=count
    if any(cursor[k]!=len(labels[k]) for k in labels):raise ValueError('Teacher brow alignment failed')
    with torch.inference_mode():
        result=[]
        for start in range(0,len(data['hidden']),128):
            h=torch.from_numpy(data['hidden'][start:start+128]);s=torch.from_numpy(data['baseline'][start:start+128])
            e=torch.from_numpy(data['emotion'][start:start+128,None,:]).expand(-1,60,-1)
            x=torch.cat([(h-adapter.feature_mean)/adapter.feature_scale,s,e],dim=-1).transpose(1,2)
            latent=adapter.blocks(F.gelu(adapter.input(x),approximate='tanh'))
            result.append(F.avg_pool1d(latent,10,10).numpy())
    size=len(data['hidden']);next_index=np.arange(size)
    for i in range(size-11):
        if data['cases'][i]==data['cases'][i+11]:next_index[i]=i+11
    joined=np.concatenate([data['valid'][:,25:30],data['valid'][next_index,:5]],axis=1)
    joined[next_index==np.arange(size)]=False
    return dict(slow=np.concatenate(result),target=teacher_target,emotion=data['emotion'],valid=data['valid'],
        voices=data['voices'],next=next_index,join_valid=joined)


def batch(data,indices):
    following=data['next'][indices]
    def rows(ix):
        return [torch.from_numpy(data[k][ix]) for k in ['slow','emotion']]
    a=rows(indices);b=rows(following)
    a[1]=a[1][:,None,:].expand(-1,60,-1);b[1]=b[1][:,None,:].expand(-1,60,-1)
    return a,b,*[torch.from_numpy(data[k][indices]) for k in ['target','valid','join_valid']]


def objective(net,a,b,target,valid,join_valid,gram):
    full=net.decode(*a);next_full=net.decode(*b);p=full[:,15:45]
    def geometry(error,mask):
        squared=torch.einsum('bti,ij,btj->bt',error,gram,error)
        return (squared*mask).sum()/mask.sum().clamp_min(1)
    error=p-target
    position=geometry(error,valid)
    coefficient=(error.square()*valid[:,:,None]).sum()/(valid.sum().clamp_min(1)*5)
    pairs=valid[:,1:]&valid[:,:-1]
    velocity=geometry(6*((p[:,1:]-p[:,:-1])-(target[:,1:]-target[:,:-1])),pairs)
    triples=valid[:,2:]&valid[:,1:-1]&valid[:,:-2]
    # Physical acceleration regularization with a 120 ms characteristic scale.
    acceleration=geometry((60*.12)**2*(p[:,2:]-2*p[:,1:-1]+p[:,:-2]),triples)
    continuity=geometry(full[:,40:50]-next_full[:,10:20],join_valid)
    total=position+.05*coefficient+.25*velocity+.05*acceleration+.2*continuity
    return total,dict(position=position,coefficient=coefficient,velocity=velocity,acceleration=acceleration,continuity=continuity)


def append_graph(net):
    # Export just the new small decoder, leaving every original operation and
    # initializer in place. Only five output columns are replaced by Concat.
    small=FIX/'brow-head.onnx'
    torch.onnx.export(net,(torch.zeros(1,64,60),torch.zeros(1,60,10)),str(small),
        input_names=['latent','controls'],output_names=['brows'],opset_version=17,dynamo=False)
    graph=onnx.load(PARENT)
    convolutions=[n for n in graph.graph.node if 'model.face.adapter.output.weight' in n.input]
    if len(convolutions)!=1:raise ValueError('Parent adapter latent is not uniquely identifiable')
    latent=convolutions[0].input[0]
    branch=onnx.compose.add_prefix(onnx.load(small),'brow_stability/')
    mapping={'brow_stability/latent':latent,'brow_stability/controls':'emotion'}
    for node in branch.graph.node:
        for i,name in enumerate(node.input):node.input[i]=mapping.get(name,name)
    # Preserve the complete parent graph. Its original blendshapes become an
    # internal value; the public output receives the two untouched slices and
    # the five newly decoded eyebrow values.
    for node in graph.graph.node:
        for i,name in enumerate(node.output):
            if name=='blendshapes':node.output[i]='brow_stability/prior_blendshapes'
        for i,name in enumerate(node.input):
            if name=='blendshapes':node.input[i]='brow_stability/prior_blendshapes'
    graph.graph.node.extend(branch.graph.node);graph.graph.initializer.extend(branch.graph.initializer)
    for name,indices in [('before',np.arange(41,dtype=np.int64)),('after',np.arange(46,52,dtype=np.int64))]:
        tensor='brow_stability/'+name+'_indices';output='brow_stability/'+name
        graph.graph.initializer.append(numpy_helper.from_array(indices,tensor))
        graph.graph.node.append(helper.make_node('Gather',['brow_stability/prior_blendshapes',tensor],[output],axis=2))
    graph.graph.node.append(helper.make_node('Concat',['brow_stability/before','brow_stability/brows','brow_stability/after'],['blendshapes'],axis=2))
    onnx.checker.check_model(graph);onnx.save(graph,FIX/'model.onnx')
    return latent


def main():
    torch.set_num_threads(4);torch.manual_seed(37);rng=np.random.default_rng(1037)
    FIX.mkdir(parents=True,exist_ok=True)
    if (FIX/'selection.json').exists():raise ValueError('Completed brow model exists; use a new output directory')
    state=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    adapter=ExpressionAdapter(np.flatnonzero(state['speech_owned'].numpy()),np.zeros(96,np.float32),np.ones(96,np.float32))
    adapter.load_state_dict(state);adapter.eval();adapter.requires_grad_(False)
    net=BrowDecoder(state['output.weight'][BROW_IDS],state['output.bias'][BROW_IDS])
    data={s:prepare(s,adapter) for s in ['train','dev']}
    with np.load(OUT/'calibration.npz') as z:
        shape=z['shape'][BROW_IDS];support=np.max(np.linalg.norm(shape,axis=2),axis=0);region=np.argsort(support)[-512:]
        selected=shape[:,region]*10
        matrix=np.einsum('ivk,jvk->ij',selected,selected,optimize=False)/(512*3)
    gram=torch.from_numpy(matrix)
    protocol=dict(seed=37,steps=2400,learning_rate=.002,batch_size=64,anchors_per_second=6,
        trainable_parameters=375,loss='position_mm2 + .05 coefficient + .25 velocity-match(100ms) + .05 acceleration(120ms) + .2 overlap continuity',
        selection='same composite objective, equally weighted development voices; initial slow decoder eligible',
        parent_model_sha256=sha(PARENT),adapter_sha256=sha(OUT/'adapter.pt'),
        dataset_sha256={s:sha(OUT/'features'/f'{s}.npz') for s in ['train','dev']},
        calibration_sha256=sha(OUT/'calibration.npz'),
        teacher_label_recipe_sha256=sha(OUT/'label-recipe.json'),
        brow_target='Absolute teacher brow weights in neutral and all ten emotion conditions; no frozen-v4 neutral gate.',
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['brow_model.py','train_brows.py']},
        contract='Only output columns 41:46 may change. Every parent operation/initializer retained; no audio clock or mouth changes.')
    freeze(FIX/'training-protocol.json',protocol)
    @torch.inference_mode()
    def evaluate():
        values=[];parts=[];dev=data['dev']
        for voice in sorted(set(dev['voices'])):
            ids=np.flatnonzero(dev['voices']==voice);accum={};count=0
            for start in range(0,len(ids),128):
                ix=ids[start:start+128];value,stats=objective(net,*batch(dev,ix),gram)
                for name,v in {'total':value,**stats}.items():accum[name]=accum.get(name,0.)+float(v)*len(ix)
                count+=len(ix)
            parts.append(dict(voice=str(voice),**{k:v/count for k,v in accum.items()}));values.append(accum['total']/count)
        return float(np.mean(values)),parts
    initial,initial_parts=evaluate();best,best_step,best_state=initial,0,copy.deepcopy(net.state_dict())
    train=data['train'];groups=[np.flatnonzero(train['voices']==v) for v in sorted(set(train['voices']))]
    optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    begun=time.monotonic();history=[]
    for step in range(1,2401):
        ids=np.concatenate([rng.choice(g,8,replace=True) for g in groups])
        value,_=objective(net,*batch(train,ids),gram)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite brow loss')
        optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for group in optimizer.param_groups:group['lr']=float(.0002+.0009*(1+np.cos(np.pi*step/2400)))
        if step%150==0:
            score,parts=evaluate()
            if score<best:best,best_step,best_state=score,step,copy.deepcopy(net.state_dict())
            history.append(dict(step=step,score=score,parts=parts,seconds=time.monotonic()-begun))
            print('brows',step,'dev',round(score,5),'best',best_step,'seconds',round(time.monotonic()-begun),flush=True)
    net.load_state_dict(best_state);net.eval();torch.save(net.state_dict(),FIX/'brow-decoder.pt')
    latent=append_graph(net)
    report=dict(best_step=best_step,initial_dev_score=initial,selected_dev_score=best,initial_parts=initial_parts,
        history=history,bytes=(FIX/'model.onnx').stat().st_size,model_sha256=sha(FIX/'model.onnx'),
        parent_sha256=sha(PARENT),latent_value=latent,protocol_sha256=sha(FIX/'training-protocol.json'))
    freeze(FIX/'selection.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
