"""Can the unchanged student memorize four known training recordings?"""
from pathlib import Path
import argparse, copy, importlib.util, json, os, sys, time
import numpy as np
import torch
from torch.nn import functional as F
APP=Path(__file__).resolve().parents[3]
OUT=Path(os.environ.get('A2F_SCALE_OUTPUT',APP/'.local/compact-scale-v1')).resolve()
WORK=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
if os.environ.get('A2F_ONNX_SITE_PACKAGES'):sys.path.append(os.environ['A2F_ONNX_SITE_PACKAGES'])
import onnx
from onnx import numpy_helper


def main():
    p=argparse.ArgumentParser();p.add_argument('--objective',choices=['mixed','fit-mse'],required=True);args=p.parse_args()
    root=OUT/'memorization';run=root/args.objective;run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():raise ValueError('Completed run exists')
    protocol=json.loads((root/'protocol.json').read_text())
    import hashlib
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(__file__)!=protocol['trainer_sha256']:raise ValueError('Frozen trainer changed')
    if digest(root/'manifest.json')!=protocol['manifest_sha256'] or digest(root/'sources/model.py')!=protocol['model_source_sha256']:raise ValueError('Changed diagnostic input')
    for row in json.loads((root/'manifest.json').read_text()):
        if digest(row['audio'])!=protocol['audio'][row['stem']] or digest(root/'labels'/(row['stem']+'.npz'))!=protocol['labels'][row['stem']]:raise ValueError('Changed waveform/label')
    torch.set_num_threads(4);torch.manual_seed(37)
    os.environ['A2F_ABLATION_OUTPUT']=str(root/'feature-cache')
    sys.path.insert(0,str(APP/'scripts/research/compact_ablation'));import train as source
    spec=importlib.util.spec_from_file_location('memorization_base',root/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    graph=onnx.load(protocol['baseline']);values={v.name:numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    if digest(protocol['baseline'])!=protocol['baseline_sha256']:raise ValueError('Changed baseline')
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active']
    student=base.Student(96,values['student.mean'],values['student.scale'],active)
    student.load_state_dict({k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()})
    arrays=source.dataset(root,'train',active,16000);x,y,g,m=[torch.from_numpy(v) for v in arrays]
    basis=torch.from_numpy(rig['gap_basis'][active]);rest=torch.from_numpy(rig['rest']+rig['gap_offset'])
    target_gaps=y@basis+rest
    names=json.loads((WORK/'arkit-student/channels.json').read_text());lips=torch.tensor([1.7 if names[i].startswith(('mouth','jaw')) else 1. for i in active])
    def prediction():return student(x)[:,15:45]
    def mixed(pred):
        w=m[...,None];count=m.sum().clamp(min=1)
        shape=(F.smooth_l1_loss((pred-y)/student.scale,torch.zeros_like(pred),reduction='none',beta=.5)*w*lips).sum()/(count*len(active))
        geom=pred.clamp(0,1)@basis+rest
        gap=(F.smooth_l1_loss(geom,g,reduction='none',beta=.14)*w).sum()/(count*9)
        adjacent=m[:,1:]&m[:,:-1]
        velocity=(((pred.diff(dim=1)-y.diff(dim=1)).abs()/student.scale)*adjacent[...,None]).sum()/(count*len(active))
        actual,estimate=g.median(-1).values,geom.median(-1).values;contact=actual<=.1
        closure=(F.binary_cross_entropy_with_logits((.1-estimate)/.16,contact.float(),reduction='none')*(1+contact)*m).sum()/count
        near=((actual[:,1:]<.35)|(actual[:,:-1]<.35))&adjacent
        timing=(F.smooth_l1_loss(estimate.diff(dim=1),actual.diff(dim=1),reduction='none',beta=.1)*near).sum()/near.sum().clamp(min=1)
        return shape+1.2*gap+.12*velocity+.22*closure+.45*timing
    @torch.no_grad()
    def evaluate():
        pred=prediction().clamp(0,1);geom=pred@basis+rest
        a,b=g[m].median(-1).values<=.1,geom[m].median(-1).values<=.1
        return dict(student_to_fit_rms_mm=float(10*(geom[m]-target_gaps[m]).square().mean().sqrt()),
                    student_to_raw_rms_mm=float(10*(geom[m]-g[m]).square().mean().sqrt()),
                    fitting_rms_mm=float(10*(target_gaps[m]-g[m]).square().mean().sqrt()),
                    normalized_coefficient_mae=float(((pred-y)/student.scale)[m].abs().mean()),
                    contact_f1=float(2*(a&b).sum()/(a.sum()+b.sum()).clamp(min=1)))
    initial=evaluate();best=initial;best_step=0;state=copy.deepcopy(student.state_dict());history=[]
    optimizer=torch.optim.AdamW(student.parameters(),lr=.0003,weight_decay=0)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1500,eta_min=.00003)
    began=time.monotonic()
    for step in range(1,1501):
        optimizer.zero_grad(set_to_none=True);pred=prediction()
        loss=mixed(pred) if args.objective=='mixed' else (((pred-y)/student.scale).square()*m[...,None]).sum()/(m.sum()*len(active))
        if not torch.isfinite(loss):raise ValueError('Invalid loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);optimizer.step();schedule.step()
        if step==1 or step%50==0:
            metrics=evaluate();history.append(dict(step=step,loss=float(loss.detach()),**metrics))
            if metrics['student_to_fit_rms_mm']<best['student_to_fit_rms_mm']:best,best_step,state=metrics,step,copy.deepcopy(student.state_dict())
            (run/'progress.json').write_text(json.dumps(dict(initial=initial,best=best,best_step=best_step,history=history),indent=2))
            print(args.objective,step,'fit',round(metrics['student_to_fit_rms_mm'],4),'raw',round(metrics['student_to_raw_rms_mm'],4),'seconds',round(time.monotonic()-began),flush=True)
            torch.save(dict(student=student.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),step=step,best_state=state,protocol=protocol),run/'checkpoint.pt')
    student.load_state_dict(state)
    torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(run/'model.onnx'),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False)
    passed=best['student_to_fit_rms_mm']<=.5 and best['student_to_raw_rms_mm']<=1.0
    result=dict(objective=args.objective,initial=initial,best=best,best_step=best_step,memorization_pass=passed,training_only=True,
                steps=1500,clips=4,windows=len(x),frames=int(m.sum()),bytes=(run/'model.onnx').stat().st_size,model_sha256=digest(run/'model.onnx'),history=history,seconds=time.monotonic()-began)
    (run/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':main()
