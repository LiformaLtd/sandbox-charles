"""Matched history/no-history decoder fine-tunes with a frozen v4 backbone."""
import argparse, copy, json, math, time, sys, os
import numpy as np
import torch
from torch.nn import functional as F
from common import OUT, WORK, BASELINE, base_module, sha, freeze
from decoder import Decoder, waveform_class
from data import backbone, dataset


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=['control', 'temporal'], required=True); p.add_argument('--seed', type=int, default=37)
    args = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(args.seed)
    run = OUT / args.arm; run.mkdir(exist_ok=True)
    if (run / 'model.onnx').exists(): raise ValueError('Completed run exists')
    norm = np.load(OUT / 'normalization.npz'); decoder = Decoder(norm['center'], norm['spread'], args.arm)
    student, active = backbone()
    train = dataset('primary-train'); dev = dataset('primary-dev'); olddev = dataset('replay-dev')
    replay = [dataset(n) for n in ['replay-train', 'extra-train']]
    replay = {k: np.concatenate([v[k] for v in replay]) if k == 'voice' else torch.cat([v[k] for v in replay]) for k in train}
    rig = np.load(WORK / 'arkit-student/rig-metrics.npz'); basis = torch.from_numpy(rig['gap_basis'][active]); rest = torch.from_numpy(rig['rest'] + rig['gap_offset'])
    names = json.loads((WORK / 'arkit-student/channels.json').read_text()); lips = torch.tensor([1.7 if names[i].startswith(('mouth', 'jaw')) else 1. for i in active])
    def loss(pred, target, gaps, valid):
        w = valid[..., None]; count = valid.sum().clamp(min=1)
        shape = (F.smooth_l1_loss((pred-target)/student.scale, torch.zeros_like(pred), reduction='none', beta=.5)*w*lips).sum()/(count*len(active))
        geom = pred.clamp(0,1) @ basis + rest
        gap = (F.smooth_l1_loss(geom,gaps,reduction='none',beta=.14)*w).sum()/(count*9)
        adjacent = valid[:,1:] & valid[:,:-1]
        velocity = (((pred.diff(dim=1)-target.diff(dim=1)).abs()/student.scale)*adjacent[...,None]).sum()/(count*len(active))
        actual, estimate = gaps.median(-1).values, geom.median(-1).values
        contact = actual <= .1
        closure = (F.binary_cross_entropy_with_logits((.1-estimate)/.16,contact.float(),reduction='none')*(1+contact)*valid).sum()/count
        near = ((actual[:,1:]<.35)|(actual[:,:-1]<.35)) & adjacent
        timing = (F.smooth_l1_loss(estimate.diff(dim=1),actual.diff(dim=1),reduction='none',beta=.1)*near).sum()/near.sum().clamp(min=1)
        return shape+1.2*gap+.12*velocity+.22*closure+.45*timing
    @torch.no_grad()
    def evaluate():
        result=[]
        for cohort in [dev, olddev]:
            pred=torch.cat([(b+decoder(c)*student.scale).clamp(0,1) for c,b in zip(cohort['contexts'].split(128),cohort['baseline'].split(128))])
            voices=np.unique(cohort['voice']) if cohort is dev else ['easton_guard']
            for voice in voices:
                mask=np.nonzero(cohort['voice']==voice)[0] if cohort is dev else np.arange(len(pred))
                valid=cohort['valid'][mask]; truth=cohort['gaps'][mask][valid]; estimate=(pred[mask]@basis+rest)[valid]
                a,b=truth.median(-1).values<=.1,estimate.median(-1).values<=.1
                result.append(dict(voice=str(voice),rms=float((estimate-truth).square().mean().sqrt()),f1=float(2*(a&b).sum()/(a.sum()+b.sum()).clamp(min=1))))
        return float(np.mean([v['rms']+.25*(1-v['f1']) for v in result])),result
    groups=[torch.from_numpy(np.nonzero(train['voice']==v)[0]) for v in np.unique(train['voice'])]
    assert len(groups)==8
    rng=torch.Generator().manual_seed(args.seed+1000)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=.0003,weight_decay=.015)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,744,eta_min=.00003)
    best,initial=evaluate(); best_epoch=0; best_state=copy.deepcopy(decoder.state_dict())
    recipe=dict(arm=args.arm,seed=args.seed,baseline_sha256=sha(BASELINE),data_lock_sha256=sha(OUT/'data-lock.json'),normalization_sha256=sha(OUT/'normalization.npz'),code_sha256=sha(__file__),decoder_sha256=sha(__import__('pathlib').Path(__file__).with_name('decoder.py')),epochs=24,steps=31,lr=.0003,weight_decay=.015,backbone='frozen',replay='48 balanced primary + 16 historical',selection='macro voice RMS + .25*(1-F1), including Easton guard')
    log=dict(recipe=recipe,initial_dev=initial,history=[]); began=time.monotonic()
    for epoch in range(24):
        losses=[]
        for _ in range(31):
            ix=torch.cat([g[torch.randint(len(g),(6,),generator=rng)] for g in groups]); old=torch.randint(len(replay['baseline']),(16,),generator=rng)
            c,b,y,g,m=[torch.cat((train[k][ix],replay[k][old])) for k in ['contexts','baseline','weights','gaps','valid']]
            optimizer.zero_grad(set_to_none=True); pred=b+decoder(c)*student.scale
            objective=loss(pred,y,g,m)
            if not torch.isfinite(objective): raise ValueError('Invalid loss')
            objective.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True); optimizer.step(); schedule.step(); losses.append(float(objective.detach()))
        score,scores=evaluate()
        if score<best:best,best_epoch,best_state=score,epoch+1,copy.deepcopy(decoder.state_dict())
        log['history'].append(dict(epoch=epoch+1,selection=score,dev=scores,loss=float(np.mean(losses))))
        torch.save(dict(model=decoder.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),rng=rng.get_state(),epoch=epoch+1,best=best,best_epoch=best_epoch,best_state=best_state,recipe=recipe),run/'checkpoint.pt')
        (run/'training.json').write_text(json.dumps(log,indent=2)); print(args.arm,epoch+1,round(score,5),'best',best_epoch,'seconds',round(time.monotonic()-began),flush=True)
    decoder.load_state_dict(best_state); torch.save(decoder.state_dict(),run/'decoder.pt')
    model=waveform_class(base_module())(student,decoder).eval()
    torch.onnx.export(model,(torch.zeros(1,16000),torch.zeros(1,7,96),torch.tensor([15]),torch.tensor([30])),str(run/'model.onnx'),input_names=['pcm','history','first','count'],output_names=['blendshapes','next_history'],opset_version=17,dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(run/'model.onnx'))
    log.update(best_epoch=best_epoch,best_selection=best,bytes=(run/'model.onnx').stat().st_size,model_sha256=sha(run/'model.onnx'),decoder_parameters=sum(v.numel() for v in decoder.parameters()),parameters=sum(v.numel() for v in model.parameters()),seconds=time.monotonic()-began)
    (run/'training.json').write_text(json.dumps(log,indent=2))


if __name__=='__main__':main()
