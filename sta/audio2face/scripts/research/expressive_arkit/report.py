"""Coverage, simple expression-template control, and exact speech regression."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from fit import mm
from infer import read_audio, sequence
from train import load, evaluate
from model import ExpressionAdapter


def main():
    torch.set_num_threads(4)
    selection=json.loads((OUT/'selection.json').read_text())
    train,test=load('train'),load('test')
    state=torch.load(OUT/'adapter.pt',map_location='cpu',weights_only=True)
    net=ExpressionAdapter(train['protected'],np.zeros(96,np.float32),np.ones(96,np.float32))
    net.load_state_dict(state);net.eval()
    # Compare with a much simpler per-emotion constant residual. This tests
    # whether the learned dynamic adapter adds anything beyond expression poses.
    templates=[]
    for index in range(11):
        indices=np.arange(index,len(train['target']),11)
        residual=train['target'][indices]-train['baseline'][indices,15:45]
        templates.append(residual[train['valid'][indices]].mean(0))
    accum={}
    for i in range(len(test['target'])):
        e=0 if not test['emotion'][i].any() else int(test['emotion'][i].argmax())+1
        p=np.clip(test['baseline'][i,15:45]+templates[e],0,1)
        error=p[test['valid'][i]]-test['target'][i,test['valid'][i]]
        stat=accum.setdefault(str(test['voices'][i]),np.zeros(53,np.float64))
        stat[:52]+=np.square(error).sum(0);stat[52]+=len(error)
    template_channels=np.mean([np.sqrt(s[:52]/s[52]) for s in accum.values()],axis=0)
    coverage=json.loads((OUT/'coverage.json').read_text())
    names=json.loads((OUT/'channels.json').read_text())
    wav=Path('/Users/charleswiles/Source/liforma/sandboxes/sandbox-charles/sta/audio2face-2d/static/assets/speech/appointment-supertonic-m1.wav')
    pcm=read_audio(wav)
    old=sequence(BASELINE,pcm)
    with np.load(OUT/'calibration.npz') as z:gap_basis=z['gap_basis'];rest=z['gap_neutral']
    old_gaps=mm(old,gap_basis)+rest
    reference_contact=np.median(old_gaps,axis=1)<=.1
    regression=[]
    tracks={}
    for i,name in enumerate(['neutral',*EMOTIONS]):
        control=np.zeros(10,np.float32)
        if i:control[i-1]=1
        w=sequence(OUT/'model.onnx',pcm,control)
        gaps=mm(w,gap_basis)+rest
        contact=np.median(gaps,axis=1)<=.1
        regression.append(dict(emotion=name,frames=len(w),
            old_39_max_drift=float(np.max(np.abs(w[:,train['active']]-old[:,train['active']]))),
            mouth_gap_max_drift_mm=float(np.max(np.abs(gaps-old_gaps)))*10,
            baseline_contact_frames=int(reference_contact.sum()),contact_frames=int(contact.sum()),
            lost_baseline_contacts=int((reference_contact&~contact).sum()),
            added_contact_frames=int((~reference_contact&contact).sum())))
        tracks[name]=w
    if regression[0]['old_39_max_drift']>3e-5 or regression[0]['mouth_gap_max_drift_mm']>.0001:
        raise ValueError('Neutral appointment regression failed')
    np.savez_compressed(OUT/'appointment-tracks.npz',baseline=old,**tracks)
    protected=json.loads((OUT/'protected-before.json').read_text())
    differences=[p for p,digest in protected.items() if sha(p)!=digest]
    if differences:raise ValueError('Protected production files changed: '+str(differences))
    report=dict(selection=selection,coverage=coverage,
        template_test_rms=float(template_channels.mean()),template_channels=template_channels.tolist(),
        appointment_audio=str(wav),appointment_audio_sha256=sha(wav),appointment_regression=regression,
        protected_files_checked=len(protected),protected_files_changed=differences,
        full_52_teacher_coverage_passed=not bool(coverage['train']['limited_range']),
        release_ready=False,
        reason='Prototype only: coverage, semantic calibration, expressive lip contacts and visual/browser/2.5D checks remain separate gates.')
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(12,4))
    ix=np.arange(52);width=.26
    ax.bar(ix-width,selection['test_initial_channels'],width,label='Frozen speech only')
    ax.bar(ix,template_channels,width,label='Constant pose per emotion')
    ax.bar(ix+width,selection['test_selected_channels'],width,label='Trained adapter')
    ax.set_xticks(ix,names,rotation=90,fontsize=7);ax.set_ylabel('Coefficient RMS, held-out voices')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'channel-errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    time=np.arange(len(old))/60
    for ax,channel in zip(axes,[41,43,44]):
        for name in ['neutral','amazement','anger','joy']:ax.plot(time,tracks[name][:,channel],label=name)
        ax.set_ylabel(names[channel]);ax.set_ylim(-.02,1.02)
    axes[0].legend(ncol=4);axes[-1].set_xlabel('Audio time (seconds)')
    fig.suptitle('Same appointment speech; different supplied emotion controls')
    fig.tight_layout();fig.savefig(OUT/'appointment-brows.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(test_rms=selection['test_selected_rms'],template_rms=float(template_channels.mean()),
                          coverage_pass=report['full_52_teacher_coverage_passed'],
                          appointment=regression,protected_files_checked=len(protected)),indent=2))


if __name__=='__main__':main()
