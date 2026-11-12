"""Readable results and fixed-case teacher-gap traces; never modifies training outputs."""
from common import *
import numpy as np

def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    reports={}
    for p in OUT.glob('evaluation-*.json'):reports[p.stem.removeprefix('evaluation-')]=json.loads(p.read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    destination=Path(os.environ['A2F_REPORT_OUTPUT']);destination.mkdir(parents=True,exist_ok=True)
    cohorts=['feature_confirmation','scale_confirmation','historical']
    names=['baseline','mentor']+[f'{a}-s{s}' for s in [37,73,101] for a in ['control','distill']]
    text=['# Intermediate animation teacher / feature distillation experiment','',
          'LIFORMA-104 · 11 November 2026 · offline research only','',
          '**Outcome:** '+('Promising regression screen; fresh confirmation and visual playback still required.' if summary['promising'] else 'The feature-distillation pilot did not pass the predeclared improvement screen. Keep the current model, 2.5D mapping and Player unchanged.'),'',
          '## Design','',
          'Current compact baseline: finetune-r4, SHA256 `'+EXPECTED+'`. A 192-channel mentor starts with the original 96-channel prediction path plus independent random extra features; its extra channels are not copied replicas. It receives 7,440 supervised updates. Development selection keeps epoch zero eligible and selects step 6,200. The selected mentor passes the predeclared development gate.', '',
          'For each of seeds 37, 73 and 101, two width-96 learners receive identical 3,720-update schedules, sample order, starting student weights and train-only projection initialization. Both retain the original animation losses. Distillation alone ramps a 0.1 cosine loss on two frozen mentor animation features. Projections and mentor are excluded from compact ONNX exports. All compact exports retain 331,047 learned parameters and 2,251,827 bytes.', '',
          'All selections are development-only, locked before compact held-out scoring. Cohorts below are previously inspected held-out recordings: they are regression screens, not fresh independent confirmation. The three seeds vary fine-tuning, not original pretraining or teacher training.', '',
          '## Results','',
          '| Model | Feature confirmation gap RMS (mm) | Contact F1 | Scale confirmation RMS (mm) | Historical RMS (mm) |','| --- | ---: | ---: | ---: | ---: |']
    for name in names:
        r=reports[name];v=[r[c]['conditions']['original'] for c in cohorts]
        text.append(f"| {name} | {v[0]['macro_gap_rmse_mm']:.4f} | {v[0]['macro_contact_f1']:.4f} | {v[1]['macro_gap_rmse_mm']:.4f} | {v[2]['macro_gap_rmse_mm']:.4f} |")
    text += ['', '## Paired distillation effects','', 'Positive gap reduction means distillation improves on its paired ordinary-fine-tuning control.', '', '| Cohort | Mean gap reduction | Seed37 / 73 / 101 gap reductions | Mean contact F1 change |', '| --- | ---: | --- | ---: |']
    for c,e in summary['effects'].items():text.append(f"| {c} | {100*e['mean_relative_gap_reduction']:.2f}% | "+' / '.join(f'{100*x:.2f}%' for x in e['relative_gap_reduction_by_seed'])+f" | {e['mean_contact_f1_delta']:+.5f} |")
    text += ['', 'The predeclared primary gap threshold is 3% mean reduction, with contact, opening, engine and closure guards. This is an engineering screen, not a statistical proof. Individual seed variation matters; do not select the best held-out seed or retune on these results.', '', 'Checks: `'+json.dumps(summary['checks'])+'`.', '',
             '## Interpretation and limits','',
             'A stronger intermediate predictor is not enough: its internal features must also provide a useful learning signal for this compact architecture. Further ordinary fine-tuning can account for shared gains, which is why the paired controls are required. This trial uses cosine matching through two train-only projections, not EA’s exact fully-connected feature-MSE formulation. It tests this specified recipe, not every possible intermediate-teacher method.', '',
             'No fresh speech synthesis, blind preference study or 2.5D retargeting experiment was used to establish superiority. Teacher labels remain a geometric reference, not ground-truth human articulation. No export was installed or published.', '',
             '## Checks and artifacts','',
             'Five boundary tests passed. All exports are checked with ONNX; matched sampler and initial-parameter hashes are required. MLX-versus-ONNX parity is measured on three training clips per export, separately from quality evaluation. Protected primary model and Player speech-shape hashes are checked after the run. Source, data, model, selection and environment hashes remain in the isolated ignored experiment bundle.', '',
             'Workspace: `'+str(APP.parents[1])+'`', 'Branch: `research/intermediate-animation-teacher`', 'Results: `'+str(OUT)+'`', '',
             'Fixed-case gap plots use seed37, selected by metadata before compact scoring; they visualize raw teacher lip gaps and fitted-model predictions, not full-avatar appearance or a perceptual preference test.', '']
    (destination/'intermediate_teacher_results.md').write_text('\n'.join(text))
    rows={r['stem']:r for r in json.loads((DATA/'.local/compact-feature-v1/confirmation/variants/manifest.json').read_text())}
    cases=json.loads((OUT/'visual-cases.json').read_text())['cases']
    rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset']
    fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    for ax,case in zip(axes.flat,cases):
        row=rows[case['stem']]
        with np.load(DATA/'.local/compact-feature-v1/confirmation/variants/labels'/(row['stem']+'.npz')) as z:truth=z['gaps']
        t=np.arange(len(truth))/60;ax.plot(t,10*np.median(truth,-1),color='black',lw=1.4,label='Raw teacher')
        for name,color in [('baseline','#888888'),('mentor','#208050'),('control-s37','#2563aa'),('distill-s37','#cb6332')]:
            prediction=np.load(OUT/'predictions'/reports[name]['sha256']/(row['sha256_audio']+'.npy'))
            gaps=np.einsum('tc,ck->tk',prediction,basis,optimize=False)+rest
            ax.plot(t,10*np.median(gaps,-1),lw=.9,color=color,label=name)
        ax.axhline(1.,color='grey',ls=':',lw=.6);ax.set_title(case['engine']+' · '+case['kind']+'\n'+case['text'],fontsize=8);ax.set(xlabel='Audio time (s)',ylabel='Median lip gap (mm)');ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7,ncol=2)
    fig.savefig(destination/'intermediate_teacher_gap_traces.png',dpi=145);plt.close(fig)
    for name in ['summary.json','protocol.json','selection-lock.json','environment.json']:(destination/('intermediate_teacher_'+name)).write_bytes((OUT/name).read_bytes())
    print('Wrote report, summary/provenance and fixed-case gap traces.',flush=True)
if __name__=='__main__':main()
