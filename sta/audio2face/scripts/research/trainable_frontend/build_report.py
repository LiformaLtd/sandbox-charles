"""Scientific readout; fixed-case lip-gap traces and learned bank inspection."""
from common import *
import numpy as np,onnx
from onnx import numpy_helper

def main():
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 dest=Path(os.environ['A2F_REPORT_OUTPUT']);dest.mkdir(parents=True,exist_ok=True)
 reports={p.stem.removeprefix('evaluation-'):json.loads(p.read_text()) for p in OUT.glob('evaluation-*.json')}
 summary=json.loads((OUT/'summary.json').read_text());names=['baseline']+[f'{a}-s{s}' for s in [37,73,101] for a in ['fixed','learned']]
 cohorts=['feature_confirmation','scale_confirmation','historical'];selections=json.loads((OUT/'selection-lock.json').read_text())
 text=['# Trainable mel-filter frontend experiment','', 'LIFORMA-105 · 11 November2026 · offline research only','',
       '**Outcome:** '+('Promising regression screen; fresh independent recordings, browser measurements and visual acceptance still required.' if summary['promising'] else 'The learned mel-filter bank did not pass the predeclared improvement screen. Keep the current model, 2.5D mapping and Player unchanged.'),'',
       '## What was tested','',
       'This is adaptation of the existing spectral filter bank, not a fully learned waveform encoder. FFT, Hann window, power computation, log compression, width96 temporal predictor, input context, output normalization and animation losses keep their existing form. Both copies of the predictor are fine-tuned. The frontend starts with the identical existing matrix; learned arm updates direct bank weights and projects them nonnegative after each update. Fixed control uses stop-gradient on identical initial bank weights.', '',
       'Three matched pairs (seeds37/73/101) use the same data batches,3,720 updates and development-only selection with epoch zero eligible. Predictor/frontend have separate AdamW states and norm1 clipping, with a cosine schedule and frontend learning rate .01 times predictor rate; frontend weight decay is zero. There is no additional loss. Power caches reproduce all original log-mel arrays exactly. Eight boundary tests check alignment, initial features/predictions/gradients, gradient access and nonnegative finite spectra.', '',
       'Current baseline is finetune-r4, SHA256 `'+EXPECTED+'`. Selected exports retain the original graph and2,251,827-byte artifact. The predictor still has331,047 learned parameters. Learning the existing64x257 initializer adds16,448 learned values, so the learned model has347,495 learned scalars; this is not a claim of unchanged learned capacity.', '',
       'Held-out cohorts were already inspected in earlier research. They are regression screens, not fresh confirmation. No candidate is automatically promoted.', '',
       '## Selected-model results','', '| Model | Primary original gap RMS (mm) | Contact F1 | Scale original RMS (mm) | Historical original RMS (mm) |', '| --- | ---: | ---: | ---: | ---: |']
 for name in names:
  v=[reports[name][c]['conditions']['original'] for c in cohorts]
  text.append(f"| {name} | {v[0]['macro_gap_rmse_mm']:.4f} | {v[0]['macro_contact_f1']:.4f} | {v[1]['macro_gap_rmse_mm']:.4f} | {v[2]['macro_gap_rmse_mm']:.4f} |")
 text += ['', '## Paired effects','', 'Positive reduction means learned frontend beats its paired fixed-frontend fine-tuning control.', '', '| Cohort | Mean gap reduction | Seed37 /73 /101 gap reduction | Mean contact F1 delta |', '| --- | ---: | --- | ---: |']
 for c,e in summary['effects'].items():text.append(f"| {c} | {100*e['mean_relative_gap_reduction']:.3f}% | "+' / '.join(f'{100*x:.3f}%' for x in e['relative_gap_reduction_by_seed'])+f" | {e['mean_contact_f1_delta']:+.5f} |")
 text += ['', 'Predeclared primary threshold:3% mean gap gain, with contact/opening/timed-closure/engine/gain and historical guards. These are pilot engineering criteria, not statistical proof.', '', 'Checks: `'+json.dumps(summary['checks'])+'`.', '', '## Did the frontend actually change?','', '| Seed | Selected update | Bank delta RMS | Positive weights: fixed → learned |', '| --- | ---: | ---: | ---: |']
 for seed in [37,73,101]:
  r=selections[f'learned-s{seed}'];e=r['export'];text.append(f"| {seed} | {r['best_step']} | {e['bank_delta_rms']:.7f} | {e['base_positive_entries']} → {e['bank_positive_entries']} |")
 text += ['', '## Interpretation and runtime implications','',
       'The same ONNX tensor and existing JS frontend can represent the learned bank. However, spreading weights into formerly zero frequency bins can widen each band’s summation range. File size alone therefore says nothing about frontend execution cost. A separate warm Node CPU microbenchmark is saved where available; it is not a browser, GPU or mobile performance measurement.', '',
       'This experiment does not test trainable Fourier filters, phase-sensitive waveform features, a different compression function, longer context, or a larger predictor. A negative result applies to this projected nonnegative bank-adaptation recipe. Controls are newly trained on the same power→mel computation, rather than reused from a different numerical frontend.', '',
       'No new independent confirmation, full-avatar perceptual study, live-model replacement, 2.5D mapping change or Player deployment was performed for a failed pilot. Raw teacher geometry is an agreement reference, not filmed-human ground truth.', '',
       '## Artifacts','', 'Workspace: `'+str(APP.parents[1])+'`','Branch: `research/trainable-audio-frontend`','Results: `'+str(OUT)+'`','', 'The frozen protocol, array/model hashes, development-selected checkpoints, paired sampler locks, ONNX parity reports, full per-clip evaluations and logs are retained in the ignored research bundle. No large weights enter Git. Fixed-case plots use metadata-selected seed37 examples, not a best-test-seed selection.']
 (dest/'trainable_frontend_results.md').write_text('\n'.join(text)+'\n')
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset']
 root=DATA/'.local/compact-feature-v1/confirmation/variants';rows={r['stem']:r for r in json.loads((root/'manifest.json').read_text())};cases=json.loads((OUT/'visual-cases.json').read_text())['cases']
 fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
 for ax,case in zip(axes.flat,cases):
  row=rows[case['stem']]
  with np.load(root/'labels'/(row['stem']+'.npz')) as z:truth=z['gaps']
  t=np.arange(len(truth))/60;ax.plot(t,10*np.median(truth,-1),color='black',label='Raw teacher',lw=1.3)
  for name,color in [('baseline','#888888'),('fixed-s37','#2563aa'),('learned-s37','#cb6332')]:
   pred=np.load(OUT/'predictions'/reports[name]['sha256']/(row['sha256_audio']+'.npy'));gap=np.einsum('tc,ck->tk',pred,basis,optimize=False)+rest
   ax.plot(t,10*np.median(gap,-1),color=color,label=name,lw=.9)
  ax.axhline(1.,color='grey',ls=':',lw=.6);ax.set_title(case['engine']+' · '+case['kind']+'\n'+case['text'],fontsize=8);ax.set(xlabel='Audio time (s)',ylabel='Median lip gap (mm)');ax.grid(alpha=.15)
 axes.flat[0].legend(fontsize=7,ncol=2);fig.savefig(dest/'trainable_frontend_gap_traces.png',dpi=145);plt.close(fig)
 values={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(BASELINE).graph.initializer};bank=values['mel'][:,:,0]
 learned={t.name:numpy_helper.to_array(t).copy() for t in onnx.load(OUT/'learned-s37/model.onnx').graph.initializer}['mel'][:,:,0]
 fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for ax,matrix,title in zip(axes,[bank,learned,learned-bank],['Fixed mel bank','Selected learned bank (seed37)','Learned − fixed']):
  lim=np.max(np.abs(matrix));im=ax.imshow(matrix,aspect='auto',origin='lower',extent=(0,8000,0,64),cmap='coolwarm' if '−' in title else 'viridis',vmin=-lim if '−' in title else 0,vmax=lim);ax.set(title=title,xlabel='Frequency (Hz)',ylabel='Band');fig.colorbar(im,ax=ax)
 fig.savefig(dest/'trainable_frontend_filter_bank.png',dpi=145);plt.close(fig)
 bench=OUT/'frontend-benchmark';bench.mkdir(exist_ok=True);values['filters'][0,0].astype('<f4').tofile(bench/'window.f32')
 first=next(r for r in json.loads((SCALE/'datasets/old-train/manifest.json').read_text()) if r['split']=='train');audio=np.fromfile(first['audio'],'<f4');pcm=np.zeros(16000,np.float32);n=min(16000,len(audio));pcm[:n]=audio[:n];pcm.tofile(bench/'pcm.f32')
 models=[]
 for name,path in [('baseline',BASELINE)]+[(f'learned-s{s}',OUT/f'learned-s{s}/model.onnx') for s in [37,73,101]]:
  matrix=next(numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer if t.name=='mel');target=bench/(name+'.f32');matrix.astype('<f4').tofile(target);models.append(dict(name=name,mel=str(target)))
 write(bench/'fixtures.json',dict(window=str(bench/'window.f32'),pcm=str(bench/'pcm.f32'),models=models))
 for name in ['summary.json','protocol.json','selection-lock.json']:(dest/('trainable_frontend_'+name)).write_bytes((OUT/name).read_bytes())
 print('Wrote report, bank/gap plots, benchmark fixtures and provenance.',flush=True)
if __name__=='__main__':main()
