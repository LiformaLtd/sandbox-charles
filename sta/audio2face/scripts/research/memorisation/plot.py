"""Standalone learning curves; memorised training data only."""
from common import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for arm,label in [('mixed','Existing mixed loss'),('fit-mse','Fitted-target MSE')]:
  r=json.loads((OUT/arm/'result.json').read_text());history=[dict(step=0,**r['initial'])]+r['history'];steps=[h['step'] for h in history]
  axes[0].plot(steps,[h['student_to_raw_rms_mm'] for h in history],label=label);axes[1].plot(steps,[h['contact_f1'] for h in history],label=label)
 axes[0].axhline(r['initial']['fitting_rms_mm'],color='black',ls='--',label='Subset fitted reference')
 axes[0].set(ylabel='Raw-teacher lip-gap RMS (mm)',xlabel='Full-batch updates',title='Training-set gap error');axes[0].set_ylim(bottom=0)
 axes[1].set(ylabel='Contact F1',xlabel='Full-batch updates',title='Training-set contact accuracy');axes[1].set_ylim(top=1.005)
 for ax in axes:ax.legend(fontsize=8)
 fig.suptitle('Unchanged v4, eight fixed training clips — not held-out quality')
 fig.savefig(OUT/'learning-curves.png',dpi=150)
if __name__=='__main__':main()
