from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes())
 sources=[HERE/n for n in ['common.py','loss.py','train.py','test_loss.py','prepare_confirmation.py','run_confirmation.py','freeze_training.py','evaluate_experiment.py','inference.py']]
 sources += [APP/'scripts/research/data_scale'/n for n in ['features.py','mlx_backend.py','evaluate.py']]
 sources += [APP/'scripts/research/multivoice/closure_metrics.py',APP/'docs/plans/compact-loss-balance.md']
 arrays={}
 prior=json.loads((SCALE/'training-protocol.json').read_text())
 for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest;arrays[name]=digest
 freeze(OUT/'training-protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,allowed_seeds=[37],steps=3720,shape_coefficients=dict(control=1.,balanced=4.),source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},array_sha256=arrays,confirmation_audit_sha256=sha(OUT/'confirmation/audit.json')))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Frozen both losses, full existing corpus, training sources and fresh confirmation.',flush=True)
if __name__=='__main__':main()
