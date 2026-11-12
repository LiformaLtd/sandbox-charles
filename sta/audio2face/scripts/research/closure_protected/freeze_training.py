from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 conflict=json.loads((OUT/'conflicts.json').read_text());assert conflict['proceed']
 assert json.loads((OUT/'projection-tests.json').read_text())['passes']
 steps=json.loads((OUT/'step-tests.json').read_text());assert steps['passes'] and steps['nonzero_steps']>=8
 pre=json.loads((OUT/'preflight-protocol.json').read_text())
 for name,digest in pre['sources'].items():assert sha(HERE/name)==digest
 freeze(OUT/'sources/model.py',(SCALE/'sources/model.py').read_bytes())
 sources=[HERE/n for n in ['common.py','loss.py','protection.py','step.py','preflight.py','test_protection.py','check_steps.py','train.py','prepare_confirmation.py','run_confirmation.py','freeze_training.py','evaluate_experiment.py','inference.py']]
 sources += [APP/'scripts/research/data_scale'/n for n in ['features.py','mlx_backend.py','evaluate.py']]+[APP/'scripts/research/multivoice/closure_metrics.py',APP/'docs/plans/compact-closure-protected.md']
 arrays={}
 for name,digest in json.loads((SCALE/'training-protocol.json').read_text())['array_sha256'].items():assert sha(SCALE/name)==digest;arrays[name]=digest
 freeze(OUT/'training-protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,allowed_seeds=[37],steps=3720,method='Two-guard projection of actual AdamW displacement, nonlinear same-batch backtracking; control total loss unchanged',source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},array_sha256=arrays,confirmation_audit_sha256=sha(OUT/'confirmation/audit.json'),preflight_sha256={n:sha(OUT/n) for n in ['conflicts.json','projection-tests.json','step-tests.json']}))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Frozen both arms, update protection, data/selection and fresh confirmation.',flush=True)
if __name__=='__main__':main()
