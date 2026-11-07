from common import *
def main():
 assert sha(BASELINE)==EXPECTED
 prior=json.loads((SCALE/'training-protocol.json').read_text())
 for name,digest in prior['array_sha256'].items():assert sha(SCALE/name)==digest,name
 sources=[HERE/n for n in ['common.py','prepare_corpus.py','run_corpus.py','assemble.py','build_features.py','train.py','freeze_training.py','evaluate_experiment.py']]
 sources += [APP/'docs/plans/compact-targeted-speech.md',APP/'scripts/research/data_scale/mlx_backend.py',APP/'scripts/research/data_scale/features.py',APP/'scripts/research/data_scale/prepare_shard.py']
 arrays={}
 for name in ['broad-train','targeted-train']:
  for p in (OUT/'features'/name).glob('*.npy'):arrays[str(p.relative_to(OUT))]=sha(p)
 freeze(OUT/'training-protocol.json',dict(baseline=str(BASELINE),baseline_sha256=EXPECTED,source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},array_sha256=arrays,allowed_seeds=[37],steps=3720,prior_protocol_sha256=sha(SCALE/'training-protocol.json'),assembly_sha256=sha(OUT/'assembly-lock.json'),corpus_protocol_sha256=sha(OUT/'corpus/protocol.json'),selection='unchanged common dev+Easton, macro RMS+.25*(1-F1)'))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Three matched arms frozen before optimization.',flush=True)
if __name__=='__main__':main()
