"""Verify measurement adapter equivalence and the refit's unchanged runtime graph."""
from common import *
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
from evaluate import evaluate_cohort
from measure import score_records
sys.path.insert(0,str(APP/'scripts/research/syncnet_loss'))
from inference import infer
import metrics
metrics.infer=infer

def main():
 for base in [OUT,OUT/'head-evaluation']:
  protocol=json.loads((base/'protocol.json').read_text())
  for name,digest in protocol['sources'].items():assert sha(APP/name)==digest,name
  for value in protocol['models'].values():assert sha(value['path'])==value['sha256']
 baseline=onnx.load(MODELS['control']);refit=onnx.load(OUT/'head-refit/model.onnx');ops=lambda m:[(n.op_type,list(n.input),list(n.output)) for n in m.graph.node];assert ops(baseline)==ops(refit)
 old={v.name:numpy_helper.to_array(v) for v in baseline.graph.initializer};new={v.name:numpy_helper.to_array(v) for v in refit.graph.initializer};changed=[]
 for key in old:
  assert old[key].shape==new[key].shape
  if not np.array_equal(old[key],new[key]):changed.append(key)
 assert set(changed)=={'student.output.weight','student.output.bias'}
 rig=np.load(WORK/'arkit-student/rig-metrics.npz');root=SCALE/'datasets/old-train';rows=json.loads((root/'manifest.json').read_text());options=ort.SessionOptions();options.intra_op_num_threads=2;s=ort.InferenceSession(str(MODELS['control']),sess_options=options,providers=['CPUExecutionProvider']);cache=OUT/'predictions'/sha(MODELS['control']);reference=evaluate_cohort(s,rows,root/'labels',cache,rig['gap_basis'],rig['rest']+rig['gap_offset'],rig['active']);actual=json.loads((OUT/'evaluation-control.json').read_text())['cohorts']['old-train']
 for condition,r in reference['conditions'].items():
  for key in ['macro_gap_rmse_mm','macro_contact_f1']:assert abs(r[key]-actual['conditions'][condition][key])<1e-12,(condition,key)
  for phone in ['m','b','p']:assert r['closure_audit'][phone]['well_timed_hits']==actual['conditions'][condition]['closure_audit'][phone]['well_timed_hits']
 reports=[json.loads((OUT/f'evaluation-{a}.json').read_text()) for a in MODELS];reports += [json.loads((OUT/'head-evaluation/evaluation-head-refit.json').read_text())];delta=max(r['delta'] for report in reports for r in report['native_parity']);assert delta<5e-5
 freeze(OUT/'validation.json',dict(source_sha256=sha(__file__),scorer_matches_existing_evaluator=True,comparison_clips=len(rows),changed_initializers=changed,graph_identical=True,bytes=(OUT/'head-refit/model.onnx').stat().st_size,updated_parameters=3783,max_native_coefficient_delta=delta,initial_measurement_protocol_sha256=sha(OUT/'protocol.json'),head_protocol_sha256=sha(OUT/'head-protocol.json')));print((OUT/'validation.json').read_text())
if __name__=='__main__':main()
