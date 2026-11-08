from common import *
import importlib.util,numpy as np,torch,onnx
from onnx import numpy_helper

def main():
 torch.set_num_threads(2);spec=importlib.util.spec_from_file_location('base',OUT/'sources/model.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);values={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(BASELINE).graph.initializer};active=np.load(WORK/'arkit-student/rig-metrics.npz')['active']
 for arm in ['mixed','fit-mse']:
  report=json.loads((OUT/arm/'result.json').read_text());assert sha(OUT/arm/'best.npz')==report['best_sha256'];student=base.Student(96,values['student.mean'],values['student.scale'],active);state={k:torch.from_numpy(values['student.'+k].copy()) for k in student.state_dict()}
  with np.load(OUT/arm/'best.npz') as z:
   for k in z.files:state[k]=torch.from_numpy(z[k].copy())
  student.load_state_dict(state);student.eval();path=OUT/arm/'model.onnx';torch.onnx.export(base.WaveformModel(student).eval(),torch.zeros(1,16000),str(path),input_names=['pcm'],output_names=['blendshapes'],opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(path));assert path.stat().st_size==2251827
  freeze(OUT/arm/'export.json',dict(bytes=path.stat().st_size,model_sha256=sha(path),parameters=sum(v.numel() for v in student.parameters())))
if __name__=='__main__':main()
