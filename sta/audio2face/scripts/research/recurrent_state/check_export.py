from common import *
import numpy as np,onnxruntime as ort

def main():
 z=np.load(OUT/'native-reference.npz');report={}
 for arm in ['control','recurrent']:
  options=ort.SessionOptions();options.intra_op_num_threads=2;s=ort.InferenceSession(str(OUT/arm/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider']);h=np.zeros((1,32),np.float32);errors=[];state_errors=[]
  for t,wave in enumerate(z['pcm']):
   valid=z['valid'][t];first=int(np.flatnonzero(valid)[0]+15);count=int(valid.sum());out,h=s.run(None,dict(pcm=wave[None],state=h,first=np.array([first],np.int64),count=np.array([count],np.int64)))
   errors.append(float(np.abs(out[0,:count]-z[arm][t,valid]).max()));state_errors.append(float(np.abs(h-z[arm+'_states'][t]).max()))
  assert max(errors)<5e-5 and max(state_errors)<5e-5,(arm,errors,state_errors)
  # Empty emitted crop leaves recurrent state exactly unchanged.
  hx=np.ones((1,32),np.float32);_,after=s.run(None,dict(pcm=z['pcm'][0,None],state=hx,first=np.array([15],np.int64),count=np.array([0],np.int64)));np.testing.assert_array_equal(hx,after)
  report[arm]=dict(max_coefficient_delta=max(errors),max_state_delta=max(state_errors),empty_crop_unchanged=True)
 freeze(OUT/'export-parity.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
