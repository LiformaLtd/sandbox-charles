from pathlib import Path
import os,sys,json
import numpy as np
import onnxruntime as ort
h=Path(os.environ['A2F_ABLATION_OUTPUT']).resolve()
work=Path(os.environ['A2F_RESEARCH_WORK']).resolve()
for name in sys.argv[1:]:
    path=work/'arkit-student-v3/james-arkit-student-v3.onnx' if name=='baseline' else h/(name+'.onnx')
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    n=session.get_inputs()[0].shape[1]
    row=json.loads((work/'arkit-student-v3/manifest.json').read_text())[0]
    pcm=np.fromfile(row['audio'],'<f4')[:n];pcm=np.pad(pcm,(0,max(0,n-len(pcm))))
    rows=[]
    for x in [np.zeros(n,np.float32),np.random.default_rng(19).normal(0,.035,n).astype(np.float32),pcm]:
        expected=session.run(None,{'pcm':x[None]})[0][0]
        assert np.isfinite(expected).all()
        rows.append({'pcm':x.tolist(),'expected':expected.ravel().tolist()})
    (h/(name+'-parity.json')).write_text(json.dumps(rows))
