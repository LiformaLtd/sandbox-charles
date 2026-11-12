"""Training-only residual structure; correlation is not out-of-sample predictability."""
from common import *
import numpy as np,onnx
from onnx import numpy_helper
from cache_features import FrozenDataset

def main():
    v={t.name:numpy_helper.to_array(t) for t in onnx.load(BASELINE).graph.initializer};w=v['student.output.weight'][:,:,0];bias=v['student.output.bias'];scale=v['student.scale'];mean=v['student.mean'];s=np.zeros(39);q=np.zeros((39,39));n=0;lag=np.zeros((5,39));pairs=0
    for part in ['old-train','new-train']:
        dataset=FrozenDataset([part]);data=dataset.parts[0]
        for start in range(0,dataset.length,128):
            h=data['mel'][start:start+128,15:45];valid=data['valid'][start:start+128];prediction=h@w.T+bias;error=((data['weights'][start:start+128]-mean)/scale-prediction).astype(np.float64);e=error[valid];s+=e.sum(0);q+=e.T@e;n+=len(e)
            adjacent=valid[:,1:]&valid[:,:-1];a=error[:,:-1][adjacent];b=error[:,1:][adjacent];lag+=np.stack([a.sum(0),b.sum(0),(a*a).sum(0),(b*b).sum(0),(a*b).sum(0)]);pairs+=len(a)
    mu=s/n;cov=q/n-np.outer(mu,mu);eigen=np.maximum(np.linalg.eigvalsh(cov),0)[::-1];correlation=(lag[4]-lag[0]*lag[1]/pairs)/np.maximum(np.sqrt((lag[2]-lag[0]**2/pairs)*(lag[3]-lag[1]**2/pairs)),1e-12);active=np.load(WORK/'arkit-student/rig-metrics.npz')['active'];names=json.loads((WORK/'arkit-student/channels.json').read_text())
    result=dict(source_sha256=sha(__file__),baseline_sha256=sha(BASELINE),frames=n,adjacent_pairs=pairs,coefficient_normalization='v4 fixed scale, errors before output clipping; all-primary gains, frame weighted',bias_fraction_of_normalized_mse=float((mu@mu)/np.trace(q/n)),median_lag1_correlation=float(np.median(correlation)),top5_variance_fraction=float(eigen[:5].sum()/eigen.sum()),channels={names[int(i)]:dict(mean_normalized_residual=float(m),lag1_correlation=float(c),variance=float(var)) for i,m,c,var in zip(active,mu,correlation,np.diag(cov))},note='Structured/smooth residuals alone do not establish predictability from the available frozen features, or distinguish missing history from optimization/capacity.')
    freeze(OUT/'residual-structure.json',result);print(json.dumps({k:v for k,v in result.items() if k!='channels'},indent=2))

if __name__=='__main__':main()
