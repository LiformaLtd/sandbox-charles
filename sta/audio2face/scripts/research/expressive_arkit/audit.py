"""Test complete rig recovery and record exact official model interfaces."""
import json
from pathlib import Path
import numpy as np
import onnx
from common import *
from fit import FullFaceFit, gaze_coefficients, EYE_ENDPOINT_DEGREES


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fitter=FullFaceFit(ASSETS,WORK/'arkit-student/james_arkit_v3.ma',APP)
    # One channel at a time; opposed gaze directions deliberately remain exclusive.
    coefficients=np.vstack([np.zeros((1,52)),np.eye(52)*.5]).astype(np.float32)
    geometry=np.zeros((53,88831),np.float32)
    skin=np.einsum('fc,cvk->fvk',coefficients,fitter.shape,optimize=False)+fitter.neutral-fitter.model_neutral
    geometry[:,:72006]=skin.reshape(53,-1)
    tongue=np.einsum('fc,cvk->fvk',coefficients,fitter.tongue_mouth,optimize=False)
    tongue+=coefficients[:,51,None,None]*fitter.tongue_basis[fitter.tongue_out_index].reshape(1,5602,3)
    geometry[:,72006:88812]=(tongue+fitter.tongue_neutral-fitter.model_tongue).reshape(53,-1)
    geometry[:,-4]=EYE_ENDPOINT_DEGREES*(coefficients[:,8]-coefficients[:,11])
    geometry[:,-3]=EYE_ENDPOINT_DEGREES*(coefficients[:,9]-coefficients[:,10])
    geometry[:,-2]=EYE_ENDPOINT_DEGREES*(coefficients[:,1]-coefficients[:,4])
    geometry[:,-1]=EYE_ENDPOINT_DEGREES*(coefficients[:,3]-coefficients[:,2])
    fitted,extra=fitter(geometry)
    error=np.abs(fitted-coefficients)
    reports=[dict(name=name,max_error=float(error[i+1].max()),
                  own_recovered=float(fitted[i+1,i]),skin_rms=float(extra['skin_fit_rms'][i+1]))
             for i,name in enumerate(fitter.names)]
    graph=onnx.load(TEACHER/'models_cache/audio2face-v3/network.onnx',load_external_data=False)
    inputs={v.name:[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim] for v in graph.graph.input}
    outputs={v.name:[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim] for v in graph.graph.output}
    report=dict(interface=dict(inputs=inputs,outputs=outputs),
        output_components=dict(skin=72006,tongue=16806,jaw=15,eyes=4,head=0),
        synthetic_own_rig_roundtrip=reports,max_coefficient_error=float(error.max()),
        neutral_error=float(error[0].max()),eye_endpoint_degrees=EYE_ENDPOINT_DEGREES,
        limitations=['Synthetic self-consistency does not validate ARKit semantic calibration.',
          'Gaze endpoints are a declared 25-degree research calibration; visual sign/scale check still required.',
          'tongueOut uses tongueStretch as a protrusion proxy, not a supplied ARKit tongueOut target.',
          'No official NVIDIA head-motion checkpoint was found in the linked releases; head deferred.'],
        official_sources=['https://arxiv.org/html/2508.16401v1#S9.SS2',
          'https://github.com/NVIDIA/Audio2Face-3D/blob/release/README.md',
          'https://huggingface.co/nvidia/Audio2Face-3D-v3.0/tree/b74132732fd9a9d29b237bec193ded64c9745e91'])
    freeze(OUT/'rig-audit.json',report)
    print(json.dumps(report,indent=2))
    if error.max()>.06 or error[0].max()>.003:
        raise ValueError('Research rig fails synthetic recovery tolerance')
    bad=np.array([[float('nan'),0,0,0]],np.float32)
    try:gaze_coefficients(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid gaze accepted')


if __name__=='__main__':main()
