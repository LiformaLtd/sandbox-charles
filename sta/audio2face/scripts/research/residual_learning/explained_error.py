"""Post-selection measurement of recoverable residual energy on fresh originals."""
from common import *
import numpy as np,onnx
from onnx import numpy_helper

def main():
    models={'v4':BASELINE,**{a:OUT/a/'model.onnx' for a in ['control','residual']}};hashes={a:sha(p) for a,p in models.items()};rig=np.load(WORK/'arkit-student/rig-metrics.npz');active=rig['active'];basis=rig['gap_basis'];rest=rig['rest']+rig['gap_offset'];v={t.name:numpy_helper.to_array(t) for t in onnx.load(BASELINE).graph.initializer};rows=[r for r in json.loads((OUT/'confirmation/variants/manifest.json').read_text()) if r['condition']=='original'];stats={}
    for row in rows:
        pred={a:np.load(OUT/'predictions'/h/(row['sha256_audio']+'.npy')) for a,h in hashes.items()}
        with np.load(OUT/'confirmation/variants/labels'/(row['stem']+'.npz')) as z:target=z['weights'];gaps=z['gaps']
        residual=(target[:,active]-pred['v4'][:,active])/v['student.scale'];rawgap=pred['v4']@basis+rest-gaps;key=row['engine']+':'+row['voice'];s=stats.setdefault(key,{a:np.zeros(4) for a in ['control','residual']})
        for arm in s:
            remaining=(target[:,active]-pred[arm][:,active])/v['student.scale'];gap=pred[arm]@basis+rest-gaps;s[arm] += [np.square(residual).sum(),np.square(remaining).sum(),np.square(rawgap).sum(),np.square(gap).sum()]
    voices={voice:{a:dict(normalized_coefficient_residual_energy_explained=float(1-s[1]/s[0]),raw_geometry_gap_residual_energy_explained=float(1-s[3]/s[2])) for a,s in arms.items()} for voice,arms in stats.items()}
    result=dict(source_sha256=sha(__file__),model_sha256=hashes,clips=len(rows),voices=voices,macro={a:{key:float(np.mean([v[a][key] for v in voices.values()])) for key in next(iter(voices.values()))[a]} for a in ['control','residual']},note='Held-out original clips, voice-macro fraction of total squared v4 error reduced after output clipping; not conventional mean-centered R-squared, irreducible-noise estimate or a checkpoint selection criterion.')
    freeze(OUT/'residual-explained.json',result);print(json.dumps(result['macro'],indent=2))

if __name__=='__main__':main()
