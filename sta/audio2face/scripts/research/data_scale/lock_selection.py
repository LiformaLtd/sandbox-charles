"""Audit all four exports and lock their hashes before the first fresh score."""
import json
import onnx
from prepare_corpus import OUT,freeze,sha


def main():
    result={}
    protocol=json.loads((OUT/'training-protocol.json').read_text());baseline=onnx.load(protocol['baseline'])
    operators=lambda g:[(n.op_type,list(n.input),list(n.output)) for n in g.graph.node]
    for name in ['original-short','original-long','expanded-short','expanded-long']:
        root=OUT/name;row=json.loads((root/'selection.json').read_text());graph=onnx.load(root/'model.onnx')
        if sha(root/'model.onnx')!=row['model_sha256'] or row['protocol_sha256']!=sha(OUT/'training-protocol.json'):raise ValueError('Changed export/protocol')
        if row['bytes']!=2251827 or row['parameters']!=331047 or operators(graph)!=operators(baseline):raise ValueError('Runtime topology changed')
        if row['best_step']>row['budget']:raise ValueError('Checkpoint selected beyond its budget')
        result[name]=row
    for name in ['original','expanded']:
        if not (OUT/'trajectories'/name/'complete.json').exists():raise ValueError('Incomplete trajectory')
    freeze(OUT/'selection-lock.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
