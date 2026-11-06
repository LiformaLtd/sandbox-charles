import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


MODEL_SHA256 = '7be7a4491c1e4678d9433be0cc47766e9288172430b594fbfa9991095c6c04d3'
models = Path(__file__).resolve().parents[1] / 'static' / 'models'
model_bytes = (models / 'model.onnx').read_bytes()
if hashlib.sha256(model_bytes).hexdigest() != MODEL_SHA256:
    raise SystemExit('Expected the pinned TinyAligner model. Run npm run assets.')

dictionary = json.loads((models / 'lexicon.json').read_text())
phone_to_index = dictionary['phone2idx']
phones = {'sil', '<blank>'}
phones.update(phone for pronunciation in dictionary['lexicon'].values() for phone in pronunciation)
symbols = sorted(phones, key=phone_to_index.__getitem__)
columns = [phone_to_index[symbol] for symbol in symbols]

model = onnx.load_model_from_string(model_bytes)
for weight in model.graph.initializer:
    if weight.name in {'onnx::MatMul_243', 'model.fc.bias'}:
        values = numpy_helper.to_array(weight)[..., columns]
        weight.CopyFrom(numpy_helper.from_array(values, weight.name))

del model.graph.value_info[:]
model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = len(columns)
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, models / 'model-english.onnx')
(models / 'phones-english.json').write_text(json.dumps({
    symbol: index for index, symbol in enumerate(symbols)
}))
print(f'Built English model with {len(symbols)} of {len(phone_to_index)} output columns.')
