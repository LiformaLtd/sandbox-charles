export const onnxVersions = {
	fp32: { name: 'Full precision · FP32', size: '165.8 MB' },
	fp16: { name: 'Half precision · FP16', size: '83.0 MB' },
	int8: { name: 'INT8', size: '64.0 MB' },
	q4: { name: 'Q4', size: '51.3 MB' },
	q4f16: { name: 'Q4 with FP16', size: '34.6 MB' }
} as const;

export type OnnxVersion = keyof typeof onnxVersions;
export type OnnxDevice = 'wasm' | 'webgpu';
export type RuntimeMode = 'browser' | 'native' | 'mlx' | `onnx-${OnnxVersion}`;

export function onnxVersion(mode: RuntimeMode): OnnxVersion | undefined {
	return mode.startsWith('onnx-') ? mode.slice(5) as OnnxVersion : undefined;
}

export const onnxRepository = 'onnx-community/phonon-31m-en-ONNX';
// Pin graphs, tokenizer and voice layout together; the Hub's main branch can change.
export const onnxRevision = 'f0f522a750bf515227cb478abb97b2ffca7e6d8c';
export const onnxBase = `https://huggingface.co/${onnxRepository}/resolve/${onnxRevision}/`;
