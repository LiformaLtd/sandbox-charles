export const revision = '3cadd1ee6394adea1bd021217a0e650ede09a323';
export const modelBase = `https://huggingface.co/Supertone/supertonic-3/resolve/${revision}/`;
export const graphs = ['duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder'];
export const voices = ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'];
export const languages = ['en', 'ko', 'ja', 'ar', 'bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'tr', 'uk', 'vi'];
export const files = [...graphs.map(name => `onnx/${name}.onnx`), 'onnx/tts.json', 'onnx/unicode_indexer.json', ...voices.map(name => `voice_styles/${name}.json`), 'LICENSE'];
export interface Settings { text: string; voice: string; language: string; steps: number; speed: number }
export interface Packet { status?: string; setup?: number; generation?: number; pcm?: Float32Array<ArrayBuffer>; sampleRate?: number; error?: string; done?: boolean }
export function segments(text: string, language: string) {
 const limit = ['ja','ko'].includes(language) ? 100 : 180;
 const sentences = Array.from(new Intl.Segmenter(language, { granularity: 'sentence' }).segment(text), s => s.segment.trim()).filter(Boolean);
 const result: string[] = [];
 for (let sentence of sentences) {
  while (sentence.length > limit) {
   const space = sentence.lastIndexOf(' ', limit);
   const cut = space > limit / 2 ? space : limit;
   result.push(sentence.slice(0, cut)); sentence = sentence.slice(cut).trim();
  }
  if (sentence) result.push(sentence);
 }
 return result;
}
