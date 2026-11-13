import models from './models.json';
import type { ModelSpec } from 'phonon-tts';

export { models };
export type ModelKey = keyof typeof models;
export type Language = 'en' | 'fr' | 'de' | 'es' | 'pt';
export const languages: Record<Language, string> = { en: 'English', fr: 'French', de: 'German', es: 'Spanish', pt: 'Portuguese' };
export function modelSource(key: ModelKey): ModelSpec | string {
	const model = models[key];
	if (!model.local) return `hf://${model.repo}@${model.revision}`;
	const base = `/api/model-assets/${key}/`;
	const url = (file: string) => `${base}${file}?revision=${model.revision}`;
	return { weights: { q8: url('model.q8.gguf') }, tokenizer: url('tokenizer.json'), config: url('config.json') };
}

export const sampleText: Record<Language, string> = {
	en: 'The rain had stopped by morning. Across the river, a small café was already open, and the smell of fresh bread drifted down the street.',
	fr: 'La pluie avait cessé au petit matin. De l’autre côté de la rivière, un petit café était déjà ouvert et une odeur de pain frais flottait dans la rue.',
	de: 'Am Morgen hatte der Regen aufgehört. Auf der anderen Seite des Flusses war ein kleines Café bereits geöffnet, und der Duft von frischem Brot zog durch die Straße.',
	es: 'La lluvia había cesado por la mañana. Al otro lado del río, una pequeña cafetería ya estaba abierta y el olor a pan recién hecho recorría la calle.',
	pt: 'A chuva tinha parado de manhã. Do outro lado do rio, um pequeno café já estava aberto, e o cheiro de pão fresco se espalhava pela rua.'
};
