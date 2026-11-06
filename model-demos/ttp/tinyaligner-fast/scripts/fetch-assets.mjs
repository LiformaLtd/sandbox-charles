import { mkdir, writeFile } from 'node:fs/promises';

const base = 'https://raw.githubusercontent.com/krmanik/tiny-aligner/1f07ad6d186c70baac4fae26c605cc84660d65f7/browser/public/';
const destination = new URL('../static/models/', import.meta.url);

async function download(file) {
	const response = await fetch(base + file, { signal: AbortSignal.timeout(60000) });
	if (!response.ok) throw new Error(`Could not download ${file} (HTTP ${response.status})`);
	return response;
}

const [model, dictionary] = await Promise.all([download('model.onnx'), download('lexicon.json')]);
const [{ lexicon, phone2idx }, weights] = await Promise.all([dictionary.json(), model.arrayBuffer()]);
const english = Object.fromEntries(Object.entries(lexicon).filter(([word, phones]) =>
	/^[A-Z]+(?:'[A-Z]+)*$/.test(word) && phones.length &&
	phones.every((phone) => /^[A-Z]+$/.test(phone) && Object.hasOwn(phone2idx, phone))
));
await mkdir(destination, { recursive: true });
await Promise.all([
	writeFile(new URL('model.onnx', destination), new Uint8Array(weights)),
	writeFile(new URL('lexicon.json', destination), JSON.stringify({ lexicon: english, phone2idx }))
]);
console.log('TinyAligner model and English dictionary are ready.');
