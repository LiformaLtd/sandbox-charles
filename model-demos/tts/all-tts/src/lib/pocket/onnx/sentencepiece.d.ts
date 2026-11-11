declare module '@sctg/sentencepiece-js' {
	export class SentencePieceProcessor {
		loadFromB64StringModel(model: string): Promise<void>;
		encodeIds(text: string): number[];
	}
}
