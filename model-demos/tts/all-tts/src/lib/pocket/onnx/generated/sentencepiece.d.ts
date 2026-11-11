interface Disposable { delete(): void }
interface StringView extends Disposable { getView(): Disposable }
interface TokenIds extends Disposable { size(): number; get(index: number): number }
interface Processor {
  Load(modelPath: Disposable): Disposable;
  EncodeAsIds(text: Disposable): TokenIds;
}
interface SentencePieceModule {
  FS: { writeFile(path: string, bytes: Uint8Array): void; unlink(path: string): void };
  StringView: new (text: string) => StringView;
  SentencePieceProcessor: new () => Processor;
}
export default function createModule(): Promise<SentencePieceModule>;
