import createModule from './generated/sentencepiece.js';

/** Browser-only interface to the pinned upstream SentencePiece engine. */
export async function createTokenizer(bytes: ArrayBuffer) {
  const module = await createModule();
  const filename = '/tokenizer.model';
  module.FS.writeFile(filename, new Uint8Array(bytes));
  const path = new module.StringView(filename);
  const view = path.getView();
  const processor = new module.SentencePieceProcessor();
  try {
    const status = processor.Load(view);
    status.delete();
  } finally {
    view.delete();
    path.delete();
    module.FS.unlink(filename);
  }
  return {
    encodeIds(text: string): number[] {
      const string = new module.StringView(text);
      const value = string.getView();
      try {
        const ids = processor.EncodeAsIds(value);
        try {
          return Array.from({ length: ids.size() }, (_, index) => ids.get(index));
        } finally {
          ids.delete();
        }
      } finally {
        value.delete();
        string.delete();
      }
    }
  };
}
