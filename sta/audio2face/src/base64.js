// Keep argument lists below WebKit's limit when encoding tokenizer bytes.
export function bytesToBase64(bytes) {
  const parts = [];
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    parts.push(String.fromCharCode(...bytes.subarray(offset, offset + 8192)));
  }
  return btoa(parts.join(''));
}
