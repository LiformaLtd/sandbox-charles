export function hannWindow(length: number): Float32Array {
  const out = new Float32Array(length);
  if (length <= 1) return out;
  for (let n = 0; n < length; n++) {
    out[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (length - 1));
  }
  return out;
}

function reverseBits(x: number, bits: number): number {
  let y = 0;
  for (let i = 0; i < bits; i++) {
    y = (y << 1) | (x & 1);
    x >>>= 1;
  }
  return y;
}

export function createRadix2FFT(length: number) {
  const bits = Math.log2(length);
  if (!Number.isInteger(bits)) {
    throw new Error(`FFT input length must be power of 2, got ${length}`);
  }

  const swaps: number[] = [];
  for (let index = 0; index < length; index++) {
    const reversed = reverseBits(index, bits);
    if (reversed > index) swaps.push(index, reversed);
  }

  const stages: { size: number; half: number; twiddleReal: Float64Array; twiddleImag: Float64Array }[] = [];
  for (let size = 2; size <= length; size *= 2) {
    const half = size >>> 1;
    const theta = (-2 * Math.PI) / size;
    const wMulR = Math.cos(theta);
    const wMulI = Math.sin(theta);
    const twiddleReal = new Float64Array(half);
    const twiddleImag = new Float64Array(half);
    let weightReal = 1;
    let weightImag = 0;
    for (let offset = 0; offset < half; offset++) {
      twiddleReal[offset] = weightReal;
      twiddleImag[offset] = weightImag;
      const nextReal = weightReal * wMulR - weightImag * wMulI;
      weightImag = weightReal * wMulI + weightImag * wMulR;
      weightReal = nextReal;
    }
    stages.push({ size, half, twiddleReal, twiddleImag });
  }

  return (real: Float32Array, imag: Float32Array): void => {
    for (let offset = 0; offset < swaps.length; offset += 2) {
      const left = swaps[offset];
      const right = swaps[offset + 1];
      const savedReal = real[left];
      const savedImag = imag[left];
      real[left] = real[right];
      imag[left] = imag[right];
      real[right] = savedReal;
      imag[right] = savedImag;
    }

    for (const { size, half, twiddleReal, twiddleImag } of stages) {
      for (let start = 0; start < length; start += size) {
        for (let offset = 0; offset < half; offset++) {
          const left = start + offset;
          const right = left + half;
          const weightReal = twiddleReal[offset];
          const weightImag = twiddleImag[offset];
          const productReal = weightReal * real[right] - weightImag * imag[right];
          const productImag = weightReal * imag[right] + weightImag * real[right];

          real[right] = real[left] - productReal;
          imag[right] = imag[left] - productImag;
          real[left] = real[left] + productReal;
          imag[left] = imag[left] + productImag;
        }
      }
    }
  };
}
