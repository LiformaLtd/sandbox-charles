// Adapted from the Pocket TTS ONNX Web Demo by KevinAHM (Apache-2.0, see POCKET-LICENSE).
// https://huggingface.co/spaces/KevinAHM/pocket-tts-web/tree/d0c0c79b7712256a32d691c67f20b8ae2e020d00
import * as ort from 'onnxruntime-web/wasm';

const base = '/assets/pocket';
const CHUNK_GAP_SEC = 0.25;
const MAX_FRAMES = 500;
const LSD_STEPS = 1;
let bundleMetadata;
let tokenizerProcessor;
let voiceState;
let textConditionerSession;
let flowLmMainSession;
let flowLmFlowSession;
let mimiDecoderSession;
let currentSampleRate;
let currentLatentDim;
let currentConditioningDim;
let currentMaxTokenPerChunk;
let stTensors;

async function asset(name) {
    const response = await fetch(`${base}/${name}`);
    if (!response.ok) throw new Error(`Missing Pocket TTS asset ${name}. Run sh scripts/fetch_onnx_tts.sh.`);
    return response;
}

async function load() {
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 1, 4);
    bundleMetadata = await (await asset('bundle.json')).json();
    currentSampleRate = bundleMetadata.sample_rate;
    currentLatentDim = bundleMetadata.latent_dim;
    currentConditioningDim = bundleMetadata.conditioning_dim;
    currentMaxTokenPerChunk = bundleMetadata.max_token_per_chunk;
    const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const session = (name) => ort.InferenceSession.create(`${base}/${name}_int8.onnx`, options);
    [textConditionerSession, flowLmMainSession, flowLmFlowSession, mimiDecoderSession] = await Promise.all([
        session('text_conditioner'), session('flow_lm_main'), session('flow_lm_flow'), session('mimi_decoder'),
    ]);

    const tokenizerUrl = `${base}/sentencepiece.js`;
    const { SentencePieceProcessor } = await import(/* @vite-ignore */ tokenizerUrl);
    const tokenizerBuffer = await (await asset(bundleMetadata.tokenizer_file)).arrayBuffer();
    tokenizerProcessor = new SentencePieceProcessor();
    await tokenizerProcessor.loadFromB64StringModel(btoa(String.fromCharCode(...new Uint8Array(tokenizerBuffer))));
    const voices = parseVoiceStatesBin(await (await asset('voices.bin')).arrayBuffer());
    voiceState = stateFromVoiceRecord(voices.alba);
    precomputeFlowBuffers();
}

function emitAudio(pcm) {
    postMessage({ type: 'audio', pcm, rate: currentSampleRate }, [pcm.buffer]);
}

let loading;
self.onmessage = async ({ data: text }) => {
    try {
        if (!loading) postMessage({ type: 'status', message: 'Loading Pocket TTS in your browser…' });
        loading ??= load();
        await loading;
        postMessage({ type: 'status', message: 'Generating speech with Pocket TTS…' });
        const { chunks, framesAfterEos } = splitIntoBestSentences(text);
        await runGenerationPipeline(chunks, framesAfterEos);
        postMessage({ type: 'done' });
    } catch (error) {
        postMessage({ type: 'error', message: error.message });
    }
};

function makeFilledArray(shape, dtype, fill) {
    const size = shape.reduce((a, b) => a * b, 1);
    let data;

    if (dtype === "int64") {
        data = new BigInt64Array(size);
    } else if (dtype === "bool") {
        data = new Uint8Array(size);
    } else {
        data = new Float32Array(size);
        if (fill === "nan") {
            data.fill(NaN);
        } else if (fill === "ones") {
            data.fill(1);
        }
    }

    return data;
}

function createTensor(dtype, data, dims) {
    return new ort.Tensor(dtype, data, dims);
}

function initStateFromManifest(manifest) {
    const state = {};
    for (const entry of manifest) {
        state[entry.input_name] = createTensor(
            entry.dtype,
            makeFilledArray(entry.shape, entry.dtype, entry.fill),
            entry.shape
        );
    }
    return state;
}

function cloneState(state) {
    return { ...state };
}

function updateStateFromManifestOutputs(state, result, manifest) {
    for (const entry of manifest) {
        state[entry.input_name] = result[entry.output_name];
    }
}

function groupVoiceRecordByModule(record) {
    const grouped = {};
    for (const [key, value] of Object.entries(record)) {
        const slash = key.indexOf("/");
        if (slash === -1) continue;
        const moduleName = key.slice(0, slash);
        const tensorKey = key.slice(slash + 1);
        if (!grouped[moduleName]) {
            grouped[moduleName] = {};
        }
        grouped[moduleName][tensorKey] = value;
    }
    return grouped;
}

function adaptTypedArray(source, entry) {
    const targetShape = entry.shape;
    const targetSize = targetShape.reduce((a, b) => a * b, 1);
    const target = makeFilledArray(targetShape, entry.dtype, entry.fill);

    if (source.shape.length === targetShape.length) {
        const exactShape = source.shape.every((dim, idx) => dim === targetShape[idx]);
        if (exactShape) {
            if (entry.dtype === "int64") {
                return new BigInt64Array(source.data);
            }
            if (entry.dtype === "bool") {
                return new Uint8Array(source.data);
            }
            return new Float32Array(source.data);
        }
    }

    if (source.data.length === targetSize) {
        if (entry.dtype === "int64") {
            return new BigInt64Array(source.data);
        }
        if (entry.dtype === "bool") {
            return new Uint8Array(source.data);
        }
        return new Float32Array(source.data);
    }

    if (source.shape.length !== targetShape.length) {
        return target;
    }

    const strides = [];
    let stride = 1;
    for (let i = source.shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= source.shape[i];
    }

    const indices = new Array(source.shape.length).fill(0);
    const maxIndices = source.shape.map((dim, idx) => Math.min(dim, targetShape[idx]));

    function targetIndex(coords) {
        let idx = 0;
        let tStride = 1;
        for (let i = targetShape.length - 1; i >= 0; i--) {
            idx += coords[i] * tStride;
            tStride *= targetShape[i];
        }
        return idx;
    }

    let done = false;
    while (!done) {
        let sourceIdx = 0;
        for (let i = 0; i < indices.length; i++) {
            sourceIdx += indices[i] * strides[i];
        }
        target[targetIndex(indices)] = source.data[sourceIdx];

        for (let dim = indices.length - 1; dim >= 0; dim--) {
            indices[dim] += 1;
            if (indices[dim] < maxIndices[dim]) {
                break;
            }
            indices[dim] = 0;
            if (dim === 0) {
                done = true;
            }
        }
    }

    return target;
}

function deriveStep(moduleState) {
    if (moduleState.step) {
        return { data: BigInt64Array.from([BigInt(moduleState.step.data[0])]), shape: [1], dtype: "int64" };
    }
    if (moduleState.offset && !moduleState.end_offset) {
        return { data: BigInt64Array.from([BigInt(moduleState.offset.data[0])]), shape: [1], dtype: "int64" };
    }
    if (moduleState.current_end) {
        return { data: BigInt64Array.from([BigInt(moduleState.current_end.shape[0])]), shape: [1], dtype: "int64" };
    }
    return { data: BigInt64Array.from([0n]), shape: [1], dtype: "int64" };
}

function stateFromVoiceRecord(record) {
    const grouped = groupVoiceRecordByModule(record);
    const state = initStateFromManifest(bundleMetadata.flow_lm_state_manifest);

    for (const entry of bundleMetadata.flow_lm_state_manifest) {
        const moduleState = grouped[entry.module] || {};
        let source = moduleState[entry.key];
        if (!source && entry.key === "step") {
            source = deriveStep(moduleState);
        }
        if (!source) {
            continue;
        }

        const data = adaptTypedArray(source, entry);
        state[entry.input_name] = createTensor(entry.dtype, data, entry.shape);
    }

    return state;
}

function parseVoiceStatesBin(buffer) {
    const view = new DataView(buffer);
    let offset = 0;
    const magic = new TextDecoder().decode(new Uint8Array(buffer, offset, 5));
    offset += 5;
    if (magic !== "PTVB1") {
        throw new Error("Invalid voices.bin header");
    }

    const voices = {};
    const voiceCount = view.getUint32(offset, true);
    offset += 4;

    for (let voiceIndex = 0; voiceIndex < voiceCount; voiceIndex++) {
        const nameLen = view.getUint16(offset, true);
        offset += 2;
        const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen));
        offset += nameLen;

        const tensorCount = view.getUint16(offset, true);
        offset += 2;
        const tensors = {};

        for (let tensorIndex = 0; tensorIndex < tensorCount; tensorIndex++) {
            const keyLen = view.getUint16(offset, true);
            offset += 2;
            const key = new TextDecoder().decode(new Uint8Array(buffer, offset, keyLen));
            offset += keyLen;

            const dtypeCode = view.getUint8(offset);
            offset += 1;
            const rank = view.getUint8(offset);
            offset += 1;

            const shape = [];
            for (let dimIndex = 0; dimIndex < rank; dimIndex++) {
                shape.push(view.getUint32(offset, true));
                offset += 4;
            }

            const byteLength = view.getUint32(offset, true);
            offset += 4;

            let data;
            if (dtypeCode === 0) {
                data = new Float32Array(buffer.slice(offset, offset + byteLength));
            } else if (dtypeCode === 1) {
                data = new BigInt64Array(buffer.slice(offset, offset + byteLength));
            } else if (dtypeCode === 2) {
                data = new Uint8Array(buffer.slice(offset, offset + byteLength));
            } else {
                throw new Error(`Unsupported voices.bin dtype code: ${dtypeCode}`);
            }
            offset += byteLength;

            tensors[key] = {
                data,
                shape,
                dtype: dtypeCode === 0 ? "float32" : dtypeCode === 1 ? "int64" : "bool",
            };
        }

        voices[name] = tensors;
    }

    return voices;
}

function prepareTextPrompt(text) {
    let prompt = text.trim();
    if (!prompt) {
        return { text: "", framesAfterEos: 1 };
    }

    prompt = prompt.replace(/\r/g, " ").replace(/\n/g, " ").replace(/\s+/g, " ");
    if (bundleMetadata.remove_semicolons) {
        prompt = prompt.replace(/;/g, ",");
    }

    const wordCount = prompt.split(/\s+/).filter(Boolean).length;
    let framesAfterEos = wordCount <= 4 ? 3 : 1;
    if (bundleMetadata.model_recommended_frames_after_eos != null) {
        framesAfterEos = Number(bundleMetadata.model_recommended_frames_after_eos);
    }

    if (prompt && !/[A-ZÀ-Þ]/.test(prompt[0])) {
        prompt = prompt[0].toUpperCase() + prompt.slice(1);
    }
    if (prompt && /[0-9A-Za-zÀ-ÿ]/.test(prompt[prompt.length - 1])) {
        prompt += ".";
    }
    if (bundleMetadata.pad_with_spaces_for_short_inputs && wordCount < 5) {
        prompt = "        " + prompt;
    }

    return { text: prompt, framesAfterEos };
}

const SENTENCE_SPLIT_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

function splitTextIntoSentences(text) {
    const matches = text.match(SENTENCE_SPLIT_RE);
    if (!matches) return [];
    return matches.map((sentence) => sentence.trim()).filter(Boolean);
}

function splitTokenIdsIntoChunks(tokenIds, maxTokens) {
    const chunks = [];
    for (let i = 0; i < tokenIds.length; i += maxTokens) {
        const chunkText = tokenizerProcessor.decodeIds(tokenIds.slice(i, i + maxTokens)).trim();
        if (chunkText) {
            chunks.push(chunkText);
        }
    }
    return chunks;
}

function splitIntoBestSentences(text) {
    const prepared = prepareTextPrompt(text);
    if (!prepared.text) {
        return { chunks: [], framesAfterEos: prepared.framesAfterEos };
    }

    const sentences = splitTextIntoSentences(prepared.text);
    if (!sentences.length) {
        return { chunks: [prepared.text], framesAfterEos: prepared.framesAfterEos };
    }

    const chunks = [];
    let currentChunk = "";

    for (const sentenceText of sentences) {
        const sentenceTokenIds = tokenizerProcessor.encodeIds(sentenceText);
        const sentenceTokens = sentenceTokenIds.length;

        if (sentenceTokens > currentMaxTokenPerChunk) {
            if (currentChunk) {
                chunks.push(currentChunk.trim());
                currentChunk = "";
            }
            const splitChunks = splitTokenIdsIntoChunks(sentenceTokenIds, currentMaxTokenPerChunk);
            for (const splitChunk of splitChunks) {
                if (splitChunk) {
                    chunks.push(splitChunk.trim());
                }
            }
            continue;
        }

        if (!currentChunk) {
            currentChunk = sentenceText;
            continue;
        }

        const combined = `${currentChunk} ${sentenceText}`;
        const combinedTokens = tokenizerProcessor.encodeIds(combined).length;
        if (combinedTokens > currentMaxTokenPerChunk) {
            chunks.push(currentChunk.trim());
            currentChunk = sentenceText;
        } else {
            currentChunk = combined;
        }
    }

    if (currentChunk) {
        chunks.push(currentChunk.trim());
    }

    return { chunks, framesAfterEos: prepared.framesAfterEos };
}

function precomputeFlowBuffers() {
    stTensors = [];
    const dt = 1.0 / LSD_STEPS;
    for (let step = 0; step < LSD_STEPS; step++) {
        const s = step / LSD_STEPS;
        const t = s + dt;
        stTensors.push({
            s: createTensor("float32", new Float32Array([s]), [1, 1]),
            t: createTensor("float32", new Float32Array([t]), [1, 1]),
        });
    }
}

async function runGenerationPipeline(chunks, framesAfterEos) {
    let mimiState = initStateFromManifest(bundleMetadata.mimi_state_manifest);
    const emptySeq = createTensor("float32", new Float32Array(0), [1, 0, currentLatentDim]);
    const emptyTextEmb = createTensor("float32", new Float32Array(0), [1, 0, currentConditioningDim]);
    const baseFlowState = voiceState;
    let flowLmState = cloneState(baseFlowState);

    const firstChunkFrames = 3;
    const normalChunkFrames = 12;
    let isFirstAudioChunk = true;

    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {

        if (chunkIdx > 0) {
            flowLmState = cloneState(baseFlowState);
        }
        if (chunkIdx > 0) {
            mimiState = initStateFromManifest(bundleMetadata.mimi_state_manifest);
        }

        const chunkText = chunks[chunkIdx];
        const tokenIds = tokenizerProcessor.encodeIds(chunkText);
        const textInput = createTensor(
            "int64",
            BigInt64Array.from(tokenIds.map((token) => BigInt(token))),
            [1, tokenIds.length]
        );

        let textEmb = (await textConditionerSession.run({ token_ids: textInput }))[textConditionerSession.outputNames[0]];
        if (textEmb.dims.length === 2) {
            textEmb = createTensor("float32", new Float32Array(textEmb.data), [1, textEmb.dims[0], textEmb.dims[1]]);
        }

        const condResult = await flowLmMainSession.run({
            sequence: emptySeq,
            text_embeddings: textEmb,
            ...flowLmState,
        });
        updateStateFromManifestOutputs(flowLmState, condResult, bundleMetadata.flow_lm_state_manifest);

        const chunkLatents = [];
        let chunkDecodedFrames = 0;
        let currentLatent = createTensor("float32", new Float32Array(currentLatentDim).fill(NaN), [1, 1, currentLatentDim]);
        let eosStep = null;
        let chunkEnded = false;

        for (let step = 0; step < MAX_FRAMES; step++) {
            const arResult = await flowLmMainSession.run({
                sequence: currentLatent,
                text_embeddings: emptyTextEmb,
                ...flowLmState,
            });

            const conditioning = arResult.conditioning;
            const eosLogit = arResult.eos_logit.data[0];
            const isEos = eosLogit > -4.0;
            if (isEos && eosStep == null) {
                eosStep = step;
            }
            const shouldStop = eosStep != null && step >= eosStep + framesAfterEos;

            const temperature = 0.7;
            const std = Math.sqrt(temperature);
            const latentData = new Float32Array(currentLatentDim);
            for (let i = 0; i < currentLatentDim; i++) {
                let u = 0;
                let v = 0;
                while (u === 0) u = Math.random();
                while (v === 0) v = Math.random();
                latentData[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
            }

            const dt = 1.0 / LSD_STEPS;
            for (let lsdIndex = 0; lsdIndex < LSD_STEPS; lsdIndex++) {
                const flowResult = await flowLmFlowSession.run({
                    c: conditioning,
                    s: stTensors[lsdIndex].s,
                    t: stTensors[lsdIndex].t,
                    x: createTensor("float32", latentData, [1, currentLatentDim]),
                });
                const flowDir = flowResult.flow_dir.data;
                for (let i = 0; i < currentLatentDim; i++) {
                    latentData[i] += flowDir[i] * dt;
                }
            }

            chunkLatents.push(new Float32Array(latentData));
            currentLatent = createTensor("float32", latentData, [1, 1, currentLatentDim]);
            updateStateFromManifestOutputs(flowLmState, arResult, bundleMetadata.flow_lm_state_manifest);

            const pending = chunkLatents.length - chunkDecodedFrames;
            let decodeSize = 0;
            if (shouldStop || step === MAX_FRAMES - 1) {
                decodeSize = pending;
            } else if (isFirstAudioChunk && pending >= firstChunkFrames) {
                decodeSize = firstChunkFrames;
            } else if (pending >= normalChunkFrames) {
                decodeSize = normalChunkFrames;
            }

            if (decodeSize > 0) {
                const decodeLatents = new Float32Array(decodeSize * currentLatentDim);
                for (let frame = 0; frame < decodeSize; frame++) {
                    decodeLatents.set(chunkLatents[chunkDecodedFrames + frame], frame * currentLatentDim);
                }

                const decodeResult = await mimiDecoderSession.run({
                    latent: createTensor("float32", decodeLatents, [1, decodeSize, currentLatentDim]),
                    ...mimiState,
                });

                for (const entry of bundleMetadata.mimi_state_manifest) {
                    mimiState[entry.input_name] = decodeResult[entry.output_name];
                }

                chunkDecodedFrames += decodeSize;
                const audioFloat32 = new Float32Array(decodeResult[mimiDecoderSession.outputNames[0]].data);

                emitAudio(audioFloat32);

                isFirstAudioChunk = false;
            }

            if (shouldStop) {
                chunkEnded = true;
                break;
            }
        }

        if (chunkEnded && chunkIdx < chunks.length - 1) {
            const gapSamples = Math.max(1, Math.floor(CHUNK_GAP_SEC * currentSampleRate));
            const silence = new Float32Array(gapSamples);
            emitAudio(silence);
        }
    }

}
