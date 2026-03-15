import { pipeline, env } from '@huggingface/transformers';
import type { STTLanguage, TranscriptEntry, ModelLoadStatus, ModelProgressEvent, TranscriptUpdateEvent, VoiceInputCompleteEvent } from './types';

// Cache models in IndexedDB — first load ~26MB, subsequent loads instant
env.cacheDir = 'indexeddb://orchestra-models';
env.allowLocalModels = false;

/**
 * Purge the entire model cache from IndexedDB.
 * Called when an ONNX inference error indicates a corrupt cache entry.
 * The models will be re-downloaded on the next loadModel() call.
 */
async function purgeModelCache(): Promise<void> {
  try {
    // The transformers.js cache DB name matches the cacheDir setting (without the scheme)
    await new Promise<void>((resolve, reject) => {
      const req = indexedDB.deleteDatabase('orchestra-models');
      req.onsuccess = () => { console.log('[OrchestraSTT] Model cache purged'); resolve(); };
      req.onerror = () => reject(req.error);
      req.onblocked = () => { console.warn('[OrchestraSTT] Cache purge blocked — close other tabs'); resolve(); };
    });
  } catch (err) {
    console.warn('[OrchestraSTT] Cache purge failed (non-fatal):', err);
  }
}

const MODEL_IDS: Record<STTLanguage, string> = {
  en: 'Xenova/whisper-tiny.en',   // ~40MB, English-only, fast
  ar: 'Xenova/whisper-base',       // ~139MB, multilingual, much better Arabic accuracy
};

export type STTEventListener = (
  event: ModelProgressEvent | TranscriptUpdateEvent | VoiceInputCompleteEvent
) => void;

export class OrchestraSTT {
  private models: Partial<Record<STTLanguage, any>> = {};
  private activeModel: any = null;
  private activeLang: STTLanguage = 'en';
  private transcript: TranscriptEntry[] = [];
  private listeners: STTEventListener[] = [];
  private audioContext: AudioContext | null = null;
  /** True when this instance created the AudioContext (owns it); false when shared externally. */
  private ownsAudioContext = false;
  private isCapturing = false;
  private scriptProcessor: ScriptProcessorNode | null = null;

  onEvent(listener: STTEventListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private emit(event: ModelProgressEvent | TranscriptUpdateEvent | VoiceInputCompleteEvent) {
    this.listeners.forEach(l => l(event));
  }

  async loadModel(lang: STTLanguage = 'en'): Promise<void> {
    if (this.models[lang]) {
      this.activeModel = await this.models[lang]!;
      this.activeLang = lang;
      return;
    }

    const modelId = MODEL_IDS[lang];

    this.emit({ type: 'model_progress', lang, status: 'loading', progress: 0 });

    // Try WebGPU first, fall back to WASM
    let device: 'webgpu' | 'wasm' = 'wasm';
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) device = 'webgpu';
      } catch {
        // WebGPU not available
      }
    }

    const modelPromise = pipeline(
      'automatic-speech-recognition',
      modelId,
      {
        device,
        dtype: device === 'webgpu' ? 'fp32' : 'q8',
        progress_callback: (p: any) => {
          const progress = p.progress ?? (p.status === 'done' ? 100 : 0);
          this.emit({ type: 'model_progress', lang, status: 'loading', progress, message: p.file });
        },
      }
    );

    this.models[lang] = modelPromise as any;
    this.activeModel = await modelPromise;
    this.activeLang = lang;

    this.emit({ type: 'model_progress', lang, status: 'ready', progress: 100 });
  }

  async transcribe(
    audioFloat32: Float32Array,
    options: { language?: STTLanguage; chunk_length_s?: number; stride_length_s?: number } = {}
  ): Promise<TranscriptEntry> {
    if (!this.activeModel) {
      throw new Error('No STT model loaded. Call loadModel() first.');
    }

    const lang = options.language ?? this.activeLang;

    // English-only model (.en suffix) must NOT receive language/task params.
    // Multilingual model: don't force a language — let Whisper auto-detect.
    const isMultilingual = !MODEL_IDS[lang].endsWith('.en');

    const inferenceOptions = {
      return_timestamps: true,
      chunk_length_s: options.chunk_length_s ?? 30,
      stride_length_s: options.stride_length_s ?? 5,
      ...(isMultilingual ? { task: 'transcribe' } : {}),
    };

    let result: any;
    try {
      result = await (this.activeModel as any)(audioFloat32, inferenceOptions);
    } catch (err) {
      // ONNX runtime error (e.g. 65945296) — cache is corrupt; purge IndexedDB and reload
      console.warn('[OrchestraSTT] Inference failed (corrupt cache?), purging and reloading:', err);
      delete this.models[lang];
      this.activeModel = null;
      await purgeModelCache();
      try {
        await this.loadModel(lang);
        result = await (this.activeModel as any)(audioFloat32, inferenceOptions);
      } catch (retryErr) {
        console.error('[OrchestraSTT] Retry also failed after cache purge:', retryErr);
        throw retryErr;
      }
    }

    const entry: TranscriptEntry = {
      timestamp: Date.now(),
      text: result.text as string,
      chunks: result.chunks,
    };

    this.transcript.push(entry);
    return entry;
  }

  async startCapture(streamId: string, chunkLengthS = 10): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // @ts-ignore — Chrome-specific constraint
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId,
        },
      },
    });

    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(stream);

    // Use AudioWorklet if available, fall back to ScriptProcessor
    try {
      await this.audioContext.audioWorklet.addModule(
        new URL('./audio-chunk-processor.js', import.meta.url).href
      );
      const processor = new AudioWorkletNode(this.audioContext, 'audio-chunk-processor');
      let buffer = new Float32Array(0);
      const CHUNK_SIZE = 16000 * chunkLengthS;
      const OVERLAP = 3200; // 200ms

      processor.port.onmessage = async (e: MessageEvent<Float32Array>) => {
        const newData = e.data;
        const combined = new Float32Array(buffer.length + newData.length);
        combined.set(buffer);
        combined.set(newData, buffer.length);
        buffer = combined;

        if (buffer.length >= CHUNK_SIZE) {
          const chunk = buffer.slice(0, CHUNK_SIZE);
          buffer = buffer.slice(CHUNK_SIZE - OVERLAP);

          try {
            const result = await this.transcribe(chunk);
            this.emit({ type: 'transcript_update', text: result.text, timestamp: result.timestamp });
          } catch (err) {
            console.error('[OrchestraSTT] transcribe error:', err);
          }
        }
      };

      source.connect(processor);
      processor.connect(this.audioContext.destination);
    } catch {
      // ScriptProcessor fallback for offscreen documents
      this._startScriptProcessor(source, chunkLengthS);
    }

    this.isCapturing = true;
  }

  private _startScriptProcessor(source: MediaStreamAudioSourceNode, chunkLengthS: number) {
    const CHUNK_SIZE = 16000 * chunkLengthS;
    const OVERLAP = 3200;
    let buffer = new Float32Array(0);

    const processor = this.audioContext!.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = async (e: AudioProcessingEvent) => {
      const newData = e.inputBuffer.getChannelData(0);
      const combined = new Float32Array(buffer.length + newData.length);
      combined.set(buffer);
      combined.set(newData, buffer.length);
      buffer = combined;

      if (buffer.length >= CHUNK_SIZE) {
        const chunk = buffer.slice(0, CHUNK_SIZE);
        buffer = buffer.slice(CHUNK_SIZE - OVERLAP);

        try {
          const result = await this.transcribe(chunk);
          this.emit({ type: 'transcript_update', text: result.text, timestamp: result.timestamp });
        } catch (err) {
          console.error('[OrchestraSTT] transcribe error:', err);
        }
      }
    };

    source.connect(processor);
    processor.connect(this.audioContext!.destination);
  }

  async startMicCapture(options: {
    threshold?: number;
    silenceMs?: number;
    stream?: MediaStream;
    audioContext?: AudioContext;
  } = {}): Promise<void> {
    const SILENCE_THRESHOLD = options.threshold ?? 0.01;
    const SILENCE_DURATION = options.silenceMs ?? 800;

    const stream = options.stream ?? await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // Reuse caller's AudioContext if provided; otherwise create one at the
    // stream's native sample rate so the browser doesn't need to resample.
    this.ownsAudioContext = !options.audioContext;
    const audioCtx = options.audioContext ?? new AudioContext();
    this.audioContext = audioCtx;

    // Resume context if suspended (browser autoplay policy)
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const source = audioCtx.createMediaStreamSource(stream);

    let silenceTimer: ReturnType<typeof setTimeout> | null = null;
    let audioBuffer: Float32Array[] = [];
    let audioBufferLen = 0;
    let speaking = false;

    // ScriptProcessor still works in all browsers; AudioWorklet would be
    // better but requires a separate file. Use 2048 frames for lower latency.
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);

    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const data = e.inputBuffer.getChannelData(0);
      // Use max amplitude instead of RMS — more responsive to transients
      let maxAmp = 0;
      for (let i = 0; i < data.length; i++) {
        const abs = Math.abs(data[i]);
        if (abs > maxAmp) maxAmp = abs;
      }

      if (maxAmp > SILENCE_THRESHOLD) {
        speaking = true;
        if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }

        // Copy frame into buffer
        const copy = new Float32Array(data);
        audioBuffer.push(copy);
        audioBufferLen += copy.length;
      } else if (speaking && !silenceTimer) {
        silenceTimer = setTimeout(async () => {
          speaking = false;
          silenceTimer = null;

          if (audioBufferLen < 8000) { // ignore < 500ms noise
            audioBuffer = [];
            audioBufferLen = 0;
            return;
          }

          // Flatten chunks into single Float32Array
          const flat = new Float32Array(audioBufferLen);
          let offset = 0;
          for (const chunk of audioBuffer) { flat.set(chunk, offset); offset += chunk.length; }
          audioBuffer = [];
          audioBufferLen = 0;

          // Downsample to 16kHz if the AudioContext is at a different rate
          const inputRate = audioCtx.sampleRate;
          const pcm16k = inputRate !== 16000 ? this._resampleTo16k(flat, inputRate) : flat;

          console.log('[OrchestraSTT] Transcribing', pcm16k.length, 'samples at 16kHz');
          try {
            const result = await this.transcribe(pcm16k);
            console.log('[OrchestraSTT] Transcript:', result.text);
            if (result.text.trim()) {
              this.emit({ type: 'voice_input_complete', text: result.text.trim() });
            }
          } catch (err) {
            console.error('[OrchestraSTT] mic transcribe error:', err);
          }
        }, SILENCE_DURATION);
      }
    };

    source.connect(processor);
    // Must connect to destination for onaudioprocess to fire
    processor.connect(audioCtx.destination);
    this.scriptProcessor = processor;
    this.isCapturing = true;
    console.log('[OrchestraSTT] Mic capture started, ctx rate:', audioCtx.sampleRate, 'threshold:', SILENCE_THRESHOLD);
  }

  private _resampleTo16k(input: Float32Array, fromRate: number): Float32Array {
    const ratio = fromRate / 16000;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      out[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }
    return out;
  }

  stopCapture(): void {
    // Disconnect the ScriptProcessor to stop processing (works even on shared context)
    if (this.scriptProcessor) {
      try { this.scriptProcessor.disconnect(); } catch { /* already disconnected */ }
      this.scriptProcessor = null;
    }
    // Only close the AudioContext if this instance created it.
    // If it was passed in from outside (shared with waveform analyser), leave it open.
    if (this.ownsAudioContext && this.audioContext) {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
    this.ownsAudioContext = false;
    this.isCapturing = false;
  }

  getFullTranscript(): string {
    return this.transcript.map(t => t.text).join(' ');
  }

  getTranscriptEntries(): TranscriptEntry[] {
    return [...this.transcript];
  }

  clear(): void {
    this.transcript = [];
  }

  get capturing(): boolean {
    return this.isCapturing;
  }
}
