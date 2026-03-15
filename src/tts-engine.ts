import { pipeline, env } from '@huggingface/transformers';
import type { TTSLanguage, ModelLoadStatus, ModelProgressEvent } from './types';

env.cacheDir = 'indexeddb://orchestra-models';
env.allowLocalModels = false;

async function purgeModelCache(): Promise<void> {
  try {
    await new Promise<void>((resolve, reject) => {
      const req = indexedDB.deleteDatabase('orchestra-models');
      req.onsuccess = () => { console.log('[OrchestraTTS] Model cache purged'); resolve(); };
      req.onerror = () => reject(req.error);
      req.onblocked = () => { console.warn('[OrchestraTTS] Cache purge blocked'); resolve(); };
    });
  } catch (err) {
    console.warn('[OrchestraTTS] Cache purge failed (non-fatal):', err);
  }
}

/** Wails v3: call Go methods via direct HTTP POST to /wails/runtime */
async function wailsCall<T>(method: string, ...args: unknown[]): Promise<T> {
  const id = Math.random().toString(36).slice(2) + Date.now().toString(36);
  const resp = await fetch('/wails/runtime', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      object: 0,  // objectNames.Call
      method: 0,  // CallBinding
      args: { 'call-id': id, methodName: method, args },
    }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  const ct = resp.headers.get('Content-Type') ?? '';
  return (ct.includes('application/json') ? resp.json() : resp.text()) as Promise<T>;
}

function hasWailsTTS(): boolean {
  // window._wails is injected synchronously by Wails before the page loads.
  try {
    return typeof window !== 'undefined' && typeof (window as any)._wails !== 'undefined';
  } catch {
    return false;
  }
}

export type TTSBackend = 'native' | 'kokoro' | 'webspeech';

export type TTSEventListener = (event: ModelProgressEvent) => void;

export class OrchestraTTS {
  private mmsArabicEngine: any = null;
  private audioContext: AudioContext | null = null;
  private listeners: TTSEventListener[] = [];
  private isSpeaking = false;
  private webSpeechVoice: string | null = null;
  private backend: TTSBackend = 'native';

  onEvent(listener: TTSEventListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private emit(event: ModelProgressEvent) {
    this.listeners.forEach(l => l(event));
  }

  private getAudioContext(): AudioContext {
    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /** Set the active TTS backend. */
  setBackend(b: TTSBackend) {
    this.backend = b;
  }

  getBackend(): TTSBackend {
    return this.backend;
  }

  // ── Loading ──────────────────────────────────────────────────────────────

  /**
   * "Load" for native/webspeech — just checks availability and emits ready.
   * For kokoro, no pre-load needed; Kokoro loads on first Speak call.
   */
  async loadKokoro(): Promise<void> {
    this.emit({ type: 'model_progress', lang: 'en', status: 'loading', progress: 50 });
    await this._waitForVoices();
    this.emit({ type: 'model_progress', lang: 'en', status: 'ready', progress: 100 });
  }

  private _waitForVoices(): Promise<void> {
    return new Promise((resolve) => {
      if (typeof window === 'undefined' || !window.speechSynthesis) {
        resolve();
        return;
      }
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) { resolve(); return; }
      const onChanged = () => { window.speechSynthesis.removeEventListener('voiceschanged', onChanged); resolve(); };
      window.speechSynthesis.addEventListener('voiceschanged', onChanged);
      setTimeout(resolve, 1500);
    });
  }

  // ── Native voices (macOS / Windows via Wails Go binding) ────────────────

  /** Pick the best available Web Speech API English voice (fallback only). */
  private _getBestEnglishVoice(): SpeechSynthesisVoice | null {
    if (typeof window === 'undefined' || !window.speechSynthesis) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;

    if (this.webSpeechVoice) {
      const match = voices.find(v => v.name === this.webSpeechVoice);
      if (match) return match;
    }

    const enVoices = voices.filter(v => v.lang.startsWith('en'));

    const premium = enVoices.find(v => v.localService && (v.name.includes('Premium') || v.name.includes('Enhanced')));
    if (premium) return premium;

    const macHighQuality = ['Samantha', 'Alex', 'Karen', 'Daniel', 'Moira', 'Rishi', 'Tessa', 'Veena'];
    for (const name of macHighQuality) {
      const v = enVoices.find(v => v.localService && v.name.startsWith(name));
      if (v) return v;
    }

    const win = enVoices.find(v => v.localService && v.name.includes('Microsoft') && v.name.includes('Natural'));
    if (win) return win;
    const winAny = enVoices.find(v => v.localService && v.name.includes('Microsoft'));
    if (winAny) return winAny;

    const anyLocal = enVoices.find(v => v.localService);
    if (anyLocal) return anyLocal;

    return enVoices[0] ?? null;
  }

  /** Set preferred Web Speech voice by name (used for webspeech backend). */
  setVoice(name: string) {
    this.webSpeechVoice = name || null;
  }

  listVoices(): SpeechSynthesisVoice[] {
    if (typeof window === 'undefined' || !window.speechSynthesis) return [];
    return window.speechSynthesis.getVoices();
  }

  listEnglishVoices(): SpeechSynthesisVoice[] {
    return this.listVoices().filter(v => v.lang.startsWith('en'));
  }

  logAvailableVoices(): void {
    const voices = this.listVoices();
    const selected = this._getBestEnglishVoice();
    console.log('[OrchestraTTS] Available voices:', voices.map(v => `${v.name} (${v.lang}) local=${v.localService}`));
    console.log('[OrchestraTTS] Selected voice:', selected?.name ?? 'none');
  }

  // ── English TTS dispatch ────────────────────────────────────────────────

  async speakEnglish(
    text: string,
    options: { voice?: string; speed?: number } = {}
  ): Promise<void> {
    if (options.voice) this.webSpeechVoice = options.voice;

    const backend = this.backend;

    // Native backend: use Go/Wails NativeTTSService.Speak (macOS say / Windows SAPI)
    if (backend === 'native' && hasWailsTTS()) {
      this.isSpeaking = true;
      try {
        await wailsCall<void>('github.com/orchestra-mcp/framework/app/desktop.NativeTTSService.Speak', text, this.webSpeechVoice ?? '', 'native');
      } catch (err) {
        console.warn('[OrchestraTTS] Native speak failed, falling back to Web Speech:', err);
        await this._speakWebSpeech(text, 'en-US', options.speed);
      } finally {
        this.isSpeaking = false;
      }
      return;
    }

    // Kokoro backend: use SpeakKokoroSync → get base64 PCM → play via AudioContext
    if (backend === 'kokoro' && hasWailsTTS()) {
      this.isSpeaking = true;
      try {
        const b64 = await wailsCall<string>('github.com/orchestra-mcp/framework/app/desktop.NativeTTSService.SpeakKokoroSync', text, this.webSpeechVoice ?? '');
        if (b64) {
          const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const view = new DataView(raw.buffer);
          const sampleRate = view.getUint32(0, true);
          const sampleCount = (raw.length - 4) / 4;
          const samples = new Float32Array(sampleCount);
          for (let i = 0; i < sampleCount; i++) {
            samples[i] = view.getFloat32(4 + i * 4, true);
          }
          await this.playAudioData(samples, sampleRate);
        }
      } catch (err) {
        console.warn('[OrchestraTTS] Kokoro failed, falling back to native/Web Speech:', err);
        await this._speakWebSpeech(text, 'en-US', options.speed);
      } finally {
        this.isSpeaking = false;
      }
      return;
    }

    // Web Speech API fallback (always works in browser/Wails but quality varies)
    await this._speakWebSpeech(text, 'en-US', options.speed);
  }

  private _speakWebSpeech(text: string, lang = 'en-US', rate = 1.0): Promise<void> {
    return new Promise((resolve) => {
      if (typeof window === 'undefined' || !window.speechSynthesis) {
        resolve();
        return;
      }
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = lang;
      utter.rate = rate ?? 1.0;

      const voice = lang.startsWith('en') ? this._getBestEnglishVoice() : null;
      if (voice) {
        utter.voice = voice;
        console.log('[OrchestraTTS] Web Speech voice:', voice.name, `(${voice.lang})`);
      }

      this.isSpeaking = true;
      utter.onend = () => { this.isSpeaking = false; resolve(); };
      utter.onerror = () => { this.isSpeaking = false; resolve(); };
      window.speechSynthesis.speak(utter);
    });
  }

  // ── Arabic: MMS-TTS ───────────────────────────────────────────────────

  async loadPiperArabic(): Promise<void> {
    if (this.mmsArabicEngine) return;

    this.emit({ type: 'model_progress', lang: 'ar', status: 'loading', progress: 0 });

    this.mmsArabicEngine = await pipeline(
      'text-to-speech',
      'Xenova/mms-tts-ara',
      {
        dtype: 'q8',
        progress_callback: (p: any) => {
          const progress = p.progress ?? (p.status === 'done' ? 100 : 0);
          this.emit({ type: 'model_progress', lang: 'ar', status: 'loading', progress, message: p.file });
        },
      }
    );

    this.emit({ type: 'model_progress', lang: 'ar', status: 'ready', progress: 100 });
  }

  async speakArabic(
    text: string,
    options: { speed?: number } = {}
  ): Promise<void> {
    if (!this.mmsArabicEngine) {
      await this._speakWebSpeech(text, 'ar-SA');
      return;
    }

    this.isSpeaking = true;
    try {
      const sentences = this.splitIntoSentences(text);
      for (const sentence of sentences) {
        if (!sentence.trim()) continue;
        try {
          const output = await (this.mmsArabicEngine as any)(sentence);
          await this.playAudioData(output.audio, output.sampling_rate ?? 16000);
        } catch (err) {
          console.warn('[OrchestraTTS] MMS Arabic failed, purging cache and falling back:', err);
          this.mmsArabicEngine = null;
          await purgeModelCache();
          await this._speakWebSpeech(sentence, 'ar-SA');
          return;
        }
      }
    } finally {
      this.isSpeaking = false;
    }
  }

  // ── Unified API ──────────────────────────────────────────────────────────

  detectLanguage(text: string): 'ar' | 'en' {
    return /[\u0600-\u06FF\u0750-\u077F]/.test(text) ? 'ar' : 'en';
  }

  async speak(
    text: string,
    lang: TTSLanguage = 'auto',
    options: { voice?: string; speed?: number; onChunk?: (audio: any) => void } = {}
  ): Promise<void> {
    const resolved = lang === 'auto' ? this.detectLanguage(text) : lang;

    if (resolved === 'ar') {
      return this.speakArabic(text, options);
    } else {
      return this.speakEnglish(text, options);
    }
  }

  stop(): void {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (hasWailsTTS()) {
      wailsCall<void>('github.com/orchestra-mcp/framework/app/desktop.NativeTTSService.Stop').catch(() => {});
    }
    this.audioContext?.close().catch(() => {});
    this.audioContext = null;
    this.isSpeaking = false;
  }

  get speaking(): boolean {
    return this.isSpeaking;
  }

  // ── Audio playback helper ────────────────────────────────────────────────

  private async playAudioData(data: Float32Array | ArrayBuffer, sampleRate = 24000): Promise<void> {
    const ctx = this.getAudioContext();

    const floatData =
      data instanceof Float32Array
        ? data
        : new Float32Array(data as ArrayBuffer);

    const buffer = ctx.createBuffer(1, floatData.length, sampleRate);
    buffer.getChannelData(0).set(floatData);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    return new Promise<void>(resolve => {
      source.onended = () => resolve();
      source.start();
    });
  }

  private splitIntoSentences(text: string): string[] {
    const parts = text.match(/[^.!?؟\n]+[.!?؟\n]*/g);
    return parts ?? [text];
  }
}
