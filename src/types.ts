export type STTLanguage = 'en' | 'ar';
export type STTModel = 'moonshine-tiny' | 'moonshine-tiny-ar' | 'whisper-base';
export type TTSLanguage = 'en' | 'ar' | 'auto';
export type ModelLoadStatus = 'unloaded' | 'loading' | 'ready' | 'error';

export interface TranscriptEntry {
  timestamp: number;
  text: string;
  chunks?: Array<{ timestamp: [number, number]; text: string }>;
}

export interface MeetingSession {
  id: string;
  title: string;
  platform: 'meet' | 'zoom' | 'teams' | 'other';
  source_url: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  language: TTSLanguage;
  status: 'recording' | 'completed' | 'error';
  stt_model: string;
  participant_count: number;
  word_count: number;
}

export interface TranscriptChunk {
  id: string;
  session_id: string;
  text: string;
  start_time_ms: number;
  end_time_ms: number;
  sequence_index: number;
  confidence?: number;
}

export interface VoiceSettings {
  stt_model: STTModel;
  tts_voice_en: string;
  tts_voice_ar: string;
  language: TTSLanguage;
  tts_enabled: boolean;
  stt_enabled: boolean;
  vad_threshold: number;
  vad_silence_ms: number;
  chunk_length_s: number;
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  stt_model: 'moonshine-tiny',
  tts_voice_en: 'af_heart',
  tts_voice_ar: 'ar_JO-kareem-medium',
  language: 'auto',
  tts_enabled: true,
  stt_enabled: true,
  vad_threshold: 0.01,
  vad_silence_ms: 1500,
  chunk_length_s: 10,
};

export interface ModelProgressEvent {
  type: 'model_progress';
  lang: STTLanguage;
  status: ModelLoadStatus;
  progress?: number;
  message?: string;
}

export interface TranscriptUpdateEvent {
  type: 'transcript_update';
  text: string;
  timestamp: number;
  sessionId?: string;
}

export interface VoiceInputCompleteEvent {
  type: 'voice_input_complete';
  text: string;
}
