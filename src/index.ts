// @orchestra-mcp/voice — Local STT + TTS engine
// All models run in-browser via WebGPU/WASM. Audio never leaves the device.

export { OrchestraSTT } from './stt-engine';
export { OrchestraTTS } from './tts-engine';
export {
  saveMeetingSession,
  updateMeetingSession,
  saveTranscriptChunk,
  listSessions,
  getSessionWithChunks,
  deleteSession,
  getFullTranscriptText,
} from './db';
export {
  DEFAULT_VOICE_SETTINGS,
} from './types';
export type {
  STTLanguage,
  TTSLanguage,
  STTModel,
  ModelLoadStatus,
  TranscriptEntry,
  MeetingSession,
  TranscriptChunk,
  VoiceSettings,
  ModelProgressEvent,
  TranscriptUpdateEvent,
  VoiceInputCompleteEvent,
} from './types';
export type { STTEventListener } from './stt-engine';
export type { TTSEventListener, TTSBackend } from './tts-engine';
