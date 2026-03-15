import { openDB, type IDBPDatabase } from 'idb';
import type { MeetingSession, TranscriptChunk } from './types';

const DB_NAME = 'orchestra-voice';
const DB_VERSION = 1;

let _db: IDBPDatabase | null = null;

async function getDb(): Promise<IDBPDatabase> {
  if (_db) return _db;

  _db = await openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      // MeetingSession store
      if (!db.objectStoreNames.contains('meeting_sessions')) {
        const sessions = db.createObjectStore('meeting_sessions', { keyPath: 'id' });
        sessions.createIndex('started_at', 'started_at');
        sessions.createIndex('platform', 'platform');
        sessions.createIndex('status', 'status');
      }

      // TranscriptChunk store
      if (!db.objectStoreNames.contains('transcript_chunks')) {
        const chunks = db.createObjectStore('transcript_chunks', { keyPath: 'id' });
        chunks.createIndex('session_id', 'session_id');
        chunks.createIndex('sequence_index', 'sequence_index');
      }
    },
  });

  return _db;
}

export async function saveMeetingSession(session: MeetingSession): Promise<void> {
  const db = await getDb();
  await db.put('meeting_sessions', session);
}

export async function updateMeetingSession(
  id: string,
  updates: Partial<MeetingSession>
): Promise<void> {
  const db = await getDb();
  const existing = await db.get('meeting_sessions', id);
  if (existing) {
    await db.put('meeting_sessions', { ...existing, ...updates });
  }
}

export async function saveTranscriptChunk(chunk: TranscriptChunk): Promise<void> {
  const db = await getDb();
  await db.put('transcript_chunks', chunk);
}

export async function listSessions(): Promise<MeetingSession[]> {
  const db = await getDb();
  const sessions = await db.getAll('meeting_sessions');
  return sessions.sort(
    (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
  );
}

export async function getSessionWithChunks(
  sessionId: string
): Promise<{ session: MeetingSession; chunks: TranscriptChunk[] } | null> {
  const db = await getDb();
  const session = await db.get('meeting_sessions', sessionId);
  if (!session) return null;

  const allChunks = await db.getAllFromIndex('transcript_chunks', 'session_id', sessionId);
  const chunks = allChunks.sort((a, b) => a.sequence_index - b.sequence_index);

  return { session, chunks };
}

export async function deleteSession(sessionId: string): Promise<void> {
  const db = await getDb();
  const tx = db.transaction(['meeting_sessions', 'transcript_chunks'], 'readwrite');

  await tx.objectStore('meeting_sessions').delete(sessionId);

  const chunksStore = tx.objectStore('transcript_chunks');
  const chunks = await chunksStore.index('session_id').getAllKeys(sessionId);
  await Promise.all(chunks.map(key => chunksStore.delete(key)));

  await tx.done;
}

export async function getFullTranscriptText(sessionId: string): Promise<string> {
  const db = await getDb();
  const allChunks = await db.getAllFromIndex('transcript_chunks', 'session_id', sessionId);
  const sorted = allChunks.sort((a, b) => a.sequence_index - b.sequence_index);
  return sorted.map((c: TranscriptChunk) => c.text).join(' ');
}
