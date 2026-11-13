export interface Voice {
  id: string; name: string; createdAt: string; recordingSeconds: number; usedSeconds: number; creationMs: number;
}
export interface Health { ready: boolean; busy: boolean; error: string; model: string; device: string }
export interface Speech { id: string; voiceId: string; url: string; durationSeconds: number; generationMs: number }

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/${path}`, { ...options, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The voice operation failed. Please try again.');
  }
  return data as T;
}
