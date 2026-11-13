export interface Voice {
  id: string; name: string; createdAt: string; recordingSeconds: number; usedSeconds: number; creationMs: number;
  parentId?: string; method?: string;
  transcript?: string;
  fit?: { before: FitScore; after: FitScore; heldOutBefore: FitScore; heldOutAfter: FitScore; elapsedSeconds: number; generalized: boolean; heldOutText: string; focus?: 'voice' | 'pronunciation'; algorithm?: 'linear'; components?: number; effectiveRank?: number; solveMs?: number; iterations?: number; finalError?: number };
}
interface FitScore { speakerCosine: number; loss: number }
export interface Health {
  ready: boolean; busy: boolean; error: string; model: string; device: string; refinementReady?: boolean;
  supportsPronunciation?: boolean;
  supportsLinear?: boolean; linearReady?: boolean;
  refinement?: { id: string; voiceId: string; stage: string; step: number; total: number; before?: number; best?: number; elapsedSeconds?: number };
}
export interface Speech { id: string; voiceId: string; url: string; durationSeconds: number; generationMs: number }

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/${path}`, { ...options, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The voice operation failed. Please try again.');
  }
  return data as T;
}
